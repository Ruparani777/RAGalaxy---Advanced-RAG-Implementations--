import os
import time
import sys
import traceback
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional

import pdfplumber
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)

# ---------------------------
# CONFIG
# ---------------------------
PDF_PATH = "Data/ECHOES OF HER LOVE.pdf"
COLLECTION_NAME = "FeedbackRAG_Documents"
DIM = 384
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = "llama-3.1-8b-instant"
TARGET_NS = 200_000

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("❌ ERROR: Set GROQ_API_KEY environment variable!")
    sys.exit(1)

# ---------------------------
# LATENCY UTILITIES
# ---------------------------
def format_time_ns(ns: int) -> str:
    if ns < 1_000:
        return f"{ns} ns"
    if ns < 1_000_000:
        return f"{ns/1_000:.3f} µs ({ns} ns)"
    if ns < 1_000_000_000:
        return f"{ns/1_000_000:.3f} ms ({ns} ns)"
    return f"{ns/1_000_000_000:.3f} s ({ns} ns)"

def timed_call(fn, *args, **kwargs):
    start = time.time_ns()
    result = fn(*args, **kwargs)
    elapsed = time.time_ns() - start
    return result, elapsed

def timer_ns(func):
    def wrapper(*args, **kwargs):
        start = time.time_ns()
        result = func(*args, **kwargs)
        elapsed = time.time_ns() - start
        print(f"⏱️  {func.__name__} time: {format_time_ns(elapsed)}")
        wrapper.last_elapsed_ns = elapsed
        return result
    wrapper.last_elapsed_ns = None
    return wrapper

class LatencyReport:
    def __init__(self):
        self.store = defaultdict(list)
    def add(self, component: str, ns: int):
        self.store[component].append(ns)
    def summary(self) -> Dict:
        out = {}
        for comp, vals in self.store.items():
            total = sum(vals)
            out[comp] = {
                "count": len(vals),
                "total_ns": total,
                "avg_ns": total // len(vals) if vals else 0,
                "min_ns": min(vals) if vals else 0,
                "max_ns": max(vals) if vals else 0
            }
        return out
    def pretty_print(self):
        s = self.summary()
        print("\n" + "="*70)
        print("LATENCY SUMMARY (nanoseconds)")
        print("="*70)
        for comp, stats in sorted(s.items(), key=lambda p: p[0]):
            print(f"\n📊 Component: {comp}")
            print(f"   Count:     {stats['count']}")
            print(f"   Total:     {format_time_ns(stats['total_ns'])}")
            print(f"   Average:   {format_time_ns(stats['avg_ns'])}")
            print(f"   Min:       {format_time_ns(stats['min_ns'])}")
            print(f"   Max:       {format_time_ns(stats['max_ns'])}")
        print("\n" + "="*70 + "\n")

latency_report = LatencyReport()

# ---------------------------
# PDF Loader
# ---------------------------
@timer_ns
def load_pdf(path: str) -> str:
    print(f"📄 Loading PDF: {path}")
    text = ""
    with pdfplumber.open(path) as pdf:
        for i, p in enumerate(pdf.pages):
            start_ns = time.time_ns()
            t = p.extract_text() or ""
            elapsed = time.time_ns() - start_ns
            latency_report.add("pdf_page_extract", elapsed)
            text += t + "\n"
    print(f"✅ Loaded PDF: {len(text)} characters from {len(pdf.pages)} pages")
    return text

# ---------------------------
# Chunker
# ---------------------------
@timer_ns
def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
    print("✂️  Chunking text...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    print(f"✅ Created {len(chunks)} chunks")
    return chunks

# ---------------------------
# Embeddings loader
# ---------------------------
@timer_ns
def load_embeddings(model_name: str = EMBED_MODEL) -> SentenceTransformer:
    print(f"🔢 Loading embeddings model: {model_name}")
    embedder = SentenceTransformer(model_name)
    print("✅ Embeddings model loaded")
    return embedder

# ---------------------------
# Milvus init (AUTO-ID primary key)
# ---------------------------
@timer_ns
def init_milvus(host: str, port: str, collection_name: str = COLLECTION_NAME, dim: int = DIM) -> Collection:
    print(f"🗃️  Initializing Milvus connection to {host}:{port}")
    connections.connect(host=host, port=port)

    # drop if exists
    try:
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            print(f"🗑️  Deleted existing collection '{collection_name}'")
    except Exception as e:
        print(f"⚠️  Collection check/delete: {e}")

    # primary key auto-id field first
    chunk_id_field = FieldSchema(name="chunk_id", dtype=DataType.INT64, is_primary=True, auto_id=True)
    text_field = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
    source_field = FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1024)
    embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)

    schema = CollectionSchema(fields=[chunk_id_field, text_field, source_field, embedding_field],
                              description="Feedback-Loop RAG document chunks")
    collection = Collection(name=collection_name, schema=schema)

    # create index and load
    index_params = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 1024}}
    try:
        collection.create_index(field_name="embedding", index_params=index_params)
    except Exception as e:
        print(f"⚠️  create_index: {e}")
    try:
        collection.load()
    except Exception as e:
        print(f"⚠️  load: {e}")

    print(f"✅ Milvus collection '{collection_name}' ready")
    return collection

# ---------------------------
# Insert chunks (no manual IDs)
# ---------------------------
@timer_ns
def insert_chunks(collection: Collection, embedder: SentenceTransformer, chunks: List[str]) -> None:
    print(f"⬆️  Inserting {len(chunks)} chunks into Milvus...")
    start = time.time_ns()
    vectors = embedder.encode(chunks, show_progress_bar=False)
    encode_time = time.time_ns() - start
    latency_report.add("embedding_encode_batch", encode_time)
    print(f"   ✅ Encoded in {format_time_ns(encode_time)}")

    texts = chunks
    sources = [f"chunk_{i}" for i in range(len(chunks))]
    embeddings = [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]

    start = time.time_ns()
    collection.insert([texts, sources, embeddings])
    insert_time = time.time_ns() - start
    latency_report.add("milvus_insert", insert_time)
    print(f"   ✅ Inserted {len(chunks)} vectors in {format_time_ns(insert_time)}")

    start = time.time_ns()
    collection.flush()
    flush_time = time.time_ns() - start
    latency_report.add("milvus_flush", flush_time)
    print(f"   ✅ Flushed collection ({format_time_ns(flush_time)})")

# ---------------------------
# Search
# ---------------------------
def search_milvus(collection: Collection, embedder: SentenceTransformer, query: str, limit: int = 4) -> Tuple[List[str], int]:
    start = time.time_ns()
    qvec = embedder.encode([query])[0]
    encode_time = time.time_ns() - start
    latency_report.add("query_embedding", encode_time)

    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    start = time.time_ns()
    try:
        results = collection.search(data=[qvec.tolist()], anns_field="embedding", param=search_params, limit=limit,
                                    output_fields=["text", "source", "chunk_id"])
        search_time = time.time_ns() - start
        latency_report.add("milvus_search", search_time)

        hits = []
        for hit in results[0]:
            try:
                ent = getattr(hit, "entity", None) or getattr(hit, "_fields", None) or {}
                if isinstance(ent, dict):
                    txt = ent.get("text", "")
                else:
                    txt = str(hit)
                hits.append(txt)
            except Exception:
                hits.append("")
    except Exception as e:
        search_time = time.time_ns() - start
        latency_report.add("milvus_search_error", search_time)
        print(f"⚠️ Milvus search failed: {e}")
        hits = []
    total_time = encode_time + (latency_report.store.get("milvus_search", [-1])[-1] if latency_report.store.get("milvus_search") else 0)
    return hits, total_time

# ---------------------------
# RAG with Feedback Loop
# ---------------------------
class FeedbackLoopRAG:
    """
    RAG with Feedback Loop System
    
    Workflow:
    1. RETRIEVE: Get relevant documents
    2. GENERATE: Create initial answer
    3. EVALUATE: Assess answer quality with multiple criteria
    4. FEEDBACK: Generate specific improvement suggestions
    5. REFINE: Use feedback to improve retrieval/generation
    6. REPEAT: Continue loop until quality threshold met
    
    The feedback loop enables continuous improvement through:
    - Relevance scoring
    - Completeness checking
    - Accuracy assessment
    - Context sufficiency evaluation
    - Query refinement suggestions
    """
    
    def __init__(self, llm, collection: Collection, embedder: SentenceTransformer):
        self.llm = llm
        self.collection = collection
        self.embedder = embedder
        self.feedback_history = []

    def _llm_invoke_timed(self, prompt: str, label: str) -> Tuple[str, int]:
        start = time.time_ns()
        try:
            response = self.llm.invoke(prompt)
            elapsed = time.time_ns() - start
            latency_report.add(label, elapsed)
            content = response.content if hasattr(response, "content") else str(response)
            return content, elapsed
        except Exception as e:
            elapsed = time.time_ns() - start
            latency_report.add(label + "_error", elapsed)
            print(f"LLM invoke for {label} failed: {e}")
            traceback.print_exc()
            return str(e), elapsed

    def retrieve_documents(self, query: str, k: int = 4, iteration: int = 1) -> Tuple[List[str], str, int]:
        """
        Retrieve relevant documents with optional query refinement based on feedback
        """
        print(f"   🔍 Retrieving documents (k={k})...")
        
        hits, elapsed = search_milvus(self.collection, self.embedder, query, k)
        context = "\n\n---\n\n".join([f"[Document {i+1}]\n{hit}" for i, hit in enumerate(hits)])
        
        print(f"   ✅ Retrieved {len(hits)} documents ({format_time_ns(elapsed)})")
        print(f"   📝 Total context length: {len(context)} characters")
        
        return hits, context, elapsed

    def generate_answer(self, query: str, context: str, feedback: Optional[str] = None, iteration: int = 1) -> Tuple[str, int]:
        """
        Generate answer from context, incorporating feedback from previous iteration
        """
        print(f"   ✍️  Generating answer...")
        
        if feedback and iteration > 1:
            prompt = f"""You are improving your previous answer based on feedback.

Original Question: {query}

Context from Retrieved Documents:
{context}

Previous Iteration Feedback:
{feedback}

Instructions:
1. Address the feedback points specifically
2. Use the retrieved context effectively
3. Ensure completeness and accuracy
4. Provide a well-structured, comprehensive answer

Improved Answer:"""
        else:
            prompt = f"""Answer the following question based on the retrieved context.

Question: {query}

Retrieved Context:
{context}

Instructions:
1. Answer based on the provided context
2. Be specific and detailed
3. If the context is insufficient, acknowledge it
4. Structure your response clearly

Answer:"""
        
        answer, elapsed = self._llm_invoke_timed(prompt, "llm_generate_answer")
        print(f"   ✅ Answer generated ({format_time_ns(elapsed)})")
        print(f"   📏 Answer length: {len(answer)} characters")
        
        return answer, elapsed

    def evaluate_answer(self, query: str, answer: str, context: str, iteration: int) -> Tuple[Dict[str, Any], int]:
        """
        Evaluate answer quality across multiple dimensions
        Returns scores and detailed assessment
        """
        print(f"   📊 Evaluating answer quality...")
        
        eval_prompt = f"""You are an expert evaluator. Assess this answer across multiple quality dimensions.

Question: {query}

Generated Answer:
{answer}

Available Context:
{context}

Evaluate the answer on the following criteria (score 1-10 for each):

1. RELEVANCE: Does the answer directly address the question?
2. COMPLETENESS: Are all aspects of the question covered?
3. ACCURACY: Is the information correct based on context?
4. CONTEXT_USAGE: How well does it utilize the retrieved context?
5. CLARITY: Is the answer clear and well-structured?

Provide your evaluation in this EXACT format:
RELEVANCE: [score]/10 - [brief reason]
COMPLETENESS: [score]/10 - [brief reason]
ACCURACY: [score]/10 - [brief reason]
CONTEXT_USAGE: [score]/10 - [brief reason]
CLARITY: [score]/10 - [brief reason]

OVERALL_SCORE: [average]/10
PASS: [YES/NO] (YES if overall >= 7.5, NO otherwise)

Evaluation:"""
        
        eval_text, elapsed = self._llm_invoke_timed(eval_prompt, "llm_evaluate")
        
        # Parse evaluation scores
        scores = self._parse_evaluation(eval_text)
        
        print(f"   ✅ Evaluation complete ({format_time_ns(elapsed)})")
        print(f"   📈 Overall Score: {scores['overall_score']:.1f}/10")
        print(f"   {'✅ PASS' if scores['pass_threshold'] else '❌ NEEDS IMPROVEMENT'}")
        
        return scores, elapsed

    def _parse_evaluation(self, eval_text: str) -> Dict[str, Any]:
        """Parse evaluation text into structured scores"""
        scores = {
            'relevance': 5.0,
            'completeness': 5.0,
            'accuracy': 5.0,
            'context_usage': 5.0,
            'clarity': 5.0,
            'overall_score': 5.0,
            'pass_threshold': False,
            'raw_evaluation': eval_text
        }
        
        lines = eval_text.split('\n')
        for line in lines:
            line = line.strip()
            if ':' in line:
                key_part = line.split(':')[0].upper().strip()
                value_part = line.split(':')[1].strip()
                
                # Extract numeric score
                score_str = value_part.split('/')[0].strip()
                try:
                    score = float(score_str)
                    if 'RELEVANCE' in key_part:
                        scores['relevance'] = score
                    elif 'COMPLETENESS' in key_part:
                        scores['completeness'] = score
                    elif 'ACCURACY' in key_part:
                        scores['accuracy'] = score
                    elif 'CONTEXT' in key_part or 'USAGE' in key_part:
                        scores['context_usage'] = score
                    elif 'CLARITY' in key_part:
                        scores['clarity'] = score
                    elif 'OVERALL' in key_part:
                        scores['overall_score'] = score
                except (ValueError, IndexError):
                    pass
            
            if 'PASS:' in line.upper():
                scores['pass_threshold'] = 'YES' in line.upper()
        
        # Calculate overall if not explicitly provided
        if scores['overall_score'] == 5.0:
            scores['overall_score'] = (
                scores['relevance'] + 
                scores['completeness'] + 
                scores['accuracy'] + 
                scores['context_usage'] + 
                scores['clarity']
            ) / 5.0
        
        # Set pass threshold if not explicitly provided
        if not scores['pass_threshold']:
            scores['pass_threshold'] = scores['overall_score'] >= 7.5
        
        return scores

    def generate_feedback(self, query: str, answer: str, context: str, evaluation: Dict[str, Any], iteration: int) -> Tuple[str, int]:
        """
        Generate specific, actionable feedback for improvement
        """
        print(f"   💬 Generating feedback...")
        
        scores_text = f"""
RELEVANCE: {evaluation['relevance']}/10
COMPLETENESS: {evaluation['completeness']}/10
ACCURACY: {evaluation['accuracy']}/10
CONTEXT_USAGE: {evaluation['context_usage']}/10
CLARITY: {evaluation['clarity']}/10
OVERALL: {evaluation['overall_score']:.1f}/10
"""
        
        feedback_prompt = f"""You are providing constructive feedback to improve an answer.

Question: {query}

Current Answer:
{answer}

Quality Scores:
{scores_text}

Task: Generate specific, actionable feedback to improve the answer. Focus on:

1. RETRIEVAL IMPROVEMENTS:
   - Should we search for different/additional information?
   - Are there specific topics or keywords to focus on?
   - Do we need more context or different sources?

2. ANSWER IMPROVEMENTS:
   - What specific aspects are missing or incomplete?
   - Which parts need more detail or clarification?
   - How can we better structure the response?

3. SPECIFIC ACTIONS:
   - List 2-3 concrete steps to improve the answer
   - Suggest refined search queries if needed
   - Identify gaps to address in next iteration

Provide clear, actionable feedback:"""
        
        feedback, elapsed = self._llm_invoke_timed(feedback_prompt, "llm_generate_feedback")
        
        print(f"   ✅ Feedback generated ({format_time_ns(elapsed)})")
        
        return feedback, elapsed

    def refine_query(self, original_query: str, feedback: str, iteration: int) -> Tuple[str, int]:
        """
        Refine the search query based on feedback to get better retrieval results
        """
        print(f"   🔄 Refining search query...")
        
        refine_prompt = f"""Based on feedback, create a refined search query to retrieve better information.

Original Query: {original_query}

Feedback:
{feedback}

Instructions:
1. Identify key missing information or gaps
2. Create a more specific search query
3. Focus on the most important aspects to retrieve
4. Keep the query concise but informative

If the original query is already optimal, return it unchanged.

Refined Query:"""
        
        refined_query, elapsed = self._llm_invoke_timed(refine_prompt, "llm_refine_query")
        
        # Clean up the refined query
        refined_query = refined_query.strip().strip('"').strip("'")
        
        if refined_query and refined_query != original_query:
            print(f"   ✅ Query refined ({format_time_ns(elapsed)})")
            print(f"   🔍 New query: \"{refined_query[:80]}...\"")
        else:
            print(f"   ℹ️  Query unchanged ({format_time_ns(elapsed)})")
            refined_query = original_query
        
        return refined_query, elapsed

    def query(self, question: str, max_iterations: int = 3, quality_threshold: float = 7.5) -> Dict[str, Any]:
        """
        Main feedback loop query execution
        """
        print("\n" + "="*70)
        print("🔄 RAG WITH FEEDBACK LOOP")
        print("="*70)
        print(f"❓ Question: {question}")
        print(f"🎯 Quality Threshold: {quality_threshold}/10")
        print(f"🔁 Max Iterations: {max_iterations}\n")
        
        overall_start = time.time_ns()
        iteration = 0
        current_query = question
        previous_feedback = None
        
        best_answer = ""
        best_score = 0.0
        all_iterations = []
        
        while iteration < max_iterations:
            iter_start = time.time_ns()
            iteration += 1
            
            print(f"\n{'='*70}")
            print(f"📍 ITERATION {iteration}/{max_iterations}")
            print(f"{'='*70}")
            
            # STEP 1: RETRIEVE (with query refinement if feedback exists)
            print(f"\n🔍 STEP 1: RETRIEVAL")
            print("-" * 70)
            
            if iteration > 1 and previous_feedback:
                refined_query, refine_time = self.refine_query(question, previous_feedback, iteration)
                current_query = refined_query
            
            # Increase k for later iterations to get more context
            k = 4 if iteration == 1 else min(6, 4 + iteration - 1)
            hits, context, retrieval_time = self.retrieve_documents(current_query, k, iteration)
            
            # STEP 2: GENERATE
            print(f"\n✍️  STEP 2: GENERATION")
            print("-" * 70)
            answer, gen_time = self.generate_answer(question, context, previous_feedback, iteration)
            
            # STEP 3: EVALUATE
            print(f"\n📊 STEP 3: EVALUATION")
            print("-" * 70)
            evaluation, eval_time = self.evaluate_answer(question, answer, context, iteration)
            
            # Track best answer so far
            if evaluation['overall_score'] > best_score:
                best_score = evaluation['overall_score']
                best_answer = answer
            
            # STEP 4: FEEDBACK (if not passing)
            print(f"\n💬 STEP 4: FEEDBACK")
            print("-" * 70)
            
            if evaluation['pass_threshold']:
                print("   ✅ Quality threshold met! No feedback needed.")
                feedback = "Quality threshold met. Answer is satisfactory."
                feedback_time = 0
            else:
                feedback, feedback_time = self.generate_feedback(
                    question, answer, context, evaluation, iteration
                )
                print(f"   📝 Feedback generated for next iteration")
            
            iter_elapsed = time.time_ns() - iter_start
            latency_report.add("feedback_loop_iteration", iter_elapsed)
            
            # Store iteration details
            iteration_data = {
                'iteration': iteration,
                'query_used': current_query,
                'documents_retrieved': len(hits),
                'context_length': len(context),
                'answer': answer,
                'answer_length': len(answer),
                'evaluation': evaluation,
                'feedback': feedback,
                'time_ns': iter_elapsed,
                'component_times': {
                    'retrieval': retrieval_time,
                    'generation': gen_time,
                    'evaluation': eval_time,
                    'feedback': feedback_time
                }
            }
            all_iterations.append(iteration_data)
            
            print(f"\n⏱️  Iteration {iteration} completed in {format_time_ns(iter_elapsed)}")
            print(f"   📊 Iteration Score: {evaluation['overall_score']:.1f}/10")
            
            # Check if we should continue
            if evaluation['pass_threshold']:
                print(f"\n✅ Quality threshold met! Stopping after {iteration} iteration(s)")
                break
            elif iteration >= max_iterations:
                print(f"\n⚠️  Max iterations reached. Using best answer (score: {best_score:.1f}/10)")
                answer = best_answer
                break
            else:
                print(f"\n🔄 Continuing to iteration {iteration + 1} with feedback...")
                previous_feedback = feedback
        
        total_query_ns = time.time_ns() - overall_start
        latency_report.add("feedback_loop_query_total", total_query_ns)
        
        # Display final answer
        print("\n" + "="*70)
        print("💬 FINAL ANSWER:")
        print("="*70)
        print(answer[:800])
        if len(answer) > 800:
            print("...")
        print(f"\n📊 Final Score: {best_score:.1f}/10")
        print(f"🔁 Iterations Used: {iteration}/{max_iterations}")
        print(f"⏱️  Total Time: {format_time_ns(total_query_ns)}")
        print("="*70 + "\n")
        
        return {
            "question": question,
            "final_answer": answer,
            "best_score": best_score,
            "iterations_used": iteration,
            "max_iterations": max_iterations,
            "all_iterations": all_iterations,
            "quality_threshold": quality_threshold,
            "threshold_met": best_score >= quality_threshold,
            "total_query_ns": total_query_ns,
        }

# ---------------------------
# Vader Sentiment
# ---------------------------
class VaderSentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    def analyze(self, text: str) -> Dict[str, Any]:
        scores = self.analyzer.polarity_scores(text)
        compound = scores["compound"]
        if compound >= 0.05:
            label = "POSITIVE"
            percentage = round((compound + 1) * 50, 2)
        elif compound <= -0.05:
            label = "NEGATIVE"
            percentage = round((1 - abs(compound)) * 50, 2)
        else:
            label = "NEUTRAL"
            percentage = round(50 + (compound * 50), 2)
        return {"label": label, "percentage": percentage, "compound": compound, "scores": scores}

def run_sentiment_benchmark(sa: VaderSentimentAnalyzer, examples: List[str],
                            target_ns: int = TARGET_NS, run_number: int = 1):
    print("\n" + "="*70)
    print(f"🔥 SENTIMENT BENCHMARK RUN #{run_number}")
    print("="*70)
    print(f"🎯 TARGET: < {target_ns} ns per analysis\n")
    individual_times = []
    for i, text in enumerate(examples, 1):
        start_ns = time.time_ns()
        result = sa.analyze(text)
        elapsed_ns = time.time_ns() - start_ns
        latency_report.add("vader_per_example", elapsed_ns)
        individual_times.append(elapsed_ns)
        status = "✅" if elapsed_ns < target_ns else "❌"
        print(f"[{i:2d}] {format_time_ns(elapsed_ns):25s} {status} | {result['label']:8s} | \"{text}\"")
    total_ns = sum(individual_times)
    avg_ns = total_ns // len(individual_times)
    min_ns = min(individual_times)
    max_ns = max(individual_times)
    under_target = sum(1 for t in individual_times if t < target_ns)
    print("\n📊 RUN #{run_number} STATISTICS:")
    print(f"   Total:        {format_time_ns(total_ns)}")
    print(f"   Average:      {format_time_ns(avg_ns)}")
    print(f"   Min:          {format_time_ns(min_ns)}")
    print(f"   Max:          {format_time_ns(max_ns)}")
    print(f"   < {target_ns}ns: {under_target}/{len(individual_times)} texts")
    if avg_ns < target_ns:
        print("   ✅ TARGET MET!")
    else:
        print("   ⚠️  TARGET MISSED")

# ---------------------------
# MAIN
# ---------------------------
def main():
    print("="*70)
    print("🔄 RAG WITH FEEDBACK LOOP + FULL LATENCY INSTRUMENTATION")
    print("="*70 + "\n")

    pipeline_start = time.time_ns()

    # Phase 1: Load and prepare data
    print("📚 PHASE 1: DATA PREPARATION")
    print("-" * 70)
    text, load_time = timed_call(load_pdf, PDF_PATH)
    latency_report.add("pipeline_pdf_load", load_time)

    chunks, chunk_time = timed_call(chunk_text, text, 1000, 100)
    latency_report.add("pipeline_chunking", chunk_time)

    embedder, embed_time = timed_call(load_embeddings, EMBED_MODEL)
    latency_report.add("pipeline_embeddings_load", embed_time)

    milvus_collection, milvus_time = timed_call(init_milvus, MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME, DIM)
    latency_report.add("pipeline_milvus_init", milvus_time)

    insert_time_start = time.time_ns()
    insert_chunks(milvus_collection, embedder, chunks)
    insert_time = time.time_ns() - insert_time_start
    latency_report.add("pipeline_insert_chunks", insert_time)

    # Phase 2: Initialize LLM
    print("\n📚 PHASE 2: LLM INITIALIZATION")
    print("-" * 70)
    llm_start = time.time_ns()
    llm = ChatGroq(model_name=MODEL_NAME, groq_api_key=GROQ_API_KEY, temperature=0)
    llm_time = time.time_ns() - llm_start
    latency_report.add("llm_init", llm_time)
    print(f"✅ LLM initialized in {format_time_ns(llm_time)}")

    # Initialize Feedback Loop RAG
    feedback_rag = FeedbackLoopRAG(llm, milvus_collection, embedder)
    print("\n✅ RAG with Feedback Loop system initialized!")

    # Phase 3: Run queries
    print("\n📚 PHASE 3: FEEDBACK LOOP RAG QUERIES")
    print("-" * 70)
    queries = [
        "What are the main themes in this story?",
        "Describe the character development and relationships throughout the narrative.",
        "Analyze the emotional journey and key turning points in the plot."
    ]
    results = []
    for q in queries:
        result = feedback_rag.query(q, max_iterations=3, quality_threshold=7.5)
        results.append(result)
        
        # Print iteration summary
        print(f"\n📈 QUERY SUMMARY:")
        print(f"   Question: {q[:60]}...")
        print(f"   Final Score: {result['best_score']:.1f}/10")
        print(f"   Threshold Met: {'✅ YES' if result['threshold_met'] else '❌ NO'}")
        print(f"   Iterations: {result['iterations_used']}/{result['max_iterations']}")
        print(f"   Total Time: {format_time_ns(result['total_query_ns'])}")

    # Phase 4: Sentiment benchmark
    print("\n📚 PHASE 4: VADER SENTIMENT BENCHMARK")
    print("-" * 70)
    sa = VaderSentimentAnalyzer()
    sa_init = 0
    latency_report.add("vader_init", sa_init)
    print(f"✅ VADER INIT TIME: {format_time_ns(sa_init)}\n")
    examples = [
        "I love this product!",
        "This is very bad service.",
        "It's okay, not too good, not too bad.",
        "Not great, really disappointed",
        "Amazing experience!"
    ]
    for run in range(1, 3):
        run_sentiment_benchmark(sa, examples, TARGET_NS, run)
        time.sleep(0.1)

    pipeline_total = time.time_ns() - pipeline_start
    latency_report.add("pipeline_total", pipeline_total)

    # Final Summary
    print("\n" + "="*70)
    print("📈 PIPELINE SUMMARY")
    print("="*70)
    print(f"Total pipeline time: {format_time_ns(pipeline_total)}")
    if results:
        print(f"\nQueries executed: {len(queries)}")
        avg_time = sum(r['total_query_ns'] for r in results) // len(results)
        print(f"Average query time: {format_time_ns(avg_time)}")
        
        avg_score = sum(r['best_score'] for r in results) / len(results)
        print(f"Average quality score: {avg_score:.2f}/10")
        
        threshold_met = sum(1 for r in results if r['threshold_met'])
        print(f"Threshold met: {threshold_met}/{len(results)} queries")
        
        total_iterations = sum(r['iterations_used'] for r in results)
        print(f"Total iterations used: {total_iterations}")
    
    latency_report.pretty_print()

    try:
        connections.disconnect()
    except Exception:
        pass

    print("✅ PIPELINE COMPLETE")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)