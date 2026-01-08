import os
import time
import sys
import traceback
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import json
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
COLLECTION_NAME = "MultiHopRAG_Documents"
DIM = 384
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = "llama-3.1-8b-instant"
TARGET_NS = 200_000
MAX_HOPS = 3  # Maximum number of hops for multi-hop reasoning
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
# Milvus init
# ---------------------------
@timer_ns
def init_milvus(host: str, port: str, collection_name: str = COLLECTION_NAME, dim: int = DIM) -> Collection:
    print(f"🗃️  Initializing Milvus connection to {host}:{port}")
    connections.connect(host=host, port=port)
    
    try:
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            print(f"🗑️  Deleted existing collection '{collection_name}'")
    except Exception as e:
        print(f"⚠️  Collection check/delete: {e}")
    
    chunk_id_field = FieldSchema(name="chunk_id", dtype=DataType.INT64, is_primary=True, auto_id=True)
    text_field = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
    source_field = FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1024)
    embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    
    schema = CollectionSchema(fields=[chunk_id_field, text_field, source_field, embedding_field],
                              description="Multi-Hop RAG document chunks")
    collection = Collection(name=collection_name, schema=schema)
    
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
# Insert chunks
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
        results = collection.search(
            data=[qvec.tolist()], 
            anns_field="embedding", 
            param=search_params, 
            limit=limit,
            output_fields=["text", "source", "chunk_id"]
        )
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
# Multi-Hop RAG System
# ---------------------------
class MultiHopRAG:
    def __init__(self, llm, collection: Collection, embedder: SentenceTransformer, max_hops: int = MAX_HOPS):
        self.llm = llm
        self.collection = collection
        self.embedder = embedder
        self.max_hops = max_hops
        
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
    
    def decompose_query(self, query: str) -> Tuple[List[str], int]:
        """Break down complex query into multiple sub-questions"""
        prompt = f"""You are a query decomposition expert. Break down this complex question into simpler sub-questions that need to be answered sequentially.

Original Question: {query}

Analyze if this question requires multiple pieces of information to answer. If yes, break it into 2-4 sub-questions in logical order. If it's simple, return just the original question.

Format your response as a numbered list:
1. [First sub-question]
2. [Second sub-question]
etc.

Sub-questions:"""
        
        response, elapsed = self._llm_invoke_timed(prompt, "llm_query_decomposition")
        
        # Parse sub-questions
        lines = response.strip().split('\n')
        sub_questions = []
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                # Remove numbering and clean up
                clean = line.lstrip('0123456789.-•) ').strip()
                if clean:
                    sub_questions.append(clean)
        
        # If no sub-questions found, use original
        if not sub_questions:
            sub_questions = [query]
        
        print(f"   🔍 Decomposed into {len(sub_questions)} sub-question(s)")
        for i, sq in enumerate(sub_questions, 1):
            print(f"      {i}. {sq[:80]}...")
        
        return sub_questions, elapsed
    
    def retrieve_documents(self, query: str, k: int = 3) -> Tuple[List[str], int]:
        """Retrieve documents for a specific query"""
        hits, elapsed = search_milvus(self.collection, self.embedder, query, k)
        print(f"      ✅ Retrieved {len(hits)} documents in {format_time_ns(elapsed)}")
        return hits, elapsed
    
    def answer_sub_question(self, sub_question: str, context: List[str], previous_answers: List[Dict]) -> Tuple[str, int]:
        """Answer a sub-question using retrieved context and previous answers"""
        # Build context from previous answers
        prev_context = ""
        if previous_answers:
            prev_context = "\n\nPrevious findings:\n"
            for i, prev in enumerate(previous_answers, 1):
                prev_context += f"{i}. Q: {prev['question']}\n   A: {prev['answer'][:150]}...\n"
        
        # Build document context
        doc_context = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(context)])
        
        prompt = f"""Answer the following sub-question using the provided context and previous findings.

{prev_context}

Current Document Context:
{doc_context}

Sub-Question: {sub_question}

Provide a clear, concise answer based on the context above.

Answer:"""
        
        answer, elapsed = self._llm_invoke_timed(prompt, "llm_answer_subquestion")
        return answer, elapsed
    
    def synthesize_final_answer(self, original_query: str, hop_results: List[Dict]) -> Tuple[str, int]:
        """Synthesize final answer from all hops"""
        reasoning_chain = "\n\n".join([
            f"Step {i+1} - Q: {hop['question']}\nA: {hop['answer']}"
            for i, hop in enumerate(hop_results)
        ])
        
        prompt = f"""You are synthesizing a final answer from multi-hop reasoning.

Original Question: {original_query}

Reasoning Chain:
{reasoning_chain}

Based on the step-by-step reasoning above, provide a comprehensive final answer to the original question. Integrate insights from all steps coherently.

Final Answer:"""
        
        answer, elapsed = self._llm_invoke_timed(prompt, "llm_synthesize_answer")
        print(f"   ✅ Final answer synthesized in {format_time_ns(elapsed)}")
        return answer, elapsed
    
    def query(self, question: str) -> Dict[str, Any]:
        """Process query using Multi-Hop RAG"""
        print("\n" + "="*70)
        print("🔗 MULTI-HOP RAG QUERY PROCESSING")
        print("="*70)
        print(f"❓ Original Question: {question}\n")
        
        overall_start = time.time_ns()
        
        # Step 1: Decompose query into sub-questions
        print("📋 Step 1: Query Decomposition")
        print("-" * 70)
        sub_questions, decomp_time = self.decompose_query(question)
        
        # Step 2: Multi-hop reasoning
        print("\n🔗 Step 2: Multi-Hop Reasoning")
        print("-" * 70)
        
        hop_results = []
        previous_answers = []
        
        for hop_num, sub_q in enumerate(sub_questions[:self.max_hops], 1):
            print(f"\n   🔸 HOP {hop_num}/{min(len(sub_questions), self.max_hops)}")
            print(f"   Question: {sub_q}")
            
            hop_start = time.time_ns()
            
            # Retrieve documents for this sub-question
            context, retrieval_time = self.retrieve_documents(sub_q, k=3)
            
            # Answer sub-question
            print(f"      💡 Answering sub-question...")
            answer, answer_time = self.answer_sub_question(sub_q, context, previous_answers)
            
            hop_elapsed = time.time_ns() - hop_start
            latency_report.add(f"hop_{hop_num}_total", hop_elapsed)
            
            hop_result = {
                "hop_number": hop_num,
                "question": sub_q,
                "answer": answer,
                "context": context,
                "time_ns": hop_elapsed
            }
            hop_results.append(hop_result)
            previous_answers.append({"question": sub_q, "answer": answer})
            
            print(f"      ⏱️  Hop {hop_num} completed in {format_time_ns(hop_elapsed)}")
            print(f"      📝 Answer: {answer[:100]}...")
        
        # Step 3: Synthesize final answer
        print("\n🎯 Step 3: Answer Synthesis")
        print("-" * 70)
        final_answer, synth_time = self.synthesize_final_answer(question, hop_results)
        
        # Calculate total time
        total_query_ns = time.time_ns() - overall_start
        latency_report.add("multihop_query_total", total_query_ns)
        
        print("\n" + "="*70)
        print("💬 FINAL SYNTHESIZED ANSWER:")
        print("="*70)
        print(final_answer[:800])
        if len(final_answer) > 800:
            print("...")
        
        print(f"\n📊 Multi-Hop Statistics:")
        print(f"   Total hops: {len(hop_results)}")
        print(f"   Average hop time: {format_time_ns(sum(h['time_ns'] for h in hop_results) // len(hop_results))}")
        print(f"   Total query time: {format_time_ns(total_query_ns)}")
        print("="*70 + "\n")
        
        return {
            "question": question,
            "sub_questions": sub_questions,
            "hop_results": hop_results,
            "final_answer": final_answer,
            "total_hops": len(hop_results),
            "total_query_ns": total_query_ns,
        }
    
    def explain_reasoning(self, result: Dict) -> str:
        """Generate a readable explanation of the multi-hop reasoning process"""
        explanation = f"Question: {result['question']}\n\n"
        explanation += "Reasoning Process:\n"
        explanation += "="*50 + "\n\n"
        
        for hop in result['hop_results']:
            explanation += f"Hop {hop['hop_number']}: {hop['question']}\n"
            explanation += f"Answer: {hop['answer'][:200]}...\n"
            explanation += f"Time: {format_time_ns(hop['time_ns'])}\n\n"
        
        explanation += "="*50 + "\n"
        explanation += f"Final Answer: {result['final_answer'][:300]}...\n"
        
        return explanation

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
    print("🔗 MULTI-HOP RAG + FULL LATENCY INSTRUMENTATION")
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
    
    # Initialize Multi-Hop RAG
    multi_hop_rag = MultiHopRAG(llm, milvus_collection, embedder, max_hops=MAX_HOPS)
    print(f"\n✅ Multi-Hop RAG system initialized (max hops: {MAX_HOPS})!")
    
    # Phase 3: Run multi-hop queries
    print("\n📚 PHASE 3: MULTI-HOP RAG QUERIES")
    print("-" * 70)
    
    queries = [
        "What are the main themes and how do they relate to the characters?",
        "Summarize the story and explain its emotional impact.",
        "What happens in the beginning and how does it connect to the ending?",
    ]
    
    results = []
    for q in queries:
        result = multi_hop_rag.query(q)
        results.append(result)
        
        # Show reasoning explanation
        print("\n📖 REASONING EXPLANATION:")
        print("-" * 70)
        explanation = multi_hop_rag.explain_reasoning(result)
        print(explanation[:500] + "..." if len(explanation) > 500 else explanation)
        
        time.sleep(0.5)
    
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
    
    print("\n" + "="*70)
    print("📈 PIPELINE SUMMARY")
    print("="*70)
    print(f"Total pipeline time: {format_time_ns(pipeline_total)}")
    if results:
        print(f"Queries executed: {len(queries)}")
        print(f"Average query time: {format_time_ns(sum(r['total_query_ns'] for r in results) // len(results))}")
        print(f"Average hops per query: {sum(r['total_hops'] for r in results) / len(results):.1f}")
        print(f"Total hops executed: {sum(r['total_hops'] for r in results)}")
    
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