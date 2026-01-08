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
COLLECTION_NAME = "CorrectiveRAG_Documents"
DIM = 384
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = "llama-3.1-8b-instant"
TARGET_NS = 200_000
RELEVANCE_THRESHOLD = 0.5  # Threshold for document relevance
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
                              description="Corrective RAG document chunks")
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
# Search with scores
# ---------------------------
def search_milvus_with_scores(collection: Collection, embedder: SentenceTransformer, query: str, limit: int = 5) -> Tuple[List[Dict], int]:
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
                distance = getattr(hit, "distance", 1.0)
                
                if isinstance(ent, dict):
                    txt = ent.get("text", "")
                else:
                    txt = str(hit)
                
                hits.append({
                    "text": txt,
                    "distance": distance,
                    "source": ent.get("source", "") if isinstance(ent, dict) else ""
                })
            except Exception:
                pass
    except Exception as e:
        search_time = time.time_ns() - start
        latency_report.add("milvus_search_error", search_time)
        print(f"⚠️ Milvus search failed: {e}")
        hits = []
    
    total_time = encode_time + search_time
    return hits, total_time

# ---------------------------
# Corrective RAG System
# ---------------------------
class CorrectiveRAG:
    def __init__(self, llm, collection: Collection, embedder: SentenceTransformer, 
                 relevance_threshold: float = RELEVANCE_THRESHOLD):
        self.llm = llm
        self.collection = collection
        self.embedder = embedder
        self.relevance_threshold = relevance_threshold
        
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
    
    def retrieve_documents(self, query: str, k: int = 5) -> Tuple[List[Dict], int]:
        """Initial document retrieval with relevance scores"""
        hits, elapsed = search_milvus_with_scores(self.collection, self.embedder, query, k)
        print(f"   ✅ Retrieved {len(hits)} documents in {format_time_ns(elapsed)}")
        return hits, elapsed
    
    def evaluate_relevance(self, query: str, documents: List[Dict]) -> Tuple[Dict[str, List[Dict]], int]:
        """
        Evaluate relevance of retrieved documents using LLM
        Returns: {
            'correct': [highly relevant docs],
            'ambiguous': [partially relevant docs],
            'incorrect': [irrelevant docs]
        }
        """
        if not documents:
            return {'correct': [], 'ambiguous': [], 'incorrect': []}, 0
        
        # Build evaluation prompt
        doc_texts = "\n\n".join([
            f"Document {i+1}:\n{doc['text'][:300]}..."
            for i, doc in enumerate(documents)
        ])
        
        prompt = f"""You are a document relevance evaluator. Evaluate how relevant each document is to the given query.

Query: {query}

Documents:
{doc_texts}

For each document, classify it as:
- CORRECT: Highly relevant and directly answers the query
- AMBIGUOUS: Partially relevant or contains some useful information
- INCORRECT: Not relevant or off-topic

Format your response as:
Document 1: [CORRECT/AMBIGUOUS/INCORRECT] - [brief reason]
Document 2: [CORRECT/AMBIGUOUS/INCORRECT] - [brief reason]
etc.

Evaluation:"""
        
        evaluation, elapsed = self._llm_invoke_timed(prompt, "llm_relevance_evaluation")
        
        # Parse evaluation results
        categorized = {
            'correct': [],
            'ambiguous': [],
            'incorrect': []
        }
        
        lines = evaluation.strip().split('\n')
        for i, line in enumerate(lines):
            if i >= len(documents):
                break
            
            line_upper = line.upper()
            doc = documents[i]
            
            if 'CORRECT' in line_upper and 'INCORRECT' not in line_upper:
                categorized['correct'].append(doc)
            elif 'AMBIGUOUS' in line_upper:
                categorized['ambiguous'].append(doc)
            elif 'INCORRECT' in line_upper:
                categorized['incorrect'].append(doc)
            else:
                # Default to ambiguous if unclear
                categorized['ambiguous'].append(doc)
        
        print(f"   📊 Relevance evaluation: {len(categorized['correct'])} correct, "
              f"{len(categorized['ambiguous'])} ambiguous, {len(categorized['incorrect'])} incorrect")
        
        return categorized, elapsed
    
    def apply_knowledge_refinement(self, query: str, correct_docs: List[Dict], 
                                   ambiguous_docs: List[Dict]) -> Tuple[str, int]:
        """
        Apply knowledge refinement on ambiguous documents
        Extract and refine relevant information from partially relevant documents
        """
        if not ambiguous_docs:
            return "", 0
        
        ambiguous_texts = "\n\n".join([
            f"Document {i+1}:\n{doc['text']}"
            for i, doc in enumerate(ambiguous_docs)
        ])
        
        prompt = f"""You are refining partially relevant information. Extract ONLY the information that is relevant to the query.

Query: {query}

Partially Relevant Documents:
{ambiguous_texts}

Extract and summarize only the relevant parts that help answer the query. Ignore irrelevant information.

Refined Information:"""
        
        refined_info, elapsed = self._llm_invoke_timed(prompt, "llm_knowledge_refinement")
        print(f"   🔧 Knowledge refined in {format_time_ns(elapsed)}")
        
        return refined_info, elapsed
    
    def generate_search_queries(self, original_query: str, categorized: Dict) -> Tuple[List[str], int]:
        """
        Generate alternative search queries for web search when retrieved docs are insufficient
        """
        if len(categorized['correct']) >= 2:
            return [], 0  # Sufficient correct documents found
        
        prompt = f"""The retrieved documents are insufficient to answer this query. Generate 2-3 alternative search queries that might find better information.

Original Query: {original_query}

Current situation:
- Correct documents: {len(categorized['correct'])}
- Ambiguous documents: {len(categorized['ambiguous'])}
- Incorrect documents: {len(categorized['incorrect'])}

Generate alternative search queries (one per line):"""
        
        queries_text, elapsed = self._llm_invoke_timed(prompt, "llm_generate_search_queries")
        
        # Parse queries
        search_queries = []
        for line in queries_text.strip().split('\n'):
            clean = line.strip().lstrip('0123456789.-•) ').strip()
            if clean and len(clean) > 10:
                search_queries.append(clean)
        
        if search_queries:
            print(f"   🔍 Generated {len(search_queries)} alternative search queries")
            for i, sq in enumerate(search_queries, 1):
                print(f"      {i}. {sq}")
        
        return search_queries, elapsed
    
    def fallback_web_search(self, search_queries: List[str]) -> Tuple[str, int]:
        """
        Simulate web search fallback (in real implementation, this would call a web search API)
        """
        if not search_queries:
            return "", 0
        
        start = time.time_ns()
        
        # Simulate web search - in production, this would use actual web search
        print(f"   🌐 [SIMULATED] Web search for: {search_queries[0][:50]}...")
        web_results = """[Simulated Web Search Results]
        
In a production environment, this would contain real web search results from external sources.
The system would fetch additional information to supplement the document collection when
the retrieved documents are deemed insufficient to answer the query accurately."""
        
        elapsed = time.time_ns() - start
        latency_report.add("web_search_fallback", elapsed)
        
        return web_results, elapsed
    
    def generate_corrected_answer(self, query: str, correct_docs: List[Dict], 
                                  refined_info: str, web_info: str) -> Tuple[str, int]:
        """
        Generate final answer using corrected and refined information
        """
        # Build context
        context_parts = []
        
        if correct_docs:
            correct_context = "\n\n".join([
                f"Relevant Document {i+1}:\n{doc['text']}"
                for i, doc in enumerate(correct_docs)
            ])
            context_parts.append(f"Highly Relevant Information:\n{correct_context}")
        
        if refined_info:
            context_parts.append(f"\nRefined Information:\n{refined_info}")
        
        if web_info:
            context_parts.append(f"\nAdditional Information:\n{web_info}")
        
        full_context = "\n\n".join(context_parts) if context_parts else "No relevant information found."
        
        prompt = f"""Answer the following query using the provided corrected and refined information.

Query: {query}

{full_context}

Provide a comprehensive, accurate answer based on the information above.

Answer:"""
        
        answer, elapsed = self._llm_invoke_timed(prompt, "llm_generate_corrected_answer")
        print(f"   ✅ Corrected answer generated in {format_time_ns(elapsed)}")
        
        return answer, elapsed
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Process query using Corrective RAG (CRAG)
        """
        print("\n" + "="*70)
        print("🔧 CORRECTIVE RAG (CRAG) QUERY PROCESSING")
        print("="*70)
        print(f"❓ Question: {question}\n")
        
        overall_start = time.time_ns()
        
        # Step 1: Initial Retrieval
        print("📥 Step 1: Initial Document Retrieval")
        print("-" * 70)
        documents, retrieval_time = self.retrieve_documents(question, k=5)
        
        # Step 2: Relevance Evaluation
        print("\n⚖️  Step 2: Relevance Evaluation")
        print("-" * 70)
        categorized, eval_time = self.evaluate_relevance(question, documents)
        
        # Step 3: Knowledge Refinement (for ambiguous docs)
        print("\n🔧 Step 3: Knowledge Refinement")
        print("-" * 70)
        refined_info = ""
        refine_time = 0
        if categorized['ambiguous']:
            refined_info, refine_time = self.apply_knowledge_refinement(
                question, 
                categorized['correct'], 
                categorized['ambiguous']
            )
        else:
            print("   ℹ️  No ambiguous documents to refine")
        
        # Step 4: Web Search Fallback (if needed)
        print("\n🌐 Step 4: Web Search Fallback")
        print("-" * 70)
        web_info = ""
        web_time = 0
        search_queries = []
        
        if len(categorized['correct']) < 2:
            print("   ⚠️  Insufficient relevant documents, generating search queries...")
            search_queries, sq_time = self.generate_search_queries(question, categorized)
            
            if search_queries:
                web_info, web_time = self.fallback_web_search(search_queries)
            else:
                print("   ℹ️  No additional search needed")
        else:
            print("   ✅ Sufficient relevant documents found, skipping web search")
        
        # Step 5: Generate Corrected Answer
        print("\n💡 Step 5: Generate Corrected Answer")
        print("-" * 70)
        answer, gen_time = self.generate_corrected_answer(
            question,
            categorized['correct'],
            refined_info,
            web_info
        )
        
        total_query_ns = time.time_ns() - overall_start
        latency_report.add("crag_query_total", total_query_ns)
        
        print("\n" + "="*70)
        print("💬 FINAL CORRECTED ANSWER:")
        print("="*70)
        print(answer[:800])
        if len(answer) > 800:
            print("...")
        
        print(f"\n📊 CRAG Statistics:")
        print(f"   Total documents retrieved: {len(documents)}")
        print(f"   Correct documents: {len(categorized['correct'])}")
        print(f"   Ambiguous documents: {len(categorized['ambiguous'])}")
        print(f"   Incorrect documents: {len(categorized['incorrect'])}")
        print(f"   Knowledge refined: {'Yes' if refined_info else 'No'}")
        print(f"   Web search used: {'Yes' if web_info else 'No'}")
        print(f"   Total query time: {format_time_ns(total_query_ns)}")
        print("="*70 + "\n")
        
        return {
            "question": question,
            "answer": answer,
            "categorized_docs": categorized,
            "refined_info": refined_info,
            "web_search_used": bool(web_info),
            "search_queries": search_queries,
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
    print("🔧 CORRECTIVE RAG (CRAG) + FULL LATENCY INSTRUMENTATION")
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
    
    # Initialize Corrective RAG
    crag = CorrectiveRAG(llm, milvus_collection, embedder, relevance_threshold=RELEVANCE_THRESHOLD)
    print(f"\n✅ Corrective RAG system initialized!")
    
    # Phase 3: Run CRAG queries
    print("\n📚 PHASE 3: CORRECTIVE RAG QUERIES")
    print("-" * 70)
    
    queries = [
        "What are the main themes in this story?",
        "Tell me about the character development.",
        "What is quantum computing?",  # Likely to trigger web search fallback
    ]
    
    results = []
    for q in queries:
        result = crag.query(q)
        results.append(result)
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
        correct_total = sum(len(r['categorized_docs']['correct']) for r in results)
        ambiguous_total = sum(len(r['categorized_docs']['ambiguous']) for r in results)
        incorrect_total = sum(len(r['categorized_docs']['incorrect']) for r in results)
        print(f"Total correct documents used: {correct_total}")
        print(f"Total ambiguous documents refined: {ambiguous_total}")
        print(f"Total incorrect documents filtered: {incorrect_total}")
        web_searches = sum(1 for r in results if r['web_search_used'])
        print(f"Web searches triggered: {web_searches}/{len(results)}")
    
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