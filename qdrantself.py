#!/usr/bin/env python3
"""
qdrant_selfrag_latency.py
Self-RAG with Qdrant and comprehensive nanosecond latency instrumentation.

Features:
- Full pipeline timing (PDF load, chunking, embeddings, vectorstore)
- Per-component latency tracking
- Query-level performance metrics
- Detailed latency reports
"""

import os
import time
import sys
from collections import defaultdict
from typing import List, Dict, Any, Tuple

import pdfplumber
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# =========================================================
# CONFIG
# =========================================================
PDF_PATH = "Data/ECHOES OF HER LOVE.pdf"
COLLECTION = "rag_collection"
DIM = 384
MODEL_NAME = "llama-3.1-8b-instant"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("❌ ERROR: Set GROQ_API_KEY environment variable!")
    sys.exit(1)

# =========================================================
# LATENCY UTILITIES
# =========================================================
def format_time_ns(ns: int) -> str:
    """Return human-readable representation of nanoseconds."""
    if ns < 1_000:
        return f"{ns} ns"
    if ns < 1_000_000:
        return f"{ns/1_000:.3f} µs ({ns} ns)"
    if ns < 1_000_000_000:
        return f"{ns/1_000_000:.3f} ms ({ns} ns)"
    return f"{ns/1_000_000_000:.3f} s ({ns} ns)"

def timed_call(fn, *args, **kwargs):
    """Call fn(*args, **kwargs) and return (result, elapsed_ns)."""
    start = time.time_ns()
    result = fn(*args, **kwargs)
    elapsed = time.time_ns() - start
    return result, elapsed

def timer_ns(func):
    """Decorator that prints elapsed ns and stores last_elapsed_ns on wrapper."""
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
    """Aggregates and reports latency metrics"""
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

# =========================================================
# LOAD PDF WITH TIMING
# =========================================================
@timer_ns
def load_pdf(path: str) -> str:
    """Load PDF with per-page timing"""
    print(f"📄 Loading PDF: {path}")
    text = ""
    
    with pdfplumber.open(path) as pdf:
        pages = pdf.pages
        for i, p in enumerate(pages):
            start_ns = time.time_ns()
            t = p.extract_text() or ""
            elapsed = time.time_ns() - start_ns
            latency_report.add("pdf_page_extract", elapsed)
            text += t + "\n"
    
    print(f"✅ Loaded PDF: {len(text)} characters from {len(pages)} pages")
    return text

# =========================================================
# CHUNK TEXT WITH TIMING
# =========================================================
@timer_ns
def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
    """Chunk text with timing"""
    print(f"✂️  Chunking text...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(text)
    print(f"✅ Created {len(chunks)} chunks")
    return chunks

# =========================================================
# LOAD EMBEDDINGS WITH TIMING
# =========================================================
@timer_ns
def load_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load embedding model with timing"""
    print(f"🔢 Loading embeddings model: {model_name}")
    embedder = SentenceTransformer(model_name)
    print(f"✅ Embeddings model loaded")
    return embedder

# =========================================================
# INIT QDRANT WITH TIMING
# =========================================================
@timer_ns
def init_qdrant(collection_name: str = COLLECTION, dim: int = DIM) -> QdrantClient:
    """Initialize Qdrant with timing"""
    print(f"🗃️  Initializing Qdrant in-memory DB")
    
    start = time.time_ns()
    qdrant = QdrantClient(":memory:")
    init_time = time.time_ns() - start
    latency_report.add("qdrant_client_init", init_time)
    
    # Remove previous collection if exists
    if qdrant.collection_exists(collection_name):
        start = time.time_ns()
        qdrant.delete_collection(collection_name)
        delete_time = time.time_ns() - start
        latency_report.add("qdrant_delete_collection", delete_time)
    
    # Create collection
    start = time.time_ns()
    qdrant.create_collection(
        collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
    )
    create_time = time.time_ns() - start
    latency_report.add("qdrant_create_collection", create_time)
    
    print(f"✅ Qdrant collection '{collection_name}' ready")
    return qdrant

# =========================================================
# INSERT CHUNKS WITH TIMING
# =========================================================
@timer_ns
def insert_chunks(qdrant: QdrantClient, embedder: SentenceTransformer, 
                  chunks: List[str], collection_name: str = COLLECTION) -> None:
    """Insert chunks into Qdrant with detailed timing"""
    print(f"⬆️  Inserting {len(chunks)} chunks into Qdrant...")
    
    # Encode chunks (batch embedding)
    print(f"   🔢 Encoding {len(chunks)} chunks...")
    start = time.time_ns()
    vectors = embedder.encode(chunks, show_progress_bar=False)
    encode_time = time.time_ns() - start
    latency_report.add("embedding_encode_batch", encode_time)
    print(f"   ✅ Encoded in {format_time_ns(encode_time)}")
    
    # Create points
    print(f"   📦 Creating point structures...")
    start = time.time_ns()
    points = [
        PointStruct(
            id=i,
            vector=vectors[i].tolist(),
            payload={"text": chunks[i], "chunk_id": i}
        )
        for i in range(len(chunks))
    ]
    point_creation_time = time.time_ns() - start
    latency_report.add("qdrant_point_creation", point_creation_time)
    print(f"   ✅ Points created in {format_time_ns(point_creation_time)}")
    
    # Upsert to Qdrant
    print(f"   💾 Upserting to Qdrant...")
    start = time.time_ns()
    qdrant.upsert(collection_name=collection_name, points=points)
    upsert_time = time.time_ns() - start
    latency_report.add("qdrant_upsert", upsert_time)
    print(f"   ✅ Upserted in {format_time_ns(upsert_time)}")
    
    print(f"✅ All chunks inserted successfully!")

# =========================================================
# SEARCH QDRANT WITH TIMING
# =========================================================
def search_qdrant(qdrant: QdrantClient, embedder: SentenceTransformer, 
                  query: str, limit: int = 4, collection_name: str = COLLECTION) -> Tuple[List[str], int]:
    """Search Qdrant with timing"""
    
    # Encode query
    start = time.time_ns()
    qvec = embedder.encode([query])[0]
    encode_time = time.time_ns() - start
    latency_report.add("query_embedding", encode_time)
    
    # Query Qdrant
    start = time.time_ns()
    response = qdrant.query_points(
        collection_name=collection_name,
        query=qvec.tolist(),
        limit=limit
    )
    search_time = time.time_ns() - start
    latency_report.add("qdrant_search", search_time)
    
    # Extract texts
    hits = [p.payload.get("text", "") for p in response.points]
    
    total_time = encode_time + search_time
    
    return hits, total_time

# =========================================================
# SELF-RAG WITH TIMING
# =========================================================
class SelfRAG:
    """Self-RAG system with comprehensive timing"""
    
    def __init__(self, llm, qdrant: QdrantClient, embedder: SentenceTransformer, 
                 collection_name: str = COLLECTION):
        self.llm = llm
        self.qdrant = qdrant
        self.embedder = embedder
        self.collection_name = collection_name
    
    def retrieve_decision(self, query: str) -> Tuple[bool, str, int]:
        """Decide if retrieval is needed"""
        prompt = f"""You are a helpful assistant. Decide if you need to retrieve information from a document to answer this question.

Question: {query}

Think step by step:
1. Can you answer this from general knowledge?
2. Does it require specific document information?

Answer with ONLY 'RETRIEVE' or 'NO_RETRIEVE' and a brief reason.

Decision:"""
        
        start = time.time_ns()
        resp = self.llm.invoke(prompt)
        elapsed = time.time_ns() - start
        latency_report.add("llm_retrieve_decision", elapsed)
        
        decision_text = resp.content if hasattr(resp, "content") else str(resp)
        needs_retrieval = 'RETRIEVE' in decision_text.upper() and 'NO_RETRIEVE' not in decision_text.upper()
        
        return needs_retrieval, decision_text, elapsed
    
    def retrieve_documents(self, query: str, k: int = 4) -> Tuple[str, int]:
        """Retrieve documents from Qdrant"""
        print(f"   🔍 Retrieving documents...")
        
        hits, elapsed = search_qdrant(self.qdrant, self.embedder, query, k, self.collection_name)
        context = "\n\n".join(hits)
        
        print(f"   ✅ Retrieved {len(hits)} documents in {format_time_ns(elapsed)}")
        
        return context, elapsed
    
    def generate_answer(self, query: str, context: str = "") -> Tuple[str, int]:
        """Generate answer with timing"""
        if context:
            prompt = f"""Answer the question based on the following context:

Context:
{context}

Question: {query}

Provide a detailed answer based on the context above.

Answer:"""
        else:
            prompt = f"""Answer the following question based on your general knowledge:

Question: {query}

Answer:"""
        
        print(f"   💡 Generating answer...")
        start = time.time_ns()
        resp = self.llm.invoke(prompt)
        elapsed = time.time_ns() - start
        latency_report.add("llm_generate_answer", elapsed)
        
        answer = resp.content if hasattr(resp, "content") else str(resp)
        print(f"   ✅ Answer generated in {format_time_ns(elapsed)}")
        
        return answer, elapsed
    
    def self_critique(self, query: str, answer: str, context: str = "") -> Tuple[str, bool, int]:
        """Self-critique the answer"""
        critique_prompt = f"""You are a critical evaluator. Evaluate the following answer.

Question: {query}

Answer: {answer}

Context Available: {'Yes' if context else 'No'}

Rate the answer on a scale of 1-10 and provide:
1. Relevance Score (1-10)
2. Completeness Score (1-10)
3. Accuracy Assessment
4. Should we retrieve more information? (YES/NO)

Evaluation:"""
        
        print(f"   🔍 Self-critiquing answer...")
        start = time.time_ns()
        resp = self.llm.invoke(critique_prompt)
        elapsed = time.time_ns() - start
        latency_report.add("llm_self_critique", elapsed)
        
        critique_text = resp.content if hasattr(resp, "content") else str(resp)
        needs_more = 'YES' in critique_text.upper() and 'RETRIEVE' in critique_text.upper()
        
        print(f"   ✅ Critique completed in {format_time_ns(elapsed)}")
        
        return critique_text, needs_more, elapsed
    
    def query(self, question: str, max_iterations: int = 2) -> Dict[str, Any]:
        """Process query with Self-RAG pipeline"""
        print(f"\n{'='*70}")
        print(f"🚀 SELF-RAG QUERY PROCESSING")
        print(f"{'='*70}")
        print(f"❓ Question: {question}\n")
        
        overall_start = time.time_ns()
        
        iteration = 0
        context = ""
        answer = ""
        per_iteration_times = []
        
        while iteration < max_iterations:
            iter_start = time.time_ns()
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")
            
            # Step 1: Decide if retrieval is needed
            if iteration == 1:
                needs_retrieval, decision_reason, decision_time = self.retrieve_decision(question)
                print(f"🤔 Retrieval Decision: {'RETRIEVE' if needs_retrieval else 'NO_RETRIEVE'}")
                print(f"   Reasoning: {decision_reason.strip()[:100]}...")
                print(f"   Time: {format_time_ns(decision_time)}")
            else:
                needs_retrieval = True  # Force retrieval when refining
            
            # Step 2: Retrieve if needed
            if needs_retrieval:
                context, retrieval_time = self.retrieve_documents(question)
                print(f"   📝 Context length: {len(context)} characters")
            
            # Step 3: Generate answer
            answer, gen_time = self.generate_answer(question, context)
            print(f"   📄 Answer length: {len(answer)} characters")
            
            # Step 4: Self-critique
            critique_text, needs_more, critique_time = self.self_critique(question, answer, context)
            
            iter_elapsed = time.time_ns() - iter_start
            per_iteration_times.append(iter_elapsed)
            latency_report.add("selfrag_iteration", iter_elapsed)
            
            print(f"\n⏱️  Iteration {iteration} total time: {format_time_ns(iter_elapsed)}")
            
            # Step 5: Decide loop break
            if not needs_more or iteration >= max_iterations:
                print(f"\n✅ Self-RAG completed after {iteration} iteration(s)")
                break
            else:
                print(f"\n🔄 Refinement needed, starting iteration {iteration + 1}...")
        
        total_query_ns = time.time_ns() - overall_start
        latency_report.add("selfrag_query_total", total_query_ns)
        
        print(f"\n{'='*70}")
        print(f"💬 FINAL ANSWER:")
        print(f"{'='*70}")
        print(answer[:800])
        if len(answer) > 800:
            print("...")
        print(f"\n⏱️  Total query time: {format_time_ns(total_query_ns)}")
        print(f"{'='*70}\n")
        
        return {
            'question': question,
            'answer': answer,
            'context': context,
            'iterations': iteration,
            'critique': critique_text,
            'per_iteration_times': per_iteration_times,
            'total_query_ns': total_query_ns
        }

# =========================================================
# VADER SENTIMENT BENCHMARK
# =========================================================
class VaderSentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    
    def analyze(self, text: str) -> Dict[str, Any]:
        scores = self.analyzer.polarity_scores(text)
        compound = scores['compound']
        
        if compound >= 0.05:
            label = "POSITIVE"
            percentage = round((compound + 1) * 50, 2)
        elif compound <= -0.05:
            label = "NEGATIVE"
            percentage = round((1 - abs(compound)) * 50, 2)
        else:
            label = "NEUTRAL"
            percentage = round(50 + (compound * 50), 2)
        
        return {
            'label': label,
            'percentage': percentage,
            'compound': compound,
            'scores': scores
        }

def run_sentiment_benchmark(sa: VaderSentimentAnalyzer, examples: List[str], 
                            target_ns: int = 200_000, run_number: int = 1):
    """Run sentiment analysis benchmark"""
    print(f"\n{'='*70}")
    print(f"🔥 SENTIMENT BENCHMARK RUN #{run_number}")
    print(f"{'='*70}")
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
    
    print(f"\n📊 RUN #{run_number} STATISTICS:")
    print(f"   Total:        {format_time_ns(total_ns)}")
    print(f"   Average:      {format_time_ns(avg_ns)}")
    print(f"   Min:          {format_time_ns(min_ns)}")
    print(f"   Max:          {format_time_ns(max_ns)}")
    print(f"   < {target_ns}ns: {under_target}/{len(individual_times)} texts")
    
    if avg_ns < target_ns:
        print(f"   ✅ TARGET MET!")
    else:
        print(f"   ⚠️  TARGET MISSED")

# =========================================================
# MAIN PROGRAM
# =========================================================
def main():
    print("="*70)
    print("🚀 SELF-RAG QDRANT + FULL LATENCY INSTRUMENTATION")
    print("="*70)
    print()
    
    pipeline_start = time.time_ns()
    
    # Phase 1: Load and prepare data
    print("📚 PHASE 1: DATA PREPARATION")
    print("-"*70)
    
    text, load_time = timed_call(load_pdf, PDF_PATH)
    latency_report.add("pipeline_pdf_load", load_time)
    
    chunks, chunk_time = timed_call(chunk_text, text, 1000, 100)
    latency_report.add("pipeline_chunking", chunk_time)
    
    embedder, embed_time = timed_call(load_embeddings, "sentence-transformers/all-MiniLM-L6-v2")
    latency_report.add("pipeline_embeddings_load", embed_time)
    
    qdrant, qdrant_time = timed_call(init_qdrant, COLLECTION, DIM)
    latency_report.add("pipeline_qdrant_init", qdrant_time)
    
    insert_time_start = time.time_ns()
    insert_chunks(qdrant, embedder, chunks, COLLECTION)
    insert_time = time.time_ns() - insert_time_start
    latency_report.add("pipeline_insert_chunks", insert_time)
    
    # Phase 2: Initialize LLM
    print(f"\n📚 PHASE 2: LLM INITIALIZATION")
    print("-"*70)
    
    llm_start = time.time_ns()
    llm = ChatGroq(
        model_name=MODEL_NAME,
        groq_api_key=GROQ_API_KEY,
        temperature=0
   # Phase 3: Initialize Tools
print(f"\n📚 PHASE 3: TOOL INITIALIZATION")
print("-"*70)

tools = [
    DocumentSearchTool(qdrant, embedder, COLLECTION),
    CalculatorTool(),
    SentimentAnalyzerTool(),
    SummarizerTool(llm)
]

print("✅ Available tools:")
for tool in tools:
    print(f"   🔧 {tool.name}: {tool.description[:60]}...")

# Initialize Tool-Calling RAG
tool_rag = ToolCallingRAG(llm, tools, max_tool_calls=5)
print(f"\n✅ Tool-Calling RAG system initialized!")

# Phase 4: Run Tool-Calling queries
print(f"\n📚 PHASE 4: TOOL-CALLING RAG QUERIES")
print("-"*70)

queries = [
    "What are the main themes in this story?",
    "Calculate the square root of 144 and then search for information about love in the document",
    "Analyze the sentiment of this text: 'This is a beautiful and touching story'"
]

results = []
for q in queries:
    result = tool_rag.query(q)
    results.append(result)

# Phase 5: Sentiment benchmark
print(f"\n📚 PHASE 5: VADER SENTIMENT BENCHMARK")
print("-"*70)

sa_start = time.time_ns()
sa = VaderSentimentAnalyzer()
sa_init = time.time_ns() - sa_start
latency_report.add("vader_init", sa_init)
print(f"✅ VADER INIT TIME: {format_time_ns(sa_init)}\n")

examples = [
    "I love this product!",
    "This is very bad service.",
    "It's okay, not too good, not too bad.",
    "Not great, really disappointed",
    "Amazing experience!"
]

run_sentiment_benchmark(sa, examples, 200_000, 1)

# Final summary
pipeline_total = time.time_ns() - pipeline_start
latency_report.add("pipeline_total", pipeline_total)

print(f"\n{'='*70}")
print(f"📈 PIPELINE SUMMARY")
print(f"{'='*70}")
print(f"Total pipeline time: {format_time_ns(pipeline_total)}")
print(f"Queries executed: {len(queries)}")
print(f"Average query time: {format_time_ns(sum(r['total_query_ns'] for r in results) // len(results))}")

print(f"\n🔧 Tool Usage Statistics:")
total_tool_calls = sum(r['tool_calls'] for r in results)
print(f"   Total tool calls: {total_tool_calls}")
print(f"   Average per query: {total_tool_calls / len(results):.1f}")

for i, r in enumerate(results, 1):
    print(f"   Query {i}: {r['tool_calls']} tool(s), {format_time_ns(r['total_query_ns'])}")

# Detailed latency report
latency_report.pretty_print()

print("✅ PIPELINE COMPLETE")