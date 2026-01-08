#!/usr/bin/env python3
"""
Milvusselfrag_full_ready.py
Complete Self-RAG pipeline (Milvus + embeddings + Groq LLM) with full latency instrumentation.

Assumptions:
- Milvus running at localhost:19530
- SentenceTransformers available
- GROQ_API_KEY environment variable set
- Python venv activated
"""

import os
import time
import sys
import traceback
from collections import defaultdict
from typing import List, Dict, Any, Tuple

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
COLLECTION_NAME = "SelfRAG_Documents"
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
                              description="Self-RAG document chunks")
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
    collection.insert([texts, sources, embeddings])  # chunk_id auto-generated
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
                # newer pymilvus returns entity as a dict in .entity or ._fields
                ent = getattr(hit, "entity", None) or getattr(hit, "_fields", None) or {}
                if isinstance(ent, dict):
                    txt = ent.get("text", "")
                else:
                    # fallback: hit._raw or str
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
# Self-RAG LLM wrappers
# ---------------------------
class SelfRAG:
    def __init__(self, llm, collection: Collection, embedder: SentenceTransformer):
        self.llm = llm
        self.collection = collection
        self.embedder = embedder

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

    def retrieve_decision(self, query: str) -> Tuple[bool, str, int]:
        prompt = f"""You are a helpful assistant. Decide if you need to retrieve information from a document to answer this question.

Question: {query}

Think step by step:
1. Can you answer this from general knowledge?
2. Does it require specific document information?

Answer with ONLY 'RETRIEVE' or 'NO_RETRIEVE' and a brief reason.

Decision:"""
        decision_text, elapsed = self._llm_invoke_timed(prompt, "llm_retrieve_decision")
        needs_retrieval = "RETRIEVE" in decision_text.upper() and "NO_RETRIEVE" not in decision_text.upper()
        print(f"🤔 Retrieval Decision: {'RETRIEVE' if needs_retrieval else 'NO_RETRIEVE'}")
        print(f"   Reasoning: {decision_text.strip()[:200]}...")
        return needs_retrieval, decision_text, elapsed

    def retrieve_documents(self, query: str, k: int = 4) -> Tuple[str, int]:
        hits, elapsed = search_milvus(self.collection, self.embedder, query, k)
        context = "\n\n".join(hits)
        print(f"   ✅ Retrieved {len(hits)} documents in {format_time_ns(elapsed)}")
        return context, elapsed

    def generate_answer(self, query: str, context: str = "") -> Tuple[str, int]:
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
        answer, elapsed = self._llm_invoke_timed(prompt, "llm_generate_answer")
        print(f"   ✅ Answer generated in {format_time_ns(elapsed)}")
        return answer, elapsed

    def self_critique(self, query: str, answer: str, context: str = "") -> Tuple[str, bool, int]:
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
        critique_text, elapsed = self._llm_invoke_timed(critique_prompt, "llm_self_critique")
        needs_more = "YES" in critique_text.upper() and "RETRIEVE" in critique_text.upper()
        print(f"   ✅ Critique completed in {format_time_ns(elapsed)}")
        return critique_text, needs_more, elapsed

    def query(self, question: str, max_iterations: int = 2) -> Dict[str, Any]:
        print("\n" + "="*70)
        print("🚀 SELF-RAG QUERY PROCESSING")
        print("="*70)
        print(f"❓ Question: {question}\n")
        overall_start = time.time_ns()
        iteration = 0
        context = ""
        answer = ""
        per_iteration_times = []
        critique_text = ""
        while iteration < max_iterations:
            iter_start = time.time_ns()
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")
            if iteration == 1:
                needs_retrieval, decision_reason, decision_time = self.retrieve_decision(question)
            else:
                needs_retrieval = True
            if needs_retrieval:
                context, retrieval_time = self.retrieve_documents(question)
                print(f"   📝 Context length: {len(context)} characters")
            answer, gen_time = self.generate_answer(question, context)
            critique_text, needs_more, critique_time = self.self_critique(question, answer, context)
            iter_elapsed = time.time_ns() - iter_start
            per_iteration_times.append(iter_elapsed)
            latency_report.add("selfrag_iteration", iter_elapsed)
            print(f"\n⏱️  Iteration {iteration} total time: {format_time_ns(iter_elapsed)}")
            if not needs_more or iteration >= max_iterations:
                print(f"\n✅ Self-RAG completed after {iteration} iteration(s)")
                break
            else:
                print(f"\n🔄 Refinement needed, starting iteration {iteration + 1}...")
        total_query_ns = time.time_ns() - overall_start
        latency_report.add("selfrag_query_total", total_query_ns)
        print("\n" + "="*70)
        print("💬 FINAL ANSWER:")
        print("="*70)
        print(answer[:800])
        if len(answer) > 800:
            print("...")
        print(f"\n⏱️  Total query time: {format_time_ns(total_query_ns)}")
        print("="*70 + "\n")
        return {
            "question": question,
            "answer": answer,
            "context": context,
            "iterations": iteration,
            "critique": critique_text,
            "per_iteration_times": per_iteration_times,
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
    print("🚀 SELF-RAG MILVUS + FULL LATENCY INSTRUMENTATION")
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

    # Initialize Self-RAG
    self_rag = SelfRAG(llm, milvus_collection, embedder)
    print("\n✅ Self-RAG system initialized!")

    # Phase 3: Run queries
    print("\n📚 PHASE 3: SELF-RAG QUERIES")
    print("-" * 70)
    queries = [
        "What are the main themes in this story?",
        "Summarize the key events in the document.",
        "What is the capital of France?"
    ]
    results = []
    for q in queries:
        result = self_rag.query(q, max_iterations=2)
        results.append(result)

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
