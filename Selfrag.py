#!/usr/bin/env python3
import os
import sys
"""
selfrag_full.py
Full pipeline with comprehensive nanosecond latency instrumentation.

- Use environment variables for secrets:
    PINECONE_API_KEY, GROQ_API_KEY

Notes:
- This file keeps the same high-level structure as your original script but:
  * Replaces time.time() with time.time_ns()
  * Adds timers for each major component and LLM invocation
  * Prints a final latency report
  * Avoids hard-coded API keys (use env vars instead)
"""

import os
import time
import sys
import math
import pprint
import traceback
from collections import defaultdict

# third-party imports (same as your original requirements)
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone, ServerlessSpec
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------
# Config (use env vars; no secrets in source)
# ---------------------------
PDF_PATH = "Data/ECHOES OF HER LOVE.pdf"
INDEX_NAME = "pinecone"
DIM = 384  # MiniLM embedding dimension
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.1-8b-instant"
TARGET_NS = 200_000  # example target for sentiment inference

if PINECONE_API_KEY is None:
    print("ERROR: Set PINECONE_API_KEY environment variable before running.")
    sys.exit(1)

# ---------------------------
# Utilities: time formatting + timing helpers
# ---------------------------
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
        print(f"⏱️ {func.__name__} time: {format_time_ns(elapsed)}")
        wrapper.last_elapsed_ns = elapsed
        return result
    wrapper.last_elapsed_ns = None
    return wrapper

# A simple latency aggregator to produce final report
class LatencyReport:
    def __init__(self):
        self.store = defaultdict(list)
    def add(self, component, ns):
        self.store[component].append(ns)
    def summary(self):
        out = {}
        for comp, vals in self.store.items():
            total = sum(vals)
            out[comp] = {
                "count": len(vals),
                "total_ns": total,
                "avg_ns": total // len(vals),
                "min_ns": min(vals),
                "max_ns": max(vals)
            }
        return out
    def pretty_print(self):
        s = self.summary()
        print("\n" + "="*60)
        print("LATENCY SUMMARY (nanoseconds)")
        print("="*60)
        for comp, stats in sorted(s.items(), key=lambda p: p[0]):
            print(f"\nComponent: {comp}")
            print(f"  Count: {stats['count']}")
            print(f"  Total: {format_time_ns(stats['total_ns'])}")
            print(f"  Avg:   {format_time_ns(stats['avg_ns'])}")
            print(f"  Min:   {format_time_ns(stats['min_ns'])}")
            print(f"  Max:   {format_time_ns(stats['max_ns'])}")
        print("\n" + "="*60 + "\n")


latency_report = LatencyReport()

# ---------------------------
# PDF load / chunking / embeddings / pinecone
# ---------------------------
@timer_ns
def load_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        pages = pdf.pages
        # read page by page and time per-page (helpful for large PDFs)
        page_texts = []
        for i, p in enumerate(pages):
            start_ns = time.time_ns()
            t = p.extract_text() or ""
            elapsed = time.time_ns() - start_ns
            latency_report.add("pdf_page_extract", elapsed)
            page_texts.append(t)
        text = "\n".join(page_texts)
    print(f"📄 Loaded PDF, total length: {len(text)} chars")
    return text

@timer_ns
def chunk_text(text, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    print(f"📄 Total Chunks: {len(chunks)}")
    return chunks

@timer_ns
def get_embeddings_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    # instantiate embedding model and return wrapper
    emb = HuggingFaceEmbeddings(model_name=model_name)
    return emb

def init_pinecone(api_key, index_name=INDEX_NAME, dim=DIM):
    start = time.time_ns()
    pc = Pinecone(api_key=api_key)
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    latency_report.add("pinecone_list_indexes", time.time_ns() - start)

    if index_name in existing_indexes:
        print(f"🗑️  Deleting existing index '{index_name}'...")
        start = time.time_ns()
        pc.delete_index(index_name)
        latency_report.add("pinecone_delete_index", time.time_ns() - start)
        time.sleep(2)

    print(f"🆕 Creating index '{index_name}'...")
    start = time.time_ns()
    pc.create_index(
        name=index_name,
        dimension=dim,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    latency_report.add("pinecone_create_index", time.time_ns() - start)
    time.sleep(2)
    print(f"✅ Index '{index_name}' created")
    return pc

@timer_ns
def create_vectorstore(embed_model, chunks, index_name=INDEX_NAME):
    # measure embedding transformation + upsert to pinecone in blocks
    # Note: from_texts will call embedding model internally - we measure the wrapper call
    start = time.time_ns()
    vectorstore = PineconeVectorStore.from_texts(
        texts=chunks,
        embedding=embed_model,
        index_name=index_name,
        namespace="",
        metadatas=[{"source": f"chunk_{i}", "chunk_id": i} for i in range(len(chunks))]
    )
    elapsed = time.time_ns() - start
    latency_report.add("pinecone_upsert_total", elapsed)
    print(f"✅ Created vector store with {len(chunks)} chunks")
    return vectorstore

# ---------------------------
# Vader Sentiment (benchmark)
# ---------------------------
class VaderSentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    def analyze(self, text):
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

def run_sentiment_benchmark(run_number, sa, examples, target_ns=TARGET_NS):
    print(f"\n{'='*70}")
    print(f"🔥 SENTIMENT RUN #{run_number}")
    print(f"{'='*70}")
    individual_times = []
    for i, text in enumerate(examples, 1):
        start_ns = time.time_ns()
        result = sa.analyze(text)
        elapsed_ns = time.time_ns() - start_ns
        latency_report.add("vader_per_example", elapsed_ns)
        individual_times.append(elapsed_ns)
        status = "✅" if elapsed_ns < target_ns else "❌"
        print(f"[{i:2d}] {format_time_ns(elapsed_ns):20s} {status} | {result['label']:8s} | \"{text}\"")
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
    return {
        'run': run_number,
        'times': individual_times,
        'total': total_ns,
        'avg': avg_ns,
        'min': min_ns,
        'max': max_ns,
        'under_target': under_target
    }

# ---------------------------
# SELF-RAG with detailed LLM + retriever timings
# ---------------------------
class SelfRAG:
    """Self-Reflective RAG with full timing instrumentation."""
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
        # create retriever object (if supported)
        self.retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    def _llm_invoke_timed(self, prompt, label):
        """Invoke LLM and record elapsed time under label."""
        # some LLMs provide .invoke(...) returning .content - keep robust
        start = time.time_ns()
        try:
            response = self.llm.invoke(prompt)
            elapsed = time.time_ns() - start
            latency_report.add(label, elapsed)
            # try to extract textual content
            content = response.content if hasattr(response, 'content') else str(response)
            return content, elapsed
        except Exception as e:
            elapsed = time.time_ns() - start
            latency_report.add(label + "_error", elapsed)
            print(f"LLM invoke for {label} failed: {e}")
            traceback.print_exc()
            return str(e), elapsed

    def retrieve_decision(self, question):
        prompt = f"""You are a helpful assistant. Decide if you need to retrieve information from a document to answer this question.

Question: {question}

Think step by step:
1. Can you answer this from general knowledge?
2. Does it require specific document information?

Answer with ONLY 'RETRIEVE' or 'NO_RETRIEVE' and a brief reason.

Decision:"""
        decision_text, elapsed = self._llm_invoke_timed(prompt, "llm_retrieve_decision")
        needs_retrieval = 'RETRIEVE' in decision_text.upper() and 'NO_RETRIEVE' not in decision_text.upper()
        print(f"🤔 Retrieval Decision: {'RETRIEVE' if needs_retrieval else 'NO_RETRIEVE'}")
        print(f"   Reasoning: {decision_text.strip()}")
        return needs_retrieval, decision_text

    def retrieve_documents(self, question, k=4):
        start = time.time_ns()
        try:
            docs = self.retriever.invoke(question)
            elapsed = time.time_ns() - start
            latency_report.add("retriever_search", elapsed)
            # docs might be list-like
            count = len(docs) if hasattr(docs, "__len__") else 1
            print(f"📚 Retrieved {count} documents in {format_time_ns(elapsed)}")
            return docs, elapsed
        except Exception as e:
            elapsed = time.time_ns() - start
            latency_report.add("retriever_search_error", elapsed)
            print(f"Retriever failed: {e}")
            traceback.print_exc()
            return [], elapsed

    def generate_answer(self, question, context=""):
        if context:
            prompt = f"""Answer the question based on the following context:

Context:
{context}

Question: {question}

Provide a detailed answer based on the context above.

Answer:"""
        else:
            prompt = f"""Answer the following question based on your general knowledge:

Question: {question}

Answer:"""
        answer_text, elapsed = self._llm_invoke_timed(prompt, "llm_generate_answer")
        print(f"\n💬 Generated Answer (took {format_time_ns(elapsed)}):\n{answer_text[:1000]}")  # truncate long prints
        return answer_text, elapsed

    def self_critique(self, question, answer, context=""):
        critique_prompt = f"""You are a critical evaluator. Evaluate the following answer.

Question: {question}

Answer: {answer}

Context Available: {'Yes' if context else 'No'}

Rate the answer on a scale of 1-10 and provide:
1. Relevance Score (1-10)
2. Completeness Score (1-10)
3. Accuracy Assessment
4. Should we retrieve more information? (YES/NO)

Evaluation:"""
        critique_text, elapsed = self._llm_invoke_timed(critique_prompt, "llm_self_critique")
        print(f"\n🔍 Self-Critique (took {format_time_ns(elapsed)}):\n{critique_text.strip()}")
        # decide if more retrieval is suggested
        needs_more = 'YES' in critique_text.upper() and 'RETRIEVE' in critique_text.upper()
        return critique_text, needs_more, elapsed

    def query(self, question, max_iterations=2):
        print(f"\n{'='*70}")
        print(f"🚀 SELF-RAG QUERY PROCESSING")
        print(f"{'='*70}")
        print(f"❓ Question: {question}\n")

        iteration = 0
        context = ""
        answer = ""
        per_iteration_times = []

        overall_start = time.time_ns()
        while iteration < max_iterations:
            iter_start = time.time_ns()
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")

            # Step 1: Decide if retrieval is needed
            if iteration == 1:
                needs_retrieval, decision_reason = self.retrieve_decision(question)
            else:
                needs_retrieval = True  # force retrieval when refining

            # Step 2: Retrieve if needed
            retriever_elapsed = 0
            if needs_retrieval:
                docs, retriever_elapsed = self.retrieve_documents(question)
                if docs:
                    # join docs' content (robust extraction)
                    context = "\n\n".join([
                        getattr(doc, "page_content", None) or getattr(doc, "content", None) or str(doc)
                        for doc in docs
                    ])
                    latency_report.add("context_build", len(context))  # note: storing char count as metric
                    print(f"📝 Context length: {len(context)} characters")

            # Step 3: Generate answer
            gen_answer, gen_elapsed = self.generate_answer(question, context)

            # Step 4: Self-critique
            critique_text, needs_more, critique_elapsed = self.self_critique(question, gen_answer, context)

            iter_elapsed = time.time_ns() - iter_start
            per_iteration_times.append(iter_elapsed)
            latency_report.add("selfrag_iteration", iter_elapsed)
            print(f"\n⏱️ Iteration {iteration} total time: {format_time_ns(iter_elapsed)}")

            # Step 5: Decide loop break
            if not needs_more or iteration >= max_iterations:
                print(f"\n✅ Self-RAG completed after {iteration} iteration(s)")
                break
            else:
                print(f"\n🔄 Refinement needed, starting iteration {iteration + 1}...")

        total_query_ns = time.time_ns() - overall_start
        latency_report.add("selfrag_query_total", total_query_ns)
        return {
            'question': question,
            'answer': gen_answer,
            'context': context,
            'iterations': iteration,
            'critique': critique_text,
            'per_iteration_times': per_iteration_times,
            'total_query_ns': total_query_ns
        }

# ---------------------------
# Main pipeline orchestration
# ---------------------------
def main():
    print("="*70)
    print("🚀 SELF-RAG PIPELINE + FULL LATENCY INSTRUMENTATION")
    print("="*70)

    # Phase 1: Load PDF + chunk + embeddings + pinecone init + upsert
    start_total = time.time_ns()
    try:
        pdf_text, t_pdf = timed_call(load_pdf, PDF_PATH)
        latency_report.add("pdf_load", t_pdf)

        chunks, t_chunks = timed_call(chunk_text, pdf_text, 1000, 100)
        latency_report.add("chunking", t_chunks)

        embed_model, t_emb_load = timed_call(get_embeddings_model, "sentence-transformers/all-MiniLM-L6-v2")
        latency_report.add("embedding_model_init", t_emb_load)

        # pinecone init + index create
        pc_start = time.time_ns()
        pc = init_pinecone(PINECONE_API_KEY, INDEX_NAME, DIM)
        pc_elapsed = time.time_ns() - pc_start
        latency_report.add("pinecone_init_total", pc_elapsed)

        # create vectorstore (embedding + upsert)
        vs_start = time.time_ns()
        vectorstore = create_vectorstore(embed_model, chunks, INDEX_NAME)
        vs_elapsed = time.time_ns() - vs_start
        latency_report.add("vectorstore_create_total", vs_elapsed)

    except Exception as e:
        print("Error in Phase 1:", e)
        traceback.print_exc()
        return

    # Initialize LLM
    try:
        start_ns = time.time_ns()
        llm = ChatGroq(model_name=MODEL_NAME, temperature=0, groq_api_key=GROQ_API_KEY)
        elapsed_ns = time.time_ns() - start_ns
        latency_report.add("llm_init", elapsed_ns)
        print(f"✅ LLM initialized in {format_time_ns(elapsed_ns)}")
    except Exception as e:
        print("LLM init failed:", e)
        traceback.print_exc()
        return

    # Initialize Self-RAG
    self_rag = SelfRAG(vectorstore, llm)
    print("\n✅ Self-RAG system initialized!")

    # Phase 2: Self-RAG queries (example set)
    print("\n\n📚 PHASE 2: SELF-RAG QUERIES")
    queries = [
        "What are the main themes in this story?",
        "Summarize the key events in the document.",
        "What is the capital of France?"  # general knowledge
    ]
    rag_results = []
    for q in queries:
        q_start = time.time_ns()
        res = self_rag.query(q, max_iterations=2)
        q_elapsed = time.time_ns() - q_start
        latency_report.add("query_loop", q_elapsed)
        rag_results.append(res)
        print(f"\n{'='*70}\n")

    # Phase 3: VADER Sentiment Benchmark
    print("\n📚 PHASE 3: VADER SENTIMENT BENCHMARK")
    print("-"*70)
    print(f"🎯 TARGET: < {TARGET_NS} ns per analysis\n")

    sa_start = time.time_ns()
    sa = VaderSentimentAnalyzer()
    sa_init_ns = time.time_ns() - sa_start
    latency_report.add("vader_init", sa_init_ns)
    print(f"✅ VADER INIT TIME: {format_time_ns(sa_init_ns)}\n")

    examples = [
        "I love this product!",
        "This is very bad service.",
        "It's okay, not too good, not too bad.",
        "Not great, really disappointed",
        "Amazing experience!"
    ]
    runs = []
    for run in range(1, 4):
        r = run_sentiment_benchmark(run, sa, examples, TARGET_NS)
        runs.append(r)
        time.sleep(0.1)

    # Final aggregation & report
    overall_ns = time.time_ns() - start_total
    latency_report.add("pipeline_total", overall_ns)
    print("\n📈 AGGREGATE STATISTICS")
    print(f"   Full pipeline time: {format_time_ns(overall_ns)}")
    print(f"   Queries executed: {len(queries)}")
    latency_report.pretty_print()

    # Optionally pretty-print RAG results
    print("\nSample RAG results (truncated answers):")
    for r in rag_results:
        print("-" * 40)
        print(f"Q: {r['question']}")
        ans_preview = (r['answer'] or "")[:800]
        print(f"A (preview): {ans_preview}")
        print(f"Iterations: {r['iterations']}, Query time: {format_time_ns(r['total_query_ns'])}")
    print("\n✅ PIPELINE COMPLETE")

if __name__ == "__main__":
    main()
