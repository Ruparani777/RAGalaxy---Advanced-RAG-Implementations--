import os
import time
import sys
import traceback
from collections import defaultdict
from typing import List, Dict, Any, Tuple
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
COLLECTION_NAME = "PlanSolveRAG_Documents"
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
                              description="Plan-and-Solve RAG document chunks")
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
# Plan-and-Solve Agentic RAG
# ---------------------------
class PlanAndSolveRAG:
    """
    Plan-and-Solve Agentic RAG System
    
    Workflow:
    1. PLAN: Decompose complex query into sub-tasks
    2. SOLVE: Execute each sub-task with retrieval
    3. SYNTHESIZE: Combine results into final answer
    4. VERIFY: Check completeness and quality
    """
    
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

    def plan_decomposition(self, query: str) -> Tuple[List[Dict[str, str]], int]:
        """
        STEP 1: Decompose the query into sub-tasks
        Returns: List of sub-tasks with types and queries
        """
        prompt = f"""You are a strategic planning agent. Analyze this query and decompose it into logical sub-tasks.

Query: {query}

Decompose this into 2-4 sub-tasks. For each sub-task, specify:
1. task_id: (e.g., "task_1", "task_2")
2. task_type: ("retrieve" for document search, "analyze" for reasoning, "synthesize" for combining)
3. description: What needs to be done
4. query: The specific question or search query for this sub-task

Format your response as a numbered list:
1. [task_id] (task_type): description
   Query: specific query

Example:
1. [task_1] (retrieve): Find information about main themes
   Query: What are the main themes?
2. [task_2] (analyze): Identify character relationships
   Query: How do characters relate to each other?

Sub-tasks:"""
        
        response, elapsed = self._llm_invoke_timed(prompt, "llm_plan_decomposition")
        
        # Parse the response into structured sub-tasks
        sub_tasks = self._parse_plan(response)
        
        print(f"📋 PLAN: Generated {len(sub_tasks)} sub-tasks")
        for i, task in enumerate(sub_tasks, 1):
            print(f"   [{i}] {task['task_type'].upper()}: {task['description'][:60]}...")
        
        return sub_tasks, elapsed

    def _parse_plan(self, plan_text: str) -> List[Dict[str, str]]:
        """Parse the plan text into structured sub-tasks"""
        sub_tasks = []
        lines = plan_text.strip().split('\n')
        
        current_task = {}
        task_counter = 1
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line starts with a number (new task)
            if line[0].isdigit() and '.' in line[:3]:
                # Save previous task if exists
                if current_task and 'description' in current_task:
                    sub_tasks.append(current_task)
                
                # Start new task
                current_task = {
                    'task_id': f"task_{task_counter}",
                    'task_type': 'retrieve',  # default
                    'description': '',
                    'query': ''
                }
                task_counter += 1
                
                # Parse task type and description
                if '(retrieve)' in line.lower():
                    current_task['task_type'] = 'retrieve'
                elif '(analyze)' in line.lower():
                    current_task['task_type'] = 'analyze'
                elif '(synthesize)' in line.lower():
                    current_task['task_type'] = 'synthesize'
                
                # Extract description (after the colon)
                if ':' in line:
                    current_task['description'] = line.split(':', 1)[1].strip()
                else:
                    current_task['description'] = line
                    
            elif line.lower().startswith('query:'):
                # Extract query
                current_task['query'] = line.split(':', 1)[1].strip()
        
        # Add last task
        if current_task and 'description' in current_task:
            sub_tasks.append(current_task)
        
        # If parsing failed, create a default task
        if not sub_tasks:
            sub_tasks = [{
                'task_id': 'task_1',
                'task_type': 'retrieve',
                'description': 'Answer the query',
                'query': plan_text[:200]
            }]
        
        return sub_tasks

    def solve_subtask(self, sub_task: Dict[str, str]) -> Tuple[str, int]:
        """
        STEP 2: Solve a single sub-task
        """
        task_type = sub_task['task_type']
        query = sub_task['query'] or sub_task['description']
        
        print(f"\n   🔧 Solving: {sub_task['task_id']} ({task_type})")
        
        total_time = 0
        
        if task_type == 'retrieve':
            # Retrieve relevant documents
            context, retrieval_time = self.retrieve_documents(query, k=3)
            total_time += retrieval_time
            
            # Generate answer based on context
            answer, gen_time = self._generate_with_context(query, context)
            total_time += gen_time
            
        elif task_type == 'analyze':
            # May retrieve for analysis, but focus on reasoning
            context, retrieval_time = self.retrieve_documents(query, k=2)
            total_time += retrieval_time
            
            analyze_prompt = f"""Analyze the following information and provide insights:

Context: {context if context else "No specific context available."}

Task: {sub_task['description']}
Query: {query}

Provide a thoughtful analysis:"""
            
            answer, gen_time = self._llm_invoke_timed(analyze_prompt, "llm_analyze_subtask")
            total_time += gen_time
            
        else:  # synthesize or default
            # Use general knowledge or previous context
            answer, gen_time = self._generate_with_context(query, "")
            total_time += gen_time
        
        print(f"   ✅ Completed in {format_time_ns(total_time)}")
        return answer, total_time

    def retrieve_documents(self, query: str, k: int = 4) -> Tuple[str, int]:
        """Retrieve relevant documents from vector store"""
        hits, elapsed = search_milvus(self.collection, self.embedder, query, k)
        context = "\n\n".join(hits)
        return context, elapsed

    def _generate_with_context(self, query: str, context: str) -> Tuple[str, int]:
        """Generate answer with or without context"""
        if context:
            prompt = f"""Based on the following context, answer the question:

Context:
{context}

Question: {query}

Answer:"""
        else:
            prompt = f"""Answer the following question:

Question: {query}

Answer:"""
        
        return self._llm_invoke_timed(prompt, "llm_generate_answer")

    def synthesize_results(self, query: str, sub_results: List[Dict[str, Any]]) -> Tuple[str, int]:
        """
        STEP 3: Synthesize all sub-task results into final answer
        """
        print(f"\n   🔄 Synthesizing {len(sub_results)} results...")
        
        # Compile all sub-results
        results_text = ""
        for i, res in enumerate(sub_results, 1):
            results_text += f"\n\nSub-task {i} ({res['task_id']}):\n"
            results_text += f"Description: {res['description']}\n"
            results_text += f"Result: {res['answer'][:300]}...\n"
        
        synthesis_prompt = f"""You are synthesizing multiple sub-task results to answer the original query.

Original Query: {query}

Sub-task Results:
{results_text}

Instructions:
1. Combine insights from all sub-tasks
2. Create a coherent, comprehensive answer
3. Highlight key findings
4. Ensure logical flow

Final Synthesized Answer:"""
        
        final_answer, elapsed = self._llm_invoke_timed(synthesis_prompt, "llm_synthesize")
        print(f"   ✅ Synthesis completed in {format_time_ns(elapsed)}")
        
        return final_answer, elapsed

    def verify_solution(self, query: str, final_answer: str, sub_results: List[Dict]) -> Tuple[str, bool, int]:
        """
        STEP 4: Verify the solution quality and completeness
        """
        verify_prompt = f"""You are a quality verification agent. Evaluate the answer for completeness and accuracy.

Original Query: {query}

Final Answer:
{final_answer}

Number of sub-tasks completed: {len(sub_results)}

Evaluate:
1. Completeness Score (1-10): Does it fully address the query?
2. Coherence Score (1-10): Is it well-organized and logical?
3. Accuracy Score (1-10): Is the information accurate based on available evidence?
4. Missing Elements: What's missing, if anything?
5. Recommendation: ACCEPT or REPLAN (if significant gaps exist)

Verification Report:"""
        
        verification, elapsed = self._llm_invoke_timed(verify_prompt, "llm_verify")
        
        # Determine if we need to replan
        needs_replan = "REPLAN" in verification.upper()
        
        print(f"   ✅ Verification completed in {format_time_ns(elapsed)}")
        print(f"   Status: {'🔄 REPLAN NEEDED' if needs_replan else '✅ ACCEPTED'}")
        
        return verification, needs_replan, elapsed

    def query(self, question: str, max_iterations: int = 2) -> Dict[str, Any]:
        """
        Main Plan-and-Solve query execution
        """
        print("\n" + "="*70)
        print("🚀 PLAN-AND-SOLVE AGENTIC RAG")
        print("="*70)
        print(f"❓ Question: {question}\n")
        
        overall_start = time.time_ns()
        iteration = 0
        final_answer = ""
        verification_report = ""
        all_iterations = []
        
        while iteration < max_iterations:
            iter_start = time.time_ns()
            iteration += 1
            
            print(f"\n{'='*70}")
            print(f"ITERATION {iteration}")
            print(f"{'='*70}")
            
            # STEP 1: PLAN - Decompose query
            print("\n🎯 STEP 1: PLANNING")
            print("-" * 70)
            sub_tasks, plan_time = self.plan_decomposition(question)
            
            # STEP 2: SOLVE - Execute sub-tasks
            print(f"\n⚙️  STEP 2: SOLVING ({len(sub_tasks)} sub-tasks)")
            print("-" * 70)
            sub_results = []
            solve_start = time.time_ns()
            
            for sub_task in sub_tasks:
                answer, task_time = self.solve_subtask(sub_task)
                sub_results.append({
                    'task_id': sub_task['task_id'],
                    'task_type': sub_task['task_type'],
                    'description': sub_task['description'],
                    'query': sub_task['query'],
                    'answer': answer,
                    'time_ns': task_time
                })
            
            solve_time = time.time_ns() - solve_start
            
            # STEP 3: SYNTHESIZE - Combine results
            print(f"\n🔗 STEP 3: SYNTHESIS")
            print("-" * 70)
            final_answer, synthesis_time = self.synthesize_results(question, sub_results)
            
            # STEP 4: VERIFY - Check quality
            print(f"\n✔️  STEP 4: VERIFICATION")
            print("-" * 70)
            verification_report, needs_replan, verify_time = self.verify_solution(
                question, final_answer, sub_results
            )
            
            iter_elapsed = time.time_ns() - iter_start
            latency_report.add("plan_solve_iteration", iter_elapsed)
            
            all_iterations.append({
                'iteration': iteration,
                'plan': sub_tasks,
                'results': sub_results,
                'final_answer': final_answer,
                'verification': verification_report,
                'time_ns': iter_elapsed
            })
            
            print(f"\n⏱️  Iteration {iteration} total time: {format_time_ns(iter_elapsed)}")
            
            # Check if we should continue
            if not needs_replan or iteration >= max_iterations:
                print(f"\n✅ Plan-and-Solve completed after {iteration} iteration(s)")
                break
            else:
                print(f"\n🔄 Replanning needed, starting iteration {iteration + 1}...")
        
        total_query_ns = time.time_ns() - overall_start
        latency_report.add("plan_solve_query_total", total_query_ns)
        
        # Display final answer
        print("\n" + "="*70)
        print("💬 FINAL ANSWER:")
        print("="*70)
        print(final_answer[:800])
        if len(final_answer) > 800:
            print("...")
        print(f"\n⏱️  Total query time: {format_time_ns(total_query_ns)}")
        print("="*70 + "\n")
        
        return {
            "question": question,
            "final_answer": final_answer,
            "iterations": iteration,
            "all_iterations": all_iterations,
            "verification": verification_report,
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
    print("🚀 PLAN-AND-SOLVE AGENTIC RAG + FULL LATENCY INSTRUMENTATION")
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

    # Initialize Plan-and-Solve RAG
    plan_solve_rag = PlanAndSolveRAG(llm, milvus_collection, embedder)
    print("\n✅ Plan-and-Solve Agentic RAG system initialized!")

    # Phase 3: Run queries
    print("\n📚 PHASE 3: PLAN-AND-SOLVE RAG QUERIES")
    print("-" * 70)
    queries = [
        "What are the main themes and character relationships in this story?",
        "Summarize the key events and emotional arcs in the document.",
        "Compare the narrative structure and writing style used throughout."
    ]
    results = []
    for q in queries:
        result = plan_solve_rag.query(q, max_iterations=2)
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