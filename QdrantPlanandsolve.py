#!/usr/bin/env python3
import os
"""
plan_solve_rag.py - Plan-and-Solve RAG
Creates a comprehensive plan first, then executes it systematically

Plan-and-Solve RAG Process:
1. PLAN: Break down query into logical sub-tasks
2. SOLVE: Execute each sub-task in sequence
3. AGGREGATE: Combine all results
4. ANSWER: Generate comprehensive final answer

Advantages:
- More structured than ReAct
- Better handling of complex queries
- Predictable execution flow
- Easier debugging
"""

import os
import time
import sys
import json
import re
import traceback
from collections import defaultdict
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------
# CONFIG
# ---------------------------
PDF_PATH = "Data/ECHOES OF HER LOVE.pdf"
INDEX_NAME = "new2"
DIM = 384
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.1-8b-instant"
TARGET_NS = 200_000

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# ---------------------------
# UTILITIES
# ---------------------------
def format_time_ns(ns: int) -> str:
    if ns < 1_000:
        return f"{ns} ns"
    if ns < 1_000_000:
        return f"{ns/1_000:.3f} µs"
    if ns < 1_000_000_000:
        return f"{ns/1_000_000:.3f} ms"
    return f"{ns/1_000_000_000:.3f} s"

class LatencyReport:
    def __init__(self):
        self.store = defaultdict(list)
    
    def add(self, component, ns):
        self.store[component].append(ns)
    
    def pretty_print(self):
        s = {}
        for comp, vals in self.store.items():
            total = sum(vals)
            s[comp] = {
                "count": len(vals),
                "total": format_time_ns(total),
                "avg": format_time_ns(total // len(vals) if vals else 0),
                "min": format_time_ns(min(vals) if vals else 0),
                "max": format_time_ns(max(vals) if vals else 0)
            }
        
        print("\n" + "="*70)
        print("LATENCY SUMMARY")
        print("="*70)
        for comp, stats in sorted(s.items()):
            print(f"\n📊 Component: {comp}")
            for k, v in stats.items():
                print(f"   {k.capitalize():10s} {v}")
        print("="*70 + "\n")

latency_report = LatencyReport()

# ---------------------------
# PDF/EMBEDDINGS/PINECONE
# ---------------------------
def load_pdf(path):
    start = time.time_ns()
    with pdfplumber.open(path) as pdf:
        page_texts = []
        for p in pdf.pages:
            t = p.extract_text() or ""
            page_texts.append(t)
        text = "\n".join(page_texts)
    elapsed = time.time_ns() - start
    latency_report.add("pipeline_pdf_load", elapsed)
    print(f"📄 Loaded PDF: {len(text)} chars ({format_time_ns(elapsed)})")
    return text

def chunk_text(text):
    start = time.time_ns()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)
    elapsed = time.time_ns() - start
    latency_report.add("pipeline_chunking", elapsed)
    print(f"📄 Created {len(chunks)} chunks ({format_time_ns(elapsed)})")
    return chunks

def get_embeddings_model():
    start = time.time_ns()
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    elapsed = time.time_ns() - start
    latency_report.add("pipeline_embeddings_load", elapsed)
    print(f"🧠 Embeddings loaded ({format_time_ns(elapsed)})")
    return emb

def init_pinecone(index_name):
    start = time.time_ns()
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [idx.name for idx in pc.list_indexes()]
    elapsed = time.time_ns() - start
    latency_report.add("pinecone_init", elapsed)
    
    if index_name not in existing:
        print(f"❌ ERROR: Index '{index_name}' does not exist!")
        sys.exit(1)
    
    print(f"✅ Connected to index '{index_name}' ({format_time_ns(elapsed)})")
    return pc

def create_vectorstore(embed, chunks, index_name):
    start = time.time_ns()
    vs = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embed
    )
    
    print(f"📤 Uploading {len(chunks)} chunks...")
    insert_start = time.time_ns()
    vs.add_texts(
        texts=chunks,
        metadatas=[{"chunk_id": i, "source": "plan_solve"} for i in range(len(chunks))]
    )
    insert_elapsed = time.time_ns() - insert_start
    latency_report.add("pipeline_insert_chunks", insert_elapsed)
    
    elapsed = time.time_ns() - start
    latency_report.add("pipeline_vectorstore_create", elapsed)
    print(f"✅ Vector store ready ({format_time_ns(elapsed)})")
    return vs

# ---------------------------
# VADER SENTIMENT
# ---------------------------
class VaderSentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    
    def analyze(self, text):
        scores = self.analyzer.polarity_scores(text)
        compound = scores['compound']
        if compound >= 0.05:
            return {'label': 'POSITIVE', 'compound': compound}
        elif compound <= -0.05:
            return {'label': 'NEGATIVE', 'compound': compound}
        return {'label': 'NEUTRAL', 'compound': compound}

def run_sentiment_benchmark(run_num, sa, examples):
    print(f"\n{'='*70}")
    print(f"🔥 SENTIMENT BENCHMARK RUN #{run_num}")
    print(f"{'='*70}")
    print(f"🎯 TARGET: < {TARGET_NS} ns per analysis\n")
    
    times = []
    for i, text in enumerate(examples, 1):
        start = time.time_ns()
        result = sa.analyze(text)
        elapsed = time.time_ns() - start
        times.append(elapsed)
        latency_report.add("vader_per_example", elapsed)
        
        status = "✅" if elapsed < TARGET_NS else "❌"
        print(f"[{i:2d}] {format_time_ns(elapsed):20s} {status} | {result['label']:8s} | \"{text}\"")
    
    total = sum(times)
    avg = total // len(times)
    
    print(f"\n📊 RUN #{run_num} STATISTICS:")
    print(f"   Total:        {format_time_ns(total)}")
    print(f"   Average:      {format_time_ns(avg)}")
    print(f"   Min:          {format_time_ns(min(times))}")
    print(f"   Max:          {format_time_ns(max(times))}")
    print(f"   < {TARGET_NS}ns: {sum(1 for t in times if t < TARGET_NS)}/{len(times)} texts")
    
    if avg < TARGET_NS:
        print(f"   ✅ TARGET MET")
    else:
        print(f"   ⚠️  TARGET MISSED")
    
    return avg

# ---------------------------
# PLAN-AND-SOLVE RAG
# ---------------------------
@dataclass
class SubTask:
    """Single sub-task in the plan"""
    id: int
    description: str
    type: str  # retrieve, analyze, compute, summarize
    params: Dict[str, Any]
    result: str = ""
    success: bool = False
    elapsed_ns: int = 0

class PlanAndSolveRAG:
    """
    Plan-and-Solve RAG Agent
    
    1. PLAN: Decompose query into logical sub-tasks
    2. SOLVE: Execute each sub-task sequentially
    3. AGGREGATE: Combine all results
    4. ANSWER: Generate comprehensive response
    """
    
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    def _llm_invoke(self, prompt: str, label: str) -> Tuple[str, int]:
        """Timed LLM invocation"""
        start = time.time_ns()
        try:
            response = self.llm.invoke(prompt)
            elapsed = time.time_ns() - start
            latency_report.add(label, elapsed)
            content = response.content if hasattr(response, 'content') else str(response)
            return content, elapsed
        except Exception as e:
            elapsed = time.time_ns() - start
            latency_report.add(label + "_error", elapsed)
            print(f"⚠️  LLM error: {e}")
            return str(e), elapsed
    
    def _create_plan(self, query: str) -> Tuple[List[SubTask], int]:
        """
        Phase 1: PLAN
        Decompose query into logical sub-tasks
        """
        
        prompt = f"""You are a planning assistant. Break down this query into logical sub-tasks.

Query: {query}

Available Task Types:
- retrieve: Search documents for information
- analyze: Analyze or process retrieved information
- compute: Perform calculations
- summarize: Create summaries

Create a step-by-step plan. Return as JSON:
{{
  "plan": [
    {{
      "id": 1,
      "description": "What to do",
      "type": "retrieve|analyze|compute|summarize",
      "params": {{"query": "search query" or other params}}
    }}
  ]
}}

Keep the plan simple and focused (2-5 steps).

Plan:"""
        
        plan_text, elapsed = self._llm_invoke(prompt, "llm_planning")
        
        # Parse plan
        subtasks = []
        try:
            json_match = re.search(r'\{.*\}', plan_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                plan_items = data.get('plan', [])
                
                for item in plan_items:
                    subtasks.append(SubTask(
                        id=item.get('id', len(subtasks) + 1),
                        description=item.get('description', ''),
                        type=item.get('type', 'retrieve'),
                        params=item.get('params', {})
                    ))
        except Exception as e:
            print(f"⚠️  Plan parsing error: {e}")
            # Fallback: create simple retrieve task
            subtasks.append(SubTask(
                id=1,
                description="Retrieve relevant information from documents",
                type="retrieve",
                params={"query": query}
            ))
        
        return subtasks, elapsed
    
    def _execute_retrieve(self, params: Dict) -> Tuple[str, int]:
        """Execute retrieval sub-task"""
        query = params.get('query', '')
        
        start = time.time_ns()
        try:
            docs = self.retriever.invoke(query)
            content = "\n\n".join([
                getattr(doc, "page_content", str(doc)) for doc in docs
            ])
            elapsed = time.time_ns() - start
            latency_report.add("subtask_retrieve", elapsed)
            
            return content, elapsed
        except Exception as e:
            elapsed = time.time_ns() - start
            print(f"⚠️  Retrieval error: {e}")
            return f"Error: {str(e)}", elapsed
    
    def _execute_analyze(self, params: Dict, context: str) -> Tuple[str, int]:
        """Execute analysis sub-task"""
        instruction = params.get('instruction', 'Analyze the information')
        
        prompt = f"""Analyze the following information according to the instruction.

Context:
{context[:2000]}

Instruction: {instruction}

Analysis:"""
        
        result, elapsed = self._llm_invoke(prompt, "subtask_analyze")
        latency_report.add("subtask_analyze_total", elapsed)
        return result, elapsed
    
    def _execute_compute(self, params: Dict) -> Tuple[str, int]:
        """Execute computation sub-task"""
        expression = params.get('expression', '')
        
        start = time.time_ns()
        try:
            # Safe eval
            allowed = {'abs': abs, 'round': round, 'min': min, 'max': max, 'sum': sum, 'len': len}
            result = eval(expression, {"__builtins__": {}}, allowed)
            elapsed = time.time_ns() - start
            latency_report.add("subtask_compute", elapsed)
            return str(result), elapsed
        except Exception as e:
            elapsed = time.time_ns() - start
            return f"Computation error: {str(e)}", elapsed
    
    def _execute_summarize(self, params: Dict, context: str) -> Tuple[str, int]:
        """Execute summarization sub-task"""
        max_words = params.get('max_words', 150)
        
        prompt = f"""Summarize the following in {max_words} words or less.

Content:
{context[:3000]}

Summary:"""
        
        result, elapsed = self._llm_invoke(prompt, "subtask_summarize")
        latency_report.add("subtask_summarize_total", elapsed)
        return result, elapsed
    
    def _execute_subtask(self, subtask: SubTask, accumulated_context: str) -> SubTask:
        """
        Phase 2: SOLVE
        Execute a single sub-task
        """
        
        print(f"\n   🔹 Executing Sub-task {subtask.id}")
        print(f"      Type: {subtask.type}")
        print(f"      Description: {subtask.description}")
        
        start = time.time_ns()
        
        if subtask.type == "retrieve":
            result, exec_elapsed = self._execute_retrieve(subtask.params)
            subtask.result = result
            subtask.success = bool(result and "Error" not in result)
        
        elif subtask.type == "analyze":
            result, exec_elapsed = self._execute_analyze(subtask.params, accumulated_context)
            subtask.result = result
            subtask.success = bool(result)
        
        elif subtask.type == "compute":
            result, exec_elapsed = self._execute_compute(subtask.params)
            subtask.result = result
            subtask.success = "error" not in result.lower()
        
        elif subtask.type == "summarize":
            result, exec_elapsed = self._execute_summarize(subtask.params, accumulated_context)
            subtask.result = result
            subtask.success = bool(result)
        
        else:
            result = f"Unknown task type: {subtask.type}"
            exec_elapsed = 0
            subtask.success = False
        
        subtask.elapsed_ns = time.time_ns() - start
        latency_report.add(f"subtask_{subtask.type}_total", subtask.elapsed_ns)
        
        status = "✅" if subtask.success else "❌"
        result_preview = subtask.result[:80].replace('\n', ' ')
        print(f"      {status} Result: {result_preview}...")
        print(f"      ⏱️  {format_time_ns(subtask.elapsed_ns)}")
        
        return subtask
    
    def _aggregate_results(self, subtasks: List[SubTask]) -> str:
        """
        Phase 3: AGGREGATE
        Combine all sub-task results
        """
        
        aggregated = []
        for st in subtasks:
            if st.success:
                aggregated.append(f"[Task {st.id}: {st.description}]")
                aggregated.append(st.result)
                aggregated.append("")
        
        return "\n".join(aggregated)
    
    def _generate_final_answer(self, query: str, aggregated_results: str, plan: List[SubTask]) -> Tuple[str, int]:
        """
        Phase 4: ANSWER
        Generate comprehensive final answer
        """
        
        plan_summary = "\n".join([
            f"{st.id}. {st.description} ({'✅' if st.success else '❌'})"
            for st in plan
        ])
        
        prompt = f"""Generate a comprehensive answer to the query using the results from the execution plan.

Query: {query}

Execution Plan:
{plan_summary}

Results:
{aggregated_results[:3000]}

Provide a clear, detailed answer that synthesizes all the information above.

Answer:"""
        
        answer, elapsed = self._llm_invoke(prompt, "llm_final_answer")
        return answer, elapsed
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Execute Plan-and-Solve RAG pipeline
        """
        print(f"\n{'='*70}")
        print(f"📋 PLAN-AND-SOLVE RAG")
        print(f"{'='*70}")
        print(f"❓ {question}\n")
        
        overall_start = time.time_ns()
        
        # PHASE 1: PLAN
        print(f"{'─'*70}")
        print("PHASE 1: PLANNING")
        print(f"{'─'*70}")
        
        plan, plan_time = self._create_plan(question)
        
        print(f"\n📋 Created plan with {len(plan)} sub-tasks ({format_time_ns(plan_time)}):")
        for st in plan:
            print(f"   {st.id}. [{st.type.upper()}] {st.description}")
        
        # PHASE 2: SOLVE
        print(f"\n{'─'*70}")
        print("PHASE 2: SOLVING")
        print(f"{'─'*70}")
        
        accumulated_context = ""
        
        for subtask in plan:
            subtask = self._execute_subtask(subtask, accumulated_context)
            
            # Accumulate successful results for next tasks
            if subtask.success:
                accumulated_context += f"\n\n{subtask.result}"
        
        successful_tasks = sum(1 for st in plan if st.success)
        print(f"\n   ✅ Completed {successful_tasks}/{len(plan)} tasks")
        
        # PHASE 3: AGGREGATE
        print(f"\n{'─'*70}")
        print("PHASE 3: AGGREGATING")
        print(f"{'─'*70}")
        
        aggregated = self._aggregate_results(plan)
        print(f"   ✅ Aggregated {len(aggregated)} chars of results")
        
        # PHASE 4: ANSWER
        print(f"\n{'─'*70}")
        print("PHASE 4: FINAL ANSWER")
        print(f"{'─'*70}")
        
        answer, answer_time = self._generate_final_answer(question, aggregated, plan)
        
        print(f"\n💬 ANSWER ({format_time_ns(answer_time)}):")
        print(f"{answer}\n")
        
        total_time = time.time_ns() - overall_start
        latency_report.add("plan_solve_query_total", total_time)
        
        print(f"⏱️  Total: {format_time_ns(total_time)}")
        
        return {
            'question': question,
            'answer': answer,
            'plan': [{'id': st.id, 'description': st.description, 'type': st.type, 'success': st.success} for st in plan],
            'successful_tasks': successful_tasks,
            'total_tasks': len(plan),
            'total_time': total_time
        }

# ---------------------------
# MAIN
# ---------------------------
def main():
    pipeline_start = time.time_ns()
    
    print("="*70)
    print("📋 PLAN-AND-SOLVE RAG PIPELINE")
    print("="*70 + "\n")
    
    # Setup
    text = load_pdf(PDF_PATH)
    chunks = chunk_text(text)
    embed = get_embeddings_model()
    pc = init_pinecone(INDEX_NAME)
    vs = create_vectorstore(embed, chunks, INDEX_NAME)
    
    print(f"\n✅ Initializing LLM...")
    llm_start = time.time_ns()
    llm = ChatGroq(model_name=MODEL_NAME, temperature=0, groq_api_key=GROQ_API_KEY)
    llm_elapsed = time.time_ns() - llm_start
    latency_report.add("llm_init", llm_elapsed)
    print(f"   ✅ LLM ready ({format_time_ns(llm_elapsed)})")
    
    # Initialize Plan-and-Solve RAG
    ps_rag = PlanAndSolveRAG(vs, llm)
    
    print("\n" + "="*70)
    print("PHASE 1: PLAN-AND-SOLVE RAG QUERIES")
    print("="*70)
    
    queries = [
        "What are the main themes in this story?",
        "Summarize the key events and analyze the emotional tone",
        "What is the capital of France?",
    ]
    
    results = []
    for i, q in enumerate(queries, 1):
        print(f"\n{'═'*70}")
        print(f"QUERY {i}/{len(queries)}")
        print(f"{'═'*70}")
        result = ps_rag.query(q)
        results.append(result)
        time.sleep(0.5)
    
    # VADER Benchmark
    print("\n\n" + "="*70)
    print("PHASE 2: VADER SENTIMENT BENCHMARK")
    print("="*70)
    
    vader_start = time.time_ns()
    sa = VaderSentimentAnalyzer()
    vader_init = time.time_ns() - vader_start
    latency_report.add("vader_init", vader_init)
    
    examples = [
        "I love this product!",
        "This is very bad service.",
        "It's okay, not too good, not too bad.",
        "Not great, really disappointed",
        "Amazing experience!"
    ]
    
    run_sentiment_benchmark(1, sa, examples)
    
    # Final Summary
    pipeline_total = time.time_ns() - pipeline_start
    latency_report.add("pipeline_total", pipeline_total)
    
    print("\n" + "="*70)
    print("📈 PIPELINE SUMMARY")
    print("="*70)
    
    total_tasks = sum(r['total_tasks'] for r in results)
    successful_tasks = sum(r['successful_tasks'] for r in results)
    avg_time = sum(r['total_time'] for r in results) // len(results)
    
    print(f"Total pipeline time: {format_time_ns(pipeline_total)}")
    print(f"Queries executed: {len(results)}")
    print(f"Average query time: {format_time_ns(avg_time)}")
    print(f"Total sub-tasks: {total_tasks}")
    print(f"Successful tasks: {successful_tasks}/{total_tasks} ({100*successful_tasks//total_tasks}%)")
    
    print("\n🧠 Plan-and-Solve Statistics:")
    for i, r in enumerate(results, 1):
        print(f"  Query {i}: {r['total_tasks']} tasks, {format_time_ns(r['total_time'])}")
    
    latency_report.pretty_print()
    print("✅ PIPELINE COMPLETE\n")

if __name__ == "__main__":
    main()