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
COLLECTION_NAME = "ReActRAG_Documents"
DIM = 384
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = "llama-3.1-8b-instant"
TARGET_NS = 200_000
MAX_ITERATIONS = 5  # Maximum ReAct iterations
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
                              description="ReAct RAG document chunks")
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
    
    total_time = encode_time + search_time
    return hits, total_time

# ---------------------------
# ReAct Agentic RAG System
# ---------------------------
class ReActAgenticRAG:
    """
    ReAct (Reasoning + Acting) Agent for RAG
    The agent follows the ReAct pattern:
    1. Thought: Reason about what to do next
    2. Action: Take an action (search, lookup, calculate, etc.)
    3. Observation: Observe the result
    4. Repeat until answer is found
    """
    
    def __init__(self, llm, collection: Collection, embedder: SentenceTransformer, max_iterations: int = MAX_ITERATIONS):
        self.llm = llm
        self.collection = collection
        self.embedder = embedder
        self.max_iterations = max_iterations
        self.available_tools = {
            "search_documents": self.search_documents,
            "lookup_specific": self.lookup_specific,
            "calculate": self.calculate,
            "finish": self.finish
        }
        
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
    
    def search_documents(self, query: str) -> Tuple[str, int]:
        """Tool: Search for relevant documents"""
        hits, elapsed = search_milvus(self.collection, self.embedder, query, limit=3)
        
        if hits:
            result = "Found documents:\n" + "\n---\n".join([f"Doc {i+1}: {doc[:200]}..." for i, doc in enumerate(hits)])
        else:
            result = "No relevant documents found."
        
        return result, elapsed
    
    def lookup_specific(self, keyword: str) -> Tuple[str, int]:
        """Tool: Look up specific information using keyword search"""
        # This simulates looking up specific facts
        start = time.time_ns()
        hits, _ = search_milvus(self.collection, self.embedder, keyword, limit=2)
        
        if hits:
            result = f"Specific lookup for '{keyword}':\n" + "\n".join([doc[:150] for doc in hits])
        else:
            result = f"No specific information found for '{keyword}'."
        
        elapsed = time.time_ns() - start
        latency_report.add("lookup_specific", elapsed)
        return result, elapsed
    
    def calculate(self, expression: str) -> Tuple[str, int]:
        """Tool: Perform calculations"""
        start = time.time_ns()
        try:
            # Safe evaluation of simple math expressions
            result = str(eval(expression, {"__builtins__": {}}, {}))
        except Exception as e:
            result = f"Cannot calculate: {e}"
        
        elapsed = time.time_ns() - start
        latency_report.add("calculate", elapsed)
        return result, elapsed
    
    def finish(self, answer: str) -> Tuple[str, int]:
        """Tool: Finish with final answer"""
        start = time.time_ns()
        elapsed = time.time_ns() - start
        return answer, elapsed
    
    def parse_action(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse the action from LLM output"""
        lines = text.strip().split('\n')
        action = None
        action_input = None
        
        for line in lines:
            line_stripped = line.strip()
            if line_stripped.startswith("Action:"):
                action = line_stripped.replace("Action:", "").strip()
            elif line_stripped.startswith("Action Input:"):
                action_input = line_stripped.replace("Action Input:", "").strip()
        
        return action, action_input
    
    def react_step(self, question: str, history: List[Dict]) -> Tuple[Dict, int]:
        """
        Execute one ReAct step: Thought -> Action -> Observation
        """
        # Build history context
        history_text = ""
        if history:
            for i, step in enumerate(history):
                history_text += f"\n{step['thought']}\n{step['action_text']}\n{step['observation']}\n"
        
        # Create ReAct prompt
        prompt = f"""You are a ReAct (Reasoning + Acting) agent. Answer questions by reasoning and taking actions step by step.

Question: {question}

{history_text}

Available Actions:
- search_documents: Search the document collection for relevant information
- lookup_specific: Look up specific information with a keyword
- calculate: Perform mathematical calculations
- finish: Provide the final answer when you have enough information

Use this format:
Thought: [Your reasoning about what to do next]
Action: [One of the available actions]
Action Input: [Input for the action]

Now continue:
"""
        
        response, elapsed = self._llm_invoke_timed(prompt, "llm_react_step")
        
        # Parse thought
        thought = ""
        for line in response.split('\n'):
            if line.strip().startswith("Thought:"):
                thought = line.replace("Thought:", "").strip()
                break
        
        # Parse action
        action, action_input = self.parse_action(response)
        
        return {
            "thought": f"Thought: {thought}",
            "action": action,
            "action_input": action_input,
            "action_text": f"Action: {action}\nAction Input: {action_input}",
            "raw_response": response
        }, elapsed
    
    def execute_action(self, action: str, action_input: str) -> Tuple[str, int]:
        """Execute the chosen action"""
        action_lower = action.lower() if action else ""
        
        if "search" in action_lower:
            result, elapsed = self.search_documents(action_input)
            return f"Observation: {result}", elapsed
        
        elif "lookup" in action_lower:
            result, elapsed = self.lookup_specific(action_input)
            return f"Observation: {result}", elapsed
        
        elif "calculate" in action_lower:
            result, elapsed = self.calculate(action_input)
            return f"Observation: Calculation result = {result}", elapsed
        
        elif "finish" in action_lower:
            return f"Observation: Task completed with answer: {action_input}", 0
        
        else:
            return f"Observation: Unknown action '{action}'. Please use a valid action.", 0
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Process query using ReAct Agentic RAG
        """
        print("\n" + "="*70)
        print("🤖 REACT AGENTIC RAG QUERY PROCESSING")
        print("="*70)
        print(f"❓ Question: {question}\n")
        
        overall_start = time.time_ns()
        
        history = []
        final_answer = None
        
        for iteration in range(1, self.max_iterations + 1):
            print(f"\n{'='*70}")
            print(f"🔄 ITERATION {iteration}/{self.max_iterations}")
            print(f"{'='*70}")
            
            iter_start = time.time_ns()
            
            # Step 1: Think and decide action
            print("\n💭 Step 1: Reasoning")
            print("-" * 70)
            step_info, think_time = self.react_step(question, history)
            
            print(f"   {step_info['thought']}")
            print(f"   {step_info['action_text']}")
            print(f"   ⏱️  Reasoning time: {format_time_ns(think_time)}")
            
            # Check if we should finish
            if step_info['action'] and "finish" in step_info['action'].lower():
                final_answer = step_info['action_input']
                observation = f"Observation: Task completed"
                action_time = 0
                print(f"\n✅ Agent decided to finish with answer")
            else:
                # Step 2: Execute action
                print(f"\n⚡ Step 2: Action Execution")
                print("-" * 70)
                observation, action_time = self.execute_action(
                    step_info['action'], 
                    step_info['action_input']
                )
                print(f"   {observation[:200]}...")
                print(f"   ⏱️  Action time: {format_time_ns(action_time)}")
            
            # Store in history
            step_info['observation'] = observation
            history.append(step_info)
            
            iter_elapsed = time.time_ns() - iter_start
            latency_report.add(f"react_iteration_{iteration}", iter_elapsed)
            print(f"\n⏱️  Iteration {iteration} total time: {format_time_ns(iter_elapsed)}")
            
            # Break if finished
            if final_answer:
                break
        
        # If no explicit finish, extract answer from last observation
        if not final_answer:
            print(f"\n⚠️  Max iterations reached, extracting answer...")
            final_answer = self.extract_final_answer(question, history)
        
        total_query_ns = time.time_ns() - overall_start
        latency_report.add("react_query_total", total_query_ns)
        
        print("\n" + "="*70)
        print("💬 FINAL ANSWER:")
        print("="*70)
        print(final_answer[:800])
        if len(final_answer) > 800:
            print("...")
        
        print(f"\n📊 ReAct Statistics:")
        print(f"   Total iterations: {len(history)}")
        print(f"   Actions taken: {len([h for h in history if h['action']])}")
        print(f"   Total query time: {format_time_ns(total_query_ns)}")
        print("="*70 + "\n")
        
        return {
            "question": question,
            "answer": final_answer,
            "reasoning_trace": history,
            "iterations": len(history),
            "total_query_ns": total_query_ns,
        }
    
    def extract_final_answer(self, question: str, history: List[Dict]) -> str:
        """Extract final answer when agent doesn't explicitly finish"""
        # Build context from all observations
        observations = "\n".join([step['observation'] for step in history])
        
        prompt = f"""Based on the following reasoning trace, provide a final answer to the question.

Question: {question}

Reasoning Trace:
{observations}

Final Answer:"""
        
        answer, _ = self._llm_invoke_timed(prompt, "llm_extract_answer")
        return answer
    
    def get_reasoning_trace(self, result: Dict) -> str:
        """Generate readable reasoning trace"""
        trace = f"Question: {result['question']}\n\n"
        trace += "Reasoning Trace:\n"
        trace += "="*60 + "\n\n"
        
        for i, step in enumerate(result['reasoning_trace'], 1):
            trace += f"Iteration {i}:\n"
            trace += f"{step['thought']}\n"
            trace += f"{step['action_text']}\n"
            trace += f"{step['observation'][:150]}...\n\n"
        
        trace += "="*60 + "\n"
        trace += f"Final Answer: {result['answer'][:200]}...\n"
        
        return trace

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
    print("🤖 REACT AGENTIC RAG + FULL LATENCY INSTRUMENTATION")
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
    
    # Initialize ReAct Agentic RAG
    react_agent = ReActAgenticRAG(llm, milvus_collection, embedder, max_iterations=MAX_ITERATIONS)
    print(f"\n✅ ReAct Agentic RAG system initialized (max iterations: {MAX_ITERATIONS})!")
    
    # Phase 3: Run ReAct queries
    print("\n📚 PHASE 3: REACT AGENTIC RAG QUERIES")
    print("-" * 70)
    
    queries = [
        "What are the main themes in this story?",
        "Tell me about the characters and their relationships.",
        "What is the setting and how does it influence the story?",
    ]
    
    results = []
    for q in queries:
        result = react_agent.query(q)
        results.append(result)
        
        # Show reasoning trace
        print("\n📖 REASONING TRACE SUMMARY:")
        print("-" * 70)
        trace = react_agent.get_reasoning_trace(result)
        print(trace[:500] + "..." if len(trace) > 500 else trace)
        
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
        print(f"Average iterations per query: {sum(r['iterations'] for r in results) / len(results):.1f}")
        total_actions = sum(len(r['reasoning_trace']) for r in results)
        print(f"Total actions taken: {total_actions}")
    
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