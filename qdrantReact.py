#!/usr/bin/env python3
"""
qdrant_react_rag.py
ReAct RAG (Reasoning + Acting) with Qdrant and comprehensive nanosecond latency instrumentation.

ReAct Pattern:
1. THOUGHT: Reason about what to do next
2. ACTION: Execute an action (search, calculate, etc.)
3. OBSERVATION: Observe the result
4. Repeat until answer is found

Features:
- Full pipeline timing (PDF load, chunking, embeddings, vectorstore)
- Per-component latency tracking
- Detailed ReAct cycle metrics
- Comprehensive latency reports
"""

import os
import time
import sys
import re
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional

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
# ReAct RAG SYSTEM
# =========================================================
class ReActRAG:
    """ReAct RAG system (Reasoning + Acting) with comprehensive timing"""
    
    def __init__(self, llm, qdrant: QdrantClient, embedder: SentenceTransformer, 
                 collection_name: str = COLLECTION):
        self.llm = llm
        self.qdrant = qdrant
        self.embedder = embedder
        self.collection_name = collection_name
        self.max_steps = 5
    
    def _llm_invoke(self, prompt: str, label: str) -> Tuple[str, int]:
        """Invoke LLM with timing"""
        start = time.time_ns()
        resp = self.llm.invoke(prompt)
        elapsed = time.time_ns() - start
        latency_report.add(f"llm_{label}", elapsed)
        
        content = resp.content if hasattr(resp, "content") else str(resp)
        return content, elapsed
    
    def _parse_react_response(self, response: str) -> Dict[str, str]:
        """Parse ReAct format: Thought, Action, etc."""
        result = {
            'thought': '',
            'action': '',
            'action_input': '',
            'answer': ''
        }
        
        # Extract Thought
        thought_match = re.search(r'Thought[:\s]+(.+?)(?=Action[:\s]|Answer[:\s]|$)', response, re.IGNORECASE | re.DOTALL)
        if thought_match:
            result['thought'] = thought_match.group(1).strip()
        
        # Extract Action
        action_match = re.search(r'Action[:\s]+(\w+)', response, re.IGNORECASE)
        if action_match:
            result['action'] = action_match.group(1).strip().lower()
        
        # Extract Action Input
        input_match = re.search(r'Action Input[:\s]+(.+?)(?=Thought[:\s]|Action[:\s]|Answer[:\s]|$)', response, re.IGNORECASE | re.DOTALL)
        if input_match:
            result['action_input'] = input_match.group(1).strip()
        
        # Extract Answer (final answer)
        answer_match = re.search(r'Answer[:\s]+(.+)', response, re.IGNORECASE | re.DOTALL)
        if answer_match:
            result['answer'] = answer_match.group(1).strip()
        
        return result
    
    def execute_action(self, action: str, action_input: str) -> Tuple[str, int]:
        """Execute an action and return observation"""
        action = action.lower()
        
        if action == "search":
            # Search the document
            print(f"      🔍 ACTION: Searching for '{action_input[:50]}...'")
            hits, elapsed = search_qdrant(self.qdrant, self.embedder, action_input, 3, self.collection_name)
            observation = "\n\n".join(hits)
            print(f"      ✅ Found {len(hits)} relevant passages in {format_time_ns(elapsed)}")
            return observation, elapsed
        
        elif action == "lookup":
            # Lookup specific information
            print(f"      🔎 ACTION: Looking up '{action_input[:50]}...'")
            hits, elapsed = search_qdrant(self.qdrant, self.embedder, action_input, 2, self.collection_name)
            observation = " ".join(hits)
            print(f"      ✅ Lookup completed in {format_time_ns(elapsed)}")
            return observation, elapsed
        
        elif action == "finish":
            # Final answer
            print(f"      ✓ ACTION: Finish with answer")
            return action_input, 0
        
        else:
            # Unknown action
            print(f"      ⚠️ Unknown action: {action}")
            return f"Unknown action: {action}. Available actions: search, lookup, finish", 0
    
    def query(self, question: str) -> Dict[str, Any]:
        """Process query using ReAct pattern: Thought -> Action -> Observation"""
        print(f"\n{'='*70}")
        print(f"🧠 ReAct RAG QUERY PROCESSING")
        print(f"{'='*70}")
        print(f"❓ Question: {question}\n")
        
        overall_start = time.time_ns()
        
        # Track ReAct cycles
        react_history = []
        step = 0
        
        # Build the ReAct prompt
        system_prompt = """You are an AI assistant using the ReAct (Reasoning + Acting) framework to answer questions.

Available Actions:
- search: Search the document for relevant information (use for broad queries)
- lookup: Look up specific details (use for precise information)
- finish: Provide the final answer when you have enough information

You must use this format:

Thought: [Your reasoning about what to do next]
Action: [One of: search, lookup, finish]
Action Input: [The input for the action]

After each action, you'll receive an Observation. Then continue:

Thought: [Reason about the observation]
Action: [Next action]
Action Input: [Input for next action]

When you have enough information:
Thought: [Your final reasoning]
Action: finish
Action Input: [Your comprehensive final answer]

Begin!"""
        
        conversation = f"{system_prompt}\n\nQuestion: {question}\n\n"
        
        while step < self.max_steps:
            step += 1
            print(f"\n{'─'*70}")
            print(f"🔄 ReAct Step {step}")
            print(f"{'─'*70}")
            
            step_start = time.time_ns()
            
            # Get LLM reasoning and action
            prompt = conversation + "Thought:"
            response, llm_time = self._llm_invoke(prompt, f"react_step_{step}")
            
            # Parse the response
            parsed = self._parse_react_response("Thought:" + response)
            
            thought = parsed['thought']
            action = parsed['action']
            action_input = parsed['action_input']
            final_answer = parsed['answer']
            
            print(f"   💭 THOUGHT: {thought[:150]}...")
            
            # Check if we have a final answer
            if final_answer or action == 'finish':
                answer = final_answer if final_answer else action_input
                print(f"   ✅ FINAL ANSWER REACHED")
                
                step_elapsed = time.time_ns() - step_start
                latency_report.add("react_step_total", step_elapsed)
                
                react_history.append({
                    'step': step,
                    'thought': thought,
                    'action': 'finish',
                    'action_input': answer,
                    'observation': '',
                    'elapsed_ns': step_elapsed
                })
                
                break
            
            if action:
                print(f"   🎬 ACTION: {action}")
                print(f"   📥 INPUT: {action_input[:100]}...")
                
                # Execute action
                observation, action_time = self.execute_action(action, action_input)
                
                print(f"   👁️ OBSERVATION: {observation[:150]}...")
                
                # Add to conversation history
                conversation += f"Thought: {thought}\n"
                conversation += f"Action: {action}\n"
                conversation += f"Action Input: {action_input}\n"
                conversation += f"Observation: {observation}\n\n"
                
                step_elapsed = time.time_ns() - step_start
                latency_report.add("react_step_total", step_elapsed)
                
                react_history.append({
                    'step': step,
                    'thought': thought,
                    'action': action,
                    'action_input': action_input,
                    'observation': observation,
                    'elapsed_ns': step_elapsed
                })
                
                print(f"   ⏱️ Step {step} time: {format_time_ns(step_elapsed)}")
                
                # Check if this was finish action
                if action == 'finish':
                    answer = action_input
                    break
            else:
                print(f"   ⚠️ No valid action found in response")
                # Try to extract any answer-like content
                if final_answer:
                    answer = final_answer
                    break
                # Continue to next step
                conversation += f"Thought: {thought}\n\n"
        
        # If we exhausted steps without finish, use last observation or thought
        if step >= self.max_steps and 'answer' not in locals():
            print(f"\n   ⚠️ Max steps reached without final answer")
            if react_history:
                last_obs = react_history[-1].get('observation', '')
                answer = f"Based on the search: {last_obs[:500]}" if last_obs else "Unable to determine answer"
            else:
                answer = "Unable to determine answer within step limit"
        
        total_query_ns = time.time_ns() - overall_start
        latency_report.add("react_query_total", total_query_ns)
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"📊 ReAct SUMMARY")
        print(f"{'='*70}")
        print(f"Total steps: {step}")
        print(f"Total time: {format_time_ns(total_query_ns)}")
        print(f"\n💬 FINAL ANSWER:")
        print(f"{'─'*70}")
        print(answer[:800])
        if len(answer) > 800:
            print("...")
        print(f"{'='*70}\n")
        
        return {
            'question': question,
            'answer': answer,
            'steps': step,
            'react_history': react_history,
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
    print("🧠 ReAct RAG (Reasoning + Acting) + FULL LATENCY INSTRUMENTATION")
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
    )
    llm_time = time.time_ns() - llm_start
    latency_report.add("llm_init", llm_time)
    print(f"✅ LLM initialized in {format_time_ns(llm_time)}")
    
    # Initialize ReAct RAG
    react_rag = ReActRAG(llm, qdrant, embedder, COLLECTION)
    print(f"\n✅ ReAct RAG system initialized!")
    
    # Phase 3: Run ReAct queries
    print(f"\n📚 PHASE 3: ReAct RAG QUERIES")
    print("-"*70)
    
    queries = [
        "What are the main themes in this story?",
        "Who are the main characters and what happens to them?",
        "What is the central conflict in the document?"
    ]
    
    results = []
    for q in queries:
        result = react_rag.query(q)
        results.append(result)
    
    # Phase 4: Sentiment benchmark
    print(f"\n📚 PHASE 4: VADER SENTIMENT BENCHMARK")
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
    
    for run in range(1, 2):
        run_sentiment_benchmark(sa, examples, 200_000, run)
    
    # Final summary
    pipeline_total = time.time_ns() - pipeline_start
    latency_report.add("pipeline_total", pipeline_total)
    
    print(f"\n{'='*70}")
    print(f"📈 PIPELINE SUMMARY")
    print(f"{'='*70}")
    print(f"Total pipeline time: {format_time_ns(pipeline_total)}")
    print(f"Queries executed: {len(queries)}")
    print(f"Average query time: {format_time_ns(sum(r['total_query_ns'] for r in results) // len(results))}")
    print(f"Average steps per query: {sum(r['steps'] for r in results) / len(results):.1f}")
    
    print(f"\n🧠 ReAct Statistics:")
    for i, r in enumerate(results, 1):
        print(f"  Query {i}: {r['steps']} steps, {format_time_ns(r['total_query_ns'])}")
    
    # Detailed latency report
    latency_report.pretty_print()
    
    print("✅ PIPELINE COMPLETE")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)