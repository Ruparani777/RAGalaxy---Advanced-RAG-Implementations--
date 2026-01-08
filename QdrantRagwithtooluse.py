#!/usr/bin/env python3
"""
qdrant_tool_calling_rag.py
Tool-Calling RAG with Qdrant and comprehensive nanosecond latency instrumentation.
(Adjusted/fixed indentation, minor bookkeeping, and removed stray text)
"""

import os
import time
import sys
import re
import math
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional, Callable

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
    # note: keep going if you want to debug offline, but in original script you exit here.
    # sys.exit(1)

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
    
    if not os.path.exists(path):
        print(f"⚠️  PDF not found at path: {path}")
        return ""
    
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
    # note: change as-needed for your environment; some Qdrant clients don't support ":memory:"
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
    
    if not chunks:
        print("⚠️  No chunks to insert.")
        return
    
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
            vector=vectors[i].tolist() if hasattr(vectors[i], "tolist") else list(vectors[i]),
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
    
    if not query:
        return [], 0
    
    # Encode query
    start = time.time_ns()
    qvecs = embedder.encode([query])
    qvec = qvecs[0]
    encode_time = time.time_ns() - start
    latency_report.add("query_embedding", encode_time)
    
    # Query Qdrant
    start = time.time_ns()
    response = qdrant.query_points(
        collection_name=collection_name,
        query=qvec.tolist() if hasattr(qvec, "tolist") else list(qvec),
        limit=limit
    )
    search_time = time.time_ns() - start
    latency_report.add("qdrant_search", search_time)
    
    # Extract texts
    hits = [p.payload.get("text", "") for p in response.points]
    
    total_time = encode_time + search_time
    
    return hits, total_time

# =========================================================
# TOOL DEFINITIONS
# (unchanged from original; kept for brevity)
# =========================================================
class Tool:
    """Base class for tools"""
    def __init__(self, name: str, description: str, parameters: List[str]):
        self.name = name
        self.description = description
        self.parameters = parameters
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }

class DocumentSearchTool(Tool):
    def __init__(self, qdrant: QdrantClient, embedder: SentenceTransformer, collection: str):
        super().__init__(
            name="document_search",
            description="Search the document database for relevant information. Use this when you need to find specific information from the document.",
            parameters=["query: str - The search query"]
        )
        self.qdrant = qdrant
        self.embedder = embedder
        self.collection = collection
    
    def execute(self, query: str = "", **kwargs) -> Dict[str, Any]:
        start = time.time_ns()
        
        hits, search_time = search_qdrant(self.qdrant, self.embedder, query, 4, self.collection)
        result_text = "\n\n".join(hits)
        
        elapsed = time.time_ns() - start
        latency_report.add("tool_document_search", elapsed)
        
        return {
            "success": True,
            "tool": "document_search",
            "result": result_text,
            "num_results": len(hits),
            "elapsed_ns": elapsed
        }

class CalculatorTool(Tool):
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Perform mathematical calculations. Supports basic operations (+, -, *, /) and functions (sqrt, pow, sin, cos, etc.).",
            parameters=["expression: str - Mathematical expression to evaluate (e.g., '2+2', 'sqrt(16)', 'pow(2,3)')"]
        )
    
    def execute(self, expression: str = "", **kwargs) -> Dict[str, Any]:
        start = time.time_ns()
        
        try:
            allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
            result = eval(expression, {"__builtins__": {}}, allowed)
            
            elapsed = time.time_ns() - start
            latency_report.add("tool_calculator", elapsed)
            
            return {
                "success": True,
                "tool": "calculator",
                "result": f"Calculation result: {result}",
                "value": result,
                "elapsed_ns": elapsed
            }
        except Exception as e:
            elapsed = time.time_ns() - start
            return {
                "success": False,
                "tool": "calculator",
                "result": f"Calculation error: {str(e)}",
                "error": str(e),
                "elapsed_ns": elapsed
            }

class SentimentAnalyzerTool(Tool):
    def __init__(self):
        super().__init__(
            name="sentiment_analyzer",
            description="Analyze the sentiment of text. Returns POSITIVE, NEGATIVE, or NEUTRAL with a confidence score.",
            parameters=["text: str - Text to analyze for sentiment"]
        )
        self.analyzer = SentimentIntensityAnalyzer()
    
    def execute(self, text: str = "", **kwargs) -> Dict[str, Any]:
        start = time.time_ns()
        
        scores = self.analyzer.polarity_scores(text)
        compound = scores['compound']
        
        if compound >= 0.05:
            label = "POSITIVE"
        elif compound <= -0.05:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"
        
        elapsed = time.time_ns() - start
        latency_report.add("tool_sentiment", elapsed)
        
        return {
            "success": True,
            "tool": "sentiment_analyzer",
            "result": f"Sentiment: {label} (score: {compound:.3f})",
            "label": label,
            "compound": compound,
            "elapsed_ns": elapsed
        }

class SummarizerTool(Tool):
    def __init__(self, llm):
        super().__init__(
            name="summarizer",
            description="Summarize long text into a concise summary. Useful for condensing large amounts of information.",
            parameters=["text: str - Text to summarize"]
        )
        self.llm = llm
    
    def execute(self, text: str = "", **kwargs) -> Dict[str, Any]:
        start = time.time_ns()
        
        try:
            prompt = f"""Provide a concise 2-3 sentence summary of the following text:

{text[:2000]}

Summary:"""
            
            resp = self.llm.invoke(prompt)
            summary = resp.content if hasattr(resp, "content") else str(resp)
            
            elapsed = time.time_ns() - start
            latency_report.add("tool_summarizer", elapsed)
            
            return {
                "success": True,
                "tool": "summarizer",
                "result": summary,
                "elapsed_ns": elapsed
            }
        except Exception as e:
            elapsed = time.time_ns() - start
            return {
                "success": False,
                "tool": "summarizer",
                "result": f"Summarization error: {str(e)}",
                "error": str(e),
                "elapsed_ns": elapsed
            }

# =========================================================
# TOOL-CALLING RAG SYSTEM
# (kept mostly as in your original)
# =========================================================
class ToolCallingRAG:
    def __init__(self, llm, tools: List[Tool], max_tool_calls: int = 5):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.max_tool_calls = max_tool_calls
    
    def _format_tools_for_prompt(self) -> str:
        tool_descriptions = []
        for name, tool in self.tools.items():
            params = ", ".join(tool.parameters)
            tool_descriptions.append(f"- {name}: {tool.description}\n  Parameters: {params}")
        return "\n".join(tool_descriptions)
    
    def _llm_invoke(self, prompt: str, label: str) -> Tuple[str, int]:
        start = time.time_ns()
        resp = self.llm.invoke(prompt)
        elapsed = time.time_ns() - start
        latency_report.add(f"llm_{label}", elapsed)
        
        content = resp.content if hasattr(resp, "content") else str(resp)
        return content, elapsed
    
    def _parse_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        pattern = r'USE_TOOL:\s*(\w+)\s*\((.*?)\)'
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        
        if not match:
            return None
        
        tool_name = match.group(1).strip()
        args_str = match.group(2).strip()
        
        args = {}
        if args_str:
            arg_pattern = r'(\w+)\s*=\s*["\']([^"\']*)["\']'
            for arg_match in re.finditer(arg_pattern, args_str):
                key = arg_match.group(1)
                value = arg_match.group(2)
                args[key] = value
        
        return {"tool": tool_name, "args": args}
    
    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name not in self.tools:
            return {
                "success": False,
                "tool": tool_name,
                "result": f"Tool '{tool_name}' not found",
                "error": f"Unknown tool: {tool_name}",
                "elapsed_ns": 0
            }
        
        tool = self.tools[tool_name]
        print(f"      🔧 Executing tool: {tool_name}")
        print(f"         Args: {args}")
        
        result = tool.execute(**args)
        
        if result.get("success"):
            print(f"      ✅ Tool succeeded in {format_time_ns(result.get('elapsed_ns', 0))}")
        else:
            print(f"      ❌ Tool failed: {result.get('error', 'Unknown error')}")
        
        return result
    
    def query(self, question: str) -> Dict[str, Any]:
        print(f"\n{'='*70}")
        print(f"🔧 TOOL-CALLING RAG QUERY PROCESSING")
        print(f"{'='*70}")
        print(f"❓ Question: {question}\n")
        
        overall_start = time.time_ns()
        
        tools_description = self._format_tools_for_prompt()
        
        system_prompt = f"""You are an intelligent assistant with access to tools. Answer the question by using the available tools.

Available Tools:
{tools_description}

To use a tool, write:
USE_TOOL: tool_name(param1="value1", param2="value2")

You can use multiple tools. After each tool use, I will provide the result, and you can decide to use another tool or provide the final answer.

When you have enough information, provide your final answer starting with "FINAL ANSWER:"

Question: {question}

What tools do you need to use?"""
        
        conversation_history = []
        tool_results = []
        tool_call_count = 0
        
        current_prompt = system_prompt
        
        while tool_call_count < self.max_tool_calls:
            print(f"\n{'─'*70}")
            print(f"🔄 Tool-Calling Step {tool_call_count + 1}")
            print(f"{'─'*70}")
            
            step_start = time.time_ns()
            
            response, llm_time = self._llm_invoke(current_prompt, f"tool_step_{tool_call_count + 1}")
            print(f"   🧠 LLM Response: {response[:200]}...")
            
            if "FINAL ANSWER:" in response.upper():
                final_match = re.search(r'FINAL ANSWER:\s*(.+)', response, re.IGNORECASE | re.DOTALL)
                if final_match:
                    answer = final_match.group(1).strip()
                    print(f"   ✅ Final answer provided")
                    
                    step_elapsed = time.time_ns() - step_start
                    latency_report.add("tool_calling_step", step_elapsed)
                    break
            
            tool_call = self._parse_tool_call(response)
            
            if tool_call:
                tool_call_count += 1
                tool_name = tool_call["tool"]
                tool_args = tool_call["args"]
                
                print(f"   🎯 Tool Call: {tool_name}")
                
                result = self._execute_tool(tool_name, tool_args)
                tool_results.append(result)
                
                tool_output = result.get("result", "No result")
                current_prompt = f"""Previous tool used: {tool_name}
Tool result: {tool_output}

Question: {question}

You can use another tool or provide the FINAL ANSWER: [your answer]

What's next?"""
                
                step_elapsed = time.time_ns() - step_start
                latency_report.add("tool_calling_step", step_elapsed)
                print(f"   ⏱️ Step time: {format_time_ns(step_elapsed)}")
            else:
                print(f"   ⚠️ No tool call detected")
                
                if len(response.strip()) > 50:
                    answer = response.strip()
                    break
                
                current_prompt = f"""You need to either:
1. USE_TOOL: tool_name(param="value") to use a tool
2. Provide FINAL ANSWER: [your answer]

Question: {question}

Please respond:"""
                
                step_elapsed = time.time_ns() - step_start
                latency_report.add("tool_calling_step", step_elapsed)
        
        if tool_call_count >= self.max_tool_calls and 'answer' not in locals():
            print(f"\n   ⚠️ Max tool calls reached")
            if tool_results:
                last_result = tool_results[-1].get("result", "No answer generated")
                answer = f"Based on tool results: {last_result[:500]}"
            else:
                answer = "Unable to determine answer"
        
        total_query_ns = time.time_ns() - overall_start
        latency_report.add("tool_calling_query_total", total_query_ns)
        
        print(f"\n{'='*70}")
        print(f"📊 TOOL-CALLING SUMMARY")
        print(f"{'='*70}")
        print(f"Tools used: {tool_call_count}")
        print(f"Total time: {format_time_ns(total_query_ns)}")
        
        if tool_results:
            print(f"\n🔧 Tools Executed:")
            for i, tr in enumerate(tool_results, 1):
                print(f"   {i}. {tr['tool']} - {format_time_ns(tr.get('elapsed_ns', 0))}")
        
        print(f"\n💬 FINAL ANSWER:")
        print(f"{'─'*70}")
        print(answer[:800])
        if len(answer) > 800:
            print("...")
        print(f"{'='*70}\n")
        
        return {
            'question': question,
            'answer': answer,
            'tool_calls': tool_call_count,
            'tool_results': tool_results,
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
    print("🔧 TOOL-CALLING RAG + FULL LATENCY INSTRUMENTATION")
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
    latency_report.add("pipeline_llm_init", llm_time)
    print(f"✅ LLM initialized in {format_time_ns(llm_time)}")

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
    average_query_time = (sum(r['total_query_ns'] for r in results) // len(results)) if results else 0
    print(f"Average query time: {format_time_ns(average_query_time)}")
    
    print(f"\n🔧 Tool Usage Statistics:")
    total_tool_calls = sum(r['tool_calls'] for r in results)
    print(f"   Total tool calls: {total_tool_calls}")
    print(f"   Average per query: {total_tool_calls / len(results):.1f}" if results else "   Average per query: 0")
    
    for i, r in enumerate(results, 1):
        print(f"   Query {i}: {r['tool_calls']} tool(s), {format_time_ns(r['total_query_ns'])}")
    
    # Detailed latency report
    latency_report.pretty_print()
    
    print("✅ PIPELINE COMPLETE")

if __name__ == "__main__":
    main()
