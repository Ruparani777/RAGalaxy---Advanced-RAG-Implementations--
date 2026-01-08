#!/usr/bin/env python3
"""
agentic_rag_tools.py
Agentic RAG with Tool Use - LLM decides which tools to use and when.

Tools available:
- document_search: Search the vector database
- calculator: Perform mathematical calculations
- sentiment_analyzer: Analyze sentiment of text
- web_search_simulator: Simulate web search for general knowledge
- document_summary: Get summary of retrieved documents

Environment variables needed:
    PINECONE_API_KEY, GROQ_API_KEY
"""

import os
import time
import sys
import json
import math
import re
import traceback
from collections import defaultdict
from typing import List, Dict, Any, Optional

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------
# Config
# ---------------------------
PDF_PATH = "Data/ECHOES OF HER LOVE.pdf"
INDEX_NAME = "pinecone-agentic"
DIM = 384
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.1-8b-instant"

if PINECONE_API_KEY is None or GROQ_API_KEY is None:
    print("ERROR: Set PINECONE_API_KEY and GROQ_API_KEY environment variables")
    sys.exit(1)

# ---------------------------
# Utilities
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
        print("\n" + "="*70)
        print("LATENCY SUMMARY")
        print("="*70)
        for comp, vals in sorted(self.store.items()):
            total = sum(vals)
            avg = total // len(vals) if vals else 0
            print(f"\n{comp}:")
            print(f"  Count: {len(vals)}, Total: {format_time_ns(total)}, Avg: {format_time_ns(avg)}")
        print("="*70 + "\n")

latency_report = LatencyReport()

# ---------------------------
# Tool Definitions
# ---------------------------
class Tool:
    """Base class for tools"""
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

class DocumentSearchTool(Tool):
    def __init__(self, vectorstore):
        super().__init__(
            name="document_search",
            description="Search the document database for relevant information. Input: 'query' (string)"
        )
        self.vectorstore = vectorstore
        self.retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    def execute(self, query: str = "", **kwargs) -> Dict[str, Any]:
        start = time.time_ns()
        try:
            docs = self.retriever.invoke(query)
            elapsed = time.time_ns() - start
            latency_report.add("tool_document_search", elapsed)
            
            results = []
            for doc in docs:
                content = getattr(doc, "page_content", str(doc))
                results.append(content)
            
            return {
                "success": True,
                "result": "\n\n".join(results),
                "num_docs": len(results),
                "elapsed_ns": elapsed
            }
        except Exception as e:
            elapsed = time.time_ns() - start
            return {"success": False, "error": str(e), "elapsed_ns": elapsed}

class CalculatorTool(Tool):
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Perform mathematical calculations. Input: 'expression' (string, e.g., '2+2', 'sqrt(16)', 'sin(3.14)')"
        )
    
    def execute(self, expression: str = "", **kwargs) -> Dict[str, Any]:
        start = time.time_ns()
        try:
            # Safe eval with math functions
            allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            elapsed = time.time_ns() - start
            latency_report.add("tool_calculator", elapsed)
            return {
                "success": True,
                "result": str(result),
                "elapsed_ns": elapsed
            }
        except Exception as e:
            elapsed = time.time_ns() - start
            return {"success": False, "error": str(e), "elapsed_ns": elapsed}

class SentimentAnalyzerTool(Tool):
    def __init__(self):
        super().__init__(
            name="sentiment_analyzer",
            description="Analyze sentiment of text. Input: 'text' (string)"
        )
        self.analyzer = SentimentIntensityAnalyzer()
    
    def execute(self, text: str = "", **kwargs) -> Dict[str, Any]:
        start = time.time_ns()
        try:
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
                "result": f"Sentiment: {label} (score: {compound:.3f})",
                "label": label,
                "compound": compound,
                "elapsed_ns": elapsed
            }
        except Exception as e:
            elapsed = time.time_ns() - start
            return {"success": False, "error": str(e), "elapsed_ns": elapsed}

class WebSearchSimulatorTool(Tool):
    def __init__(self):
        super().__init__(
            name="web_search",
            description="Search for general knowledge (simulated). Input: 'query' (string)"
        )
        # Simulated knowledge base
        self.knowledge = {
            "capital of france": "Paris is the capital of France.",
            "population of earth": "Earth's population is approximately 8 billion people.",
            "who invented python": "Python was created by Guido van Rossum in 1991.",
            "speed of light": "The speed of light is approximately 299,792,458 meters per second.",
        }
    
    def execute(self, query: str = "", **kwargs) -> Dict[str, Any]:
        start = time.time_ns()
        query_lower = query.lower()
        
        # Simple matching
        result = "No information found."
        for key, value in self.knowledge.items():
            if key in query_lower or any(word in query_lower for word in key.split()):
                result = value
                break
        
        elapsed = time.time_ns() - start
        latency_report.add("tool_web_search", elapsed)
        
        return {
            "success": True,
            "result": result,
            "elapsed_ns": elapsed
        }

class DocumentSummaryTool(Tool):
    def __init__(self, llm):
        super().__init__(
            name="document_summary",
            description="Get a summary of document chunks. Input: 'text' (string)"
        )
        self.llm = llm
    
    def execute(self, text: str = "", **kwargs) -> Dict[str, Any]:
        start = time.time_ns()
        try:
            prompt = f"""Provide a concise summary of the following text in 2-3 sentences:

{text[:2000]}

Summary:"""
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            elapsed = time.time_ns() - start
            latency_report.add("tool_summary", elapsed)
            
            return {
                "success": True,
                "result": content,
                "elapsed_ns": elapsed
            }
        except Exception as e:
            elapsed = time.time_ns() - start
            return {"success": False, "error": str(e), "elapsed_ns": elapsed}

# ---------------------------
# Agentic RAG System
# ---------------------------
class AgenticRAG:
    """RAG system where LLM decides which tools to use"""
    
    def __init__(self, llm, tools: List[Tool]):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.tool_descriptions = self._format_tool_descriptions()
    
    def _format_tool_descriptions(self) -> str:
        descriptions = []
        for name, tool in self.tools.items():
            descriptions.append(f"- {name}: {tool.description}")
        return "\n".join(descriptions)
    
    def _parse_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse tool call from LLM response"""
        # Look for patterns like: USE_TOOL: tool_name(arg1="value1", arg2="value2")
        pattern = r'USE_TOOL:\s*(\w+)\s*\((.*?)\)'
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        
        if not match:
            return None
        
        tool_name = match.group(1).strip()
        args_str = match.group(2).strip()
        
        # Parse arguments
        args = {}
        if args_str:
            # Simple parsing: key="value" or key='value'
            arg_pattern = r'(\w+)\s*=\s*["\']([^"\']*)["\']'
            for arg_match in re.finditer(arg_pattern, args_str):
                key = arg_match.group(1)
                value = arg_match.group(2)
                args[key] = value
        
        return {"tool": tool_name, "args": args}
    
    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool with given arguments"""
        if tool_name not in self.tools:
            return {"success": False, "error": f"Tool '{tool_name}' not found"}
        
        tool = self.tools[tool_name]
        print(f"  🔧 Executing tool: {tool_name}")
        print(f"     Args: {args}")
        
        result = tool.execute(**args)
        
        if result.get("success"):
            print(f"  ✅ Tool succeeded in {format_time_ns(result.get('elapsed_ns', 0))}")
        else:
            print(f"  ❌ Tool failed: {result.get('error', 'Unknown error')}")
        
        return result
    
    def _llm_invoke(self, prompt: str, label: str) -> str:
        """Invoke LLM and track timing"""
        start = time.time_ns()
        try:
            response = self.llm.invoke(prompt)
            elapsed = time.time_ns() - start
            latency_report.add(f"llm_{label}", elapsed)
            content = response.content if hasattr(response, 'content') else str(response)
            return content
        except Exception as e:
            elapsed = time.time_ns() - start
            latency_report.add(f"llm_{label}_error", elapsed)
            print(f"LLM error in {label}: {e}")
            return str(e)
    
    def query(self, question: str, max_steps: int = 5) -> Dict[str, Any]:
        """Process a query using agentic tool selection"""
        print(f"\n{'='*70}")
        print(f"🤖 AGENTIC RAG QUERY")
        print(f"{'='*70}")
        print(f"❓ Question: {question}\n")
        
        overall_start = time.time_ns()
        conversation_history = []
        tool_results = []
        step = 0
        
        # Initial prompt
        system_prompt = f"""You are an intelligent assistant with access to tools. To answer questions, you can:
1. Use tools by writing: USE_TOOL: tool_name(arg="value")
2. Provide a final answer when you have enough information

Available tools:
{self.tool_descriptions}

IMPORTANT: 
- Use ONE tool at a time
- After using a tool, I will give you its result
- When you have enough information, provide a FINAL ANSWER: [your answer]
- Be strategic about which tools to use

Question: {question}

What's your first action?"""
        
        current_prompt = system_prompt
        
        while step < max_steps:
            step += 1
            print(f"\n--- Step {step} ---")
            
            # Get LLM decision
            response = self._llm_invoke(current_prompt, f"step_{step}")
            print(f"🧠 LLM Response:\n{response[:500]}...")
            
            # Check for final answer
            if "FINAL ANSWER:" in response.upper():
                final_answer = re.search(r'FINAL ANSWER:\s*(.*)', response, re.IGNORECASE | re.DOTALL)
                if final_answer:
                    answer = final_answer.group(1).strip()
                    print(f"\n✅ Final answer reached at step {step}")
                    
                    total_elapsed = time.time_ns() - overall_start
                    latency_report.add("agentic_query_total", total_elapsed)
                    
                    return {
                        "question": question,
                        "answer": answer,
                        "steps": step,
                        "tool_results": tool_results,
                        "total_elapsed_ns": total_elapsed
                    }
            
            # Parse and execute tool call
            tool_call = self._parse_tool_call(response)
            if tool_call:
                tool_name = tool_call["tool"]
                tool_args = tool_call["args"]
                
                # Execute tool
                result = self._execute_tool(tool_name, tool_args)
                tool_results.append({
                    "step": step,
                    "tool": tool_name,
                    "args": tool_args,
                    "result": result
                })
                
                # Update prompt with tool result
                if result.get("success"):
                    tool_output = result.get("result", "")
                    current_prompt = f"""Previous action: Used {tool_name}
Tool result: {tool_output}

Question: {question}

Based on this information, what's your next action? (Use another tool or provide FINAL ANSWER)"""
                else:
                    error_msg = result.get("error", "Unknown error")
                    current_prompt = f"""Previous action: Tried to use {tool_name} but it failed
Error: {error_msg}

Question: {question}

What's your next action?"""
            else:
                # No tool call found, treat as reasoning step
                print("  💭 No tool call detected, continuing...")
                current_prompt = f"""Your previous response: {response[:200]}...

Remember to either:
1. USE_TOOL: tool_name(arg="value") to use a tool
2. Provide FINAL ANSWER: [your answer] when ready

Question: {question}

What's your next action?"""
        
        # Max steps reached
        print(f"\n⚠️ Max steps ({max_steps}) reached without final answer")
        total_elapsed = time.time_ns() - overall_start
        latency_report.add("agentic_query_total", total_elapsed)
        
        return {
            "question": question,
            "answer": "Could not determine answer within step limit",
            "steps": step,
            "tool_results": tool_results,
            "total_elapsed_ns": total_elapsed
        }

# ---------------------------
# PDF Processing
# ---------------------------
def load_and_process_pdf(path):
    print(f"📄 Loading PDF: {path}")
    start = time.time_ns()
    
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    
    elapsed = time.time_ns() - start
    latency_report.add("pdf_load", elapsed)
    print(f"✅ Loaded {len(text)} characters in {format_time_ns(elapsed)}")
    
    return text

def chunk_text(text, chunk_size=1000, chunk_overlap=100):
    start = time.time_ns()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    elapsed = time.time_ns() - start
    latency_report.add("chunking", elapsed)
    print(f"📄 Created {len(chunks)} chunks in {format_time_ns(elapsed)}")
    return chunks

def init_vectorstore(chunks, api_key, index_name=INDEX_NAME):
    print(f"🔧 Initializing Pinecone...")
    start = time.time_ns()
    
    pc = Pinecone(api_key=api_key)
    existing = [idx.name for idx in pc.list_indexes()]
    
    if index_name in existing:
        print(f"🗑️ Deleting existing index...")
        pc.delete_index(index_name)
        time.sleep(2)
    
    print(f"🆕 Creating index...")
    pc.create_index(
        name=index_name,
        dimension=DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    time.sleep(2)
    
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vectorstore = PineconeVectorStore.from_texts(
        texts=chunks,
        embedding=embed_model,
        index_name=index_name,
        namespace="",
        metadatas=[{"chunk_id": i} for i in range(len(chunks))]
    )
    
    elapsed = time.time_ns() - start
    latency_report.add("vectorstore_init", elapsed)
    print(f"✅ Vectorstore ready in {format_time_ns(elapsed)}")
    
    return vectorstore

# ---------------------------
# Main
# ---------------------------
def main():
    print("="*70)
    print("🚀 AGENTIC RAG WITH TOOL USE")
    print("="*70)
    
    # Setup
    text = load_and_process_pdf(PDF_PATH)
    chunks = chunk_text(text)
    vectorstore = init_vectorstore(chunks, PINECONE_API_KEY)
    
    # Initialize LLM
    print("\n🤖 Initializing LLM...")
    llm = ChatGroq(model_name=MODEL_NAME, temperature=0, groq_api_key=GROQ_API_KEY)
    
    # Initialize Tools
    print("\n🔧 Initializing Tools...")
    tools = [
        DocumentSearchTool(vectorstore),
        CalculatorTool(),
        SentimentAnalyzerTool(),
        WebSearchSimulatorTool(),
        DocumentSummaryTool(llm)
    ]
    
    for tool in tools:
        print(f"  ✅ {tool.name}: {tool.description}")
    
    # Initialize Agentic RAG
    agentic_rag = AgenticRAG(llm, tools)
    print("\n✅ Agentic RAG system ready!\n")
    
    # Test Queries
    queries = [
        "What are the main themes in the document?",
        "What is 25 * 4 + sqrt(144)?",
        "What is the capital of France?",
        "Analyze the sentiment of: 'This is absolutely wonderful!'",
        "Summarize the key points from the document about love"
    ]
    
    results = []
    for q in queries:
        result = agentic_rag.query(q, max_steps=5)
        results.append(result)
        print(f"\n📊 Query completed in {format_time_ns(result['total_elapsed_ns'])}")
        print(f"   Steps taken: {result['steps']}")
        print(f"   Tools used: {len(result['tool_results'])}")
        print(f"   Answer: {result['answer'][:200]}...")
        print("\n" + "="*70)
    
    # Final report
    latency_report.pretty_print()
    
    print("\n✅ AGENTIC RAG PIPELINE COMPLETE")

if __name__ == "__main__":
    main()