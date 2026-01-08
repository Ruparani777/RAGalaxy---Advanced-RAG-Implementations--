#!/usr/bin/env python3
import os
import sys
"""
react_rag_full.py
ReAct RAG Pipeline with comprehensive nanosecond latency instrumentation.

ReAct Framework: Thought -> Action -> Observation loop
- Thought: Agent reasons about what to do next
- Action: Execute retrieval, computation, or finish
- Observation: Examine results and decide next step

Use environment variables: PINECONE_API_KEY, GROQ_API_KEY
"""

import os
import time
import sys
import json
import re
import traceback
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# third-party imports
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
INDEX_NAME = "pinecone-react"
DIM = 384
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.1-8b-instant"
TARGET_NS = 200_000

if GROQ_API_KEY is None:
    print("ERROR: Set GROQ_API_KEY environment variable")
    sys.exit(1)

# ---------------------------
# Utilities
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
        print(f"⏱️ {func.__name__} time: {format_time_ns(elapsed)}")
        wrapper.last_elapsed_ns = elapsed
        return result
    wrapper.last_elapsed_ns = None
    return wrapper

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
                "avg_ns": total // len(vals) if vals else 0,
                "min_ns": min(vals) if vals else 0,
                "max_ns": max(vals) if vals else 0
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
# PDF/Chunking/Embeddings
# ---------------------------
@timer_ns
def load_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        page_texts = []
        for i, p in enumerate(pdf.pages):
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
# VADER Sentiment
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
# ReAct RAG Implementation
# ---------------------------
class ReActRAG:
    """
    ReAct (Reasoning + Acting) RAG Agent
    
    Loop:
    1. Thought: Reason about current state and what to do
    2. Action: Execute action (retrieve, compute, finish)
    3. Observation: Process results and decide next step
    """
    
    def __init__(self, vectorstore, llm, max_steps=5):
        self.vectorstore = vectorstore
        self.llm = llm
        self.max_steps = max_steps
        self.retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        # Available actions
        self.actions = {
            "RETRIEVE": self._action_retrieve,
            "COMPUTE": self._action_compute,
            "FINISH": self._action_finish
        }
    
    def _llm_invoke_timed(self, prompt, label):
        """Invoke LLM with timing"""
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
            print(f"LLM invoke for {label} failed: {e}")
            traceback.print_exc()
            return str(e), elapsed
    
    def _thought(self, question: str, history: List[Dict], step: int) -> Tuple[str, str]:
        """Generate reasoning about what action to take next"""
        
        # Build context from history
        context = self._build_context_from_history(history)
        
        prompt = f"""You are a ReAct agent. Analyze the question and decide your next action.

Question: {question}

Previous Steps:
{context}

Available Actions:
1. RETRIEVE - Search the document for relevant information
2. COMPUTE - Process or analyze information you have
3. FINISH - Provide final answer when you have enough information

Think step-by-step:
- What do I know so far?
- What information do I still need?
- What action should I take next?

Respond in this format:
Thought: [Your reasoning here]
Action: [RETRIEVE/COMPUTE/FINISH]
Action Input: [Specific query or instruction]

Your response:"""
        
        thought_text, elapsed = self._llm_invoke_timed(prompt, "react_thought")
        print(f"\n💭 THOUGHT (Step {step}, {format_time_ns(elapsed)}):")
        print(f"   {thought_text[:500]}...")
        
        # Parse action and input
        action, action_input = self._parse_thought(thought_text)
        
        return action, action_input
    
    def _parse_thought(self, thought_text: str) -> Tuple[str, str]:
        """Extract action and action input from thought"""
        action = "FINISH"  # default
        action_input = ""
        
        # Try to extract action
        action_match = re.search(r'Action:\s*(RETRIEVE|COMPUTE|FINISH)', thought_text, re.IGNORECASE)
        if action_match:
            action = action_match.group(1).upper()
        
        # Try to extract action input
        input_match = re.search(r'Action Input:\s*(.+?)(?:\n|$)', thought_text, re.IGNORECASE | re.DOTALL)
        if input_match:
            action_input = input_match.group(1).strip()
        
        return action, action_input
    
    def _action_retrieve(self, query: str) -> Tuple[str, int]:
        """Execute retrieval action"""
        start = time.time_ns()
        try:
            docs = self.retriever.invoke(query)
            elapsed = time.time_ns() - start
            latency_report.add("react_retrieve", elapsed)
            
            if not docs:
                return "No documents found.", elapsed
            
            # Extract content
            content = "\n\n".join([
                getattr(doc, "page_content", None) or getattr(doc, "content", None) or str(doc)
                for doc in docs
            ])
            
            result = f"Retrieved {len(docs)} documents:\n{content[:1000]}..."
            print(f"\n🔍 RETRIEVE ACTION ({format_time_ns(elapsed)}):")
            print(f"   Found {len(docs)} documents, {len(content)} chars")
            
            return result, elapsed
        except Exception as e:
            elapsed = time.time_ns() - start
            print(f"Retrieval failed: {e}")
            traceback.print_exc()
            return f"Error: {str(e)}", elapsed
    
    def _action_compute(self, instruction: str) -> Tuple[str, int]:
        """Execute computation/analysis action"""
        # Use LLM to process or analyze information
        prompt = f"""Process the following instruction:

{instruction}

Provide a clear, concise analysis or computation result.

Result:"""
        
        result, elapsed = self._llm_invoke_timed(prompt, "react_compute")
        print(f"\n🧮 COMPUTE ACTION ({format_time_ns(elapsed)}):")
        print(f"   Result: {result[:300]}...")
        
        return result, elapsed
    
    def _action_finish(self, final_answer: str) -> Tuple[str, int]:
        """Finish with final answer"""
        start = time.time_ns()
        print(f"\n✅ FINISH ACTION:")
        print(f"   Final answer prepared")
        elapsed = time.time_ns() - start
        return final_answer, elapsed
    
    def _observation(self, action: str, result: str, elapsed_ns: int) -> str:
        """Process observation from action result"""
        obs = f"Observation from {action} (took {format_time_ns(elapsed_ns)}): {result[:500]}"
        print(f"\n👁️ OBSERVATION:")
        print(f"   {obs[:300]}...")
        return obs
    
    def _build_context_from_history(self, history: List[Dict]) -> str:
        """Build context string from interaction history"""
        if not history:
            return "No previous steps."
        
        lines = []
        for i, step in enumerate(history, 1):
            lines.append(f"Step {i}:")
            lines.append(f"  Thought: {step.get('thought', '')[:100]}...")
            lines.append(f"  Action: {step['action']}")
            lines.append(f"  Observation: {step['observation'][:100]}...")
        
        return "\n".join(lines)
    
    def query(self, question: str) -> Dict:
        """
        Execute ReAct loop for answering question
        
        Returns:
            Dict with answer, steps, and timing info
        """
        print(f"\n{'='*70}")
        print(f"🤖 ReAct RAG QUERY PROCESSING")
        print(f"{'='*70}")
        print(f"❓ Question: {question}\n")
        
        overall_start = time.time_ns()
        history = []
        final_answer = ""
        
        for step in range(1, self.max_steps + 1):
            step_start = time.time_ns()
            print(f"\n{'─'*70}")
            print(f"📍 STEP {step}/{self.max_steps}")
            print(f"{'─'*70}")
            
            # 1. THOUGHT: Reason about next action
            action, action_input = self._thought(question, history, step)
            
            # 2. ACTION: Execute chosen action
            if action in self.actions:
                action_func = self.actions[action]
                
                if action == "FINISH":
                    # Generate final answer using all context
                    context = self._build_context_from_history(history)
                    final_prompt = f"""Based on the following information, provide a comprehensive answer.

Question: {question}

Context from previous steps:
{context}

Provide a clear, detailed final answer:"""
                    
                    final_answer, elapsed = self._llm_invoke_timed(final_prompt, "react_final_answer")
                    result = final_answer
                else:
                    result, elapsed = action_func(action_input)
            else:
                result = f"Unknown action: {action}"
                elapsed = 0
            
            # 3. OBSERVATION: Process result
            observation = self._observation(action, result, elapsed)
            
            # Record step
            step_elapsed = time.time_ns() - step_start
            latency_report.add("react_step", step_elapsed)
            
            history.append({
                'step': step,
                'thought': f"Action: {action}, Input: {action_input}",
                'action': action,
                'action_input': action_input,
                'observation': observation,
                'result': result,
                'step_time_ns': step_elapsed
            })
            
            print(f"\n⏱️ Step {step} total time: {format_time_ns(step_elapsed)}")
            
            # Check if we should finish
            if action == "FINISH":
                print(f"\n✅ ReAct completed after {step} step(s)")
                break
        
        total_query_ns = time.time_ns() - overall_start
        latency_report.add("react_query_total", total_query_ns)
        
        return {
            'question': question,
            'answer': final_answer or history[-1]['result'],
            'steps': history,
            'total_steps': len(history),
            'total_query_ns': total_query_ns
        }

# ---------------------------
# Main Pipeline
# ---------------------------
def main():
    print("="*70)
    print("🤖 ReAct RAG PIPELINE + FULL LATENCY INSTRUMENTATION")
    print("="*70)
    
    start_total = time.time_ns()
    
    # Phase 1: Setup
    try:
        pdf_text, t_pdf = timed_call(load_pdf, PDF_PATH)
        latency_report.add("pdf_load", t_pdf)
        
        chunks, t_chunks = timed_call(chunk_text, pdf_text, 1000, 100)
        latency_report.add("chunking", t_chunks)
        
        embed_model, t_emb = timed_call(get_embeddings_model)
        latency_report.add("embedding_model_init", t_emb)
        
        pc = init_pinecone(PINECONE_API_KEY, INDEX_NAME, DIM)
        vectorstore = create_vectorstore(embed_model, chunks, INDEX_NAME)
        
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
    
    # Initialize ReAct RAG
    react_rag = ReActRAG(vectorstore, llm, max_steps=5)
    print("\n✅ ReAct RAG system initialized!")
    
    # Phase 2: ReAct Queries
    print("\n\n📚 PHASE 2: ReAct RAG QUERIES")
    queries = [
        "What are the main themes in this story?",
        "Summarize the key events in the document.",
        "What is the capital of France?"
    ]
    
    react_results = []
    for q in queries:
        result = react_rag.query(q)
        react_results.append(result)
        print(f"\n{'='*70}\n")
    
    # Phase 3: VADER Sentiment
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
    
    for run in range(1, 4):
        run_sentiment_benchmark(run, sa, examples, TARGET_NS)
        time.sleep(0.1)
    
    # Final Report
    overall_ns = time.time_ns() - start_total
    latency_report.add("pipeline_total", overall_ns)
    
    print("\n📈 FINAL RESULTS")
    print(f"   Full pipeline time: {format_time_ns(overall_ns)}")
    print(f"   Queries executed: {len(queries)}")
    
    latency_report.pretty_print()
    
    print("\nSample ReAct Results:")
    for r in react_results:
        print("-" * 40)
        print(f"Q: {r['question']}")
        print(f"A: {r['answer'][:500]}...")
        print(f"Steps: {r['total_steps']}, Time: {format_time_ns(r['total_query_ns'])}")
    
    print("\n✅ ReAct PIPELINE COMPLETE")

if __name__ == "__main__":
    main()