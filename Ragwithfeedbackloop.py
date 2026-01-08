#!/usr/bin/env python3
"""
rag_feedback_loops_safe.py

RAG with Feedback Loops + full nanosecond latency instrumentation.

USAGE:
  1) Set environment variables (example):
     - Windows PowerShell:
         setx GROQ_API_KEY "gsk_...."
         setx PINECONE_API_KEY "pcsk_...."
     - macOS / Linux (bash/zsh):
         export GROQ_API_KEY="gsk_...."
         export PINECONE_API_KEY="pcsk_...."

  2) Install dependencies (examples):
       pip install pdfplumber langchain-text-splitters langchain-huggingface langchain-groq langchain-pinecone pinecone-client vaderSentiment

  3) Run:
       python rag_feedback_loops_safe.py

NOTES:
 - This script intentionally loads keys from environment variables for safety.
 - It instruments every major step using time.time_ns() and prints a latency summary.
"""

import os
import time
import sys
import re
import traceback
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

# Third-party imports (these must be installed)
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------
# CONFIG
# ---------------------------
PDF_PATH = "Data/ECHOES OF HER LOVE.pdf"   # change as needed
INDEX_NAME = "new2"                       # change as needed if your Pinecone index differs
DIM = 384
MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama-3.1-8b-instant")
QUALITY_THRESHOLD_DEFAULT = 85.0
MAX_CYCLES_DEFAULT = 5

# Load secrets from environment (safe)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    print("ERROR: PINECONE_API_KEY not found in environment variables.")
    print("Set it before running. Example (PowerShell):")
    print("  setx PINECONE_API_KEY \"pcsk_...\"")
    sys.exit(1)

if not GROQ_API_KEY:
    print("ERROR: GROQ_API_KEY not found in environment variables.")
    print("Set it before running. Example (PowerShell):")
    print("  setx GROQ_API_KEY \"gsk_...\"")
    sys.exit(1)

# ---------------------------
# Utilities: timing & reporting
# ---------------------------
def format_time_ns(ns: int) -> str:
    if ns < 1_000:
        return f"{ns} ns"
    if ns < 1_000_000:
        return f"{ns/1_000:.3f} µs ({ns} ns)"
    if ns < 1_000_000_000:
        return f"{ns/1_000_000:.3f} ms ({ns} ns)"
    return f"{ns/1_000_000_000:.3f} s ({ns} ns)"

class LatencyReport:
    def __init__(self):
        self.store = defaultdict(list)
    def add(self, component: str, ns: int):
        self.store[component].append(ns)
    def summary(self) -> Dict[str, Dict[str, Any]]:
        out = {}
        for comp, vals in self.store.items():
            total = sum(vals)
            count = len(vals)
            out[comp] = {
                "count": count,
                "total_ns": total,
                "avg_ns": total // count if count else 0,
                "min_ns": min(vals) if vals else 0,
                "max_ns": max(vals) if vals else 0,
            }
        return out
    def pretty_print(self):
        s = self.summary()
        print("\n" + "="*70)
        print("LATENCY SUMMARY")
        print("="*70)
        for comp, stats in sorted(s.items(), key=lambda x: x[0]):
            print(f"\nComponent: {comp}")
            print(f"  Count: {stats['count']}")
            print(f"  Total: {format_time_ns(stats['total_ns'])}")
            print(f"  Avg:   {format_time_ns(stats['avg_ns'])}")
            print(f"  Min:   {format_time_ns(stats['min_ns'])}")
            print(f"  Max:   {format_time_ns(stats['max_ns'])}")
        print("\n" + "="*70 + "\n")

latency_report = LatencyReport()

# ---------------------------
# Data classes for feedback
# ---------------------------
@dataclass
class QualityScores:
    relevance: int = 0
    completeness: int = 0
    accuracy: int = 0
    coherence: int = 0
    specificity: int = 0
    @property
    def overall(self) -> float:
        return (self.relevance + self.completeness + self.accuracy + self.coherence + self.specificity) * 2
    def __repr__(self):
        return f"QualityScores(overall={self.overall:.1f}%, rel={self.relevance}, comp={self.completeness}, acc={self.accuracy}, coh={self.coherence}, spec={self.specificity})"

@dataclass
class Feedback:
    cycle: int
    quality_scores: QualityScores
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)
    missing_aspects: List[str] = field(default_factory=list)
    retrieval_suggestions: List[str] = field(default_factory=list)
    raw_feedback: str = ""
    timestamp: int = 0
    def __repr__(self):
        return f"Feedback(cycle={self.cycle}, overall={self.quality_scores.overall:.1f}%)"

@dataclass
class FeedbackCycle:
    cycle_number: int
    query_refinement: str
    retrieved_docs: List = field(default_factory=list)
    retrieved_content: str = ""
    generated_answer: str = ""
    feedback: Optional[Feedback] = None
    elapsed_ns: int = 0
    improvements_from_previous: List[str] = field(default_factory=list)
    def __repr__(self):
        score = self.feedback.quality_scores.overall if self.feedback else 0
        return f"FeedbackCycle({self.cycle_number}, score={score:.1f}%)"

@dataclass
class FeedbackHistory:
    original_query: str
    cycles: List[FeedbackCycle] = field(default_factory=list)
    final_answer: str = ""
    total_elapsed_ns: int = 0
    convergence_reached: bool = False
    learned_patterns: Dict[str, Any] = field(default_factory=dict)
    def add_cycle(self, cycle: FeedbackCycle):
        self.cycles.append(cycle)
    def get_score_progression(self) -> List[float]:
        return [c.feedback.quality_scores.overall for c in self.cycles if c.feedback]
    def get_best_cycle(self) -> Optional[FeedbackCycle]:
        if not self.cycles:
            return None
        return max(self.cycles, key=lambda c: c.feedback.quality_scores.overall if c.feedback else 0)
    def analyze_convergence(self) -> Dict[str, Any]:
        scores = self.get_score_progression()
        if len(scores) < 2:
            return {"converged": False, "trend": "insufficient_data"}
        improvements = [scores[i] - scores[i-1] for i in range(1, len(scores))]
        avg_improvement = sum(improvements) / len(improvements)
        last_improvement = improvements[-1] if improvements else 0
        converged = abs(last_improvement) < 5  # less than 5% change
        return {
            "converged": converged,
            "trend": "improving" if avg_improvement > 0 else "declining",
            "avg_improvement": avg_improvement,
            "last_improvement": last_improvement,
            "final_score": scores[-1],
            "score_range": (min(scores), max(scores))
        }

# ---------------------------
# Core RAG with feedback loops
# ---------------------------
class FeedbackLoopRAG:
    def __init__(self, vectorstore, llm, quality_threshold: float = QUALITY_THRESHOLD_DEFAULT, max_cycles: int = MAX_CYCLES_DEFAULT):
        self.vectorstore = vectorstore
        self.llm = llm
        self.quality_threshold = quality_threshold
        self.max_cycles = max_cycles
        self.retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    def _llm_invoke(self, prompt: str, label: str) -> str:
        start = time.time_ns()
        try:
            response = self.llm.invoke(prompt)
            elapsed = time.time_ns() - start
            latency_report.add(f"llm_{label}", elapsed)
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            elapsed = time.time_ns() - start
            latency_report.add(f"llm_{label}_error", elapsed)
            print(f"LLM error ({label}): {e}")
            traceback.print_exc()
            return str(e)

    def retrieve_with_refinement(self, query: str, feedback: Optional[Feedback] = None, cycle: int = 1) -> Tuple[List, str]:
        print(f"    🔍 Retrieving documents (Cycle {cycle})...")
        refined_query = query
        if feedback and feedback.retrieval_suggestions:
            refinement_prompt = f"""Refine this search query based on feedback about what's missing:

Original Query: {query}

Feedback on what to search for:
{chr(10).join(f"- {s}" for s in feedback.retrieval_suggestions)}

Create an improved search query that addresses these gaps:

Refined Query:"""
            refined_query = self._llm_invoke(refinement_prompt, f"query_refinement_cycle_{cycle}").strip().split("\n")[0]
            print(f"    🎯 Refined query: '{refined_query}'")
        start = time.time_ns()
        try:
            docs = self.retriever.invoke(refined_query)
            elapsed = time.time_ns() - start
            latency_report.add(f"retrieval_cycle_{cycle}", elapsed)
            # build text content robustly
            parts = []
            for i, d in enumerate(docs):
                content = getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)
                parts.append(f"[Document {i+1}]\n{content}")
            full_content = "\n\n".join(parts)
            print(f"    ✅ Retrieved {len(docs)} docs in {format_time_ns(elapsed)}")
            return docs, full_content
        except Exception as e:
            elapsed = time.time_ns() - start
            latency_report.add(f"retrieval_cycle_{cycle}_error", elapsed)
            print(f"    ❌ Retrieval failed: {e}")
            traceback.print_exc()
            return [], ""

    def generate_answer(self, query: str, context: str, previous_feedback: Optional[Feedback], cycle: int = 1) -> str:
        print(f"    💡 Generating answer (Cycle {cycle})...")
        feedback_context = ""
        if previous_feedback and cycle > 1:
            feedback_context = f"""
PREVIOUS ATTEMPT FEEDBACK:
Strengths: {', '.join(previous_feedback.strengths)}
Weaknesses: {', '.join(previous_feedback.weaknesses)}

IMPROVEMENTS NEEDED:
{chr(10).join(f"- {s}" for s in previous_feedback.improvement_suggestions)}

MISSING ASPECTS:
{chr(10).join(f"- {a}" for a in previous_feedback.missing_aspects)}

ADDRESS THESE ISSUES in your answer."""
        prompt = f"""Answer the question using the provided context.{feedback_context}

Context:
{context[:3000]}

Question: {query}

Provide a comprehensive, well-structured answer:

Answer:"""
        answer = self._llm_invoke(prompt, f"generation_cycle_{cycle}")
        print(f"    ✅ Answer generated ({len(answer)} chars)")
        return answer

    def evaluate_answer(self, query: str, answer: str, context: str, cycle: int) -> Feedback:
        print(f"    📊 Evaluating answer quality (Cycle {cycle})...")
        eval_prompt = f"""You are a critical evaluator. Evaluate this answer comprehensively.

Question: {query}

Answer: {answer}

Context Available: {len(context)} characters

Provide evaluation in this EXACT format:

SCORES (rate each 0-10):
Relevance: [0-10]
Completeness: [0-10]
Accuracy: [0-10]
Coherence: [0-10]
Specificity: [0-10]

STRENGTHS:
- [strength 1]
- [strength 2]

WEAKNESSES:
- [weakness 1]
- [weakness 2]

IMPROVEMENT_SUGGESTIONS:
- [specific suggestion 1]
- [specific suggestion 2]

MISSING_ASPECTS:
- [missing aspect 1]
- [missing aspect 2]

RETRIEVAL_SUGGESTIONS:
- [what to search for 1]
- [what to search for 2]

Evaluation:"""
        eval_text = self._llm_invoke(eval_prompt, f"evaluation_cycle_{cycle}")
        # parse evaluation
        scores = QualityScores()
        strengths = []
        weaknesses = []
        improvements = []
        missing = []
        retrieval_suggestions = []
        current = None
        for line in eval_text.split("\n"):
            line = line.strip()
            if line.startswith("Relevance:"):
                scores.relevance = self._extract_score(line)
            elif line.startswith("Completeness:"):
                scores.completeness = self._extract_score(line)
            elif line.startswith("Accuracy:"):
                scores.accuracy = self._extract_score(line)
            elif line.startswith("Coherence:"):
                scores.coherence = self._extract_score(line)
            elif line.startswith("Specificity:"):
                scores.specificity = self._extract_score(line)
            elif line.startswith("STRENGTHS:"):
                current = "strengths"
            elif line.startswith("WEAKNESSES:"):
                current = "weaknesses"
            elif line.startswith("IMPROVEMENT_SUGGESTIONS:"):
                current = "improvements"
            elif line.startswith("MISSING_ASPECTS:"):
                current = "missing"
            elif line.startswith("RETRIEVAL_SUGGESTIONS:"):
                current = "retrieval"
            elif line.startswith("-") and current:
                item = line[1:].strip()
                if current == "strengths":
                    strengths.append(item)
                elif current == "weaknesses":
                    weaknesses.append(item)
                elif current == "improvements":
                    improvements.append(item)
                elif current == "missing":
                    missing.append(item)
                elif current == "retrieval":
                    retrieval_suggestions.append(item)
        feedback = Feedback(
            cycle=cycle,
            quality_scores=scores,
            strengths=strengths,
            weaknesses=weaknesses,
            improvement_suggestions=improvements,
            missing_aspects=missing,
            retrieval_suggestions=retrieval_suggestions,
            raw_feedback=eval_text,
            timestamp=time.time_ns()
        )
        print(f"    📈 Quality Score: {scores.overall:.1f}% -- Relevance {scores.relevance}/10, Completeness {scores.completeness}/10")
        return feedback

    def _extract_score(self, line: str) -> int:
        nums = re.findall(r"\d+", line)
        if nums:
            s = int(nums[0])
            return max(0, min(10, s))
        return 0

    def execute_feedback_cycle(self, query: str, history: FeedbackHistory, previous_feedback: Optional[Feedback] = None) -> FeedbackCycle:
        cycle_num = len(history.cycles) + 1
        print(f"\n  {'='*58}")
        print(f"  🔄 FEEDBACK CYCLE {cycle_num}")
        print(f"  {'='*58}")
        start = time.time_ns()
        cycle = FeedbackCycle(cycle_number=cycle_num, query_refinement=query)
        docs, content = self.retrieve_with_refinement(query, previous_feedback, cycle_num)
        cycle.retrieved_docs = docs
        cycle.retrieved_content = content
        if not content:
            print("    ⚠️ No content retrieved; will generate based on query alone.")
        answer = self.generate_answer(query, content, previous_feedback, cycle_num)
        cycle.generated_answer = answer
        feedback = self.evaluate_answer(query, answer, content, cycle_num)
        cycle.feedback = feedback
        # compute improvements vs previous
        if previous_feedback:
            prev_score = previous_feedback.quality_scores.overall
            curr_score = feedback.quality_scores.overall
            improvements = []
            if curr_score > prev_score:
                improvements.append(f"Overall +{curr_score - prev_score:.1f}%")
            prev_weak = set(previous_feedback.weaknesses)
            curr_weak = set(feedback.weaknesses)
            addressed = prev_weak - curr_weak
            if addressed:
                improvements.append(f"Addressed {len(addressed)} weaknesses")
            cycle.improvements_from_previous = improvements
            if improvements:
                print(f"    ✨ Improvements: {', '.join(improvements)}")
        cycle.elapsed_ns = time.time_ns() - start
        latency_report.add("feedback_cycle_total", cycle.elapsed_ns)
        print(f"  ✅ Cycle {cycle_num} completed in {format_time_ns(cycle.elapsed_ns)}")
        return cycle

    def query(self, question: str) -> Dict[str, Any]:
        print("\n" + "="*70)
        print("🔁 RAG WITH FEEDBACK LOOPS")
        print("="*70)
        print(f"❓ Question: {question}")
        print(f"🎯 Quality threshold: {self.quality_threshold}%")
        print(f"📊 Max cycles: {self.max_cycles}\n")
        start_overall = time.time_ns()
        history = FeedbackHistory(original_query=question)
        previous_feedback = None
        for i in range(1, self.max_cycles + 1):
            cycle = self.execute_feedback_cycle(question, history, previous_feedback)
            history.add_cycle(cycle)
            score = cycle.feedback.quality_scores.overall
            if score >= self.quality_threshold:
                print(f"\n  🎉 Quality threshold reached: {score:.1f}%")
                history.convergence_reached = True
                break
            if i >= self.max_cycles:
                print(f"\n  ⚠️ Max cycles reached ({self.max_cycles}). Final quality: {score:.1f}%")
                break
            previous_feedback = cycle.feedback
            print(f"\n  🔄 Preparing next cycle (score {score:.1f}% < {self.quality_threshold}%)")
        best = history.get_best_cycle()
        history.final_answer = best.generated_answer if best else ""
        history.total_elapsed_ns = time.time_ns() - start_overall
        latency_report.add("feedback_query_total", history.total_elapsed_ns)
        conv = history.analyze_convergence()
        history.learned_patterns = conv
        self._print_summary(history, conv)
        return {
            "question": question,
            "answer": history.final_answer,
            "history": history,
            "num_cycles": len(history.cycles),
            "converged": history.convergence_reached,
            "final_quality": best.feedback.quality_scores.overall if best else 0,
            "score_progression": history.get_score_progression(),
            "total_elapsed_ns": history.total_elapsed_ns
        }

    def _print_summary(self, history: FeedbackHistory, conv: Dict[str, Any]):
        print("\n" + "="*70)
        print("FEEDBACK LOOP SUMMARY")
        print("="*70)
        print(f"Cycles executed: {len(history.cycles)}")
        print(f"Converged: {history.convergence_reached}")
        print(f"Total time: {format_time_ns(history.total_elapsed_ns)}")
        print("\nQuality progression:")
        for idx, c in enumerate(history.cycles, 1):
            score = c.feedback.quality_scores.overall
            marker = "✅" if score >= self.quality_threshold else "📊"
            print(f"  Cycle {idx}: {score:.1f}% {marker}")
        print("\nConvergence analysis:")
        print(f"  Trend: {conv.get('trend')}")
        if conv.get("avg_improvement") is not None:
            print(f"  Avg improvement: {conv['avg_improvement']:.1f}%")
        if conv.get("final_score") is not None:
            print(f"  Final score: {conv['final_score']:.1f}%")
        if conv.get("score_range"):
            lo, hi = conv["score_range"]
            print(f"  Score range: {lo:.1f}% - {hi:.1f}%")
        best = history.get_best_cycle()
        if best:
            print(f"\nBest cycle: {best.cycle_number} with score {best.feedback.quality_scores.overall:.1f}%")
            if best.feedback.strengths:
                print(f"  Strength examples: {best.feedback.strengths[:3]}")

# ---------------------------
# PDF processing, chunking, vectorstore init
# ---------------------------
def load_and_process_pdf(path: str) -> str:
    print(f"📄 Loading PDF: {path} ...")
    start = time.time_ns()
    text_parts = []
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            t = p.extract_text() or ""
            text_parts.append(t)
            latency_report.add("pdf_page_extract", 0)  # placeholder if per-page is desired
    text = "\n".join(text_parts)
    elapsed = time.time_ns() - start
    latency_report.add("pdf_load", elapsed)
    print(f"✅ Loaded {len(text)} characters in {format_time_ns(elapsed)}")
    return text

def chunk_text_for_indexing(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
    print("🔪 Chunking text...")
    start = time.time_ns()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    elapsed = time.time_ns() - start
    latency_report.add("chunking", elapsed)
    print(f"📄 Created {len(chunks)} chunks in {format_time_ns(elapsed)}")
    return chunks

def init_vectorstore(chunks: List[str], api_key: str, index_name: str = INDEX_NAME) -> PineconeVectorStore:
    print("🔧 Initializing Pinecone vectorstore...")
    start = time.time_ns()
    pc = Pinecone(api_key=api_key)
    existing = [idx.name for idx in pc.list_indexes()]
    print(f"  Found indexes: {existing}")
    if index_name not in existing:
        print(f"  Index '{index_name}' not found. Creating it now...")
        t0 = time.time_ns()
        pc.create_index(name=index_name, dimension=DIM, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))
        latency_report.add("pinecone_create_index", time.time_ns() - t0)
        # small sleep to allow index creation to propagate
        time.sleep(2)
    else:
        print(f"  Using existing index '{index_name}'")
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    t1 = time.time_ns()
    vectorstore = PineconeVectorStore.from_texts(texts=chunks, embedding=embed_model, index_name=index_name, namespace="feedback_loop", metadatas=[{"chunk_id": i} for i in range(len(chunks))])
    latency_report.add("pinecone_upsert", time.time_ns() - t1)
    elapsed = time.time_ns() - start
    latency_report.add("vectorstore_init", elapsed)
    print(f"✅ Vectorstore initialized in {format_time_ns(elapsed)}")
    return vectorstore

# ---------------------------
# Sentiment analyzer (optional benchmarking)
# ---------------------------
class VaderSentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    def analyze(self, text: str) -> Dict[str, Any]:
        s = self.analyzer.polarity_scores(text)
        c = s["compound"]
        if c >= 0.05:
            label = "POSITIVE"
        elif c <= -0.05:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"
        return {"label": label, "scores": s, "compound": c}

# ---------------------------
# MAIN
# ---------------------------
def main():
    print("="*70)
    print("🚀 RAG WITH FEEDBACK LOOPS (SAFE MODE)")
    print("="*70)

    # Phase 0: load PDF & build vectorstore
    try:
        text = load_and_process_pdf(PDF_PATH)
        chunks = chunk_text_for_indexing(text)
        vectorstore = init_vectorstore(chunks, PINECONE_API_KEY, INDEX_NAME)
    except Exception as e:
        print(f"Setup error: {e}")
        traceback.print_exc()
        return

    # Init LLM (ChatGroq)
    print("\n🤖 Initializing LLM (ChatGroq)...")
    t0 = time.time_ns()
    try:
        llm = ChatGroq(model_name=MODEL_NAME, temperature=0, groq_api_key=GROQ_API_KEY)
    except Exception as e:
        print(f"LLM init error: {e}")
        traceback.print_exc()
        return
    latency_report.add("llm_init", time.time_ns() - t0)
    print(f"✅ LLM initialized in {format_time_ns(latency_report.summary().get('llm_init', {}).get('total_ns', 0))}")

    # Create FeedbackLoopRAG
    feedback_rag = FeedbackLoopRAG(vectorstore, llm, quality_threshold=80.0, max_cycles=4)
    print("\n✅ Feedback Loop RAG ready\n")

    # Example queries
    queries = [
        "What are the main themes in this story?",
        "Describe the main character's journey."
    ]

    results = []
    for q in queries:
        try:
            res = feedback_rag.query(q)
            results.append(res)
            print("\n" + "="*70)
            print("FINAL ANSWER (truncated):")
            print(res["answer"][:800] + ("..." if len(res["answer"]) > 800 else ""))
            print("="*70 + "\n")
        except Exception as e:
            print(f"Error during query '{q}': {e}")
            traceback.print_exc()

    # Print final latency and statistics
    latency_report.pretty_print()

    # Feedback loop aggregated stats
    total_cycles = sum(r["num_cycles"] for r in results)
    converged = sum(1 for r in results if r["converged"])
    avg_quality = sum(r["final_quality"] for r in results) / len(results) if results else 0
    print("\n" + "="*70)
    print("FEEDBACK LOOP AGGREGATE STATS")
    print("="*70)
    print(f"Total queries: {len(results)}")
    print(f"Total cycles: {total_cycles}")
    print(f"Average cycles per query: {total_cycles / len(results):.1f}" if results else "N/A")
    print(f"Converged queries: {converged}/{len(results)}")
    print(f"Average final quality: {avg_quality:.1f}%")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
