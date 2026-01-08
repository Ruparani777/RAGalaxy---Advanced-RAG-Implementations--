#!/usr/bin/env python3
"""
weaviate_plan_and_solve_with_feedback.py

Plan-and-Solve Agentic RAG built from your Feedback-RAG code style.
Replaces the single-feedback RAG with a Planner -> Step Solver (with feedback loops) -> Aggregator.
Keeps nanosecond latency instrumentation, VADER benchmark, and feedback assessments per step.

Features:
- Planner (LLM) breaks question into steps
- StepSolver runs a mini-FeedbackRAG for each step (retrieve -> generate -> assess -> reformulate)
- Per-step assessment and history retained
- Aggregator synthesizes step answers and runs a final assessment
- LatencyReport collects timings across components
- Safe LLM fallback when GROQ not available
- CLI for indexing, testing, sentiment benchmark, interactive mode

Run examples:
    python weaviate_plan_and_solve_with_feedback.py --pdf "Data/ECHOES OF HER LOVE.pdf" --question "Summarize chapter 2 and list themes"
    python weaviate_plan_and_solve_with_feedback.py --skip_index --question "Explain main idea"

"""

import os
import time
import sys
import argparse
import traceback
from typing import List, Dict, Any, Tuple
from collections import defaultdict

# Third-party libs (ensure installed)
import pdfplumber
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import MetadataQuery
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    from langchain_groq import ChatGroq
except Exception:
    ChatGroq = None
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ==========================
# CONFIG
# ==========================
PDF_PATH = "Data/ECHOES OF HER LOVE.pdf"
COLLECTION_NAME = "PlanSolve_Documents"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = "llama-3.1-8b-instant"
MAX_PLAN_STEPS = 6
MAX_STEP_FEEDBACK_LOOPS = 3
RELEVANCE_THRESHOLD = 7
COMPLETENESS_THRESHOLD = 7
CONFIDENCE_THRESHOLD = 6
TARGET_NS = 200_000

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "21ookhjbswyl5urlawqmxw.c0.asia-southeast1.gcp.weaviate.cloud")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", None)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)

# ==========================
# LATENCY UTILITIES
# ==========================

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

    def summary(self) -> Dict[str, Any]:
        out = {}
        for comp, vals in self.store.items():
            total = sum(vals)
            out[comp] = {
                "count": len(vals),
                "total_ns": total,
                "avg_ns": total // len(vals) if vals else 0,
                "min_ns": min(vals) if vals else 0,
                "max_ns": max(vals) if vals else 0,
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

# ==========================
# PDF, chunking, embeddings, weaviate
# ==========================

@staticmethod
def _safe_open_pdf(path: str):
    return pdfplumber.open(path)


def load_pdf(path: str) -> str:
    print(f"📄 Loading PDF: {path}")
    text = ""
    with pdfplumber.open(path) as pdf:
        for i, p in enumerate(pdf.pages):
            start = time.time_ns()
            t = p.extract_text() or ""
            elapsed = time.time_ns() - start
            latency_report.add("pdf_page_extract", elapsed)
            text += t + "\n"
    print(f"✅ Loaded PDF: {len(text)} characters from {i+1} pages")
    return text


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
    print("✂️  Chunking text...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    print(f"✅ Created {len(chunks)} chunks")
    return chunks


def load_embeddings(model_name: str = EMBED_MODEL) -> SentenceTransformer:
    print(f"🔢 Loading embeddings model: {model_name}")
    start = time.time_ns()
    embedder = SentenceTransformer(model_name)
    elapsed = time.time_ns() - start
    latency_report.add("embedder_load", elapsed)
    print(f"✅ Embeddings model loaded in {format_time_ns(elapsed)}")
    return embedder


def init_weaviate(url: str, api_key: str, collection_name: str = COLLECTION_NAME) -> weaviate.WeaviateClient:
    print(f"🗃️  Initializing Weaviate connection to {url}")
    start = time.time_ns()
    client = weaviate.connect_to_weaviate_cloud(cluster_url=url, auth_credentials=Auth.api_key(api_key))
    latency_report.add("weaviate_connect", time.time_ns() - start)

    try:
        if client.collections.exists(collection_name):
            start = time.time_ns()
            client.collections.delete(collection_name)
            latency_report.add("weaviate_delete_collection", time.time_ns() - start)
            print(f"🗑️  Deleted existing collection '{collection_name}'")
    except Exception as e:
        print(f"⚠️ Collection check/delete: {e}")

    try:
        start = time.time_ns()
        client.collections.create(
            name=collection_name,
            vectorizer_config=None,
            properties=[
                {"name": "text", "dataType": ["text"]},
                {"name": "chunk_id", "dataType": ["int"]},
                {"name": "source", "dataType": ["text"]},
            ],
        )
        latency_report.add("weaviate_create_collection", time.time_ns() - start)
        print(f"✅ Collection '{collection_name}' created")
    except Exception as e:
        print(f"⚠️ Collection creation: {e}")

    return client


def insert_chunks(client: weaviate.WeaviateClient, embedder: SentenceTransformer, chunks: List[str], collection_name: str = COLLECTION_NAME) -> None:
    print(f"⬆️  Inserting {len(chunks)} chunks into Weaviate...")
    start = time.time_ns()
    vectors = embedder.encode(chunks, show_progress_bar=False)
    latency_report.add("embedding_encode_batch", time.time_ns() - start)

    collection = client.collections.get(collection_name)
    start = time.time_ns()
    with collection.batch.dynamic() as batch:
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            batch.add_object(properties={"text": chunk, "chunk_id": i, "source": f"chunk_{i}"}, vector=vector.tolist())
    latency_report.add("weaviate_upsert", time.time_ns() - start)
    print(f"✅ All chunks inserted successfully!")


def search_weaviate(client: weaviate.WeaviateClient, embedder: SentenceTransformer, query: str, limit: int = 4, collection_name: str = COLLECTION_NAME) -> Tuple[List[str], int]:
    # Encode query
    start = time.time_ns()
    qvec = embedder.encode([query])[0]
    encode_time = time.time_ns() - start
    latency_report.add("query_embedding", encode_time)

    # Query Weaviate
    start = time.time_ns()
    collection = client.collections.get(collection_name)
    response = collection.query.near_vector(near_vector=qvec.tolist(), limit=limit, return_metadata=MetadataQuery(distance=True))
    search_time = time.time_ns() - start
    latency_report.add("weaviate_search", search_time)

    hits = [obj.properties.get("text", "") for obj in response.objects]
    total_time = encode_time + search_time
    return hits, total_time

# ==========================
# Feedback Assessment
# ==========================
class FeedbackAssessment:
    def __init__(self, relevance: int, completeness: int, confidence: int, issues: List[str], suggestions: List[str]):
        self.relevance = relevance
        self.completeness = completeness
        self.confidence = confidence
        self.issues = issues
        self.suggestions = suggestions

    def is_satisfactory(self) -> bool:
        return (self.relevance >= RELEVANCE_THRESHOLD and self.completeness >= COMPLETENESS_THRESHOLD and self.confidence >= CONFIDENCE_THRESHOLD)

    def get_score_summary(self) -> str:
        return f"R:{self.relevance}/10, C:{self.completeness}/10, Conf:{self.confidence}/10"

    def __repr__(self):
        return f"FeedbackAssessment(relevance={self.relevance}, completeness={self.completeness}, confidence={self.confidence})"

# ==========================
# LLM Wrapper (safe)
# ==========================
class LLMWrapper:
    def __init__(self, groq_api_key: str = None, model: str = MODEL_NAME):
        self.model = model
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.client = None
        if ChatGroq and self.groq_api_key:
            try:
                self.client = ChatGroq(api_key=self.groq_api_key, model=self.model)
            except Exception:
                try:
                    self.client = ChatGroq(api_key=self.groq_api_key)
                except Exception:
                    self.client = None

    def invoke(self, prompt: str) -> Any:
        start = time.time_ns()
        if self.client is not None:
            try:
                if hasattr(self.client, 'invoke'):
                    out = self.client.invoke(prompt)
                elif hasattr(self.client, '__call__'):
                    out = self.client(prompt)
                elif hasattr(self.client, 'generate'):
                    out = self.client.generate([prompt])
                else:
                    out = str(self.client)
                elapsed = time.time_ns() - start
                latency_report.add('llm_invoke', elapsed)
                return out
            except Exception as e:
                elapsed = time.time_ns() - start
                latency_report.add('llm_invoke_error', elapsed)
                print(f"⚠️ LLM client call failed: {e}")
                traceback.print_exc()
        # fallback stub
        class _Resp:
            def __init__(self, content):
                self.content = content
            def __str__(self):
                return self.content
        preview = '\n'.join(prompt.strip().split('\n')[:6])
        stub = f"[STUB LLM RESPONSE]\n\nPrompt preview:\n{preview}\n\nNote: fallback used."
        elapsed = time.time_ns() - start
        latency_report.add('llm_invoke_stub', elapsed)
        return _Resp(stub)

# ==========================
# Planner
# ==========================
class Planner:
    def __init__(self, llm: LLMWrapper, max_steps: int = MAX_PLAN_STEPS):
        self.llm = llm
        self.max_steps = max_steps

    def create_plan(self, question: str) -> Tuple[List[str], int]:
        prompt = f"""You are a planner. Break down the task into up to {self.max_steps} clear, independent subtasks.
Return a numbered list (1., 2., ...), each item one concise sentence.
Task: {question}
Plan:"""
        start = time.time_ns()
        resp = self.llm.invoke(prompt)
        elapsed = time.time_ns() - start
        latency_report.add('planner_invoke', elapsed)

        text = resp.content if hasattr(resp, 'content') else str(resp)
        steps = []
        for line in text.split('\n'):
            s = line.strip()
            if not s: continue
            # naive parse: look for leading number
            if s[0].isdigit():
                # remove leading numbering
                parts = s.split('.', 1)
                if len(parts) > 1:
                    steps.append(parts[1].strip())
            else:
                # also accept bullet-like responses
                if len(steps) < self.max_steps:
                    steps.append(s)
            if len(steps) >= self.max_steps:
                break
        if not steps:
            steps = [question]
        print(f"🧭 Planner produced {len(steps)} step(s)")
        return steps, elapsed

# ==========================
# Step Solver (mini-feedback RAG per step)
# ==========================
class StepSolver:
    def __init__(self, llm: LLMWrapper, client: weaviate.WeaviateClient, embedder: SentenceTransformer, collection_name: str = COLLECTION_NAME, max_loops: int = MAX_STEP_FEEDBACK_LOOPS):
        self.llm = llm
        self.client = client
        self.embedder = embedder
        self.collection_name = collection_name
        self.max_loops = max_loops

    def _llm_invoke_timed(self, prompt: str, label: str) -> Tuple[str, int]:
        start = time.time_ns()
        try:
            response = self.llm.invoke(prompt)
            elapsed = time.time_ns() - start
            latency_report.add(label, elapsed)
            content = response.content if hasattr(response, 'content') else str(response)
            return content, elapsed
        except Exception as e:
            elapsed = time.time_ns() - start
            latency_report.add(label + '_error', elapsed)
            print(f"❌ LLM invoke for {label} failed: {e}")
            traceback.print_exc()
            return str(e), elapsed

    def retrieve(self, query: str, k: int = 4) -> Tuple[str, int]:
        hits, elapsed = search_weaviate(self.client, self.embedder, query, k, self.collection_name)
        context = "\n\n".join(hits)
        return context, elapsed

    def assess_answer(self, question: str, answer: str, context: str, loop_num: int) -> Tuple[FeedbackAssessment, str, int]:
        prompt = f"""You are a strict quality evaluator. Assess this answer critically.

Question: {question}

Answer: {answer}

Context length: {len(context)}

Evaluate on these criteria (1-10):
RELEVANCE: 
COMPLETENESS: 
CONFIDENCE: 

ISSUES:
- 
SUGGESTIONS:
- 

Return the evaluation in the exact structure above (numbers and bulleted lists).

Evaluation:"""
        feedback_text, elapsed = self._llm_invoke_timed(prompt, f"llm_feedback_assessment_{loop_num}")

        # parse feedback
        relevance = completeness = confidence = 5
        issues = []
        suggestions = []
        lines = feedback_text.split('\n')
        current = None
        for line in lines:
            l = line.strip()
            up = l.upper()
            if up.startswith('RELEVANCE'):
                nums = [int(s) for s in ''.join(ch if ch.isdigit() or ch==',' else ' ' for ch in l).split() if s.isdigit()]
                if nums: relevance = max(1, min(10, nums[0]))
            elif up.startswith('COMPLETENESS'):
                nums = [int(s) for s in ''.join(ch if ch.isdigit() or ch==',' else ' ' for ch in l).split() if s.isdigit()]
                if nums: completeness = max(1, min(10, nums[0]))
            elif up.startswith('CONFIDENCE'):
                nums = [int(s) for s in ''.join(ch if ch.isdigit() or ch==',' else ' ' for ch in l).split() if s.isdigit()]
                if nums: confidence = max(1, min(10, nums[0]))
            elif up.startswith('ISSUES'):
                current = 'issues'
            elif up.startswith('SUGGESTIONS'):
                current = 'suggestions'
            elif l.startswith('-') and current == 'issues':
                issues.append(l[1:].strip())
            elif l.startswith('-') and current == 'suggestions':
                suggestions.append(l[1:].strip())

        assessment = FeedbackAssessment(relevance, completeness, confidence, issues, suggestions)
        return assessment, feedback_text, elapsed

    def reformulate_query(self, original_query: str, feedback: FeedbackAssessment, loop_num: int) -> Tuple[str, int]:
        issues_text = '\n'.join(f'- {i}' for i in feedback.issues[:3]) if feedback.issues else '- lacks depth'
        suggestions_text = '\n'.join(f'- {s}' for s in feedback.suggestions[:3]) if feedback.suggestions else '- need specifics'
        prompt = f"""Reformulate this search query to retrieve missing info.
Original: {original_query}
Issues:
{issues_text}
Needed:
{suggestions_text}
Return a short improved query under 120 chars.

Reformulated:"""
        reformulated, elapsed = self._llm_invoke_timed(prompt, f"llm_query_reformulation_{loop_num}")
        reformulated = reformulated.strip().split('\n')[0][:120]
        if len(reformulated) < 8:
            reformulated = original_query
        return reformulated, elapsed

    def generate_answer(self, question: str, context: str, previous_feedback: str = "") -> Tuple[str, int]:
        if previous_feedback:
            prompt = f"""Improve previous answer using feedback:
Previous feedback: {previous_feedback}
Question: {question}
Context:\n{context}\n
Improved Answer:"""
        else:
            prompt = f"""Answer the question using the context.
Question: {question}
Context:\n{context}\n
Answer:"""
        ans, elapsed = self._llm_invoke_timed(prompt, 'llm_generate_answer')
        return ans, elapsed

    def solve_step(self, step: str) -> Dict[str, Any]:
        print(f"\n➡️ Solving step: {step}")
        current_query = step
        feedback_history = []
        loop_results = []
        best_answer = ""
        best_assessment = None

        for loop_num in range(1, self.max_loops + 1):
            loop_start = time.time_ns()

            # retrieve
            context, retrieval_time = self.retrieve(current_query, k=4)
            latency_report.add(f"step_{loop_num}_retrieval", retrieval_time)

            # previous feedback summary
            prev_summary = ""
            if feedback_history:
                prev = feedback_history[-1]['assessment']
                prev_summary = '; '.join(prev.issues[:2]) if prev.issues else ''

            # generate
            answer, gen_time = self.generate_answer(step, context, prev_summary)
            latency_report.add(f"step_{loop_num}_generation", gen_time)

            # assess
            assessment, feedback_text, assess_time = self.assess_answer(step, answer, context, loop_num)
            latency_report.add(f"step_{loop_num}_assessment", assess_time)

            loop_elapsed = time.time_ns() - loop_start
            latency_report.add('step_feedback_loop', loop_elapsed)

            loop_results.append({
                'loop': loop_num,
                'query': current_query,
                'context_len': len(context),
                'answer': answer,
                'assessment': assessment,
                'feedback_text': feedback_text,
                'time_ns': loop_elapsed,
            })
            feedback_history.append({'assessment': assessment, 'text': feedback_text})

            print(f"      📊 Loop {loop_num} assessment: {assessment.get_score_summary()}")

            if assessment.is_satisfactory():
                print("      ✅ Step satisfied by assessment")
                best_answer = answer
                best_assessment = assessment
                break
            else:
                # keep best by simple heuristic (completeness+relevance)
                score = assessment.relevance + assessment.completeness + assessment.confidence
                if best_assessment is None or score > (best_assessment.relevance + best_assessment.completeness + best_assessment.confidence):
                    best_assessment = assessment
                    best_answer = answer

                if loop_num < self.max_loops:
                    current_query, reform_time = self.reformulate_query(step, assessment, loop_num)
                    latency_report.add(f"step_{loop_num}_reformulation", reform_time)
                    print(f"      🔄 Reformulated query for next loop: {current_query}")
                else:
                    print("      ⚠️ Max loops reached for this step")

        return {
            'step': step,
            'best_answer': best_answer,
            'best_assessment': best_assessment,
            'loops': loop_results,
        }

# ==========================
# Aggregator: synthesize and final assessment
# ==========================
class Aggregator:
    def __init__(self, llm: LLMWrapper):
        self.llm = llm

    def synthesize(self, step_results: List[Dict[str, Any]], question: str) -> Tuple[str, FeedbackAssessment]:
        # Simple concatenation + LLM synthesis for coherence
        combined = []
        for r in step_results:
            combined.append(f"Step: {r['step']}\nAnswer: {r['best_answer']}")
        combined_text = "\n\n".join(combined)

        prompt = f"""You are a synthesizer. Combine the step answers below into a single coherent final answer to the question.
Question: {question}

Step answers:
{combined_text}

Final Answer:"""
        start = time.time_ns()
        resp = self.llm.invoke(prompt)
        elapsed = time.time_ns() - start
        latency_report.add('aggregator_invoke', elapsed)
        final_answer = resp.content if hasattr(resp, 'content') else str(resp)

        # Final assessment using the same assess prompt
        assessor = StepSolver(self.llm, None, None)  # we only use assess_answer; pass Nones
        # Create a small context hint
        assessment, _, _ = assessor.assess_answer(question, final_answer, combined_text, loop_num=0)
        return final_answer, assessment

# ==========================
# VADER Sentiment Analyzer (benchmark)
# ==========================
class VaderSentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> Dict[str, Any]:
        scores = self.analyzer.polarity_scores(text)
        compound = scores['compound']
        if compound >= 0.05:
            label = 'POSITIVE'
            percentage = round((compound + 1) * 50, 2)
        elif compound <= -0.05:
            label = 'NEGATIVE'
            percentage = round((1 - abs(compound)) * 50, 2)
        else:
            label = 'NEUTRAL'
            percentage = round(50 + (compound * 50), 2)
        return {'label': label, 'percentage': percentage, 'compound': compound, 'scores': scores}


def run_sentiment_benchmark(sa: VaderSentimentAnalyzer, examples: List[str], target_ns: int = TARGET_NS):
    print(f"\n{'='*60}")
    print("🔥 VADER SENTIMENT BENCHMARK")
    print(f"Target: < {target_ns} ns per analysis")
    print(f"{'='*60}\n")
    times = []
    for i, text in enumerate(examples, 1):
        start = time.time_ns()
        res = sa.analyze(text)
        elapsed = time.time_ns() - start
        latency_report.add('vader_analysis', elapsed)
        times.append(elapsed)
        status = '✅' if elapsed < target_ns else '❌'
        print(f"[{i}] {format_time_ns(elapsed):25s} {status} | {res['label']:8s} | '{text[:50]}...'")
    total = sum(times)
    print(f"\nAvg: {format_time_ns(total//len(times))} | Total: {format_time_ns(total)}")

# ==========================
# Orchestrator: Plan-and-Solve with feedback
# ==========================
class PlanAndSolveRAG:
    def __init__(self, llm: LLMWrapper, client: weaviate.WeaviateClient, embedder: SentenceTransformer):
        self.llm = llm
        self.client = client
        self.embedder = embedder
        self.planner = Planner(llm)
        self.solver = StepSolver(llm, client, embedder)
        self.aggregator = Aggregator(llm)

    def run(self, question: str) -> Dict[str, Any]:
        print(f"\n{'='*70}")
        print("🔁 PLAN-AND-SOLVE RAG (with feedback loops)")
        print(f"Question: {question}")
        plan, plan_time = self.planner.create_plan(question)

        step_results = []
        for step in plan:
            res = self.solver.solve_step(step)
            step_results.append(res)

        final_answer, final_assessment = self.aggregator.synthesize(step_results, question)

        print("\n💬 FINAL ANSWER (synthesized):\n")
        print(final_answer[:1000])
        print("\n📊 FINAL ASSESSMENT:")
        print(f"   Relevance: {final_assessment.relevance}/10")
        print(f"   Completeness: {final_assessment.completeness}/10")
        print(f"   Confidence: {final_assessment.confidence}/10")

        return {
            'question': question,
            'plan': plan,
            'step_results': step_results,
            'final_answer': final_answer,
            'final_assessment': final_assessment,
        }

# ==========================
# CLI / main
# ==========================

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', type=str, default=PDF_PATH)
    parser.add_argument('--question', type=str, default=None)
    parser.add_argument('--skip_index', action='store_true')
    parser.add_argument('--run_sentiment', action='store_true')
    args = parser.parse_args(argv)

    # Load embedder
    embedder = load_embeddings()

    # Init weaviate
    if not WEAVIATE_API_KEY:
        print("⚠️  Set WEAVIATE_API_KEY environment variable.")
        return
    client = init_weaviate(WEAVIATE_URL, WEAVIATE_API_KEY)

    # Index PDF unless skipped
    if not args.skip_index:
        print("Indexing PDF into Weaviate...")
        text = load_pdf(args.pdf)
        chunks = chunk_text(text)
        insert_chunks(client, embedder, chunks)

    # LLM wrapper
    llm = LLMWrapper(groq_api_key=GROQ_API_KEY)

    rag = PlanAndSolveRAG(llm, client, embedder)

    # Optional VADER benchmark
    if args.run_sentiment:
        sa = VaderSentimentAnalyzer()
        examples = [
            "I absolutely loved the story, it moved me to tears.",
            "This is the worst experience I've ever had.",
            "It was okay, not great but not terrible either.",
        ]
        run_sentiment_benchmark(sa, examples)

    if args.question:
        out = rag.run(args.question)
        latency_report.pretty_print()
    else:
        print("Interactive mode. Type a question (or 'exit'):")
        while True:
            try:
                q = input('> ').strip()
            except (EOFError, KeyboardInterrupt):
                print('\nExiting.')
                break
            if not q:
                continue
            if q.lower() in ('exit', 'quit'):
                break
            try:
                rag.run(q)
                latency_report.pretty_print()
            except Exception as e:
                print(f"⚠️ Error: {e}")
                traceback.print_exc()


if __name__ == '__main__':
    main()
