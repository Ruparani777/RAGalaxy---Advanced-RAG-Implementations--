#!/usr/bin/env python3
"""
weaviate_feedback_rag.py (COMPLETED)
RAG with Feedback Loops and comprehensive nanosecond latency instrumentation.

Notes:
- This file completes the pipeline: PDF load -> chunk -> embed -> upsert -> feedback RAG loop.
- The LLM wrapper attempts to use langchain_groq.ChatGroq when available; otherwise falls back to a safe echo stub for offline testing.
- All calls are timed and recorded in LatencyReport.

Run:
    python weaviate_feedback_rag_completed.py --pdf "Data/ECHOES OF HER LOVE.pdf" --test_question "Who is the protagonist?"

"""

import os
import time
import sys
import traceback
import argparse
from collections import defaultdict
from typing import List, Dict, Any, Tuple

import pdfplumber
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import MetadataQuery
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Optional LLM import (may not be installed in all environments)
try:
    from langchain_groq import ChatGroq
except Exception:
    ChatGroq = None

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# =========================================================
# CONFIG
# =========================================================
PDF_PATH = "Data/ECHOES OF HER LOVE.pdf"
COLLECTION_NAME = "FeedbackRAG_Documents"
DIM = 384  # MiniLM embedding dimension
MODEL_NAME = "llama-3.1-8b-instant"
TARGET_NS = 200_000
MAX_FEEDBACK_LOOPS = 3  # Maximum number of feedback iterations

# Weaviate credentials (keep safe)
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "21ookhjbswyl5urlawqmxw.c0.asia-southeast1.gcp.weaviate.cloud")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "NTVWQ1dZVDI1bkptcndrZF9JRTFySVg3TEFBc1R5V0luUEtHaU9MajB6am5VQkc3aG5yVkgwWkFQVDc0PV92MjAw")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Quality thresholds
RELEVANCE_THRESHOLD = 7  # Out of 10
COMPLETENESS_THRESHOLD = 7  # Out of 10
CONFIDENCE_THRESHOLD = 6  # Out of 10

# =========================================================
# LATENCY UTILITIES (same as provided)
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
# PDF / chunk / embeddings / weaviate functions
# =========================================================

@timer_ns
def load_pdf(path: str) -> str:
    """Load PDF with per-page timing"""
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


@timer_ns
def load_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load embedding model with timing"""
    print(f"🔢 Loading embeddings model: {model_name}")
    embedder = SentenceTransformer(model_name)
    print(f"✅ Embeddings model loaded")
    return embedder


@timer_ns
def init_weaviate(url: str, api_key: str, collection_name: str = COLLECTION_NAME) -> weaviate.WeaviateClient:
    """Initialize Weaviate client and collection with timing"""
    print(f"🗃️  Initializing Weaviate connection to {url}")
    start = time.time_ns()

    # Connect to Weaviate Cloud
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=url,
        auth_credentials=Auth.api_key(api_key)
    )

    connect_time = time.time_ns() - start
    latency_report.add("weaviate_connect", connect_time)
    print(f"✅ Connected to Weaviate ({format_time_ns(connect_time)})")

    # Delete collection if exists
    try:
        if client.collections.exists(collection_name):
            start = time.time_ns()
            client.collections.delete(collection_name)
            delete_time = time.time_ns() - start
            latency_report.add("weaviate_delete_collection", delete_time)
            print(f"🗑️  Deleted existing collection '{collection_name}'")
    except Exception as e:
        print(f"⚠️  Collection check/delete: {e}")

    # Create collection
    start = time.time_ns()
    try:
        client.collections.create(
            name=collection_name,
            vectorizer_config=None,  # We'll provide vectors manually
            properties=[
                {"name": "text", "dataType": ["text"]},
                {"name": "chunk_id", "dataType": ["int"]},
                {"name": "source", "dataType": ["text"]}
            ]
        )
        create_time = time.time_ns() - start
        latency_report.add("weaviate_create_collection", create_time)
        print(f"✅ Collection '{collection_name}' created ({format_time_ns(create_time)})")
    except Exception as e:
        print(f"⚠️  Collection creation: {e}")

    return client


@timer_ns
def insert_chunks(client: weaviate.WeaviateClient, embedder: SentenceTransformer,
                  chunks: List[str], collection_name: str = COLLECTION_NAME) -> None:
    """Insert chunks into Weaviate with detailed timing"""
    print(f"⬆️  Inserting {len(chunks)} chunks into Weaviate...")

    # Encode chunks (batch embedding)
    print(f"   🔢 Encoding {len(chunks)} chunks...")
    start = time.time_ns()
    vectors = embedder.encode(chunks, show_progress_bar=False)
    encode_time = time.time_ns() - start
    latency_report.add("embedding_encode_batch", encode_time)
    print(f"   ✅ Encoded in {format_time_ns(encode_time)}")

    # Get collection
    collection = client.collections.get(collection_name)

    # Insert objects with vectors
    print(f"   💾 Upserting to Weaviate...")
    start = time.time_ns()

    with collection.batch.dynamic() as batch:
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            batch.add_object(
                properties={
                    "text": chunk,
                    "chunk_id": i,
                    "source": f"chunk_{i}"
                },
                vector=vector.tolist()
            )

    upsert_time = time.time_ns() - start
    latency_report.add("weaviate_upsert", upsert_time)
    print(f"   ✅ Upserted in {format_time_ns(upsert_time)}")

    print(f"✅ All chunks inserted successfully!")


# search_weaviate unchanged

def search_weaviate(client: weaviate.WeaviateClient, embedder: SentenceTransformer,
                    query: str, limit: int = 4, collection_name: str = COLLECTION_NAME) -> Tuple[List[str], int]:
    """Search Weaviate with timing"""
    # Encode query
    start = time.time_ns()
    qvec = embedder.encode([query])[0]
    encode_time = time.time_ns() - start
    latency_report.add("query_embedding", encode_time)

    # Query Weaviate
    start = time.time_ns()
    collection = client.collections.get(collection_name)

    response = collection.query.near_vector(
        near_vector=qvec.tolist(),
        limit=limit,
        return_metadata=MetadataQuery(distance=True)
    )

    search_time = time.time_ns() - start
    latency_report.add("weaviate_search", search_time)

    # Extract texts
    hits = [obj.properties.get("text", "") for obj in response.objects]

    total_time = encode_time + search_time

    return hits, total_time


# =========================================================
# FEEDBACK QUALITY ASSESSMENT + FeedbackRAG
# (copy as provided, minor addition: type hints and small fixes)
# =========================================================

class FeedbackAssessment:
    def __init__(self, relevance: int, completeness: int, confidence: int,
                 issues: List[str], suggestions: List[str]):
        self.relevance = relevance
        self.completeness = completeness
        self.confidence = confidence
        self.issues = issues
        self.suggestions = suggestions
    
    def is_satisfactory(self) -> bool:
        return (self.relevance >= RELEVANCE_THRESHOLD and 
                self.completeness >= COMPLETENESS_THRESHOLD and
                self.confidence >= CONFIDENCE_THRESHOLD)
    
    def get_score_summary(self) -> str:
        return f"R:{self.relevance}/10, C:{self.completeness}/10, Conf:{self.confidence}/10"

    def __repr__(self):
        return (f"FeedbackAssessment(relevance={self.relevance}, "
                f"completeness={self.completeness}, confidence={self.confidence})")


class LLMWrapper:
    """Small wrapper that exposes an .invoke(prompt) method used by FeedbackRAG.

    It tries a few ways to call the real LLM (ChatGroq or similar). If none
    available, it returns a safe placeholder response so the rest of the
    pipeline can be tested offline.
    """
    def __init__(self, groq_api_key: str = None, model: str = MODEL_NAME):
        self.model = model
        self.groq_api_key = groq_api_key
        self.client = None
        if ChatGroq and groq_api_key:
            try:
                # Try to instantiate ChatGroq (API may differ between versions)
                self.client = ChatGroq(api_key=groq_api_key, model=self.model)
            except Exception:
                try:
                    # alternative constructor
                    self.client = ChatGroq(api_key=groq_api_key)
                except Exception:
                    self.client = None

    def invoke(self, prompt: str) -> Any:
        # If we have a ChatGroq-like client, try multiple call patterns
        if self.client is not None:
            try:
                if hasattr(self.client, 'invoke'):
                    return self.client.invoke(prompt)
                if hasattr(self.client, 'generate'):
                    # some wrappers accept a list of messages or prompts
                    out = self.client.generate([prompt])
                    # attempt to extract textual content
                    if hasattr(out, 'generations'):
                        return out.generations[0].text
                    return str(out)
                if callable(self.client):
                    return self.client(prompt)
            except Exception as e:
                print(f"⚠️ LLM client call failed: {e}")
                traceback.print_exc()

        # Fallback: very simple deterministic "LLM" for testing
        class _Resp:
            def __init__(self, content):
                self.content = content
            def __str__(self):
                return self.content

        # Heuristic stub: echo prompt head + a canned completion
        head = prompt.strip().split('\n')[:6]
        preview = '\n'.join(head)
        stub = (
            f"[STUB LLM RESPONSE]\n\nPrompt preview:\n{preview}\n\n"
            "Note: this is a fallback response (no Groq key detected)."
        )
        return _Resp(stub)


class FeedbackRAG:
    # (Use exactly the class provided by the user, but accept LLMWrapper and other types)
    def __init__(self, llm, client: weaviate.WeaviateClient, embedder: SentenceTransformer,
                 collection_name: str = COLLECTION_NAME, max_loops: int = MAX_FEEDBACK_LOOPS):
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
            latency_report.add(label + "_error", elapsed)
            print(f"❌ LLM invoke for {label} failed: {e}")
            traceback.print_exc()
            return str(e), elapsed

    def retrieve_documents(self, query: str, k: int = 4) -> Tuple[str, int]:
        print(f"      🔍 Retrieving documents...")
        print(f"         Query: {query[:70]}{'...' if len(query) > 70 else ''}")
        hits, elapsed = search_weaviate(self.client, self.embedder, query, k, self.collection_name)
        context = "\n\n".join(hits)
        print(f"      ✅ Retrieved {len(hits)} docs ({len(context)} chars) in {format_time_ns(elapsed)}")
        return context, elapsed

    def generate_answer(self, query: str, context: str, previous_feedback: str = "") -> Tuple[str, int]:
        if previous_feedback:
            prompt = f"""You are improving a previous answer based on feedback.

Previous Feedback Issues:
{previous_feedback}

Question: {query}

Retrieved Context:
{context}

Instructions: Generate an IMPROVED answer that specifically addresses the feedback issues. Be thorough, relevant, and well-supported by the context.

Improved Answer:"""
        else:
            prompt = f"""Answer the following question based on the retrieved context.

Question: {query}

Context:
{context}

Provide a clear, comprehensive, and well-structured answer based on the context above.

Answer:"""

        print(f"      💡 Generating answer...")
        answer, elapsed = self._llm_invoke_timed(prompt, "llm_generate_answer")
        print(f"      ✅ Generated ({len(answer)} chars) in {format_time_ns(elapsed)}")
        return answer, elapsed

    def assess_answer_quality(self, query: str, answer: str, context: str,
                              loop_num: int) -> Tuple[FeedbackAssessment, str, int]:
        prompt = f"""You are a strict quality evaluator. Assess this answer critically.

Question: {query}

Answer: {answer}

Context Length: {len(context)} characters

Evaluate on these criteria (1-10 scale, be strict):

1. RELEVANCE: Does it directly answer the question?
2. COMPLETENESS: Does it cover all aspects needed?
3. CONFIDENCE: Is it well-supported by context?

Format your response EXACTLY as:
RELEVANCE: [number 1-10]
COMPLETENESS: [number 1-10]
CONFIDENCE: [number 1-10]

ISSUES:
- [specific problem 1]
- [specific problem 2]

SUGGESTIONS:
- [improvement 1]
- [improvement 2]

Evaluation:"""
        print(f"      🔍 Assessing quality...")
        feedback_text, elapsed = self._llm_invoke_timed(prompt, f"llm_feedback_assessment_{loop_num}")

        # Parse feedback with robust extraction
        relevance, completeness, confidence = 5, 5, 5  # defaults
        issues, suggestions = [], []

        lines = feedback_text.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            upper_line = line.upper()

            if 'RELEVANCE:' in upper_line or 'RELEVANCE =' in upper_line:
                try:
                    nums = [int(c) for c in line if c.isdigit()]
                    if nums:
                        relevance = min(10, max(1, nums[0] if len(nums) == 1 else int(''.join(map(str, nums[:2])))))
                except:
                    pass
            elif 'COMPLETENESS:' in upper_line or 'COMPLETENESS =' in upper_line:
                try:
                    nums = [int(c) for c in line if c.isdigit()]
                    if nums:
                        completeness = min(10, max(1, nums[0] if len(nums) == 1 else int(''.join(map(str, nums[:2])))))
                except:
                    pass
            elif 'CONFIDENCE:' in upper_line or 'CONFIDENCE =' in upper_line:
                try:
                    nums = [int(c) for c in line if c.isdigit()]
                    if nums:
                        confidence = min(10, max(1, nums[0] if len(nums) == 1 else int(''.join(map(str, nums[:2])))))
                except:
                    pass
            elif 'ISSUES:' in upper_line:
                current_section = 'issues'
            elif 'SUGGESTIONS:' in upper_line or 'IMPROVEMENTS:' in upper_line:
                current_section = 'suggestions'
            elif line.startswith('-') and current_section == 'issues':
                issues.append(line[1:].strip())
            elif line.startswith('-') and current_section == 'suggestions':
                suggestions.append(line[1:].strip())

        assessment = FeedbackAssessment(relevance, completeness, confidence, issues, suggestions)

        print(f"      📊 Scores: {assessment.get_score_summary()}")
        print(f"      ✅ Assessment done in {format_time_ns(elapsed)}")

        return assessment, feedback_text, elapsed

    def reformulate_query(self, original_query: str, feedback: FeedbackAssessment,
                         loop_num: int) -> Tuple[str, int]:
        issues_text = '\n'.join(f'- {issue}' for issue in feedback.issues[:3]) if feedback.issues else '- Information lacks depth'
        suggestions_text = '\n'.join(f'- {sug}' for sug in feedback.suggestions[:3]) if feedback.suggestions else '- Need more specific information'

        prompt = f"""Reformulate this search query to get better retrieval results.

Original Question: {original_query}

Current Problems:
{issues_text}

What's Needed:
{suggestions_text}

Task: Create a MORE SPECIFIC search query that will find the missing information. Focus on:
1. Key terms from issues/suggestions
2. More specific aspects needed
3. Alternative phrasings

Keep it concise (under 150 chars).

Reformulated Query:"""

        print(f"      🔄 Reformulating query...")
        reformulated, elapsed = self._llm_invoke_timed(prompt, f"llm_query_reformulation_{loop_num}")

        # Clean up
        reformulated = reformulated.strip().split('\n')[0][:150]
        if not reformulated or len(reformulated) < 10:
            reformulated = original_query  # Fallback

        print(f"      ✅ New query: {reformulated[:70]}{'...' if len(reformulated) > 70 else ''}")
        print(f"      ⏱️  Reformulation: {format_time_ns(elapsed)}")

        return reformulated, elapsed

    def query(self, question: str) -> Dict[str, Any]:
        print(f"\n{'='*70}")
        print(f"🔁 RAG WITH FEEDBACK LOOPS")
        print(f"{'='*70}")
        print(f"❓ Question: {question}\n")

        overall_start = time.time_ns()

        current_query = question
        context = ""
        answer = ""
        feedback_history = []
        loop_results = []

        for loop_num in range(1, self.max_loops + 1):
            loop_start = time.time_ns()
            print(f"\n{'─'*70}")
            print(f"🔄 FEEDBACK LOOP {loop_num}/{self.max_loops}")
            print(f"{'─'*70}")

            # Step 1: Retrieve documents
            context, retrieval_time = self.retrieve_documents(current_query, k=4)
            latency_report.add(f"loop_{loop_num}_retrieval", retrieval_time)

            # Step 2: Generate answer
            previous_feedback_summary = ""
            if feedback_history:
                prev = feedback_history[-1]['assessment']
                issues_summary = '; '.join(prev.issues[:2]) if prev.issues else 'General improvements needed'
                previous_feedback_summary = f"Previous issues: {issues_summary}"

            answer, generation_time = self.generate_answer(question, context, previous_feedback_summary)
            latency_report.add(f"loop_{loop_num}_generation", generation_time)

            # Step 3: Assess quality
            assessment, feedback_text, assessment_time = self.assess_answer_quality(
                question, answer, context, loop_num
            )
            latency_report.add(f"loop_{loop_num}_assessment", assessment_time)

            # Record loop results
            loop_elapsed = time.time_ns() - loop_start
            loop_result = {
                'loop_num': loop_num,
                'query': current_query,
                'context_length': len(context),
                'answer': answer,
                'assessment': assessment,
                'feedback_text': feedback_text,
                'time_ns': loop_elapsed
            }
            loop_results.append(loop_result)
            feedback_history.append({'assessment': assessment, 'text': feedback_text})

            latency_report.add("feedback_loop_iteration", loop_elapsed)
            print(f"\n      ⏱️  Loop {loop_num} total: {format_time_ns(loop_elapsed)}")

            # Step 4: Check if satisfactory
            if assessment.is_satisfactory():
                print(f"\n      ✅ Quality thresholds MET!")
                print(f"      🎯 Success in {loop_num} loop(s)")
                break
            else:
                print(f"\n      ⚠️  Below threshold:")
                if assessment.relevance < RELEVANCE_THRESHOLD:
                    print(f"         ❌ Relevance: {assessment.relevance}/10 (need ≥{RELEVANCE_THRESHOLD})")
                if assessment.completeness < COMPLETENESS_THRESHOLD:
                    print(f"         ❌ Completeness: {assessment.completeness}/10 (need ≥{COMPLETENESS_THRESHOLD})")
                if assessment.confidence < CONFIDENCE_THRESHOLD:
                    print(f"         ❌ Confidence: {assessment.confidence}/10 (need ≥{CONFIDENCE_THRESHOLD})")

                if assessment.issues:
                    print(f"         Issues: {assessment.issues[0][:60]}{'...' if len(assessment.issues[0]) > 60 else ''}")

                if loop_num < self.max_loops:
                    # Reformulate for next iteration
                    current_query, reform_time = self.reformulate_query(
                        question, assessment, loop_num
                    )
                    latency_report.add(f"loop_{loop_num}_reformulation", reform_time)
                    print(f"      🔄 Will retry with improved query...")
                else:
                    print(f"      ⚠️  Max loops reached. Using best available answer.")

        total_query_ns = time.time_ns() - overall_start
        latency_report.add("feedback_rag_total", total_query_ns)

        # Display final results
        final_assessment = loop_results[-1]['assessment']
        print(f"\n{'='*70}")
        print(f"💬 FINAL ANSWER (after {len(loop_results)} loop(s)):" )
        print(f"{'='*70}")
        print(answer[:800])
        if len(answer) > 800:
            print(f"... [truncated, full length: {len(answer)} chars]")

        print(f"\n📊 FINAL QUALITY SCORES:")
        print(f"   Relevance:    {final_assessment.relevance}/10 {'✅' if final_assessment.relevance >= RELEVANCE_THRESHOLD else '❌'}")
        print(f"   Completeness: {final_assessment.completeness}/10 {'✅' if final_assessment.completeness >= COMPLETENESS_THRESHOLD else '❌'}")
        print(f"   Confidence:   {final_assessment.confidence}/10 {'✅' if final_assessment.confidence >= CONFIDENCE_THRESHOLD else '❌'}")
        print(f"   Overall:      {'✅ SATISFACTORY' if final_assessment.is_satisfactory() else '⚠️  NEEDS IMPROVEMENT'}")

        print(f"\n⏱️  TIMING METRICS:")
        print(f"   Total time:   {format_time_ns(total_query_ns)}")
        print(f"   Loops:        {len(loop_results)}")
        print(f"   Avg/loop:     {format_time_ns(sum(r['time_ns'] for r in loop_results) // len(loop_results))}")
        print(f"{'='*70}\n")

        return {
            'question': question,
            'loop_results': loop_results,
            'final_answer': answer,
            'final_assessment': final_assessment,
            'num_loops': len(loop_results),
            'satisfied': final_assessment.is_satisfactory(),
            'total_query_ns': total_query_ns
        }


# =========================================================
# VADER SENTIMENT BENCHMARK (unchanged)
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
                            target_ns: int = TARGET_NS, run_number: int = 1):
    print(f"\n{'='*70}")
    print(f"🔥 SENTIMENT BENCHMARK RUN #{run_number}")
    print(f"{'='*70}")
    print(f"🎯 TARGET: < {target_ns} ns per analysis\n")

    individual_times = []
    for i, text in enumerate(examples, 1):
        start_ns = time.time_ns()
        result = sa.analyze(text)
        elapsed_ns = time.time_ns() - start_ns
        latency_report.add("vader_analysis", elapsed_ns)
        individual_times.append(elapsed_ns)

        status = "✅" if elapsed_ns < target_ns else "❌"
        print(f"[{i:2d}] {format_time_ns(elapsed_ns):25s} {status} | {result['label']:8s} | \"{text[:40]}...\"")

    total_ns = sum(individual_times)
    avg_ns = total_ns // len(individual_times)
    min_ns = min(individual_times)
    max_ns = max(individual_times)
    under_target = sum(1 for t in individual_times if t < target_ns)

    print(f"\n📊 RUN #{run_number} STATISTICS:")
    print(f"   Total:     {format_time_ns(total_ns)}")
    print(f"   Average:   {format_time_ns(avg_ns)}")
    print(f"   Min:       {format_time_ns(min_ns)}")
    print(f"   Max:       {format_time_ns(max_ns)}")
    print(f"   Success:   {under_target}/{len(individual_times)} under target")

    if avg_ns < target_ns:
        print(f"   ✅ TARGET MET!")
    else:
        print(f"   ⚠️  TARGET MISSED")


# =========================================================
# MAIN: orchestration and CLI
# =========================================================

def build_and_index(pdf_path: str, client: weaviate.WeaviateClient, embedder: SentenceTransformer,
                    collection_name: str = COLLECTION_NAME):
    text = load_pdf(pdf_path)
    chunks = chunk_text(text)
    insert_chunks(client, embedder, chunks, collection_name)
    return len(chunks)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Weaviate Feedback RAG pipeline")
    parser.add_argument('--pdf', type=str, default=PDF_PATH, help='Path to PDF to index')
    parser.add_argument('--test_question', type=str, default=None, help='Example question to run through RAG')
    parser.add_argument('--skip_index', action='store_true', help='Skip indexing and assume collection already exists')
    parser.add_argument('--run_sentiment', action='store_true', help='Run sentiment benchmark')
    args = parser.parse_args(argv)

    # Load embedder
    embedder = load_embeddings()

    # Init weaviate
    if not WEAVIATE_API_KEY or not WEAVIATE_URL:
        print("⚠️  Weaviate URL / API key not set. Exiting.")
        return

    client = init_weaviate(WEAVIATE_URL, WEAVIATE_API_KEY)

    if not args.skip_index:
        print("\nIndexing PDF into Weaviate (this may take a while)...")
        try:
            n_chunks = build_and_index(args.pdf, client, embedder)
            print(f"Indexed {n_chunks} chunks into collection {COLLECTION_NAME}")
        except Exception as e:
            print(f"⚠️  Indexing failed: {e}")
            traceback.print_exc()
            return
    else:
        print("Skipping indexing as requested (--skip_index)")

    # Instantiate LLM wrapper
    llm = LLMWrapper(groq_api_key=GROQ_API_KEY)

    # Create RAG system
    rag = FeedbackRAG(llm, client, embedder)

    # Optional sentiment benchmark
    if args.run_sentiment:
        sa = VaderSentimentAnalyzer()
        examples = [
            "I absolutely loved the story, it moved me to tears.",
            "This is the worst experience I've ever had.",
            "It was okay, not great but not terrible either.",
        ]
        run_sentiment_benchmark(sa, examples)

    # Run test question or interactive loop
    if args.test_question:
        result = rag.query(args.test_question)
        latency_report.pretty_print()
    else:
        print("\nInteractive mode. Type a question (or 'exit'):")
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
                rag.query(q)
                latency_report.pretty_print()
            except Exception as e:
                print(f"⚠️  Error running query: {e}")
                traceback.print_exc()


if __name__ == "__main__":
    main()
