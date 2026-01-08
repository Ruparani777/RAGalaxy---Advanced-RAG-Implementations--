#!/usr/bin/env python3
"""
qdrant_feedback_rag.py
RAG with Feedback Loops - Learn from user feedback to improve responses

Feedback Loop Features:
1. User Feedback Collection: Thumbs up/down, ratings, corrections
2. Feedback Analysis: Extract patterns from negative feedback
3. Response Refinement: Improve responses based on feedback
4. Quality Tracking: Monitor improvement over time
5. Adaptive Retrieval: Adjust search based on feedback history
"""

import os
import time
import sys
import json
import re
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime
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
COLLECTION = "feedback_rag_collection"
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

# =========================================================
# FEEDBACK STRUCTURES
# =========================================================
@dataclass
class UserFeedback:
    """User feedback on a response"""
    query_id: int
    question: str
    answer: str
    feedback_type: str  # positive, negative, correction, rating
    rating: Optional[int]  # 1-5 if rating
    comment: Optional[str]
    correction: Optional[str]
    timestamp: str
    
    def to_dict(self):
        return asdict(self)

@dataclass
class FeedbackPattern:
    """Identified pattern from feedback"""
    pattern_type: str  # length, detail, accuracy, relevance
    issue: str
    frequency: int
    examples: List[str]
    first_seen: str
    last_seen: str

class FeedbackManager:
    """
    Manages feedback collection and learning
    
    Features:
    - Collect user feedback
    - Analyze feedback patterns
    - Adjust retrieval and generation
    - Track quality metrics
    """
    
    def __init__(self):
        self.feedback_history: List[UserFeedback] = []
        self.feedback_patterns: Dict[str, FeedbackPattern] = {}
        self.query_counter = 0
        self.quality_metrics = {
            'total_queries': 0,
            'positive_feedback': 0,
            'negative_feedback': 0,
            'avg_rating': 0.0,
            'improvement_rate': 0.0
        }
        
        print(f"🔄 Feedback Manager initialized")
    
    def collect_feedback(self, question: str, answer: str, 
                        feedback_type: str = "positive",
                        rating: Optional[int] = None,
                        comment: Optional[str] = None,
                        correction: Optional[str] = None) -> UserFeedback:
        """Collect user feedback on a response"""
        
        start = time.time_ns()
        
        feedback = UserFeedback(
            query_id=self.query_counter,
            question=question,
            answer=answer,
            feedback_type=feedback_type,
            rating=rating,
            comment=comment,
            correction=correction,
            timestamp=datetime.now().isoformat()
        )
        
        self.feedback_history.append(feedback)
        self.query_counter += 1
        
        # Update metrics
        self.quality_metrics['total_queries'] += 1
        if feedback_type == "positive":
            self.quality_metrics['positive_feedback'] += 1
        elif feedback_type == "negative":
            self.quality_metrics['negative_feedback'] += 1
        
        if rating:
            total_ratings = sum(f.rating for f in self.feedback_history if f.rating)
            count_ratings = sum(1 for f in self.feedback_history if f.rating)
            self.quality_metrics['avg_rating'] = total_ratings / count_ratings if count_ratings else 0
        
        elapsed = time.time_ns() - start
        latency_report.add("feedback_collect", elapsed)
        
        print(f"✅ Feedback collected: {feedback_type}" + 
              (f" (rating: {rating}/5)" if rating else ""))
        
        return feedback
    
    def analyze_feedback_patterns(self, llm) -> Dict[str, FeedbackPattern]:
        """Analyze feedback to identify patterns"""
        
        if len(self.feedback_history) < 2:
            return {}
        
        start = time.time_ns()
        
        # Get recent negative feedback
        recent_negative = [
            f for f in self.feedback_history[-10:] 
            if f.feedback_type == "negative"
        ]
        
        if not recent_negative:
            return self.feedback_patterns
        
        # Analyze with LLM
        feedback_text = "\n\n".join([
            f"Q: {f.question}\nA: {f.answer[:200]}...\nComment: {f.comment or 'None'}"
            for f in recent_negative
        ])
        
        prompt = f"""Analyze these negative feedback instances and identify patterns.

Feedback Examples:
{feedback_text}

Identify common issues (e.g., too brief, lacks detail, off-topic, incorrect).

Return JSON:
{{
  "patterns": [
    {{"type": "length|detail|accuracy|relevance", "issue": "description", "frequency": 3}}
  ]
}}

Analysis:"""
        
        try:
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                patterns = data.get('patterns', [])
                
                for p in patterns:
                    pattern_key = f"{p['type']}_{p['issue'][:20]}"
                    
                    if pattern_key in self.feedback_patterns:
                        self.feedback_patterns[pattern_key].frequency += 1
                        self.feedback_patterns[pattern_key].last_seen = datetime.now().isoformat()
                    else:
                        self.feedback_patterns[pattern_key] = FeedbackPattern(
                            pattern_type=p['type'],
                            issue=p['issue'],
                            frequency=p.get('frequency', 1),
                            examples=[],
                            first_seen=datetime.now().isoformat(),
                            last_seen=datetime.now().isoformat()
                        )
                
                print(f"🔍 Identified {len(patterns)} feedback patterns")
        
        except Exception as e:
            print(f"⚠️  Pattern analysis failed: {e}")
        
        elapsed = time.time_ns() - start
        latency_report.add("feedback_analyze", elapsed)
        
        return self.feedback_patterns
    
    def get_adjustment_instructions(self) -> str:
        """Get instructions for adjusting response based on feedback"""
        
        if not self.feedback_patterns:
            return ""
        
        # Get most frequent patterns
        top_patterns = sorted(
            self.feedback_patterns.values(),
            key=lambda p: p.frequency,
            reverse=True
        )[:3]
        
        instructions = []
        for p in top_patterns:
            if p.pattern_type == "length":
                instructions.append(f"- Provide more detailed responses ({p.issue})")
            elif p.pattern_type == "detail":
                instructions.append(f"- Include more specific details ({p.issue})")
            elif p.pattern_type == "accuracy":
                instructions.append(f"- Ensure accuracy: {p.issue}")
            elif p.pattern_type == "relevance":
                instructions.append(f"- Stay focused on: {p.issue}")
        
        if instructions:
            return "Based on user feedback:\n" + "\n".join(instructions)
        return ""
    
    def get_quality_report(self) -> Dict[str, Any]:
        """Get quality metrics report"""
        
        total = self.quality_metrics['total_queries']
        if total == 0:
            return self.quality_metrics
        
        pos = self.quality_metrics['positive_feedback']
        neg = self.quality_metrics['negative_feedback']
        
        satisfaction_rate = (pos / total) * 100 if total > 0 else 0
        
        return {
            **self.quality_metrics,
            'satisfaction_rate': satisfaction_rate,
            'total_patterns': len(self.feedback_patterns)
        }

# =========================================================
# PDF/CHUNKING/EMBEDDINGS/QDRANT
# =========================================================
@timer_ns
def load_pdf(path: str) -> str:
    print(f"📄 Loading PDF: {path}")
    text = ""
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            t = p.extract_text() or ""
            text += t + "\n"
    print(f"✅ Loaded PDF: {len(text)} characters")
    return text

@timer_ns
def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
    print(f"✂️  Chunking text...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    print(f"✅ Created {len(chunks)} chunks")
    return chunks

@timer_ns
def load_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    print(f"🔢 Loading embeddings: {model_name}")
    embedder = SentenceTransformer(model_name)
    print(f"✅ Embeddings loaded")
    return embedder

@timer_ns
def init_qdrant(collection_name: str = COLLECTION, dim: int = DIM) -> QdrantClient:
    print(f"🗃️  Initializing Qdrant")
    qdrant = QdrantClient(":memory:")
    
    if qdrant.collection_exists(collection_name):
        qdrant.delete_collection(collection_name)
    
    qdrant.create_collection(
        collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
    )
    print(f"✅ Qdrant ready")
    return qdrant

@timer_ns
def insert_chunks(qdrant: QdrantClient, embedder: SentenceTransformer, 
                  chunks: List[str], collection_name: str = COLLECTION) -> None:
    print(f"⬆️  Inserting {len(chunks)} chunks...")
    
    vectors = embedder.encode(chunks, show_progress_bar=False)
    
    points = [
        PointStruct(
            id=i,
            vector=vectors[i].tolist(),
            payload={"text": chunks[i], "chunk_id": i}
        )
        for i in range(len(chunks))
    ]
    
    qdrant.upsert(collection_name=collection_name, points=points)
    print(f"✅ Chunks inserted!")

def search_qdrant(qdrant: QdrantClient, embedder: SentenceTransformer, 
                  query: str, limit: int = 4, collection_name: str = COLLECTION) -> Tuple[List[str], int]:
    start = time.time_ns()
    qvec = embedder.encode([query])[0]
    
    response = qdrant.query_points(
        collection_name=collection_name,
        query=qvec.tolist(),
        limit=limit
    )
    
    elapsed = time.time_ns() - start
    latency_report.add("qdrant_search", elapsed)
    
    hits = [p.payload.get("text", "") for p in response.points]
    return hits, elapsed

# =========================================================
# VADER SENTIMENT
# =========================================================
class VaderSentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    
    def analyze(self, text: str) -> Dict[str, Any]:
        scores = self.analyzer.polarity_scores(text)
        compound = scores['compound']
        
        if compound >= 0.05:
            label = "POSITIVE"
        elif compound <= -0.05:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"
        
        return {'label': label, 'compound': compound}

def run_sentiment_benchmark(sa: VaderSentimentAnalyzer, examples: List[str], 
                            target_ns: int = 200_000):
    print(f"\n{'='*70}")
    print(f"🔥 SENTIMENT BENCHMARK")
    print(f"{'='*70}")
    print(f"🎯 TARGET: < {target_ns} ns\n")
    
    times = []
    for i, text in enumerate(examples, 1):
        start = time.time_ns()
        result = sa.analyze(text)
        elapsed = time.time_ns() - start
        times.append(elapsed)
        
        status = "✅" if elapsed < target_ns else "❌"
        print(f"[{i:2d}] {format_time_ns(elapsed):20s} {status} | {result['label']:8s} | \"{text}\"")
    
    avg = sum(times) // len(times)
    print(f"\n📊 Average: {format_time_ns(avg)}")

# =========================================================
# RAG WITH FEEDBACK LOOPS
# =========================================================
class FeedbackRAG:
    """RAG system with feedback loop integration"""
    
    def __init__(self, llm, qdrant: QdrantClient, embedder: SentenceTransformer, 
                 feedback_manager: FeedbackManager, collection_name: str = COLLECTION):
        self.llm = llm
        self.qdrant = qdrant
        self.embedder = embedder
        self.feedback_manager = feedback_manager
        self.collection_name = collection_name
    
    def query(self, question: str, auto_feedback: bool = False,
              feedback_type: str = "positive", rating: Optional[int] = None) -> Dict[str, Any]:
        """
        Query with feedback loop integration
        
        Args:
            question: User question
            auto_feedback: Simulate automatic feedback for demo
            feedback_type: Type of feedback (positive/negative)
            rating: Optional 1-5 rating
        """
        print(f"\n{'='*70}")
        print(f"🔄 FEEDBACK RAG QUERY")
        print(f"{'='*70}")
        print(f"❓ Question: {question}\n")
        
        overall_start = time.time_ns()
        
        # Step 1: Get adjustment instructions from feedback
        adjustments = self.feedback_manager.get_adjustment_instructions()
        if adjustments:
            print("📊 Applying feedback adjustments:")
            print(f"   {adjustments[:100]}...")
        
        # Step 2: Retrieve from Qdrant
        print("\n📚 Retrieving from documents...")
        hits, ret_time = search_qdrant(self.qdrant, self.embedder, question, 4, self.collection_name)
        doc_context = "\n\n".join(hits)
        print(f"   ✓ Retrieved: {len(hits)} docs ({format_time_ns(ret_time)})")
        
        # Step 3: Generate answer with feedback adjustments
        print("💭 Generating answer...")
        
        prompt = f"""Answer the question based on the context.

{adjustments}

Context:
{doc_context}

Question: {question}

Provide a comprehensive, accurate answer.

Answer:"""
        
        start = time.time_ns()
        response = self.llm.invoke(prompt)
        gen_time = time.time_ns() - start
        latency_report.add("llm_generate_with_feedback", gen_time)
        
        answer = response.content if hasattr(response, 'content') else str(response)
        
        print(f"\n💬 ANSWER ({format_time_ns(gen_time)}):")
        print(answer[:400])
        if len(answer) > 400:
            print("...")
        
        # Step 4: Collect feedback (automatic for demo, or manual)
        if auto_feedback:
            print(f"\n📝 Collecting feedback...")
            feedback = self.feedback_manager.collect_feedback(
                question=question,
                answer=answer,
                feedback_type=feedback_type,
                rating=rating
            )
            
            # Analyze patterns periodically
            if len(self.feedback_manager.feedback_history) % 3 == 0:
                print("🔍 Analyzing feedback patterns...")
                self.feedback_manager.analyze_feedback_patterns(self.llm)
        
        total = time.time_ns() - overall_start
        latency_report.add("feedback_rag_total", total)
        
        # Get quality metrics
        quality = self.feedback_manager.get_quality_report()
        
        print(f"\n📊 Quality Metrics:")
        print(f"   Total queries: {quality['total_queries']}")
        print(f"   Satisfaction: {quality.get('satisfaction_rate', 0):.1f}%")
        print(f"   Avg rating: {quality['avg_rating']:.2f}/5")
        print(f"   Patterns identified: {quality['total_patterns']}")
        
        print(f"\n⏱️  Total: {format_time_ns(total)}")
        print(f"{'='*70}\n")
        
        return {
            'question': question,
            'answer': answer,
            'quality_metrics': quality,
            'total_time': total
        }

# =========================================================
# MAIN
# =========================================================
def main():
    print("="*70)
    print("🔄 RAG WITH FEEDBACK LOOPS")
    print("="*70)
    print()
    
    pipeline_start = time.time_ns()
    
    # Phase 1: Data preparation
    print("📚 PHASE 1: DATA PREPARATION")
    print("-"*70)
    
    text = load_pdf(PDF_PATH)
    chunks = chunk_text(text, 1000, 100)
    embedder = load_embeddings()
    qdrant = init_qdrant(COLLECTION, DIM)
    insert_chunks(qdrant, embedder, chunks, COLLECTION)
    
    # Phase 2: Initialize LLM
    print(f"\n📚 PHASE 2: LLM INITIALIZATION")
    print("-"*70)
    
    llm_start = time.time_ns()
    llm = ChatGroq(model_name=MODEL_NAME, groq_api_key=GROQ_API_KEY, temperature=0)
    llm_time = time.time_ns() - llm_start
    latency_report.add("llm_init", llm_time)
    print(f"✅ LLM initialized ({format_time_ns(llm_time)})")
    
    # Initialize Feedback Manager
    feedback_mgr = FeedbackManager()
    
    # Initialize Feedback RAG
    fb_rag = FeedbackRAG(llm, qdrant, embedder, feedback_mgr, COLLECTION)
    print(f"\n✅ Feedback RAG system initialized!")
    
    # Phase 3: Queries with feedback simulation
    print(f"\n📚 PHASE 3: RAG WITH FEEDBACK LOOPS")
    print("-"*70)
    
    # Simulate conversation with feedback
    interactions = [
        {"q": "What are the main themes?", "feedback": "positive", "rating": 5},
        {"q": "Tell me about love theme", "feedback": "negative", "rating": 2, 
         "comment": "Too brief, need more detail"},
        {"q": "Elaborate on mother-daughter relationship", "feedback": "positive", "rating": 4},
        {"q": "What are the key events?", "feedback": "positive", "rating": 5},
    ]
    
    results = []
    for i, interaction in enumerate(interactions, 1):
        print(f"\n{'─'*70}")
        print(f"INTERACTION {i}/{len(interactions)}")
        print(f"{'─'*70}")
        
        result = fb_rag.query(
            interaction['q'],
            auto_feedback=True,
            feedback_type=interaction['feedback'],
            rating=interaction.get('rating')
        )
        results.append(result)
        time.sleep(0.5)
    
    # Phase 4: Sentiment benchmark
    print(f"\n📚 PHASE 4: VADER SENTIMENT BENCHMARK")
    print("-"*70)
    
    sa = VaderSentimentAnalyzer()
    examples = [
        "I love this product!",
        "This is very bad service.",
        "It's okay, not too good, not too bad.",
        "Not great, really disappointed",
        "Amazing experience!"
    ]
    
    run_sentiment_benchmark(sa, examples, 200_000)
    
    # Final summary
    pipeline_total = time.time_ns() - pipeline_start
    latency_report.add("pipeline_total", pipeline_total)
    
    print(f"\n{'='*70}")
    print(f"📈 PIPELINE SUMMARY")
    print(f"{'='*70}")
    
    final_quality = feedback_mgr.get_quality_report()
    
    print(f"Total pipeline time: {format_time_ns(pipeline_total)}")
    print(f"Queries executed: {len(interactions)}")
    print(f"\n🎯 FINAL QUALITY METRICS:")
    print(f"   Total queries: {final_quality['total_queries']}")
    print(f"   Positive feedback: {final_quality['positive_feedback']}")
    print(f"   Negative feedback: {final_quality['negative_feedback']}")
    print(f"   Satisfaction rate: {final_quality.get('satisfaction_rate', 0):.1f}%")
    print(f"   Average rating: {final_quality['avg_rating']:.2f}/5")
    print(f"   Patterns identified: {final_quality['total_patterns']}")
    
    if feedback_mgr.feedback_patterns:
        print(f"\n🔍 Identified Patterns:")
        for i, (key, pattern) in enumerate(list(feedback_mgr.feedback_patterns.items())[:3], 1):
            print(f"   {i}. {pattern.pattern_type}: {pattern.issue[:50]}... ({pattern.frequency}x)")
    
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