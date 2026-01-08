#!/usr/bin/env python3
"""
qdrant_memory_augmented_rag.py
Memory-Augmented RAG with Qdrant and comprehensive latency instrumentation.

Memory Features:
- Short-term Memory: Last 10 conversation turns
- Long-term Memory: Extracted facts with mention counts
- Working Memory: Current context buffer
- Conversation continuity across queries
"""

import os
import time
import sys
import json
import re
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Any, Tuple

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
COLLECTION = "memory_rag_collection"
DIM = 384
MODEL_NAME = "llama-3.1-8b-instant"
SHORT_TERM_MEMORY_SIZE = 10

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
# MEMORY STRUCTURES
# =========================================================
@dataclass
class ConversationTurn:
    """Single conversation turn"""
    timestamp: str
    question: str
    answer: str
    context_used: str
    turn_id: int
    
    def to_text(self):
        return f"[Turn {self.turn_id}]\nQ: {self.question}\nA: {self.answer[:200]}..."

@dataclass
class LongTermFact:
    """Persistent fact or preference"""
    fact: str
    category: str  # preference, entity, relationship, summary
    confidence: float
    mentions: int
    first_seen: str
    last_updated: str

class MemoryManager:
    """
    Manages multiple memory types:
    - Short-term: Recent conversation history
    - Long-term: Persistent facts and preferences
    - Working: Current context buffer
    """
    
    def __init__(self, max_short_term=SHORT_TERM_MEMORY_SIZE):
        self.max_short_term = max_short_term
        self.short_term_memory: deque = deque(maxlen=max_short_term)
        self.long_term_memory: Dict[str, LongTermFact] = {}
        self.working_memory: Dict = {
            'current_topic': None,
            'entities_mentioned': set(),
            'temp_context': []
        }
        self.turn_counter = 0
        
        print(f"🧠 Memory Manager initialized (capacity: {max_short_term} turns)")
    
    def add_conversation_turn(self, question: str, answer: str, context: str = ""):
        """Add a conversation turn to short-term memory"""
        start = time.time_ns()
        
        turn = ConversationTurn(
            timestamp=datetime.now().isoformat(),
            question=question,
            answer=answer,
            context_used=context[:500],  # Store abbreviated context
            turn_id=self.turn_counter
        )
        
        self.short_term_memory.append(turn)
        self.turn_counter += 1
        
        elapsed = time.time_ns() - start
        latency_report.add("memory_add_turn", elapsed)
        
        print(f"💾 Added turn to short-term memory (Turn #{turn.turn_id})")
    
    def extract_and_store_facts(self, llm, question: str, answer: str):
        """Extract facts from conversation and store in long-term memory"""
        start = time.time_ns()
        
        prompt = f"""Extract key facts, preferences, or important information from this conversation.

Question: {question}
Answer: {answer}

Extract any:
- User preferences
- Important entities mentioned
- Key facts or relationships
- Topic summaries

Respond in JSON format:
{{
  "facts": [
    {{"text": "fact text", "category": "preference|entity|relationship|summary", "confidence": 0.8}}
  ]
}}

If no important facts, return {{"facts": []}}

Extraction:"""
        
        try:
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Try to parse JSON
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                facts = data.get('facts', [])
                
                for fact_data in facts:
                    fact_text = fact_data['text']
                    category = fact_data.get('category', 'summary')
                    confidence = fact_data.get('confidence', 0.7)
                    
                    # Check if fact already exists
                    if fact_text in self.long_term_memory:
                        # Update existing fact
                        self.long_term_memory[fact_text].mentions += 1
                        self.long_term_memory[fact_text].last_updated = datetime.now().isoformat()
                        self.long_term_memory[fact_text].confidence = max(
                            self.long_term_memory[fact_text].confidence,
                            confidence
                        )
                    else:
                        # Create new fact
                        self.long_term_memory[fact_text] = LongTermFact(
                            fact=fact_text,
                            category=category,
                            confidence=confidence,
                            mentions=1,
                            first_seen=datetime.now().isoformat(),
                            last_updated=datetime.now().isoformat()
                        )
                
                if facts:
                    print(f"🧠 Extracted {len(facts)} facts to long-term memory")
        
        except Exception as e:
            print(f"⚠️  Fact extraction failed: {e}")
        
        elapsed = time.time_ns() - start
        latency_report.add("memory_extract_facts", elapsed)
    
    def get_relevant_short_term_context(self, current_question: str, k: int = 3) -> str:
        """Get relevant recent conversation turns"""
        if not self.short_term_memory:
            return ""
        
        # Get last k turns
        recent_turns = list(self.short_term_memory)[-k:]
        
        context_parts = []
        for turn in recent_turns:
            context_parts.append(turn.to_text())
        
        return "\n\n".join(context_parts)
    
    def get_relevant_long_term_facts(self, query: str, k: int = 5) -> str:
        """Get relevant facts from long-term memory"""
        if not self.long_term_memory:
            return ""
        
        # Simple relevance: sort by mentions and confidence
        sorted_facts = sorted(
            self.long_term_memory.values(),
            key=lambda f: (f.mentions * f.confidence),
            reverse=True
        )
        
        top_facts = sorted_facts[:k]
        
        if not top_facts:
            return ""
        
        fact_texts = [
            f"• {fact.fact} ({fact.category}, {fact.mentions}x, conf: {fact.confidence:.2f})"
            for fact in top_facts
        ]
        
        return "Long-term Memory Facts:\n" + "\n".join(fact_texts)
    
    def get_memory_summary(self) -> Dict:
        """Get summary of current memory state"""
        return {
            'short_term_size': len(self.short_term_memory),
            'long_term_facts': len(self.long_term_memory),
            'total_turns': self.turn_counter,
            'current_topic': self.working_memory['current_topic'],
            'entities_tracked': len(self.working_memory['entities_mentioned'])
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
            start_ns = time.time_ns()
            t = p.extract_text() or ""
            elapsed = time.time_ns() - start_ns
            latency_report.add("pdf_page_extract", elapsed)
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
    print(f"🗃️  Initializing Qdrant in-memory DB")
    
    start = time.time_ns()
    qdrant = QdrantClient(":memory:")
    init_time = time.time_ns() - start
    latency_report.add("qdrant_client_init", init_time)
    
    if qdrant.collection_exists(collection_name):
        qdrant.delete_collection(collection_name)
    
    start = time.time_ns()
    qdrant.create_collection(
        collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
    )
    create_time = time.time_ns() - start
    latency_report.add("qdrant_create_collection", create_time)
    
    print(f"✅ Qdrant collection '{collection_name}' ready")
    return qdrant

@timer_ns
def insert_chunks(qdrant: QdrantClient, embedder: SentenceTransformer, 
                  chunks: List[str], collection_name: str = COLLECTION) -> None:
    print(f"⬆️  Inserting {len(chunks)} chunks...")
    
    start = time.time_ns()
    vectors = embedder.encode(chunks, show_progress_bar=False)
    encode_time = time.time_ns() - start
    latency_report.add("embedding_encode_batch", encode_time)
    print(f"   ✅ Encoded in {format_time_ns(encode_time)}")
    
    start = time.time_ns()
    points = [
        PointStruct(
            id=i,
            vector=vectors[i].tolist(),
            payload={"text": chunks[i], "chunk_id": i}
        )
        for i in range(len(chunks))
    ]
    point_time = time.time_ns() - start
    latency_report.add("qdrant_point_creation", point_time)
    
    start = time.time_ns()
    qdrant.upsert(collection_name=collection_name, points=points)
    upsert_time = time.time_ns() - start
    latency_report.add("qdrant_upsert", upsert_time)
    
    print(f"✅ Chunks inserted!")

def search_qdrant(qdrant: QdrantClient, embedder: SentenceTransformer, 
                  query: str, limit: int = 4, collection_name: str = COLLECTION) -> Tuple[List[str], int]:
    start = time.time_ns()
    qvec = embedder.encode([query])[0]
    encode_time = time.time_ns() - start
    latency_report.add("query_embedding", encode_time)
    
    start = time.time_ns()
    response = qdrant.query_points(
        collection_name=collection_name,
        query=qvec.tolist(),
        limit=limit
    )
    search_time = time.time_ns() - start
    latency_report.add("qdrant_search", search_time)
    
    hits = [p.payload.get("text", "") for p in response.points]
    total_time = encode_time + search_time
    
    return hits, total_time

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
        
        return {'label': label, 'compound': compound, 'scores': scores}

def run_sentiment_benchmark(sa: VaderSentimentAnalyzer, examples: List[str], 
                            target_ns: int = 200_000, run_number: int = 1):
    print(f"\n{'='*70}")
    print(f"🔥 SENTIMENT BENCHMARK RUN #{run_number}")
    print(f"{'='*70}")
    print(f"🎯 TARGET: < {target_ns} ns per analysis\n")
    
    times = []
    for i, text in enumerate(examples, 1):
        start = time.time_ns()
        result = sa.analyze(text)
        elapsed = time.time_ns() - start
        latency_report.add("vader_per_example", elapsed)
        times.append(elapsed)
        
        status = "✅" if elapsed < target_ns else "❌"
        print(f"[{i:2d}] {format_time_ns(elapsed):25s} {status} | {result['label']:8s} | \"{text}\"")
    
    avg = sum(times) // len(times)
    print(f"\n📊 RUN #{run_number} STATISTICS:")
    print(f"   Average:      {format_time_ns(avg)}")
    print(f"   Min:          {format_time_ns(min(times))}")
    print(f"   Max:          {format_time_ns(max(times))}")
    print(f"   {'✅ TARGET MET!' if avg < target_ns else '⚠️  TARGET MISSED'}")

# =========================================================
# MEMORY-AUGMENTED RAG
# =========================================================
class MemoryAugmentedRAG:
    """RAG with integrated memory system"""
    
    def __init__(self, llm, qdrant: QdrantClient, embedder: SentenceTransformer, 
                 memory: MemoryManager, collection_name: str = COLLECTION):
        self.llm = llm
        self.qdrant = qdrant
        self.embedder = embedder
        self.memory = memory
        self.collection_name = collection_name
    
    def query(self, question: str, use_memory: bool = True) -> Dict[str, Any]:
        """Query with memory augmentation"""
        print(f"\n{'='*70}")
        print(f"🧠 MEMORY-AUGMENTED RAG QUERY")
        print(f"{'='*70}")
        print(f"❓ Question: {question}\n")
        
        overall_start = time.time_ns()
        
        # Step 1: Get memory context
        memory_context = ""
        if use_memory:
            print("📖 Retrieving from memory...")
            
            short_term = self.memory.get_relevant_short_term_context(question, k=3)
            long_term = self.memory.get_relevant_long_term_facts(question, k=5)
            
            if short_term or long_term:
                memory_context = f"CONVERSATION MEMORY:\n"
                if short_term:
                    memory_context += f"\nRecent Conversation:\n{short_term}\n"
                    print(f"   ✓ Short-term: {len(short_term)} chars")
                if long_term:
                    memory_context += f"\n{long_term}\n"
                    print(f"   ✓ Long-term: {len(self.memory.long_term_memory)} facts")
        
        # Step 2: Retrieve from Qdrant
        print("📚 Retrieving from documents...")
        start = time.time_ns()
        hits, ret_time = search_qdrant(self.qdrant, self.embedder, question, 4, self.collection_name)
        doc_context = "\n\n".join(hits)
        print(f"   ✓ Retrieved: {len(hits)} docs ({format_time_ns(ret_time)})")
        
        # Step 3: Generate answer
        print("💭 Generating answer...")
        
        full_context = ""
        if memory_context:
            full_context += memory_context + "\n\n"
        if doc_context:
            full_context += f"DOCUMENT CONTEXT:\n{doc_context}"
        
        if not full_context:
            full_context = "No relevant context available."
        
        prompt = f"""Use conversation history and documents to answer.

{full_context}

Question: {question}

Based on the context (including conversation history and facts), provide a comprehensive answer.

Answer:"""
        
        start = time.time_ns()
        response = self.llm.invoke(prompt)
        gen_time = time.time_ns() - start
        latency_report.add("llm_generate_answer", gen_time)
        
        answer = response.content if hasattr(response, 'content') else str(response)
        
        print(f"\n💬 ANSWER ({format_time_ns(gen_time)}):")
        print(answer[:500])
        if len(answer) > 500:
            print("...")
        
        # Step 4: Extract and store facts
        if use_memory:
            print("\n🧠 Extracting facts for long-term memory...")
            self.memory.extract_and_store_facts(self.llm, question, answer)
        
        # Step 5: Add to short-term memory
        if use_memory:
            self.memory.add_conversation_turn(question, answer, doc_context[:500])
        
        # Step 6: Get memory summary
        mem_summary = self.memory.get_memory_summary()
        
        total = time.time_ns() - overall_start
        latency_report.add("memory_rag_query_total", total)
        
        print(f"\n📊 Memory State:")
        print(f"   Short-term: {mem_summary['short_term_size']} turns")
        print(f"   Long-term: {mem_summary['long_term_facts']} facts")
        print(f"   Total turns: {mem_summary['total_turns']}")
        print(f"\n⏱️  Total query time: {format_time_ns(total)}")
        print(f"{'='*70}\n")
        
        return {
            'question': question,
            'answer': answer,
            'memory_used': use_memory,
            'memory_summary': mem_summary,
            'total_time': total
        }

# =========================================================
# MAIN
# =========================================================
def main():
    print("="*70)
    print("🧠 MEMORY-AUGMENTED RAG + QDRANT")
    print("="*70)
    print()
    
    pipeline_start = time.time_ns()
    
    # Phase 1: Data preparation
    print("📚 PHASE 1: DATA PREPARATION")
    print("-"*70)
    
    text, load_time = timed_call(load_pdf, PDF_PATH)
    latency_report.add("pipeline_pdf_load", load_time)
    
    chunks, chunk_time = timed_call(chunk_text, text, 1000, 100)
    latency_report.add("pipeline_chunking", chunk_time)
    
    embedder, embed_time = timed_call(load_embeddings)
    latency_report.add("pipeline_embeddings_load", embed_time)
    
    qdrant, qdrant_time = timed_call(init_qdrant, COLLECTION, DIM)
    latency_report.add("pipeline_qdrant_init", qdrant_time)
    
    insert_start = time.time_ns()
    insert_chunks(qdrant, embedder, chunks, COLLECTION)
    insert_time = time.time_ns() - insert_start
    latency_report.add("pipeline_insert_chunks", insert_time)
    
    # Phase 2: Initialize LLM
    print(f"\n📚 PHASE 2: LLM INITIALIZATION")
    print("-"*70)
    
    llm_start = time.time_ns()
    llm = ChatGroq(model_name=MODEL_NAME, groq_api_key=GROQ_API_KEY, temperature=0)
    llm_time = time.time_ns() - llm_start
    latency_report.add("llm_init", llm_time)
    print(f"✅ LLM initialized in {format_time_ns(llm_time)}")
    
    # Initialize Memory Manager
    memory = MemoryManager()
    
    # Initialize Memory-Augmented RAG
    mem_rag = MemoryAugmentedRAG(llm, qdrant, embedder, memory, COLLECTION)
    print(f"\n✅ Memory-Augmented RAG system initialized!")
    
    # Phase 3: Multi-turn conversation
    print(f"\n📚 PHASE 3: MEMORY-AUGMENTED RAG QUERIES")
    print("-"*70)
    
    conversation = [
        "What are the main themes in this story?",
        "Tell me more about the love theme you mentioned.",
        "What other themes did you discuss earlier?",
        "Summarize everything we talked about."
    ]
    
    results = []
    for i, q in enumerate(conversation, 1):
        print(f"\n{'─'*70}")
        print(f"CONVERSATION TURN {i}/{len(conversation)}")
        print(f"{'─'*70}")
        result = mem_rag.query(q, use_memory=True)
        results.append(result)
        time.sleep(0.5)
    
    # Phase 4: Sentiment benchmark
    print(f"\n📚 PHASE 4: VADER SENTIMENT BENCHMARK")
    print("-"*70)
    
    sa_start = time.time_ns()
    sa = VaderSentimentAnalyzer()
    sa_init = time.time_ns() - sa_start
    latency_report.add("vader_init", sa_init)
    print(f"✅ VADER INIT: {format_time_ns(sa_init)}\n")
    
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
    
    final_mem = memory.get_memory_summary()
    print(f"Total pipeline time: {format_time_ns(pipeline_total)}")
    print(f"Queries executed: {len(conversation)}")
    print(f"Average query time: {format_time_ns(sum(r['total_time'] for r in results) // len(results))}")
    print(f"\n🧠 FINAL MEMORY STATE:")
    print(f"   Short-term: {final_mem['short_term_size']}/{SHORT_TERM_MEMORY_SIZE} turns")
    print(f"   Long-term: {final_mem['long_term_facts']} facts")
    print(f"   Total turns: {final_mem['total_turns']}")
    
    if memory.long_term_memory:
        print(f"\n🧠 Learned Facts:")
        for i, (text, fact) in enumerate(list(memory.long_term_memory.items())[:5], 1):
            print(f"   {i}. {text[:70]}... ({fact.mentions}x)")
    
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