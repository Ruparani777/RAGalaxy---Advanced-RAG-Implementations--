#!/usr/bin/env python3
"""
weaviate_memory_augmented_rag.py
Memory-Augmented RAG with Weaviate and comprehensive latency instrumentation.

Memory Features:
- Short-term Memory: Recent conversation turns (last 10 interactions)
- Long-term Memory: Persistent facts with mention tracking
- Working Memory: Current context buffer
- Conversation continuity across queries
"""

import os
import time
import sys
import json
import re
import traceback
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Any, Tuple

import pdfplumber
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import MetadataQuery
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# =========================================================
# CONFIG
# =========================================================
PDF_PATH = "Data/ECHOES OF HER LOVE.pdf"
COLLECTION_NAME = "MemoryRAG_Documents"
DIM = 384
MODEL_NAME = "llama-3.1-8b-instant"
TARGET_NS = 200_000
SHORT_TERM_MEMORY_SIZE = 10

# Weaviate credentials
WEAVIATE_URL = "21ookhjbswyl5urlawqmxw.c0.asia-southeast1.gcp.weaviate.cloud"
WEAVIATE_API_KEY = "NTVWQ1dZVDI1bkptcndrZF9JRTFySVg3TEFBc1R5V0luUEtHaU9MajB6am5VQkc3aG5yVkgwWkFQVDc0PV92MjAw"

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
            context_used=context[:500],
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
        
        prompt = f"""Extract key facts from this conversation. Return JSON only.

Question: {question}
Answer: {answer}

Extract facts, preferences, or entities mentioned.

{{
  "facts": [
    {{"text": "fact", "category": "preference|entity|summary", "confidence": 0.8}}
  ]
}}

If no important facts, return {{"facts": []}}

JSON:"""
        
        try:
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                facts = data.get('facts', [])
                
                for f in facts:
                    fact_text = f['text']
                    if fact_text in self.long_term_memory:
                        self.long_term_memory[fact_text].mentions += 1
                        self.long_term_memory[fact_text].last_updated = datetime.now().isoformat()
                    else:
                        self.long_term_memory[fact_text] = LongTermFact(
                            fact=fact_text,
                            category=f.get('category', 'summary'),
                            confidence=f.get('confidence', 0.7),
                            mentions=1,
                            first_seen=datetime.now().isoformat(),
                            last_updated=datetime.now().isoformat()
                        )
                
                if facts:
                    print(f"🧠 Extracted {len(facts)} facts → Long-term memory")
        
        except Exception as e:
            print(f"⚠️  Fact extraction error: {e}")
        
        elapsed = time.time_ns() - start
        latency_report.add("memory_extract_facts", elapsed)
    
    def get_short_term_context(self, k=3):
        if not self.short_term_memory:
            return ""
        recent = list(self.short_term_memory)[-k:]
        return "\n\n".join([turn.to_text() for turn in recent])
    
    def get_long_term_facts(self, k=5):
        if not self.long_term_memory:
            return ""
        sorted_facts = sorted(
            self.long_term_memory.values(),
            key=lambda f: f.mentions * f.confidence,
            reverse=True
        )[:k]
        
        return "Learned Facts:\n" + "\n".join([
            f"• {f.fact} ({f.category}, {f.mentions}x)"
            for f in sorted_facts
        ])
    
    def get_memory_summary(self):
        return {
            'short_term': len(self.short_term_memory),
            'long_term': len(self.long_term_memory),
            'total_turns': self.turn_counter
        }

# =========================================================
# PDF/CHUNKING/EMBEDDINGS/WEAVIATE
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
def init_weaviate(url: str, api_key: str, collection_name: str = COLLECTION_NAME) -> weaviate.WeaviateClient:
    print(f"🗃️  Initializing Weaviate")
    
    start = time.time_ns()
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=url,
        auth_credentials=Auth.api_key(api_key)
    )
    connect_time = time.time_ns() - start
    latency_report.add("weaviate_connect", connect_time)
    print(f"✅ Connected to Weaviate ({format_time_ns(connect_time)})")
    
    try:
        if client.collections.exists(collection_name):
            client.collections.delete(collection_name)
            print(f"🗑️  Deleted existing collection")
    except Exception as e:
        print(f"⚠️  Collection check: {e}")
    
    start = time.time_ns()
    try:
        client.collections.create(
            name=collection_name,
            vectorizer_config=None,
            properties=[
                {"name": "text", "dataType": ["text"]},
                {"name": "chunk_id", "dataType": ["int"]},
                {"name": "source", "dataType": ["text"]}
            ]
        )
        create_time = time.time_ns() - start
        latency_report.add("weaviate_create_collection", create_time)
        print(f"✅ Collection created ({format_time_ns(create_time)})")
    except Exception as e:
        print(f"⚠️  Collection creation: {e}")
    
    return client

@timer_ns
def insert_chunks(client: weaviate.WeaviateClient, embedder: SentenceTransformer,
                  chunks: List[str], collection_name: str = COLLECTION_NAME) -> None:
    print(f"⬆️  Inserting {len(chunks)} chunks...")
    
    start = time.time_ns()
    vectors = embedder.encode(chunks, show_progress_bar=False)
    encode_time = time.time_ns() - start
    latency_report.add("embedding_encode_batch", encode_time)
    print(f"   ✅ Encoded ({format_time_ns(encode_time)})")
    
    collection = client.collections.get(collection_name)
    
    start = time.time_ns()
    with collection.batch.dynamic() as batch:
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            batch.add_object(
                properties={"text": chunk, "chunk_id": i, "source": f"chunk_{i}"},
                vector=vector.tolist()
            )
    upsert_time = time.time_ns() - start
    latency_report.add("weaviate_upsert", upsert_time)
    print(f"   ✅ Upserted ({format_time_ns(upsert_time)})")

def search_weaviate(client: weaviate.WeaviateClient, embedder: SentenceTransformer,
                    query: str, limit: int = 4, collection_name: str = COLLECTION_NAME) -> Tuple[List[str], int]:
    start = time.time_ns()
    qvec = embedder.encode([query])[0]
    encode_time = time.time_ns() - start
    latency_report.add("query_embedding", encode_time)
    
    start = time.time_ns()
    collection = client.collections.get(collection_name)
    response = collection.query.near_vector(
        near_vector=qvec.tolist(),
        limit=limit,
        return_metadata=MetadataQuery(distance=True)
    )
    search_time = time.time_ns() - start
    latency_report.add("weaviate_search", search_time)
    
    hits = [obj.properties.get("text", "") for obj in response.objects]
    return hits, encode_time + search_time

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
                            target_ns: int = TARGET_NS, run_number: int = 1):
    print(f"\n{'='*70}")
    print(f"🔥 SENTIMENT BENCHMARK RUN #{run_number}")
    print(f"{'='*70}")
    print(f"🎯 TARGET: < {target_ns} ns\n")
    
    times = []
    for i, text in enumerate(examples, 1):
        start = time.time_ns()
        result = sa.analyze(text)
        elapsed = time.time_ns() - start
        latency_report.add("vader_per_example", elapsed)
        times.append(elapsed)
        
        status = "✅" if elapsed < target_ns else "❌"
        print(f"[{i:2d}] {format_time_ns(elapsed):20s} {status} | {result['label']:8s} | \"{text}\"")
    
    avg = sum(times) // len(times)
    print(f"\n📊 Average: {format_time_ns(avg)}")

# =========================================================
# MEMORY-AUGMENTED RAG
# =========================================================
class MemoryAugmentedRAG:
    """RAG with integrated memory system"""
    
    def __init__(self, llm, client: weaviate.WeaviateClient, embedder: SentenceTransformer,
                 memory: MemoryManager, collection_name: str = COLLECTION_NAME):
        self.llm = llm
        self.client = client
        self.embedder = embedder
        self.memory = memory
        self.collection_name = collection_name
    
    def query(self, question: str, use_memory: bool = True) -> Dict[str, Any]:
        """Query with memory augmentation"""
        print(f"\n{'='*70}")
        print(f"🧠 MEMORY-AUGMENTED RAG")
        print(f"{'='*70}")
        print(f"❓ {question}\n")
        
        overall_start = time.time_ns()
        
        # Step 1: Get memory context
        memory_context = ""
        if use_memory:
            print("📖 Retrieving from memory...")
            
            short_term = self.memory.get_short_term_context(k=3)
            long_term = self.memory.get_long_term_facts(k=5)
            
            if short_term or long_term:
                memory_context = f"CONVERSATION MEMORY:\n"
                if short_term:
                    memory_context += f"\nRecent Conversation:\n{short_term}\n"
                    print(f"   ✓ Short-term: {len(short_term)} chars")
                if long_term:
                    memory_context += f"\n{long_term}\n"
                    print(f"   ✓ Long-term: {len(self.memory.long_term_memory)} facts")
        
        # Step 2: Retrieve from Weaviate
        print("📚 Retrieving from documents...")
        hits, ret_time = search_weaviate(self.client, self.embedder, question, 4, self.collection_name)
        doc_context = "\n\n".join(hits)
        print(f"   ✓ Retrieved: {len(hits)} docs ({format_time_ns(ret_time)})")
        
        # Step 3: Generate answer
        print("💭 Generating answer...")
        
        full_context = ""
        if memory_context:
            full_context += memory_context + "\n\n"
        if doc_context:
            full_context += f"DOCUMENT CONTEXT:\n{doc_context}"
        
        prompt = f"""Use conversation history and documents to answer.

{full_context}

Question: {question}

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
            print("\n🧠 Extracting facts...")
            self.memory.extract_and_store_facts(self.llm, question, answer)
        
        # Step 5: Add to short-term memory
        if use_memory:
            self.memory.add_conversation_turn(question, answer, doc_context[:500])
        
        mem_summary = self.memory.get_memory_summary()
        
        total = time.time_ns() - overall_start
        latency_report.add("memory_rag_total", total)
        
        print(f"\n📊 Memory: {mem_summary['short_term']} turns, {mem_summary['long_term']} facts")
        print(f"⏱️  Total: {format_time_ns(total)}")
        print(f"{'='*70}\n")
        
        return {
            'question': question,
            'answer': answer,
            'memory_summary': mem_summary,
            'total_time': total
        }

# =========================================================
# MAIN
# =========================================================
def main():
    print("="*70)
    print("🧠 MEMORY-AUGMENTED RAG + WEAVIATE")
    print("="*70 + "\n")
    
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
    
    weaviate_client, weaviate_time = timed_call(init_weaviate, WEAVIATE_URL, WEAVIATE_API_KEY, COLLECTION_NAME)
    latency_report.add("pipeline_weaviate_init", weaviate_time)
    
    insert_start = time.time_ns()
    insert_chunks(weaviate_client, embedder, chunks, COLLECTION_NAME)
    insert_time = time.time_ns() - insert_start
    latency_report.add("pipeline_insert_chunks", insert_time)
    
    # Phase 2: Initialize LLM
    print(f"\n📚 PHASE 2: LLM INITIALIZATION")
    print("-"*70)
    
    llm_start = time.time_ns()
    llm = ChatGroq(model_name=MODEL_NAME, groq_api_key=GROQ_API_KEY, temperature=0)
    llm_time = time.time_ns() - llm_start
    latency_report.add("llm_init", llm_time)
    print(f"✅ LLM initialized ({format_time_ns(llm_time)})")
    
    # Initialize Memory Manager
    memory = MemoryManager()
    
    # Initialize Memory-Augmented RAG
    mem_rag = MemoryAugmentedRAG(llm, weaviate_client, embedder, memory, COLLECTION_NAME)
    print(f"\n✅ Memory-Augmented RAG initialized!")
    
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
        print(f"TURN {i}/{len(conversation)}")
        print(f"{'─'*70}")
        result = mem_rag.query(q, use_memory=True)
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
    
    run_sentiment_benchmark(sa, examples, TARGET_NS, 1)
    
    # Final summary
    pipeline_total = time.time_ns() - pipeline_start
    latency_report.add("pipeline_total", pipeline_total)
    
    print(f"\n{'='*70}")
    print(f"📈 PIPELINE SUMMARY")
    print(f"{'='*70}")
    
    final_mem = memory.get_memory_summary()
    print(f"Total pipeline time: {format_time_ns(pipeline_total)}")
    print(f"Queries executed: {len(conversation)}")
    print(f"\n🧠 FINAL MEMORY STATE:")
    print(f"   Short-term: {final_mem['short_term']}/{SHORT_TERM_MEMORY_SIZE} turns")
    print(f"   Long-term: {final_mem['long_term']} facts")
    print(f"   Total turns: {final_mem['total_turns']}")
    
    if memory.long_term_memory:
        print(f"\n🧠 Learned Facts:")
        for i, (text, fact) in enumerate(list(memory.long_term_memory.items())[:5], 1):
            print(f"   {i}. {text[:70]}... ({fact.mentions}x)")
    
    latency_report.pretty_print()
    
    # Cleanup
    weaviate_client.close()
    
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