#!/usr/bin/env python3
"""
memory_augmented_rag.py - FIXED VERSION WITH NEW INDEX
Memory-Augmented RAG with conversation history and fact learning
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

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------
# CONFIG - UPDATED WITH NEW INDEX
# ---------------------------
PDF_PATH = "Data/ECHOES OF HER LOVE.pdf"
INDEX_NAME = "new2"  # CHANGED: Use existing index
DIM = 384
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.1-8b-instant"
TARGET_NS = 200_000
SHORT_TERM_MEMORY_SIZE = 10

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# ---------------------------
# UTILITIES
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
        s = {}
        for comp, vals in self.store.items():
            total = sum(vals)
            s[comp] = {
                "count": len(vals),
                "total": format_time_ns(total),
                "avg": format_time_ns(total // len(vals)),
                "min": format_time_ns(min(vals)),
                "max": format_time_ns(max(vals))
            }
        print("\n" + "="*70)
        print("LATENCY SUMMARY")
        print("="*70)
        for comp, stats in sorted(s.items()):
            print(f"\n{comp}:")
            for k, v in stats.items():
                print(f"  {k}: {v}")
        print("="*70 + "\n")

latency_report = LatencyReport()

# ---------------------------
# MEMORY STRUCTURES
# ---------------------------
@dataclass
class ConversationTurn:
    timestamp: str
    question: str
    answer: str
    turn_id: int
    
    def to_text(self):
        return f"[Turn {self.turn_id}]\nQ: {self.question}\nA: {self.answer[:200]}..."

@dataclass
class LongTermFact:
    fact: str
    category: str
    confidence: float
    mentions: int
    first_seen: str
    last_updated: str

class MemoryManager:
    def __init__(self, max_short_term=SHORT_TERM_MEMORY_SIZE):
        self.max_short_term = max_short_term
        self.short_term_memory = deque(maxlen=max_short_term)
        self.long_term_memory = {}
        self.turn_counter = 0
        
        print(f"🧠 Memory Manager initialized (capacity: {max_short_term} turns)")
    
    def add_turn(self, question: str, answer: str):
        start = time.time_ns()
        turn = ConversationTurn(
            timestamp=datetime.now().isoformat(),
            question=question,
            answer=answer,
            turn_id=self.turn_counter
        )
        self.short_term_memory.append(turn)
        self.turn_counter += 1
        elapsed = time.time_ns() - start
        latency_report.add("memory_add_turn", elapsed)
        print(f"💾 Stored in short-term memory (Turn #{turn.turn_id})")
    
    def extract_facts(self, llm, question: str, answer: str):
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
    
    def get_summary(self):
        return {
            'short_term': len(self.short_term_memory),
            'long_term': len(self.long_term_memory),
            'total_turns': self.turn_counter
        }

# ---------------------------
# PDF/EMBEDDINGS/PINECONE - UPDATED
# ---------------------------
def load_pdf(path):
    start = time.time_ns()
    with pdfplumber.open(path) as pdf:
        text = "\n".join([p.extract_text() or "" for p in pdf.pages])
    elapsed = time.time_ns() - start
    latency_report.add("pdf_load", elapsed)
    print(f"📄 Loaded PDF: {len(text)} chars ({format_time_ns(elapsed)})")
    return text

def chunk_text(text):
    start = time.time_ns()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)
    elapsed = time.time_ns() - start
    latency_report.add("chunking", elapsed)
    print(f"📄 Created {len(chunks)} chunks ({format_time_ns(elapsed)})")
    return chunks

def get_embeddings_model():
    start = time.time_ns()
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    elapsed = time.time_ns() - start
    latency_report.add("embedding_init", elapsed)
    print(f"🧠 Embeddings loaded ({format_time_ns(elapsed)})")
    return emb

def init_pinecone(index_name):
    """Connect to existing Pinecone index - NO CREATION"""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [idx.name for idx in pc.list_indexes()]
    
    if index_name not in existing:
        print(f"❌ ERROR: Index '{index_name}' does not exist!")
        print(f"Available indexes: {existing}")
        sys.exit(1)
    
    print(f"✅ Connected to existing index '{index_name}'")
    return pc

def create_vectorstore(embed, chunks, index_name):
    """Use existing index and add documents"""
    start = time.time_ns()
    
    # Connect to existing index
    vs = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embed
    )
    
    # Add documents to existing index
    print(f"📤 Uploading {len(chunks)} chunks to '{index_name}'...")
    vs.add_texts(
        texts=chunks,
        metadatas=[{"chunk_id": i, "source": "memory_rag"} for i in range(len(chunks))]
    )
    
    elapsed = time.time_ns() - start
    latency_report.add("vectorstore_create", elapsed)
    print(f"✅ Vector store ready: {len(chunks)} chunks ({format_time_ns(elapsed)})")
    return vs

# ---------------------------
# VADER SENTIMENT
# ---------------------------
class VaderSentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    def analyze(self, text):
        scores = self.analyzer.polarity_scores(text)
        compound = scores['compound']
        if compound >= 0.05:
            return {'label': 'POSITIVE', 'compound': compound}
        elif compound <= -0.05:
            return {'label': 'NEGATIVE', 'compound': compound}
        return {'label': 'NEUTRAL', 'compound': compound}

def run_sentiment_benchmark(run_num, sa, examples):
    print(f"\n{'='*70}")
    print(f"🔥 VADER RUN #{run_num}")
    print(f"{'='*70}")
    times = []
    for i, text in enumerate(examples, 1):
        start = time.time_ns()
        result = sa.analyze(text)
        elapsed = time.time_ns() - start
        times.append(elapsed)
        latency_report.add("vader_inference", elapsed)
        status = "✅" if elapsed < TARGET_NS else "❌"
        print(f"[{i}] {format_time_ns(elapsed):15s} {status} {result['label']:8s} \"{text}\"")
    
    avg = sum(times) // len(times)
    print(f"📊 Average: {format_time_ns(avg)}")
    return avg

# ---------------------------
# MEMORY-AUGMENTED RAG
# ---------------------------
class MemoryAugmentedRAG:
    def __init__(self, vectorstore, llm, memory):
        self.vectorstore = vectorstore
        self.llm = llm
        self.memory = memory
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    def query(self, question: str):
        print(f"\n{'='*70}")
        print(f"🧠 MEMORY-AUGMENTED RAG")
        print(f"{'='*70}")
        print(f"❓ {question}\n")
        
        overall_start = time.time_ns()
        
        # Get memory context
        print("📖 Retrieving memories...")
        short_term = self.memory.get_short_term_context(k=3)
        long_term = self.memory.get_long_term_facts(k=5)
        
        memory_context = ""
        if short_term:
            memory_context += f"Recent Conversation:\n{short_term}\n\n"
            print(f"   ✓ Short-term: {len(short_term)} chars")
        if long_term:
            memory_context += f"{long_term}\n\n"
            print(f"   ✓ Long-term: {len(self.memory.long_term_memory)} facts")
        
        # Retrieve documents
        print("📚 Retrieving documents...")
        start = time.time_ns()
        docs = self.retriever.invoke(question)
        ret_time = time.time_ns() - start
        latency_report.add("retrieval", ret_time)
        
        doc_context = "\n\n".join([
            getattr(d, "page_content", str(d)) for d in docs
        ])
        print(f"   ✓ Retrieved: {len(docs)} docs ({format_time_ns(ret_time)})")
        
        # Generate answer
        print("💭 Generating answer...")
        full_context = memory_context + f"Documents:\n{doc_context}"
        
        prompt = f"""Use conversation history and documents to answer.

{full_context}

Question: {question}

Answer:"""
        
        start = time.time_ns()
        response = self.llm.invoke(prompt)
        gen_time = time.time_ns() - start
        latency_report.add("generation", gen_time)
        
        answer = response.content if hasattr(response, 'content') else str(response)
        
        print(f"\n💬 ANSWER ({format_time_ns(gen_time)}):")
        print(f"{answer}\n")
        
        # Extract facts
        print("🔍 Extracting facts...")
        self.memory.extract_facts(self.llm, question, answer)
        
        # Store turn
        self.memory.add_turn(question, answer)
        
        total = time.time_ns() - overall_start
        latency_report.add("total_query", total)
        
        mem_sum = self.memory.get_summary()
        print(f"\n📊 Memory: {mem_sum['short_term']} turns, {mem_sum['long_term']} facts")
        print(f"⏱️  Total: {format_time_ns(total)}")
        
        return {
            'question': question,
            'answer': answer,
            'memory_summary': mem_sum,
            'total_time': total
        }

# ---------------------------
# MAIN
# ---------------------------
def main():
    print("="*70)
    print("🧠 MEMORY-AUGMENTED RAG PIPELINE")
    print("="*70 + "\n")
    
    # Setup
    text = load_pdf(PDF_PATH)
    chunks = chunk_text(text)
    embed = get_embeddings_model()
    pc = init_pinecone(INDEX_NAME)
    vs = create_vectorstore(embed, chunks, INDEX_NAME)
    
    print(f"\n✅ LLM initializing...")
    llm = ChatGroq(model_name=MODEL_NAME, temperature=0, groq_api_key=GROQ_API_KEY)
    
    memory = MemoryManager()
    rag = MemoryAugmentedRAG(vs, llm, memory)
    
    print("\n" + "="*70)
    print("PHASE 1: MEMORY-AUGMENTED RAG CONVERSATION")
    print("="*70)
    
    # Multi-turn conversation
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
        result = rag.query(q)
        results.append(result)
        time.sleep(0.5)
    
    # VADER
    print("\n\n" + "="*70)
    print("PHASE 2: VADER SENTIMENT BENCHMARK")
    print("="*70)
    
    sa = VaderSentimentAnalyzer()
    examples = [
        "I love this product!",
        "This is very bad service.",
        "It's okay, not too good, not too bad.",
        "Not great, really disappointed",
        "Amazing experience!"
    ]
    
    for run in range(1, 4):
        run_sentiment_benchmark(run, sa, examples)
        time.sleep(0.1)
    
    # Final report
    print("\n" + "="*70)
    print("FINAL MEMORY STATE")
    print("="*70)
    mem_sum = memory.get_summary()
    print(f"Short-term: {mem_sum['short_term']}/{SHORT_TERM_MEMORY_SIZE} turns")
    print(f"Long-term: {mem_sum['long_term']} facts")
    print(f"Total turns: {mem_sum['total_turns']}")
    
    if memory.long_term_memory:
        print(f"\n🧠 Learned Facts:")
        for i, (text, fact) in enumerate(list(memory.long_term_memory.items())[:5], 1):
            print(f"   {i}. {text[:60]}... ({fact.mentions}x)")
    
    latency_report.pretty_print()
    print("✅ PIPELINE COMPLETE\n")

if __name__ == "__main__":
    main()