#!/usr/bin/env python3
"""
weaviate_multihop_rag.py
Multi-Hop RAG (Iterative RAG) with Weaviate and comprehensive nanosecond latency instrumentation.

Features:
- Multi-hop iterative retrieval following reasoning chains
- Query decomposition into sub-questions
- Progressive knowledge building across hops
- Weaviate vector database integration
- Full pipeline timing with per-hop latency tracking
- Detailed performance metrics and reports
"""

import os
import time
import sys
import traceback
from collections import defaultdict
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
COLLECTION_NAME = "MultiHopRAG_Documents"
DIM = 384  # MiniLM embedding dimension
MODEL_NAME = "llama-3.1-8b-instant"
TARGET_NS = 200_000
MAX_HOPS = 3  # Maximum number of retrieval hops

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
# PDF LOAD WITH TIMING
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
# INIT WEAVIATE WITH TIMING
# =========================================================
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

# =========================================================
# INSERT CHUNKS WITH TIMING
# =========================================================
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

# =========================================================
# SEARCH WEAVIATE WITH TIMING
# =========================================================
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
# MULTI-HOP RAG
# =========================================================
class MultiHopRAG:
    """Multi-Hop RAG system with iterative retrieval and comprehensive timing"""
    
    def __init__(self, llm, client: weaviate.WeaviateClient, embedder: SentenceTransformer,
                 collection_name: str = COLLECTION_NAME, max_hops: int = MAX_HOPS):
        self.llm = llm
        self.client = client
        self.embedder = embedder
        self.collection_name = collection_name
        self.max_hops = max_hops
    
    def _llm_invoke_timed(self, prompt: str, label: str) -> Tuple[str, int]:
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
    
    def decompose_query(self, query: str) -> Tuple[List[str], int]:
        """Decompose complex query into sub-questions for multi-hop retrieval"""
        prompt = f"""You are a query analyzer. Break down the following complex question into 2-3 simpler sub-questions that need to be answered sequentially to address the main question.

Main Question: {query}

Rules:
1. Each sub-question should build on the previous one
2. Sub-questions should be specific and focused
3. Number each sub-question (1., 2., 3.)
4. Keep sub-questions concise

Sub-questions:"""
        
        print(f"🔍 Decomposing query into sub-questions...")
        response_text, elapsed = self._llm_invoke_timed(prompt, "llm_query_decomposition")
        
        # Parse sub-questions
        sub_questions = []
        lines = response_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            # Match patterns like "1.", "1)", "Q1:", etc.
            if line and (line[0].isdigit() or line.startswith('Q')):
                # Remove numbering prefix
                clean = line.split('.', 1)[-1].split(')', 1)[-1].split(':', 1)[-1].strip()
                if clean and len(clean) > 10:  # Valid question
                    sub_questions.append(clean)
        
        # Fallback: use original query if decomposition fails
        if not sub_questions:
            sub_questions = [query]
        
        print(f"   ✅ Generated {len(sub_questions)} sub-questions in {format_time_ns(elapsed)}")
        for i, sq in enumerate(sub_questions, 1):
            print(f"      {i}. {sq[:80]}{'...' if len(sq) > 80 else ''}")
        
        return sub_questions, elapsed
    
    def retrieve_hop(self, query: str, hop_num: int, k: int = 4) -> Tuple[str, int]:
        """Retrieve documents for a specific hop"""
        print(f"\n   🔍 HOP {hop_num}: Retrieving documents...")
        print(f"      Query: {query[:100]}{'...' if len(query) > 100 else ''}")
        
        hits, elapsed = search_weaviate(self.client, self.embedder, query, k, self.collection_name)
        context = "\n\n".join(hits)
        
        print(f"      ✅ Retrieved {len(hits)} documents ({len(context)} chars) in {format_time_ns(elapsed)}")
        latency_report.add(f"hop_{hop_num}_retrieval", elapsed)
        
        return context, elapsed
    
    def synthesize_hop(self, sub_question: str, context: str, accumulated_knowledge: str, 
                       hop_num: int) -> Tuple[str, int]:
        """Synthesize answer for current hop using context and accumulated knowledge"""
        if accumulated_knowledge:
            prompt = f"""You are synthesizing information across multiple retrieval steps.

Previous Knowledge:
{accumulated_knowledge}

Current Sub-Question: {sub_question}

New Retrieved Context:
{context}

Task: Answer the current sub-question using BOTH the previous knowledge and the new context. Build upon what you already know.

Answer:"""
        else:
            prompt = f"""Answer the following question based on the retrieved context:

Question: {sub_question}

Context:
{context}

Provide a focused answer based on the context above.

Answer:"""
        
        print(f"      💡 Synthesizing answer for HOP {hop_num}...")
        answer, elapsed = self._llm_invoke_timed(prompt, f"llm_hop_{hop_num}_synthesis")
        
        print(f"      ✅ Synthesized in {format_time_ns(elapsed)} ({len(answer)} chars)")
        latency_report.add(f"hop_{hop_num}_synthesis", elapsed)
        
        return answer, elapsed
    
    def needs_next_hop(self, current_answer: str, remaining_questions: List[str], 
                       hop_num: int) -> Tuple[bool, str, int]:
        """Decide if next hop is needed"""
        if hop_num >= self.max_hops or not remaining_questions:
            return False, "Max hops reached or no remaining questions", 0
        
        prompt = f"""You are evaluating if we need more information retrieval.

Current Answer Summary:
{current_answer[:500]}

Remaining Sub-Questions:
{chr(10).join(f'{i+1}. {q}' for i, q in enumerate(remaining_questions[:3]))}

Question: Do we have enough information to answer the remaining questions, or do we need another retrieval hop?

Answer with:
- "CONTINUE" if we need more retrieval
- "STOP" if we have enough information
- Provide a brief reason

Decision:"""
        
        print(f"\n   🤔 Evaluating need for HOP {hop_num + 1}...")
        decision_text, elapsed = self._llm_invoke_timed(prompt, f"llm_hop_{hop_num}_decision")
        
        needs_more = 'CONTINUE' in decision_text.upper() and hop_num < self.max_hops
        
        decision_label = "CONTINUE" if needs_more else "STOP"
        print(f"      Decision: {decision_label}")
        print(f"      Reasoning: {decision_text.strip()[:100]}...")
        
        return needs_more, decision_text, elapsed
    
    def generate_final_answer(self, original_query: str, accumulated_knowledge: str) -> Tuple[str, int]:
        """Generate final comprehensive answer"""
        prompt = f"""You are providing a final comprehensive answer based on multi-hop retrieval.

Original Question: {original_query}

Accumulated Knowledge from Multiple Retrieval Hops:
{accumulated_knowledge}

Task: Provide a complete, well-structured answer to the original question using all the information gathered across multiple retrieval steps. Synthesize the information coherently.

Final Answer:"""
        
        print(f"\n   🎯 Generating final comprehensive answer...")
        answer, elapsed = self._llm_invoke_timed(prompt, "llm_final_answer")
        
        print(f"      ✅ Final answer generated in {format_time_ns(elapsed)}")
        
        return answer, elapsed
    
    def query(self, question: str) -> Dict[str, Any]:
        """Process query with Multi-Hop RAG pipeline"""
        print(f"\n{'='*70}")
        print(f"🚀 MULTI-HOP RAG QUERY PROCESSING")
        print(f"{'='*70}")
        print(f"❓ Original Question: {question}\n")
        
        overall_start = time.time_ns()
        
        # Step 1: Query Decomposition
        sub_questions, decomp_time = self.decompose_query(question)
        
        # Step 2: Multi-Hop Retrieval and Synthesis
        accumulated_knowledge = ""
        hop_results = []
        per_hop_times = []
        
        for hop_num in range(1, min(len(sub_questions) + 1, self.max_hops + 1)):
            hop_start = time.time_ns()
            print(f"\n{'─'*70}")
            print(f"🔄 HOP {hop_num}/{min(len(sub_questions), self.max_hops)}")
            print(f"{'─'*70}")
            
            current_question = sub_questions[hop_num - 1]
            
            # Retrieve documents for current hop
            context, retrieval_time = self.retrieve_hop(current_question, hop_num)
            
            # Synthesize answer for current hop
            hop_answer, synthesis_time = self.synthesize_hop(
                current_question, context, accumulated_knowledge, hop_num
            )
            
            # Update accumulated knowledge
            accumulated_knowledge += f"\n\n[Hop {hop_num} - {current_question}]\n{hop_answer}"
            
            hop_elapsed = time.time_ns() - hop_start
            per_hop_times.append(hop_elapsed)
            latency_report.add("multihop_hop_total", hop_elapsed)
            
            hop_results.append({
                'hop_num': hop_num,
                'sub_question': current_question,
                'context': context,
                'answer': hop_answer,
                'time_ns': hop_elapsed
            })
            
            print(f"\n   ⏱️  HOP {hop_num} total time: {format_time_ns(hop_elapsed)}")
            
            # Check if we need next hop
            remaining = sub_questions[hop_num:]
            if remaining:
                needs_more, decision, decision_time = self.needs_next_hop(
                    hop_answer, remaining, hop_num
                )
                if not needs_more:
                    print(f"\n   ✅ Multi-hop retrieval complete after {hop_num} hops")
                    break
            else:
                print(f"\n   ✅ All sub-questions processed")
        
        # Step 3: Generate Final Answer
        final_answer, final_time = self.generate_final_answer(question, accumulated_knowledge)
        
        total_query_ns = time.time_ns() - overall_start
        latency_report.add("multihop_query_total", total_query_ns)
        
        print(f"\n{'='*70}")
        print(f"💬 FINAL ANSWER:")
        print(f"{'='*70}")
        print(final_answer[:800])
        if len(final_answer) > 800:
            print("...")
        print(f"\n⏱️  Total query time: {format_time_ns(total_query_ns)}")
        print(f"⏱️  Number of hops: {len(hop_results)}")
        print(f"⏱️  Average hop time: {format_time_ns(sum(per_hop_times) // len(per_hop_times))}")
        print(f"{'='*70}\n")
        
        return {
            'question': question,
            'sub_questions': sub_questions,
            'hop_results': hop_results,
            'accumulated_knowledge': accumulated_knowledge,
            'final_answer': final_answer,
            'num_hops': len(hop_results),
            'per_hop_times': per_hop_times,
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
                            target_ns: int = TARGET_NS, run_number: int = 1):
    """Run sentiment analysis benchmark"""
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
    print("🚀 MULTI-HOP RAG (ITERATIVE RAG) + FULL LATENCY INSTRUMENTATION")
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
    
    embedder, embed_time = timed_call(load_embeddings)
    latency_report.add("pipeline_embeddings_load", embed_time)
    
    weaviate_client, weaviate_time = timed_call(init_weaviate, WEAVIATE_URL, WEAVIATE_API_KEY, COLLECTION_NAME)
    latency_report.add("pipeline_weaviate_init", weaviate_time)
    
    insert_time_start = time.time_ns()
    insert_chunks(weaviate_client, embedder, chunks, COLLECTION_NAME)
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
    latency_report.add("llm_init", llm_time)
    print(f"✅ LLM initialized in {format_time_ns(llm_time)}")
    
    # Initialize Multi-Hop RAG
    multihop_rag = MultiHopRAG(llm, weaviate_client, embedder, COLLECTION_NAME, MAX_HOPS)
    print(f"\n✅ Multi-Hop RAG system initialized (max hops: {MAX_HOPS})!")
    
    # Phase 3: Run queries
    print(f"\n📚 PHASE 3: MULTI-HOP RAG QUERIES")
    print("-"*70)
    
    queries = [
        "What are the main themes in this story and how do they relate to the characters' development?",
        "Summarize the key events and explain their significance to the overall narrative.",
        "Who are the main characters and what are their relationships with each other?"
    ]
    
    results = []
    for q in queries:
        result = multihop_rag.query(q)
        results.append(result)
    
    # Phase 4: Sentiment benchmark
    print(f"\n📚 PHASE 4: VADER SENTIMENT BENCHMARK")
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
    
    for run in range(1, 3):
        run_sentiment_benchmark(sa, examples, TARGET_NS, run)
        time.sleep(0.1)
    
    # Final summary
    pipeline_total = time.time_ns() - pipeline_start
    latency_report.add("pipeline_total", pipeline_total)
    
    print(f"\n{'='*70}")
    print(f"📈 MULTI-HOP RAG PIPELINE SUMMARY")
    print(f"{'='*70}")
    print(f"Total pipeline time: {format_time_ns(pipeline_total)}")
    print(f"Queries executed: {len(queries)}")
    print(f"Total hops across all queries: {sum(r['num_hops'] for r in results)}")
    print(f"Average hops per query: {sum(r['num_hops'] for r in results) / len(results):.1f}")
    print(f"Average query time: {format_time_ns(sum(r['total_query_ns'] for r in results) // len(results))}")
    
    # Per-query breakdown
    print(f"\n📊 PER-QUERY BREAKDOWN:")
    for i, result in enumerate(results, 1):
        print(f"\n   Query {i}: {result['question'][:60]}...")
        print(f"      Hops: {result['num_hops']}")
        print(f"      Time: {format_time_ns(result['total_query_ns'])}")
        print(f"      Avg hop time: {format_time_ns(sum(result['per_hop_times']) // len(result['per_hop_times']))}")
    
    # Detailed latency report
    latency_report.pretty_print()
    
    # Cleanup
    weaviate_client.close()
    
    print("✅ MULTI-HOP RAG PIPELINE COMPLETE")

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