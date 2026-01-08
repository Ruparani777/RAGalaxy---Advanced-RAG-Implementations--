import os
import time
import sys
import traceback
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import json
import pdfplumber
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)

# ---------------------------
# CONFIG
# ---------------------------
PDF_PATH = "Data/ECHOES OF HER LOVE.pdf"
COLLECTION_NAME = "MemoryRAG_Documents"
MEMORY_COLLECTION_NAME = "MemoryRAG_Memory"
DIM = 384
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = "llama-3.1-8b-instant"
TARGET_NS = 200_000
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("❌ ERROR: Set GROQ_API_KEY environment variable!")
    sys.exit(1)

# ---------------------------
# LATENCY UTILITIES
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

# ---------------------------
# PDF Loader
# ---------------------------
@timer_ns
def load_pdf(path: str) -> str:
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

# ---------------------------
# Chunker
# ---------------------------
@timer_ns
def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
    print("✂️  Chunking text...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    print(f"✅ Created {len(chunks)} chunks")
    return chunks

# ---------------------------
# Embeddings loader
# ---------------------------
@timer_ns
def load_embeddings(model_name: str = EMBED_MODEL) -> SentenceTransformer:
    print(f"🔢 Loading embeddings model: {model_name}")
    embedder = SentenceTransformer(model_name)
    print("✅ Embeddings model loaded")
    return embedder

# ---------------------------
# Milvus init for Documents
# ---------------------------
@timer_ns
def init_milvus_documents(host: str, port: str, collection_name: str = COLLECTION_NAME, dim: int = DIM) -> Collection:
    print(f"🗃️  Initializing Milvus connection to {host}:{port}")
    connections.connect(host=host, port=port)
    
    try:
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            print(f"🗑️  Deleted existing collection '{collection_name}'")
    except Exception as e:
        print(f"⚠️  Collection check/delete: {e}")
    
    chunk_id_field = FieldSchema(name="chunk_id", dtype=DataType.INT64, is_primary=True, auto_id=True)
    text_field = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
    source_field = FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1024)
    embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    
    schema = CollectionSchema(fields=[chunk_id_field, text_field, source_field, embedding_field],
                              description="Memory-RAG document chunks")
    collection = Collection(name=collection_name, schema=schema)
    
    index_params = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 1024}}
    try:
        collection.create_index(field_name="embedding", index_params=index_params)
    except Exception as e:
        print(f"⚠️  create_index: {e}")
    
    try:
        collection.load()
    except Exception as e:
        print(f"⚠️  load: {e}")
    
    print(f"✅ Milvus collection '{collection_name}' ready")
    return collection

# ---------------------------
# Milvus init for Memory Store
# ---------------------------
@timer_ns
def init_milvus_memory(host: str, port: str, collection_name: str = MEMORY_COLLECTION_NAME, dim: int = DIM) -> Collection:
    print(f"🧠 Initializing Memory collection: {collection_name}")
    
    try:
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            print(f"🗑️  Deleted existing memory collection '{collection_name}'")
    except Exception as e:
        print(f"⚠️  Memory collection check/delete: {e}")
    
    memory_id_field = FieldSchema(name="memory_id", dtype=DataType.INT64, is_primary=True, auto_id=True)
    query_field = FieldSchema(name="query", dtype=DataType.VARCHAR, max_length=65535)
    answer_field = FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=65535)
    context_field = FieldSchema(name="context", dtype=DataType.VARCHAR, max_length=65535)
    feedback_field = FieldSchema(name="feedback", dtype=DataType.VARCHAR, max_length=1024)
    timestamp_field = FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=256)
    embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    
    schema = CollectionSchema(
        fields=[memory_id_field, query_field, answer_field, context_field, feedback_field, timestamp_field, embedding_field],
        description="Memory-RAG interaction memory"
    )
    collection = Collection(name=collection_name, schema=schema)
    
    index_params = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 512}}
    try:
        collection.create_index(field_name="embedding", index_params=index_params)
    except Exception as e:
        print(f"⚠️  memory create_index: {e}")
    
    try:
        collection.load()
    except Exception as e:
        print(f"⚠️  memory load: {e}")
    
    print(f"✅ Memory collection '{collection_name}' ready")
    return collection

# ---------------------------
# Insert chunks
# ---------------------------
@timer_ns
def insert_chunks(collection: Collection, embedder: SentenceTransformer, chunks: List[str]) -> None:
    print(f"⬆️  Inserting {len(chunks)} chunks into Milvus...")
    
    start = time.time_ns()
    vectors = embedder.encode(chunks, show_progress_bar=False)
    encode_time = time.time_ns() - start
    latency_report.add("embedding_encode_batch", encode_time)
    print(f"   ✅ Encoded in {format_time_ns(encode_time)}")
    
    texts = chunks
    sources = [f"chunk_{i}" for i in range(len(chunks))]
    embeddings = [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]
    
    start = time.time_ns()
    collection.insert([texts, sources, embeddings])
    insert_time = time.time_ns() - start
    latency_report.add("milvus_insert", insert_time)
    print(f"   ✅ Inserted {len(chunks)} vectors in {format_time_ns(insert_time)}")
    
    start = time.time_ns()
    collection.flush()
    flush_time = time.time_ns() - start
    latency_report.add("milvus_flush", flush_time)
    print(f"   ✅ Flushed collection ({format_time_ns(flush_time)})")

# ---------------------------
# Search Documents
# ---------------------------
def search_milvus(collection: Collection, embedder: SentenceTransformer, query: str, limit: int = 4) -> Tuple[List[str], int]:
    start = time.time_ns()
    qvec = embedder.encode([query])[0]
    encode_time = time.time_ns() - start
    latency_report.add("query_embedding", encode_time)
    
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    start = time.time_ns()
    try:
        results = collection.search(
            data=[qvec.tolist()], 
            anns_field="embedding", 
            param=search_params, 
            limit=limit,
            output_fields=["text", "source", "chunk_id"]
        )
        search_time = time.time_ns() - start
        latency_report.add("milvus_search", search_time)
        
        hits = []
        for hit in results[0]:
            try:
                ent = getattr(hit, "entity", None) or getattr(hit, "_fields", None) or {}
                if isinstance(ent, dict):
                    txt = ent.get("text", "")
                else:
                    txt = str(hit)
                hits.append(txt)
            except Exception:
                hits.append("")
    except Exception as e:
        search_time = time.time_ns() - start
        latency_report.add("milvus_search_error", search_time)
        print(f"⚠️ Milvus search failed: {e}")
        hits = []
    
    total_time = encode_time + (latency_report.store.get("milvus_search", [-1])[-1] if latency_report.store.get("milvus_search") else 0)
    return hits, total_time

# ---------------------------
# Memory-Augmented RAG System
# ---------------------------
class MemoryAugmentedRAG:
    def __init__(self, llm, doc_collection: Collection, memory_collection: Collection, embedder: SentenceTransformer):
        self.llm = llm
        self.doc_collection = doc_collection
        self.memory_collection = memory_collection
        self.embedder = embedder
        self.conversation_history = []
        
    def _llm_invoke_timed(self, prompt: str, label: str) -> Tuple[str, int]:
        start = time.time_ns()
        try:
            response = self.llm.invoke(prompt)
            elapsed = time.time_ns() - start
            latency_report.add(label, elapsed)
            content = response.content if hasattr(response, "content") else str(response)
            return content, elapsed
        except Exception as e:
            elapsed = time.time_ns() - start
            latency_report.add(label + "_error", elapsed)
            print(f"LLM invoke for {label} failed: {e}")
            traceback.print_exc()
            return str(e), elapsed
    
    def retrieve_from_memory(self, query: str, k: int = 3) -> Tuple[List[Dict], int]:
        """Retrieve similar past interactions from memory"""
        start = time.time_ns()
        qvec = self.embedder.encode([query])[0]
        encode_time = time.time_ns() - start
        latency_report.add("memory_query_embedding", encode_time)
        
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        start = time.time_ns()
        try:
            results = self.memory_collection.search(
                data=[qvec.tolist()],
                anns_field="embedding",
                param=search_params,
                limit=k,
                output_fields=["query", "answer", "context", "feedback", "timestamp"]
            )
            search_time = time.time_ns() - start
            latency_report.add("memory_search", search_time)
            
            memories = []
            for hit in results[0]:
                try:
                    ent = getattr(hit, "entity", None) or getattr(hit, "_fields", None) or {}
                    if isinstance(ent, dict):
                        memories.append({
                            "query": ent.get("query", ""),
                            "answer": ent.get("answer", ""),
                            "context": ent.get("context", ""),
                            "feedback": ent.get("feedback", ""),
                            "timestamp": ent.get("timestamp", ""),
                            "distance": getattr(hit, "distance", 0)
                        })
                except Exception:
                    pass
        except Exception as e:
            search_time = time.time_ns() - start
            latency_report.add("memory_search_error", search_time)
            print(f"⚠️ Memory search failed: {e}")
            memories = []
        
        total_time = encode_time + search_time
        return memories, total_time
    
    def retrieve_documents(self, query: str, k: int = 4) -> Tuple[str, int]:
        """Retrieve relevant documents"""
        hits, elapsed = search_milvus(self.doc_collection, self.embedder, query, k)
        context = "\n\n".join(hits)
        print(f"   ✅ Retrieved {len(hits)} documents in {format_time_ns(elapsed)}")
        return context, elapsed
    
    def generate_with_memory(self, query: str, doc_context: str, memories: List[Dict]) -> Tuple[str, int]:
        """Generate answer using both document context and memory"""
        memory_context = ""
        if memories:
            memory_context = "Previous similar interactions:\n"
            for i, mem in enumerate(memories[:2], 1):
                memory_context += f"\n{i}. Q: {mem['query'][:100]}...\n   A: {mem['answer'][:150]}...\n"
        
        prompt = f"""You are an intelligent assistant with memory of past interactions.

{memory_context}

Current Document Context:
{doc_context}

Current Question: {query}

Use the document context to answer the question. If similar questions were asked before (shown above), learn from those interactions to provide better answers. Provide a comprehensive and accurate answer.

Answer:"""
        
        answer, elapsed = self._llm_invoke_timed(prompt, "llm_generate_with_memory")
        print(f"   ✅ Answer generated with memory in {format_time_ns(elapsed)}")
        return answer, elapsed
    
    def store_interaction(self, query: str, answer: str, context: str, feedback: str = "neutral") -> int:
        """Store the interaction in memory for future reference"""
        start = time.time_ns()
        
        # Encode query for memory storage
        qvec = self.embedder.encode([query])[0]
        timestamp = datetime.now().isoformat()
        
        try:
            self.memory_collection.insert([
                [query],
                [answer],
                [context[:65000]],  # Truncate if needed
                [feedback],
                [timestamp],
                [qvec.tolist()]
            ])
            self.memory_collection.flush()
            elapsed = time.time_ns() - start
            latency_report.add("memory_store", elapsed)
            print(f"   💾 Interaction stored in memory ({format_time_ns(elapsed)})")
        except Exception as e:
            elapsed = time.time_ns() - start
            latency_report.add("memory_store_error", elapsed)
            print(f"⚠️ Failed to store memory: {e}")
        
        return elapsed
    
    def query(self, question: str, use_memory: bool = True) -> Dict[str, Any]:
        """Process query using Memory-Augmented RAG"""
        print("\n" + "="*70)
        print("🧠 MEMORY-AUGMENTED RAG QUERY PROCESSING")
        print("="*70)
        print(f"❓ Question: {question}\n")
        
        overall_start = time.time_ns()
        
        # Step 1: Retrieve from memory
        memories = []
        if use_memory:
            print("🔍 Step 1: Retrieving from memory...")
            memories, memory_time = self.retrieve_from_memory(question, k=3)
            if memories:
                print(f"   📝 Found {len(memories)} similar past interactions")
            else:
                print(f"   📝 No similar past interactions found")
        
        # Step 2: Retrieve relevant documents
        print("\n🔍 Step 2: Retrieving relevant documents...")
        doc_context, retrieval_time = self.retrieve_documents(question, k=4)
        print(f"   📝 Document context length: {len(doc_context)} characters")
        
        # Step 3: Generate answer with memory and context
        print("\n💡 Step 3: Generating answer with memory augmentation...")
        answer, gen_time = self.generate_with_memory(question, doc_context, memories)
        
        # Step 4: Store interaction in memory
        print("\n💾 Step 4: Storing interaction in memory...")
        store_time = self.store_interaction(query=question, answer=answer, context=doc_context)
        
        # Add to conversation history
        self.conversation_history.append({
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        })
        
        total_query_ns = time.time_ns() - overall_start
        latency_report.add("memory_rag_query_total", total_query_ns)
        
        print("\n" + "="*70)
        print("💬 FINAL ANSWER:")
        print("="*70)
        print(answer[:800])
        if len(answer) > 800:
            print("...")
        print(f"\n⏱️  Total query time: {format_time_ns(total_query_ns)}")
        print("="*70 + "\n")
        
        return {
            "question": question,
            "answer": answer,
            "doc_context": doc_context,
            "memories_used": len(memories),
            "total_query_ns": total_query_ns,
            "memory_count": len(self.conversation_history)
        }
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about stored memories"""
        try:
            stats = self.memory_collection.num_entities
            return {
                "total_memories": stats,
                "conversation_length": len(self.conversation_history)
            }
        except Exception as e:
            print(f"⚠️ Failed to get memory stats: {e}")
            return {"total_memories": 0, "conversation_length": len(self.conversation_history)}

# ---------------------------
# Vader Sentiment
# ---------------------------
class VaderSentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    
    def analyze(self, text: str) -> Dict[str, Any]:
        scores = self.analyzer.polarity_scores(text)
        compound = scores["compound"]
        
        if compound >= 0.05:
            label = "POSITIVE"
            percentage = round((compound + 1) * 50, 2)
        elif compound <= -0.05:
            label = "NEGATIVE"
            percentage = round((1 - abs(compound)) * 50, 2)
        else:
            label = "NEUTRAL"
            percentage = round(50 + (compound * 50), 2)
        
        return {"label": label, "percentage": percentage, "compound": compound, "scores": scores}

def run_sentiment_benchmark(sa: VaderSentimentAnalyzer, examples: List[str],
                            target_ns: int = TARGET_NS, run_number: int = 1):
    print("\n" + "="*70)
    print(f"🔥 SENTIMENT BENCHMARK RUN #{run_number}")
    print("="*70)
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
    
    print("\n📊 RUN #{run_number} STATISTICS:")
    print(f"   Total:        {format_time_ns(total_ns)}")
    print(f"   Average:      {format_time_ns(avg_ns)}")
    print(f"   Min:          {format_time_ns(min_ns)}")
    print(f"   Max:          {format_time_ns(max_ns)}")
    print(f"   < {target_ns}ns: {under_target}/{len(individual_times)} texts")
    
    if avg_ns < target_ns:
        print("   ✅ TARGET MET!")
    else:
        print("   ⚠️  TARGET MISSED")

# ---------------------------
# MAIN
# ---------------------------
def main():
    print("="*70)
    print("🧠 MEMORY-AUGMENTED RAG + FULL LATENCY INSTRUMENTATION")
    print("="*70 + "\n")
    
    pipeline_start = time.time_ns()
    
    # Phase 1: Load and prepare data
    print("📚 PHASE 1: DATA PREPARATION")
    print("-" * 70)
    
    text, load_time = timed_call(load_pdf, PDF_PATH)
    latency_report.add("pipeline_pdf_load", load_time)
    
    chunks, chunk_time = timed_call(chunk_text, text, 1000, 100)
    latency_report.add("pipeline_chunking", chunk_time)
    
    embedder, embed_time = timed_call(load_embeddings, EMBED_MODEL)
    latency_report.add("pipeline_embeddings_load", embed_time)
    
    doc_collection, doc_milvus_time = timed_call(init_milvus_documents, MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME, DIM)
    latency_report.add("pipeline_milvus_doc_init", doc_milvus_time)
    
    memory_collection, mem_milvus_time = timed_call(init_milvus_memory, MILVUS_HOST, MILVUS_PORT, MEMORY_COLLECTION_NAME, DIM)
    latency_report.add("pipeline_milvus_mem_init", mem_milvus_time)
    
    insert_time_start = time.time_ns()
    insert_chunks(doc_collection, embedder, chunks)
    insert_time = time.time_ns() - insert_time_start
    latency_report.add("pipeline_insert_chunks", insert_time)
    
    # Phase 2: Initialize LLM
    print("\n📚 PHASE 2: LLM INITIALIZATION")
    print("-" * 70)
    
    llm_start = time.time_ns()
    llm = ChatGroq(model_name=MODEL_NAME, groq_api_key=GROQ_API_KEY, temperature=0)
    llm_time = time.time_ns() - llm_start
    latency_report.add("llm_init", llm_time)
    print(f"✅ LLM initialized in {format_time_ns(llm_time)}")
    
    # Initialize Memory-Augmented RAG
    memory_rag = MemoryAugmentedRAG(llm, doc_collection, memory_collection, embedder)
    print("\n✅ Memory-Augmented RAG system initialized!")
    
    # Phase 3: Run queries with memory
    print("\n📚 PHASE 3: MEMORY-AUGMENTED RAG QUERIES")
    print("-" * 70)
    
    queries = [
        "What are the main themes in this story?",
        "Can you tell me more about the themes?",  # Similar query to test memory
        "Summarize the key events in the document.",
        "What happened in the story?",  # Similar query
        "Who are the main characters?",
    ]
    
    results = []
    for q in queries:
        result = memory_rag.query(q, use_memory=True)
        results.append(result)
        
        # Show memory stats
        stats = memory_rag.get_memory_stats()
        print(f"📊 Memory Stats: {stats['total_memories']} stored interactions")
        time.sleep(0.5)  # Small delay between queries
    
    # Phase 4: Sentiment benchmark
    print("\n📚 PHASE 4: VADER SENTIMENT BENCHMARK")
    print("-" * 70)
    
    sa = VaderSentimentAnalyzer()
    sa_init = 0
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
    
    pipeline_total = time.time_ns() - pipeline_start
    latency_report.add("pipeline_total", pipeline_total)
    
    print("\n" + "="*70)
    print("📈 PIPELINE SUMMARY")
    print("="*70)
    print(f"Total pipeline time: {format_time_ns(pipeline_total)}")
    if results:
        print(f"Queries executed: {len(queries)}")
        print(f"Average query time: {format_time_ns(sum(r['total_query_ns'] for r in results) // len(results))}")
        print(f"Total memories stored: {results[-1]['memory_count']}")
    
    latency_report.pretty_print()
    
    try:
        connections.disconnect()
    except Exception:
        pass
    
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