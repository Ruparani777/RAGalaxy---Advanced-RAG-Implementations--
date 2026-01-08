import time
import os
from collections import defaultdict

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# ---------------------------
# CONFIG
# ---------------------------
PDF_PATH = "Data/bank_loan_recoverydata.pdf"
INDEX_NAME = "pinecone-vanilla"
DIM = 384
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_HOST = "https://new-pyn5anr.svc.aped-4627-b74a.pinecone.io"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.1-8b-instant"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TARGET_NS = 200_000

# ---------------------------
# UTILITIES
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
# PDF / Chunking / Embeddings
# ---------------------------
@timer_ns
def load_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        page_texts = []
        for i, p in enumerate(pdf.pages):
            t = p.extract_text() or ""
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
def get_embeddings_model(model_name=EMBEDDING_MODEL_NAME):
    emb = HuggingFaceEmbeddings(model_name=model_name)
    print("✅ Embedding model initialized")
    return emb

def init_pinecone(api_key, host, index_name=INDEX_NAME, dim=DIM):
    pc = Pinecone(api_key=api_key, environment=host)
    # Skip creating index since it exists
    print(f"✅ Connected to Pinecone index '{index_name}' at {host}")
    return pc

@timer_ns
def create_vectorstore(embed_model, chunks, index_name=INDEX_NAME):
    vectorstore = PineconeVectorStore.from_texts(
        texts=chunks,
        embedding=embed_model,
        index_name=index_name,
        namespace="",
        metadatas=[{"source": f"chunk_{i}", "chunk_id": i} for i in range(len(chunks))]
    )
    print(f"✅ Created vector store with {len(chunks)} chunks")
    return vectorstore

# ---------------------------
# Vanilla RAG Query
# ---------------------------
class VanillaRAG:
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
        self.retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    
    def query(self, question: str) -> str:
        print(f"\n❓ Question: {question}")
        docs = self.retriever.invoke(question)
        context = "\n\n".join([getattr(doc, "page_content", "") for doc in docs])
        prompt = f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {question}\nAnswer:"
        response = self.llm.invoke(prompt)
        answer = response.content if hasattr(response, 'content') else str(response)
        print(f"✅ Answer: {answer[:300]}...")
        return answer

# ---------------------------
# Main Pipeline
# ---------------------------
def main():
    start_total = time.time_ns()

    # Phase 1: Load PDF, chunk, embeddings
    pdf_text, t_pdf = timed_call(load_pdf, PDF_PATH)
    latency_report.add("pdf_load", t_pdf)
    
    chunks, t_chunks = timed_call(chunk_text, pdf_text)
    latency_report.add("chunking", t_chunks)
    
    embed_model, t_emb = timed_call(get_embeddings_model)
    latency_report.add("embedding_model_init", t_emb)

    pc = init_pinecone(PINECONE_API_KEY, PINECONE_HOST, INDEX_NAME)
    vectorstore = create_vectorstore(embed_model, chunks, INDEX_NAME)

    # Phase 2: Initialize LLM
    llm_start = time.time_ns()
    llm = ChatGroq(model_name=MODEL_NAME, temperature=0, groq_api_key=GROQ_API_KEY)
    latency_report.add("llm_init", time.time_ns() - llm_start)
    print(f"✅ LLM initialized")

    # Phase 3: Vanilla RAG Queries
    rag = VanillaRAG(vectorstore, llm)
    queries = [
        "What are the main themes in this story?",
        "Summarize the key events in the document.",
        "Who are the main characters?"
    ]
    for q in queries:
        rag.query(q)

    # Phase 4: Final Latency Report
    overall_ns = time.time_ns() - start_total
    latency_report.add("pipeline_total", overall_ns)
    latency_report.pretty_print()
    print(f"\n✅ Vanilla RAG PIPELINE COMPLETE in {format_time_ns(overall_ns)}")

if __name__ == "__main__":
    main()
