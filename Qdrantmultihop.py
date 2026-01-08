#!/usr/bin/env python3
"""
qdrant_multihop_rag_fast.py

Optimized rewrite of your Multi-Hop RAG pipeline with:
- optional ultra-fast in-memory NumPy search ("mem" mode)
- controlled Qdrant fallback ("qdrant" mode)
- caching of embeddings to .npy to avoid recompute
- reduced logging in hot paths
- batched encoding and reduced Python overhead
- preserved multi-hop/coT flow and latency instrumentation

Notes:
- This script still cannot make LLM or embedding calls complete in 3,000 ns.
- For true microsecond latency you'd need tiny on-device models and compiled code (C/C++, Rust) running on specialized hardware.
"""

import os
import sys
import time
import json
from collections import defaultdict
from typing import List, Tuple, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer

# Optional imports for Qdrant compatibility
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import VectorParams, Distance, PointStruct
    QDRANT_AVAILABLE = True
except Exception:
    QDRANT_AVAILABLE = False

# Minimal LLM shim (keeps your llm.invoke(prompt).content interface)
# Replace ChatGroq with your real client in production.
class DummyLLM:
    def __init__(self, latency_ms: float = 20.0):
        self.latency_ms = latency_ms

    def invoke(self, prompt: str):
        # emulate response object with .content
        time.sleep(self.latency_ms / 1000.0)
        class R:
            def __init__(self, c): self.content = c
        # simple heuristic response — replace with real LLM client
        return R("COG: quick thought\nDECISION: RETRIEVE" if "Decide" in prompt else
                 "COG: short\nANSWER: dummy answer\nHOPS: []")

# ---------------------------
# Latency utilities
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
    def add(self, k: str, ns: int):
        self.store[k].append(ns)
    def summary(self):
        out = {}
        for k, vals in self.store.items():
            s = sum(vals)
            out[k] = {"count": len(vals), "total_ns": s,
                      "avg_ns": s // len(vals) if vals else 0,
                      "min_ns": min(vals) if vals else 0,
                      "max_ns": max(vals) if vals else 0}
        return out
    def pretty_print(self):
        print("\n" + "="*60)
        print("LATENCY SUMMARY")
        for k, v in sorted(self.summary().items()):
            print(f"{k:25s} count={v['count']:3d} avg={format_time_ns(v['avg_ns'])} total={format_time_ns(v['total_ns'])}")
        print("="*60 + "\n")

latency = LatencyReport()

# ---------------------------
# Fast text chunker (lightweight)
# ---------------------------
def chunk_text_fast(text: str, chunk_size: int=1000, overlap: int=100) -> List[str]:
    # Very cheap splitter: slide window on whitespace boundaries
    if not text:
        return []
    words = text.split()
    chunks = []
    i = 0
    n = len(words)
    while i < n:
        j = i + chunk_size
        chunk = " ".join(words[i: min(j, n)])
        chunks.append(chunk)
        i = j - overlap
        if i <= 0:
            i = j
    return chunks

# ---------------------------
# Embedding utils with caching
# ---------------------------
def load_or_compute_embeddings(embedder: SentenceTransformer, chunks: List[str], cache_path: str):
    """
    Compute embeddings once and cache to .npy. Returns numpy array shape (N, D).
    """
    if os.path.exists(cache_path):
        start = time.time_ns()
        mat = np.load(cache_path)
        latency.add("emb_cache_load", time.time_ns() - start)
        return mat
    # batch encode for memory efficiency
    start = time.time_ns()
    vectors = embedder.encode(chunks, show_progress_bar=False, batch_size=128)
    encode_ns = time.time_ns() - start
    latency.add("embed_encode_full", encode_ns)
    mat = np.asarray(vectors, dtype=np.float32)
    # persist
    np.save(cache_path, mat)
    latency.add("emb_cache_save", 0)
    return mat

# ---------------------------
# In-memory search (very fast)
# ---------------------------
class MemANN:
    def __init__(self, vectors: np.ndarray):
        # normalize once for cosine similarity
        self.v = vectors.astype(np.float32)
        norms = np.linalg.norm(self.v, axis=1, keepdims=True)
        # avoid division by zero
        norms[norms==0] = 1.0
        self.vn = self.v / norms

    def query(self, qvec: np.ndarray, top_k: int = 4) -> Tuple[List[int], List[float], int]:
        # qvec shape (D,)
        t0 = time.time_ns()
        qn = qvec.astype(np.float32)
        qn = qn / (np.linalg.norm(qn) + 1e-12)
        # compute dot product fast
        sims = self.vn.dot(qn)
        if top_k >= sims.shape[0]:
            idx = np.argsort(-sims)
        else:
            # partial selection is faster for large N
            idx = np.argpartition(-sims, top_k)[:top_k]
            idx = idx[np.argsort(-sims[idx])]
        top_scores = sims[idx].tolist()
        elapsed = time.time_ns() - t0
        latency.add("memann_query", elapsed)
        return idx.tolist(), top_scores, elapsed

# ---------------------------
# Qdrant helper (kept minimal)
# ---------------------------
def init_qdrant_minimal(collection: str, dim: int, in_memory: bool = True) -> QdrantClient:
    if not QDRANT_AVAILABLE:
        raise RuntimeError("Qdrant client not installed")
    q = QdrantClient(":memory:") if in_memory else QdrantClient()
    if q.collection_exists(collection):
        q.delete_collection(collection)
    q.create_collection(collection_name=collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
    return q

# ---------------------------
# Search wrapper supporting mem/qdrant
# ---------------------------
class FastSearcher:
    def __init__(self, mode: str, vectors: np.ndarray = None, qdrant_client: QdrantClient = None, chunks: List[str] = None):
        self.mode = mode
        self.vectors = vectors  # numpy matrix
        self.qdrant = qdrant_client
        self.chunks = chunks or []
        if mode == "mem":
            assert vectors is not None
            self.ann = MemANN(vectors)
        elif mode == "qdrant":
            assert qdrant_client is not None

    def search(self, embedder: SentenceTransformer, query: str, top_k: int = 4) -> Tuple[List[str], int]:
        # embed query (inline)
        t0 = time.time_ns()
        qvec = embedder.encode([query], show_progress_bar=False)[0]
        embed_ns = time.time_ns() - t0
        latency.add("query_encode", embed_ns)
        if self.mode == "mem":
            idxs, scores, q_ns = self.ann.query(np.asarray(qvec, dtype=np.float32), top_k=top_k)
            hits = [self.chunks[i] for i in idxs]
            return hits, q_ns + embed_ns
        else:
            # minimal qdrant query using client.query_points
            t1 = time.time_ns()
            resp = self.qdrant.query_points(collection_name="rag_collection", query=np.asarray(qvec).tolist(), limit=top_k)
            q_ns = time.time_ns() - t1
            hits = [p.payload.get("text", "") for p in resp.points]
            latency.add("qdrant_query", q_ns)
            return hits, embed_ns + q_ns

# ---------------------------
# MultiHopRAG (unchanged structure, micro-optimized)
# ---------------------------
class MultiHopRAG:
    def __init__(self, llm, searcher: FastSearcher, embedder: SentenceTransformer):
        self.llm = llm
        self.searcher = searcher
        self.embedder = embedder

    def decide_and_chain(self, question: str) -> Tuple[bool, str, int]:
        prompt = f"You are an assistant that reasons step-by-step (chain-of-thought).\nDecide whether to retrieve document context to answer the question. Show brief chain-of-thought and then the decision.\nQuestion: {question}\nProvide:\nCOG:\nDECISION:\n"
        t0 = time.time_ns()
        r = self.llm.invoke(prompt)
        t_ns = time.time_ns() - t0
        latency.add("llm_decide", t_ns)
        text = r.content if hasattr(r, "content") else str(r)
        return ('RETRIEVE' in text.upper()), text, t_ns

    def generate_and_hop(self, question: str, context: str = "") -> Tuple[str, List[str], int]:
        if context:
            prompt = f"Context:\n{context}\nQuestion: {question}\nProvide COG, ANSWER, HOPS as JSON array."
        else:
            prompt = f"Question: {question}\nProvide COG, ANSWER, HOPS as JSON array."
        t0 = time.time_ns()
        r = self.llm.invoke(prompt)
        t_ns = time.time_ns() - t0
        latency.add("llm_generate", t_ns)
        text = r.content if hasattr(r, "content") else str(r)

        # Lightweight parse: attempt JSON extraction for HOPS
        cog = ""
        answer = ""
        hops = []
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        for ln in lines:
            if ln.upper().startswith("COG:"):
                cog = ln.split(":",1)[1].strip()
            elif ln.upper().startswith("ANSWER:"):
                answer = ln.split(":",1)[1].strip()
            elif ln.upper().startswith("HOPS:"):
                # try JSON parse
                try:
                    hops = json.loads(ln.split(":",1)[1].strip())
                except Exception:
                    # crude split fallback
                    part = ln.split(":",1)[1].strip()
                    part = part.strip("[]")
                    if part:
                        hops = [p.strip().strip('"').strip("'") for p in part.split(",") if p.strip()]
        return answer, hops, t_ns

    def self_critique(self, question: str, answer: str, context: str = "") -> Tuple[str, bool, int]:
        prompt = f"Given question and answer indicate NEED_MORE or ENOUGH. Question: {question}\nAnswer: {answer}\nContext: {'provided' if context else 'none'}"
        t0 = time.time_ns()
        r = self.llm.invoke(prompt)
        t_ns = time.time_ns() - t0
        latency.add("llm_critique", t_ns)
        text = r.content if hasattr(r, "content") else str(r)
        return text, ('NEED_MORE' in text.upper()), t_ns

    def query(self, question: str, max_hops: int = 3, top_k: int = 4) -> Dict[str, Any]:
        t0 = time.time_ns()
        needs_retrieval, dec_text, dec_ns = self.decide_and_chain(question)

        combined_context = []
        retrieved_snippets = []

        if needs_retrieval:
            hits, rt_ns = self.searcher.search(self.embedder, question, top_k=top_k)
            combined_context.append("\n\n".join(hits))
            retrieved_snippets.extend(hits)

        answer, hops, gen_ns = self.generate_and_hop(question, "\n\n".join(combined_context))
        hops_done = 0
        per_hop = []

        while hops and hops_done < max_hops:
            hop_start = time.time_ns()
            next_hops = []
            for hq in hops:
                hits, rt_ns = self.searcher.search(self.embedder, hq, top_k=top_k)
                if hits:
                    combined_context.append("\n\n".join(hits))
                    retrieved_snippets.extend(hits)
            merged_ctx = "\n\n".join(combined_context)
            answer2, next_hops, synth_ns = self.generate_and_hop(f"Refine: {question}", merged_ctx)
            if answer2.strip():
                answer = answer2
            hop_elapsed = time.time_ns() - hop_start
            per_hop.append(hop_elapsed)
            latency.add("multihop_hop_total", hop_elapsed)
            hops = next_hops
            hops_done += 1

        critique_text, need_more, crt_ns = self.self_critique(question, answer, "\n\n".join(combined_context))
        total_ns = time.time_ns() - t0
        latency.add("multihop_total", total_ns)

        return {
            "question": question,
            "answer": answer,
            "hops_done": hops_done,
            "retrieved_count": len(retrieved_snippets),
            "retrieved": retrieved_snippets,
            "critique": critique_text,
            "per_hop_ns": per_hop,
            "total_ns": total_ns
        }

# ---------------------------
# Main optimized pipeline
# ---------------------------
def main():
    # CONFIG - tweak these for experimentation
    PDF_PATH = "Data/ECHOES OF HER LOVE.pdf"
    CACHE_EMB_NPY = "embeddings_cache.npy"
    MODE = os.getenv("FAST_MODE", "mem")   # "mem" or "qdrant"
    MODEL_EMBED = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    DIM = 384

    print("FAST RAG START (mode=%s)" % MODE)
    pipeline_start = time.time_ns()

    # PHASE 1: Load text quickly or assume pre-supplied text
    # If the PDF is large, user should pre-extract text and provide path to .txt to avoid repeated PDF parsing.
    text = ""
    if os.path.exists(PDF_PATH):
        # lazy fast read: attempt to read text file first (if user provided pre-extracted)
        if PDF_PATH.lower().endswith(".txt"):
            with open(PDF_PATH, "r", encoding="utf8") as f:
                text = f.read()
        else:
            # Attempt fast PDF extraction via pdfplumber only once (keeps compatibility)
            try:
                import pdfplumber
                t0 = time.time_ns()
                with pdfplumber.open(PDF_PATH) as pdf:
                    pages = pdf.pages
                    parts = []
                    for p in pages:
                        parts.append(p.extract_text() or "")
                text = "\n".join(parts)
                latency.add("pdf_load", time.time_ns() - t0)
            except Exception:
                text = ""
    else:
        # no PDF; assume user will provide text via other means
        text = ""

    # Chunk
    chunks = chunk_text_fast(text, chunk_size=200, overlap=20) if text else ["(empty document)"]
    # Embedding model load (once)
    t0 = time.time_ns()
    embedder = SentenceTransformer(MODEL_EMBED)
    latency.add("embedder_load", time.time_ns() - t0)

    # Compute / load embeddings
    vectors = load_or_compute_embeddings(embedder, chunks, CACHE_EMB_NPY)

    # Init searcher
    if MODE == "mem":
        searcher = FastSearcher(mode="mem", vectors=vectors, chunks=chunks)
    else:
        if not QDRANT_AVAILABLE:
            raise RuntimeError("Qdrant not available; install qdrant-client or switch mode=mem")
        qdrant = init_qdrant_minimal("rag_collection", vectors.shape[1], in_memory=True)
        # create minimal point structs and upsert in batch (fast)
        ids = list(range(vectors.shape[0]))
        points = [PointStruct(id=i, vector=vectors[i].tolist(), payload={"text": chunks[i]}) for i in ids]
        t0 = time.time_ns()
        qdrant.upsert(collection_name="rag_collection", points=points)
        latency.add("qdrant_upsert_all", time.time_ns() - t0)
        searcher = FastSearcher(mode="qdrant", qdrant_client=qdrant, chunks=chunks)

    # LLM: replace DummyLLM with real ChatGroq (or other) that returns .content quickly
    llm = DummyLLM(latency_ms=10.0)  # emulate a 10ms LLM call for benching
    rag = MultiHopRAG(llm, searcher, embedder)

    # Run a few queries
    queries = [
        "What are the main themes in this story?",
        "Summarize the key events in the document.",
        "What is the capital of France?"
    ]
    results = []
    for q in queries:
        r = rag.query(q, max_hops=2, top_k=4)
        results.append(r)
        print(f"Q: {q[:60]} -> total {format_time_ns(r['total_ns'])}")

    # Final summary
    pipeline_total = time.time_ns() - pipeline_start
    latency.add("pipeline_total", pipeline_total)
    print(f"Pipeline total: {format_time_ns(pipeline_total)}")
    latency.pretty_print()

if __name__ == "__main__":
    main()
