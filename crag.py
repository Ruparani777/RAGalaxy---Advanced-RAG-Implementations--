#!/usr/bin/env python3
"""
crag_full.py - Corrective RAG (CRAG) - FIXED VERSION
Self-correcting RAG that evaluates retrieved documents and takes corrective actions
"""

import os
import time
import sys
import traceback
from collections import defaultdict
from typing import Dict, List, Tuple

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
    
    # Add documents with namespace to avoid conflicts
    print(f"📤 Uploading {len(chunks)} chunks to '{index_name}'...")
    vs.add_texts(
        texts=chunks,
        metadatas=[{"chunk_id": i, "source": "crag"} for i in range(len(chunks))]
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
# CORRECTIVE RAG (CRAG)
# ---------------------------
class CorrectiveRAG:
    """
    CRAG: Self-correcting RAG with document evaluation
    
    Steps:
    1. Initial Retrieval
    2. Relevance Evaluation (CORRECT/INCORRECT/AMBIGUOUS)
    3. Corrective Action
    4. Generate answer with corrected context
    """
    
    def __init__(self, vectorstore, llm, max_corrections=2):
        self.vectorstore = vectorstore
        self.llm = llm
        self.max_corrections = max_corrections
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    def _llm_invoke(self, prompt, label):
        """Timed LLM invocation"""
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
            print(f"LLM error: {e}")
            return str(e), elapsed
    
    def _retrieve_documents(self, query: str) -> Tuple[List, int]:
        """Retrieve documents from vector store"""
        start = time.time_ns()
        try:
            docs = self.retriever.invoke(query)
            elapsed = time.time_ns() - start
            latency_report.add("crag_retrieve", elapsed)
            return docs, elapsed
        except Exception as e:
            elapsed = time.time_ns() - start
            print(f"Retrieval error: {e}")
            return [], elapsed
    
    def _evaluate_relevance(self, query: str, docs: List) -> Tuple[str, float, int]:
        """
        Evaluate if retrieved documents are relevant
        Returns: (evaluation, confidence, elapsed_ns)
        """
        if not docs:
            return "INCORRECT", 0.0, 0
        
        doc_texts = []
        for doc in docs[:3]:
            text = getattr(doc, "page_content", str(doc))
            doc_texts.append(text[:300])
        
        combined_docs = "\n\n".join([f"Doc {i+1}: {t}" for i, t in enumerate(doc_texts)])
        
        prompt = f"""Evaluate if these documents are relevant to answer the question.

Question: {query}

Documents:
{combined_docs}

Evaluate the relevance:
- CORRECT: Documents contain relevant information
- INCORRECT: Documents are not relevant
- AMBIGUOUS: Documents partially relevant

Respond ONLY:
Evaluation: [CORRECT/INCORRECT/AMBIGUOUS]
Confidence: [0.0-1.0]
Reason: [brief]

Evaluation:"""
        
        eval_text, elapsed = self._llm_invoke(prompt, "crag_evaluate")
        
        evaluation = "AMBIGUOUS"
        confidence = 0.5
        
        if "CORRECT" in eval_text and "INCORRECT" not in eval_text:
            evaluation = "CORRECT"
            confidence = 0.9
        elif "INCORRECT" in eval_text:
            evaluation = "INCORRECT"
            confidence = 0.8
        elif "AMBIGUOUS" in eval_text:
            evaluation = "AMBIGUOUS"
            confidence = 0.6
        
        return evaluation, confidence, elapsed
    
    def _refine_query(self, original_query: str, feedback: str) -> Tuple[str, int]:
        """Refine query based on feedback"""
        prompt = f"""Refine this query to get better search results.

Original: {original_query}
Issue: {feedback}

Return ONLY the refined query.

Refined Query:"""
        
        refined, elapsed = self._llm_invoke(prompt, "crag_refine_query")
        refined = refined.strip().replace('"', '')
        return refined, elapsed
    
    def _decompose_query(self, query: str) -> Tuple[List[str], int]:
        """Decompose ambiguous query"""
        prompt = f"""Break this query into 2-3 specific sub-queries.

Query: {query}

Return numbered list:
1. [sub-query 1]
2. [sub-query 2]
3. [sub-query 3]

Sub-queries:"""
        
        result, elapsed = self._llm_invoke(prompt, "crag_decompose")
        
        import re
        lines = result.split('\n')
        sub_queries = []
        for line in lines:
            match = re.match(r'\d+\.\s*(.+)', line.strip())
            if match:
                sub_queries.append(match.group(1))
        
        return sub_queries[:3], elapsed
    
    def _generate_answer(self, query: str, context: str, correction_history: str) -> Tuple[str, int]:
        """Generate final answer"""
        prompt = f"""Answer using the context.

{correction_history}

Context:
{context}

Question: {query}

Answer:"""
        
        answer, elapsed = self._llm_invoke(prompt, "crag_generate")
        return answer, elapsed
    
    def query(self, question: str) -> Dict:
        """Execute CRAG pipeline"""
        print(f"\n{'='*70}")
        print(f"🔧 CORRECTIVE RAG (CRAG)")
        print(f"{'='*70}")
        print(f"❓ {question}\n")
        
        overall_start = time.time_ns()
        correction_count = 0
        correction_history = []
        final_context = ""
        evaluation = "AMBIGUOUS"
        
        current_query = question
        
        while correction_count < self.max_corrections:
            iteration = correction_count + 1
            print(f"\n{'─'*70}")
            print(f"ITERATION {iteration}")
            print(f"{'─'*70}")
            
            # Retrieve
            print(f"📚 Retrieving: '{current_query[:60]}...'")
            docs, ret_time = self._retrieve_documents(current_query)
            print(f"   ✓ Retrieved {len(docs)} docs ({format_time_ns(ret_time)})")
            
            if not docs:
                print("   ⚠️  No documents")
                correction_history.append(f"Iter {iteration}: No results")
                correction_count += 1
                continue
            
            # Evaluate
            print(f"🔍 Evaluating relevance...")
            evaluation, confidence, eval_time = self._evaluate_relevance(current_query, docs)
            print(f"   ✓ {evaluation} (conf: {confidence:.2f}) ({format_time_ns(eval_time)})")
            
            correction_history.append(f"Iter {iteration}: {evaluation} ({confidence:.2f})")
            
            # Corrective action
            if evaluation == "CORRECT":
                print(f"   ✅ Documents relevant")
                final_context = "\n\n".join([
                    getattr(doc, "page_content", str(doc)) for doc in docs
                ])
                break
            
            elif evaluation == "INCORRECT":
                print(f"   ❌ Not relevant, refining...")
                refined_query, refine_time = self._refine_query(current_query, "Not relevant")
                print(f"   ✓ Refined: '{refined_query[:60]}...' ({format_time_ns(refine_time)})")
                current_query = refined_query
                correction_count += 1
            
            elif evaluation == "AMBIGUOUS":
                print(f"   ⚠️  Ambiguous, decomposing...")
                sub_queries, decomp_time = self._decompose_query(current_query)
                print(f"   ✓ {len(sub_queries)} sub-queries ({format_time_ns(decomp_time)})")
                
                all_sub_docs = []
                for i, sq in enumerate(sub_queries, 1):
                    print(f"      {i}. {sq[:50]}...")
                    sub_docs, _ = self._retrieve_documents(sq)
                    all_sub_docs.extend(sub_docs)
                
                final_context = "\n\n".join([
                    getattr(doc, "page_content", str(doc)) for doc in all_sub_docs[:6]
                ])
                print(f"   ✓ Retrieved {len(all_sub_docs)} docs total")
                break
        
        # Generate answer
        print(f"\n💭 Generating answer...")
        history_text = "\nCorrections:\n" + "\n".join(correction_history)
        answer, gen_time = self._generate_answer(question, final_context, history_text)
        
        print(f"\n💬 ANSWER ({format_time_ns(gen_time)}):")
        print(f"{answer}\n")
        
        total_time = time.time_ns() - overall_start
        latency_report.add("crag_query_total", total_time)
        
        print(f"📊 Corrections: {correction_count}/{self.max_corrections}")
        print(f"⏱️  Total: {format_time_ns(total_time)}")
        
        return {
            'question': question,
            'answer': answer,
            'corrections': correction_count,
            'final_evaluation': evaluation,
            'correction_history': correction_history,
            'total_time': total_time
        }

# ---------------------------
# MAIN
# ---------------------------
def main():
    print("="*70)
    print("🔧 CORRECTIVE RAG (CRAG) PIPELINE")
    print("="*70 + "\n")
    
    # Setup
    text = load_pdf(PDF_PATH)
    chunks = chunk_text(text)
    embed = get_embeddings_model()
    pc = init_pinecone(INDEX_NAME)
    vs = create_vectorstore(embed, chunks, INDEX_NAME)
    
    print(f"\n✅ LLM initializing...")
    llm = ChatGroq(model_name=MODEL_NAME, temperature=0, groq_api_key=GROQ_API_KEY)
    
    crag = CorrectiveRAG(vs, llm, max_corrections=2)
    
    print("\n" + "="*70)
    print("PHASE 1: CORRECTIVE RAG QUERIES")
    print("="*70)
    
    queries = [
        "What are the main themes in this story?",
        "Tell me about quantum physics",
        "What happens in the story?",
        "Describe the mother-daughter relationship",
    ]
    
    results = []
    for i, q in enumerate(queries, 1):
        print(f"\n{'═'*70}")
        print(f"QUERY {i}/{len(queries)}")
        print(f"{'═'*70}")
        result = crag.query(q)
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
    print("CRAG STATISTICS")
    print("="*70)
    
    total_corrections = sum(r['corrections'] for r in results)
    print(f"Total queries: {len(results)}")
    print(f"Total corrections: {total_corrections}")
    print(f"Avg: {total_corrections/len(results):.1f} corrections/query")
    
    print("\nResults:")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['question'][:50]}...")
        print(f"   Corrections: {r['corrections']}, Final: {r['final_evaluation']}")
    
    latency_report.pretty_print()
    print("✅ CRAG COMPLETE\n")

if __name__ == "__main__":
    main()