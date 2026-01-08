#!/usr/bin/env python3
"""
iterative_multihop_rag.py
Iterative Multi-Hop RAG - Performs multiple sequential retrievals, each building on previous results.

The system:
1. Initial retrieval based on query
2. Analyzes results and generates follow-up questions
3. Performs subsequent retrievals (hops) based on information gaps
4. Chains information across hops
5. Synthesizes final answer from all hops

Environment variables needed:
    PINECONE_API_KEY, GROQ_API_KEY
"""

import os
import time
import sys
import re
import traceback
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------
# Config
# ---------------------------
PDF_PATH = "Data/ECHOES OF HER LOVE.pdf"
INDEX_NAME = "pinecone-multihop"
DIM = 384
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.1-8b-instant"

if PINECONE_API_KEY is None or GROQ_API_KEY is None:
    print("ERROR: Set PINECONE_API_KEY and GROQ_API_KEY environment variables")
    sys.exit(1)

# ---------------------------
# Utilities
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
        print("\n" + "="*70)
        print("LATENCY SUMMARY")
        print("="*70)
        for comp, vals in sorted(self.store.items()):
            total = sum(vals)
            avg = total // len(vals) if vals else 0
            print(f"\n{comp}:")
            print(f"  Count: {len(vals)}, Total: {format_time_ns(total)}, Avg: {format_time_ns(avg)}")
        print("="*70 + "\n")

latency_report = LatencyReport()

# ---------------------------
# Hop Data Structure
# ---------------------------
class Hop:
    """Represents a single retrieval hop in the multi-hop chain"""
    def __init__(self, hop_number: int, query: str, context_from_previous: str = ""):
        self.hop_number = hop_number
        self.query = query
        self.context_from_previous = context_from_previous
        self.retrieved_docs = []
        self.retrieved_content = ""
        self.analysis = ""
        self.follow_up_questions = []
        self.information_gain = 0.0
        self.elapsed_ns = 0
        self.timestamp = time.time_ns()
    
    def __repr__(self):
        return f"Hop({self.hop_number}: '{self.query[:50]}...')"

class MultiHopChain:
    """Represents the complete multi-hop retrieval chain"""
    def __init__(self, original_query: str):
        self.original_query = original_query
        self.hops = []
        self.final_answer = ""
        self.total_elapsed_ns = 0
        self.information_coverage = 0.0
    
    def add_hop(self, hop: Hop):
        self.hops.append(hop)
    
    def get_all_context(self) -> str:
        """Get concatenated context from all hops"""
        return "\n\n--- HOP SEPARATOR ---\n\n".join([
            f"HOP {h.hop_number} (Query: {h.query}):\n{h.retrieved_content}"
            for h in self.hops
        ])
    
    def __repr__(self):
        return f"MultiHopChain(query='{self.original_query}', hops={len(self.hops)})"

# ---------------------------
# Iterative Multi-Hop RAG System
# ---------------------------
class IterativeMultiHopRAG:
    """RAG system that performs multiple sequential retrievals (hops)"""
    
    def __init__(self, vectorstore, llm, max_hops: int = 4):
        self.vectorstore = vectorstore
        self.llm = llm
        self.max_hops = max_hops
        self.retriever = vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 3}
        )
    
    def _llm_invoke(self, prompt: str, label: str) -> str:
        """Invoke LLM with timing"""
        start = time.time_ns()
        try:
            response = self.llm.invoke(prompt)
            elapsed = time.time_ns() - start
            latency_report.add(f"llm_{label}", elapsed)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            elapsed = time.time_ns() - start
            latency_report.add(f"llm_{label}_error", elapsed)
            print(f"❌ LLM error in {label}: {e}")
            return str(e)
    
    def retrieve_documents(self, query: str, hop_number: int) -> Tuple[List, str]:
        """Retrieve documents for a given query"""
        print(f"    🔍 Retrieving documents for hop {hop_number}...")
        start = time.time_ns()
        
        try:
            docs = self.retriever.invoke(query)
            elapsed = time.time_ns() - start
            latency_report.add(f"retrieval_hop_{hop_number}", elapsed)
            
            # Extract content
            content_parts = []
            for i, doc in enumerate(docs):
                content = getattr(doc, "page_content", str(doc))
                content_parts.append(f"[Document {i+1}]\n{content}")
            
            full_content = "\n\n".join(content_parts)
            
            print(f"    ✅ Retrieved {len(docs)} documents in {format_time_ns(elapsed)}")
            print(f"    📄 Total content length: {len(full_content)} characters")
            
            return docs, full_content
            
        except Exception as e:
            elapsed = time.time_ns() - start
            latency_report.add(f"retrieval_hop_{hop_number}_error", elapsed)
            print(f"    ❌ Retrieval failed: {e}")
            return [], ""
    
    def analyze_information_gap(self, original_query: str, current_context: str, 
                                hop_number: int) -> Dict[str, Any]:
        """Analyze what information is missing and generate follow-up questions"""
        print(f"    🧠 Analyzing information gaps...")
        
        analysis_prompt = f"""You are analyzing information gathered so far to answer a question.

Original Question: {original_query}

Information gathered so far (from {hop_number} hop(s)):
{current_context[:2000]}

Analyze:
1. What key information have we found?
2. What information is still missing to fully answer the question?
3. What specific follow-up questions would help fill these gaps?
4. Rate information completeness (0-100%): How much of the answer do we have?

Format your response as:
FOUND: [what we learned]
MISSING: [what's still needed]
FOLLOW_UP: [specific question 1] | [specific question 2] | [specific question 3]
COMPLETENESS: [percentage]

Analysis:"""
        
        analysis = self._llm_invoke(analysis_prompt, f"gap_analysis_hop_{hop_number}")
        
        # Parse the analysis
        found = ""
        missing = ""
        follow_ups = []
        completeness = 0
        
        for line in analysis.split('\n'):
            if line.strip().startswith('FOUND:'):
                found = line.split('FOUND:', 1)[1].strip()
            elif line.strip().startswith('MISSING:'):
                missing = line.split('MISSING:', 1)[1].strip()
            elif line.strip().startswith('FOLLOW_UP:'):
                follow_up_text = line.split('FOLLOW_UP:', 1)[1].strip()
                follow_ups = [q.strip() for q in follow_up_text.split('|') if q.strip()]
            elif line.strip().startswith('COMPLETENESS:'):
                completeness_text = line.split('COMPLETENESS:', 1)[1].strip()
                # Extract number
                numbers = re.findall(r'\d+', completeness_text)
                if numbers:
                    completeness = int(numbers[0])
        
        print(f"    📊 Information completeness: {completeness}%")
        print(f"    ❓ Generated {len(follow_ups)} follow-up questions")
        
        return {
            'found': found,
            'missing': missing,
            'follow_ups': follow_ups,
            'completeness': completeness,
            'full_analysis': analysis
        }
    
    def determine_next_hop_query(self, original_query: str, follow_up_questions: List[str],
                                 chain: MultiHopChain) -> Optional[str]:
        """Determine what to search for in the next hop"""
        if not follow_up_questions:
            return None
        
        print(f"    🎯 Determining next hop query...")
        
        # Use the most promising follow-up question
        # For simplicity, we'll use the first one, but could rank them
        next_query = follow_up_questions[0]
        
        # Optionally refine it
        refinement_prompt = f"""Given the original question and current follow-up question, create an optimal search query.

Original Question: {original_query}

Follow-up Question: {next_query}

Previous queries used: {[h.query for h in chain.hops]}

Create a concise, specific search query that will find the missing information without repeating previous searches:

Search Query:"""
        
        refined_query = self._llm_invoke(refinement_prompt, f"query_refinement_hop_{len(chain.hops)}")
        
        # Clean up the refined query
        refined_query = refined_query.strip().split('\n')[0]
        
        print(f"    📝 Next hop query: '{refined_query}'")
        
        return refined_query
    
    def perform_hop(self, hop_number: int, query: str, chain: MultiHopChain) -> Hop:
        """Perform a single retrieval hop"""
        print(f"\n  {'='*66}")
        print(f"  🏃 HOP {hop_number}: {query}")
        print(f"  {'='*66}")
        
        hop_start = time.time_ns()
        
        # Create hop object
        context_from_previous = chain.get_all_context() if chain.hops else ""
        hop = Hop(hop_number, query, context_from_previous)
        
        # Retrieve documents
        docs, content = self.retrieve_documents(query, hop_number)
        hop.retrieved_docs = docs
        hop.retrieved_content = content
        
        # Quick analysis of this hop's contribution
        if content:
            contribution_prompt = f"""Rate how much NEW information this retrieval adds (0-100%):

Query: {query}
Retrieved: {content[:500]}...

Previous context available: {'Yes' if context_from_previous else 'No'}

NEW_INFORMATION_SCORE: [0-100]"""
            
            contribution = self._llm_invoke(contribution_prompt, f"contribution_hop_{hop_number}")
            
            # Extract score
            numbers = re.findall(r'\d+', contribution)
            if numbers:
                hop.information_gain = int(numbers[0])
            
            print(f"    💡 Information gain: {hop.information_gain}%")
        
        hop.elapsed_ns = time.time_ns() - hop_start
        latency_report.add("hop_total", hop.elapsed_ns)
        
        print(f"  ✅ Hop {hop_number} completed in {format_time_ns(hop.elapsed_ns)}")
        
        return hop
    
    def synthesize_final_answer(self, chain: MultiHopChain) -> str:
        """Synthesize final answer from all hops"""
        print(f"\n  {'='*66}")
        print(f"  🔄 SYNTHESIZING FINAL ANSWER FROM {len(chain.hops)} HOPS")
        print(f"  {'='*66}")
        
        all_context = chain.get_all_context()
        
        synthesis_prompt = f"""Synthesize a comprehensive answer from multiple retrieval hops.

Original Question: {chain.original_query}

Information gathered across {len(chain.hops)} retrieval hops:

{all_context[:4000]}

Instructions:
1. Integrate information from ALL hops
2. Resolve any contradictions by noting them
3. Cite which hop provided key information (e.g., "According to Hop 2...")
4. Provide a complete, coherent answer
5. Note if any information is still missing

Final Answer:"""
        
        final_answer = self._llm_invoke(synthesis_prompt, "final_synthesis")
        
        print(f"  ✅ Final answer synthesized")
        print(f"  📝 Answer length: {len(final_answer)} characters")
        
        return final_answer
    
    def query(self, question: str, target_completeness: int = 85) -> Dict[str, Any]:
        """Main query method: performs iterative multi-hop retrieval"""
        print(f"\n{'='*70}")
        print(f"🔗 ITERATIVE MULTI-HOP RAG QUERY")
        print(f"{'='*70}")
        print(f"❓ Question: {question}")
        print(f"🎯 Target completeness: {target_completeness}%")
        print(f"📊 Max hops: {self.max_hops}\n")
        
        overall_start = time.time_ns()
        
        # Initialize chain
        chain = MultiHopChain(question)
        
        # Initial hop - use original question
        current_query = question
        
        # Perform hops
        for hop_num in range(1, self.max_hops + 1):
            # Perform the hop
            hop = self.perform_hop(hop_num, current_query, chain)
            chain.add_hop(hop)
            
            # Analyze what we have so far
            current_context = chain.get_all_context()
            gap_analysis = self.analyze_information_gap(question, current_context, hop_num)
            
            hop.analysis = gap_analysis['full_analysis']
            hop.follow_up_questions = gap_analysis['follow_ups']
            chain.information_coverage = gap_analysis['completeness']
            
            # Check if we have enough information
            if gap_analysis['completeness'] >= target_completeness:
                print(f"\n  ✅ Target completeness reached: {gap_analysis['completeness']}%")
                print(f"  🎉 Stopping after {hop_num} hops")
                break
            
            # Check if we're at max hops
            if hop_num >= self.max_hops:
                print(f"\n  ⚠️ Max hops reached ({self.max_hops})")
                print(f"  📊 Current completeness: {gap_analysis['completeness']}%")
                break
            
            # Determine next hop query
            next_query = self.determine_next_hop_query(question, gap_analysis['follow_ups'], chain)
            
            if not next_query:
                print(f"\n  🛑 No more follow-up queries generated")
                print(f"  📊 Final completeness: {gap_analysis['completeness']}%")
                break
            
            current_query = next_query
        
        # Synthesize final answer from all hops
        final_answer = self.synthesize_final_answer(chain)
        chain.final_answer = final_answer
        
        overall_elapsed = time.time_ns() - overall_start
        chain.total_elapsed_ns = overall_elapsed
        latency_report.add("multihop_query_total", overall_elapsed)
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"📊 MULTI-HOP QUERY SUMMARY")
        print(f"{'='*70}")
        print(f"Total hops performed: {len(chain.hops)}")
        print(f"Information coverage: {chain.information_coverage}%")
        print(f"Total time: {format_time_ns(overall_elapsed)}")
        print(f"Average time per hop: {format_time_ns(overall_elapsed // len(chain.hops))}")
        
        print(f"\n🔗 HOP CHAIN:")
        for hop in chain.hops:
            print(f"  Hop {hop.hop_number}: '{hop.query[:60]}...'")
            print(f"    → Info gain: {hop.information_gain}%, Time: {format_time_ns(hop.elapsed_ns)}")
        
        return {
            'question': question,
            'answer': final_answer,
            'chain': chain,
            'num_hops': len(chain.hops),
            'information_coverage': chain.information_coverage,
            'total_elapsed_ns': overall_elapsed
        }

# ---------------------------
# PDF Processing (same as original)
# ---------------------------
def load_and_process_pdf(path):
    print(f"📄 Loading PDF: {path}")
    start = time.time_ns()
    
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    
    elapsed = time.time_ns() - start
    latency_report.add("pdf_load", elapsed)
    print(f"✅ Loaded {len(text)} characters in {format_time_ns(elapsed)}")
    
    return text

def chunk_text(text, chunk_size=1000, chunk_overlap=100):
    start = time.time_ns()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    elapsed = time.time_ns() - start
    latency_report.add("chunking", elapsed)
    print(f"📄 Created {len(chunks)} chunks in {format_time_ns(elapsed)}")
    return chunks

def init_vectorstore(chunks, api_key, index_name=INDEX_NAME):
    print(f"🔧 Initializing Pinecone...")
    start = time.time_ns()
    
    pc = Pinecone(api_key=api_key)
    existing = [idx.name for idx in pc.list_indexes()]
    
    if index_name in existing:
        print(f"🗑️ Deleting existing index...")
        pc.delete_index(index_name)
        time.sleep(2)
    
    print(f"🆕 Creating index...")
    pc.create_index(
        name=index_name,
        dimension=DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    time.sleep(2)
    
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vectorstore = PineconeVectorStore.from_texts(
        texts=chunks,
        embedding=embed_model,
        index_name=index_name,
        namespace="",
        metadatas=[{"chunk_id": i} for i in range(len(chunks))]
    )
    
    elapsed = time.time_ns() - start
    latency_report.add("vectorstore_init", elapsed)
    print(f"✅ Vectorstore ready in {format_time_ns(elapsed)}")
    
    return vectorstore

# ---------------------------
# Main
# ---------------------------
def main():
    print("="*70)
    print("🚀 ITERATIVE MULTI-HOP RAG")
    print("="*70)
    
    # Setup
    text = load_and_process_pdf(PDF_PATH)
    chunks = chunk_text(text)
    vectorstore = init_vectorstore(chunks, PINECONE_API_KEY)
    
    # Initialize LLM
    print("\n🤖 Initializing LLM...")
    llm = ChatGroq(model_name=MODEL_NAME, temperature=0, groq_api_key=GROQ_API_KEY)
    
    # Initialize Multi-Hop RAG
    multihop_rag = IterativeMultiHopRAG(vectorstore, llm, max_hops=4)
    print("\n✅ Multi-Hop RAG system ready!\n")
    
    # Test Queries - these benefit from multi-hop retrieval
    queries = [
        "What are the main themes in this story and how do they connect to the character development?",
        "Describe the relationship between the main characters and how it evolves throughout the narrative.",
        "What is the significance of love in the document and how is it portrayed in different contexts?"
    ]
    
    results = []
    for q in queries:
        result = multihop_rag.query(q, target_completeness=85)
        results.append(result)
        
        print(f"\n{'='*70}")
        print(f"FINAL ANSWER:")
        print(f"{'='*70}")
        print(result['answer'][:600])
        print("...\n")
        print(f"{'='*70}\n")
    
    # Final report
    latency_report.pretty_print()
    
    # Multi-hop statistics
    print("\n" + "="*70)
    print("MULTI-HOP STATISTICS")
    print("="*70)
    total_hops = sum(r['num_hops'] for r in results)
    avg_hops = total_hops / len(results)
    avg_coverage = sum(r['information_coverage'] for r in results) / len(results)
    
    print(f"Total queries: {len(results)}")
    print(f"Total hops: {total_hops}")
    print(f"Average hops per query: {avg_hops:.1f}")
    print(f"Average information coverage: {avg_coverage:.1f}%")
    print("="*70)
    
    print("\n✅ ITERATIVE MULTI-HOP RAG PIPELINE COMPLETE")

if __name__ == "__main__":
    main()