#!/usr/bin/env python3
"""
rag_plan_and_solve.py
RAG with Planning (Plan-and-Solve) - Creates a plan first, then executes it step by step.

The system:
1. PLAN: Breaks down the query into subtasks
2. EXECUTE: Executes each subtask sequentially
3. SYNTHESIZE: Combines results into final answer
4. REFLECT: Evaluates plan quality and execution

Environment variables needed:
    PINECONE_API_KEY, GROQ_API_KEY
"""

import os
import time
import sys
import json
import re
import traceback
from collections import defaultdict
from typing import List, Dict, Any, Optional
from enum import Enum

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
INDEX_NAME = "pinecone-planning"
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
# Task Types for Plan Execution
# ---------------------------
class TaskType(Enum):
    RETRIEVE = "retrieve"
    ANALYZE = "analyze"
    CALCULATE = "calculate"
    SUMMARIZE = "summarize"
    COMPARE = "compare"
    SYNTHESIZE = "synthesize"

class Task:
    """Represents a single task in the plan"""
    def __init__(self, task_id: int, task_type: TaskType, description: str, 
                 dependencies: List[int] = None, parameters: Dict = None):
        self.task_id = task_id
        self.task_type = task_type
        self.description = description
        self.dependencies = dependencies or []
        self.parameters = parameters or {}
        self.result = None
        self.status = "pending"  # pending, running, completed, failed
        self.elapsed_ns = 0
    
    def __repr__(self):
        return f"Task({self.task_id}: {self.task_type.value} - {self.description})"

class Plan:
    """Represents the complete execution plan"""
    def __init__(self, query: str, tasks: List[Task]):
        self.query = query
        self.tasks = tasks
        self.created_at = time.time_ns()
    
    def get_ready_tasks(self) -> List[Task]:
        """Get tasks that are ready to execute (dependencies completed)"""
        ready = []
        for task in self.tasks:
            if task.status != "pending":
                continue
            # Check if all dependencies are completed
            deps_completed = all(
                self.tasks[dep_id].status == "completed" 
                for dep_id in task.dependencies
            )
            if deps_completed:
                ready.append(task)
        return ready
    
    def is_complete(self) -> bool:
        """Check if all tasks are completed or failed"""
        return all(task.status in ["completed", "failed"] for task in self.tasks)
    
    def __repr__(self):
        return f"Plan(query='{self.query}', tasks={len(self.tasks)})"

# ---------------------------
# Plan Executor Components
# ---------------------------
class TaskExecutor:
    """Executes individual tasks based on their type"""
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
        self.retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
    
    def execute_retrieve(self, task: Task, context: Dict) -> str:
        """Execute a retrieval task"""
        query = task.parameters.get("query", task.description)
        print(f"    🔍 Retrieving: {query}")
        
        start = time.time_ns()
        docs = self.retriever.invoke(query)
        elapsed = time.time_ns() - start
        latency_report.add("executor_retrieve", elapsed)
        
        results = []
        for doc in docs:
            content = getattr(doc, "page_content", str(doc))
            results.append(content)
        
        return "\n\n".join(results)
    
    def execute_analyze(self, task: Task, context: Dict) -> str:
        """Execute an analysis task using LLM"""
        # Get input from dependencies or parameters
        input_text = self._get_input_from_dependencies(task, context)
        analysis_focus = task.parameters.get("focus", "general analysis")
        
        prompt = f"""Analyze the following information with focus on: {analysis_focus}

Information:
{input_text}

Provide a detailed analysis:"""
        
        print(f"    🧠 Analyzing: {analysis_focus}")
        start = time.time_ns()
        response = self.llm.invoke(prompt)
        elapsed = time.time_ns() - start
        latency_report.add("executor_analyze", elapsed)
        
        return response.content if hasattr(response, 'content') else str(response)
    
    def execute_calculate(self, task: Task, context: Dict) -> str:
        """Execute a calculation task"""
        calculation = task.parameters.get("calculation", "")
        print(f"    🔢 Calculating: {calculation}")
        
        start = time.time_ns()
        try:
            # Simple calculation support
            result = eval(calculation, {"__builtins__": {}}, {})
            elapsed = time.time_ns() - start
            latency_report.add("executor_calculate", elapsed)
            return f"Calculation result: {result}"
        except Exception as e:
            return f"Calculation error: {str(e)}"
    
    def execute_summarize(self, task: Task, context: Dict) -> str:
        """Execute a summarization task"""
        input_text = self._get_input_from_dependencies(task, context)
        
        prompt = f"""Provide a concise summary of the following information:

{input_text[:2000]}

Summary:"""
        
        print(f"    📝 Summarizing...")
        start = time.time_ns()
        response = self.llm.invoke(prompt)
        elapsed = time.time_ns() - start
        latency_report.add("executor_summarize", elapsed)
        
        return response.content if hasattr(response, 'content') else str(response)
    
    def execute_compare(self, task: Task, context: Dict) -> str:
        """Execute a comparison task"""
        input_text = self._get_input_from_dependencies(task, context)
        comparison_aspects = task.parameters.get("aspects", "general comparison")
        
        prompt = f"""Compare and contrast the following information focusing on: {comparison_aspects}

Information:
{input_text}

Comparison:"""
        
        print(f"    ⚖️ Comparing: {comparison_aspects}")
        start = time.time_ns()
        response = self.llm.invoke(prompt)
        elapsed = time.time_ns() - start
        latency_report.add("executor_compare", elapsed)
        
        return response.content if hasattr(response, 'content') else str(response)
    
    def execute_synthesize(self, task: Task, context: Dict) -> str:
        """Execute a synthesis task - combines multiple results"""
        input_text = self._get_input_from_dependencies(task, context)
        
        prompt = f"""Synthesize the following information into a coherent, comprehensive answer:

{input_text}

Original Question: {context.get('original_query', '')}

Synthesized Answer:"""
        
        print(f"    🔄 Synthesizing final answer...")
        start = time.time_ns()
        response = self.llm.invoke(prompt)
        elapsed = time.time_ns() - start
        latency_report.add("executor_synthesize", elapsed)
        
        return response.content if hasattr(response, 'content') else str(response)
    
    def _get_input_from_dependencies(self, task: Task, context: Dict) -> str:
        """Get input text from task dependencies"""
        inputs = []
        for dep_id in task.dependencies:
            dep_task = context['plan'].tasks[dep_id]
            if dep_task.result:
                inputs.append(f"From {dep_task.description}:\n{dep_task.result}")
        return "\n\n".join(inputs) if inputs else task.description
    
    def execute_task(self, task: Task, context: Dict) -> str:
        """Execute a task based on its type"""
        task.status = "running"
        start = time.time_ns()
        
        try:
            if task.task_type == TaskType.RETRIEVE:
                result = self.execute_retrieve(task, context)
            elif task.task_type == TaskType.ANALYZE:
                result = self.execute_analyze(task, context)
            elif task.task_type == TaskType.CALCULATE:
                result = self.execute_calculate(task, context)
            elif task.task_type == TaskType.SUMMARIZE:
                result = self.execute_summarize(task, context)
            elif task.task_type == TaskType.COMPARE:
                result = self.execute_compare(task, context)
            elif task.task_type == TaskType.SYNTHESIZE:
                result = self.execute_synthesize(task, context)
            else:
                result = f"Unknown task type: {task.task_type}"
            
            task.result = result
            task.status = "completed"
            task.elapsed_ns = time.time_ns() - start
            
            return result
        except Exception as e:
            task.status = "failed"
            task.elapsed_ns = time.time_ns() - start
            task.result = f"Error: {str(e)}"
            print(f"    ❌ Task failed: {str(e)}")
            return task.result

# ---------------------------
# Plan and Solve RAG System
# ---------------------------
class PlanAndSolveRAG:
    """RAG system that creates a plan and executes it step by step"""
    
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
        self.executor = TaskExecutor(vectorstore, llm)
    
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
            return str(e)
    
    def create_plan(self, query: str) -> Plan:
        """Create an execution plan for the query"""
        print(f"\n{'='*70}")
        print(f"📋 CREATING EXECUTION PLAN")
        print(f"{'='*70}")
        print(f"Query: {query}\n")
        
        planning_prompt = f"""You are a planning expert. Break down this query into a step-by-step execution plan.

Query: {query}

Create a plan with numbered steps. Each step should be one of these types:
- RETRIEVE: Search for information in documents
- ANALYZE: Analyze retrieved information
- CALCULATE: Perform calculations
- SUMMARIZE: Summarize information
- COMPARE: Compare different pieces of information
- SYNTHESIZE: Combine results into final answer

Format each step as:
Step X [TYPE]: Description
Dependencies: [step numbers this depends on, or "none"]
Parameters: key=value (if needed)

Example:
Step 1 [RETRIEVE]: Find information about themes
Dependencies: none
Parameters: query="main themes"

Step 2 [ANALYZE]: Analyze the themes found
Dependencies: 1
Parameters: focus="literary themes"

Step 3 [SYNTHESIZE]: Create final answer
Dependencies: 2

Now create the plan:"""
        
        plan_text = self._llm_invoke(planning_prompt, "planning")
        print(f"🤖 Generated Plan:\n{plan_text}\n")
        
        # Parse the plan into Task objects
        tasks = self._parse_plan(plan_text)
        
        if not tasks:
            # Fallback: create a simple plan
            print("⚠️ Could not parse plan, creating default plan")
            tasks = [
                Task(0, TaskType.RETRIEVE, "Retrieve relevant information", [], {"query": query}),
                Task(1, TaskType.SYNTHESIZE, "Synthesize answer", [0], {})
            ]
        
        plan = Plan(query, tasks)
        
        print(f"✅ Plan created with {len(tasks)} tasks:")
        for task in tasks:
            deps = f" (depends on: {task.dependencies})" if task.dependencies else ""
            print(f"  {task.task_id}. [{task.task_type.value}] {task.description}{deps}")
        
        return plan
    
    def _parse_plan(self, plan_text: str) -> List[Task]:
        """Parse plan text into Task objects"""
        tasks = []
        current_task = None
        
        lines = plan_text.split('\n')
        for line in lines:
            line = line.strip()
            
            # Match step definition: Step X [TYPE]: Description
            step_match = re.match(r'Step\s+(\d+)\s+\[(\w+)\]:\s*(.+)', line, re.IGNORECASE)
            if step_match:
                task_id = int(step_match.group(1)) - 1  # 0-indexed
                task_type_str = step_match.group(2).upper()
                description = step_match.group(3).strip()
                
                # Map string to TaskType
                task_type_map = {
                    "RETRIEVE": TaskType.RETRIEVE,
                    "ANALYZE": TaskType.ANALYZE,
                    "CALCULATE": TaskType.CALCULATE,
                    "SUMMARIZE": TaskType.SUMMARIZE,
                    "COMPARE": TaskType.COMPARE,
                    "SYNTHESIZE": TaskType.SYNTHESIZE
                }
                task_type = task_type_map.get(task_type_str, TaskType.RETRIEVE)
                
                current_task = Task(task_id, task_type, description)
                tasks.append(current_task)
            
            # Parse dependencies
            elif current_task and re.match(r'Dependencies?:', line, re.IGNORECASE):
                deps_text = line.split(':', 1)[1].strip().lower()
                if deps_text != "none":
                    # Extract numbers
                    deps = [int(d)-1 for d in re.findall(r'\d+', deps_text)]
                    current_task.dependencies = deps
            
            # Parse parameters
            elif current_task and re.match(r'Parameters?:', line, re.IGNORECASE):
                params_text = line.split(':', 1)[1].strip()
                # Parse key=value pairs
                params = {}
                for param in re.findall(r'(\w+)=["\'"]?([^"\'",]+)["\'"]?', params_text):
                    params[param[0]] = param[1].strip()
                current_task.parameters = params
        
        return tasks
    
    def execute_plan(self, plan: Plan) -> Dict[str, Any]:
        """Execute the plan step by step"""
        print(f"\n{'='*70}")
        print(f"⚙️ EXECUTING PLAN")
        print(f"{'='*70}\n")
        
        execution_start = time.time_ns()
        context = {
            'plan': plan,
            'original_query': plan.query,
            'results': {}
        }
        
        step_count = 0
        while not plan.is_complete():
            ready_tasks = plan.get_ready_tasks()
            
            if not ready_tasks:
                print("⚠️ No ready tasks but plan not complete - possible dependency issue")
                break
            
            for task in ready_tasks:
                step_count += 1
                print(f"\n  Step {step_count}: Task {task.task_id}")
                print(f"  Type: {task.task_type.value}")
                print(f"  Description: {task.description}")
                
                result = self.executor.execute_task(task, context)
                context['results'][task.task_id] = result
                
                print(f"  ✅ Completed in {format_time_ns(task.elapsed_ns)}")
                print(f"  Result preview: {result[:150]}...")
        
        execution_elapsed = time.time_ns() - execution_start
        latency_report.add("plan_execution_total", execution_elapsed)
        
        print(f"\n✅ Plan execution completed in {format_time_ns(execution_elapsed)}")
        
        return context
    
    def reflect_on_execution(self, plan: Plan, context: Dict) -> Dict[str, Any]:
        """Reflect on the plan execution quality"""
        print(f"\n{'='*70}")
        print(f"🔍 REFLECTING ON EXECUTION")
        print(f"{'='*70}\n")
        
        # Gather execution statistics
        completed = sum(1 for t in plan.tasks if t.status == "completed")
        failed = sum(1 for t in plan.tasks if t.status == "failed")
        total_time = sum(t.elapsed_ns for t in plan.tasks)
        
        # Get final result
        final_task = plan.tasks[-1]
        final_answer = final_task.result if final_task.status == "completed" else "No answer generated"
        
        # LLM-based reflection
        reflection_prompt = f"""Evaluate the execution of this plan:

Original Query: {plan.query}

Plan had {len(plan.tasks)} tasks:
- Completed: {completed}
- Failed: {failed}
- Total time: {format_time_ns(total_time)}

Final Answer: {final_answer[:500]}

Evaluate:
1. Plan Quality (1-10): Was the plan well-structured?
2. Execution Success (1-10): How well was it executed?
3. Answer Quality (1-10): How good is the final answer?
4. Improvements: What could be improved?

Evaluation:"""
        
        reflection = self._llm_invoke(reflection_prompt, "reflection")
        
        print(f"📊 Execution Statistics:")
        print(f"  Tasks: {len(plan.tasks)} (✅ {completed}, ❌ {failed})")
        print(f"  Total time: {format_time_ns(total_time)}")
        print(f"\n💭 Reflection:\n{reflection}\n")
        
        return {
            "statistics": {
                "total_tasks": len(plan.tasks),
                "completed": completed,
                "failed": failed,
                "total_time_ns": total_time
            },
            "reflection": reflection,
            "final_answer": final_answer
        }
    
    def query(self, question: str) -> Dict[str, Any]:
        """Main query method: Plan -> Execute -> Reflect"""
        print(f"\n{'='*70}")
        print(f"🚀 PLAN-AND-SOLVE RAG QUERY")
        print(f"{'='*70}")
        print(f"❓ Question: {question}\n")
        
        overall_start = time.time_ns()
        
        # Phase 1: Planning
        plan = self.create_plan(question)
        
        # Phase 2: Execution
        context = self.execute_plan(plan)
        
        # Phase 3: Reflection
        reflection_result = self.reflect_on_execution(plan, context)
        
        overall_elapsed = time.time_ns() - overall_start
        latency_report.add("query_total", overall_elapsed)
        
        return {
            "question": question,
            "plan": plan,
            "answer": reflection_result["final_answer"],
            "reflection": reflection_result,
            "total_elapsed_ns": overall_elapsed
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
    print("🚀 RAG WITH PLANNING (PLAN-AND-SOLVE)")
    print("="*70)
    
    # Setup
    text = load_and_process_pdf(PDF_PATH)
    chunks = chunk_text(text)
    vectorstore = init_vectorstore(chunks, PINECONE_API_KEY)
    
    # Initialize LLM
    print("\n🤖 Initializing LLM...")
    llm = ChatGroq(model_name=MODEL_NAME, temperature=0, groq_api_key=GROQ_API_KEY)
    
    # Initialize Plan-and-Solve RAG
    plan_solve_rag = PlanAndSolveRAG(vectorstore, llm)
    print("\n✅ Plan-and-Solve RAG system ready!\n")
    
    # Test Queries
    queries = [
        "What are the main themes in this story and how do they relate to each other?",
        "Summarize the key events and analyze their significance.",
        "What is the emotional arc of the main character throughout the document?"
    ]
    
    results = []
    for q in queries:
        result = plan_solve_rag.query(q)
        results.append(result)
        
        print(f"\n{'='*70}")
        print(f"📊 QUERY SUMMARY")
        print(f"{'='*70}")
        print(f"Question: {q}")
        print(f"Plan tasks: {len(result['plan'].tasks)}")
        print(f"Total time: {format_time_ns(result['total_elapsed_ns'])}")
        print(f"\nFinal Answer:\n{result['answer'][:400]}...")
        print(f"\n{'='*70}\n")
    
    # Final report
    latency_report.pretty_print()
    
    print("\n✅ PLAN-AND-SOLVE RAG PIPELINE COMPLETE")

if __name__ == "__main__":
    main()