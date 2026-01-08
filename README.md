# RAGalaxy: Advanced RAG Implementations

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

A comprehensive collection of Retrieval-Augmented Generation (RAG) implementations exploring cutting-edge architectures and multiple vector databases. This repository serves as a research and development hub for advanced RAG techniques, providing ready-to-use implementations for various use cases.

## 🌟 Features

- **Multiple Vector Databases**: Support for Pinecone, Milvus, Qdrant, and Weaviate
- **Diverse RAG Architectures**: From basic to advanced agentic implementations
- **Modular Design**: Easily extensible and customizable components
- **Performance Tracking**: Built-in latency monitoring and optimization
- **Multimodal Support**: Extensions for handling different data types
- **Jupyter Notebooks**: Interactive versions of all implementations

## 🗄️ Supported Vector Databases

### Pinecone
Cloud-native vector database optimized for similarity search and AI applications.

### Milvus
Open-source vector database with high performance and scalability.

### Qdrant
Vector search engine with advanced filtering and payload support.

### Weaviate
Open-source vector database with semantic search capabilities.

## 🏗️ RAG Architectures

### 1. Vanilla RAG
**File**: `vanilla.py` / `vanilla.ipynb`

Basic RAG implementation following the standard pipeline:
- Document loading and chunking
- Embedding generation
- Vector storage and retrieval
- LLM-powered answer generation

### 2. Corrective RAG (CRAG)
**Files**: `crag.py`, `MilvusCrag.py`, `QdrantCrag.py`, `weaviatecrag.py`

Enhances vanilla RAG with relevance checking and corrective mechanisms to improve answer quality through iterative refinement.

### 3. Memory-Augmented RAG
**Files**: `memory_augementedrag.py`, `MilvusMemoryAugmented.py`, `QdrantMemoryAugmented.py`, `weaviatememoryaugmentedrag.py`

Maintains conversation history and extracts long-term facts for contextual, personalized responses.

### 4. Multi-Hop RAG
**Files**: `MultiHopRag.py`, `Milvusmultihop.py`, `Qdrantmultihop.py`, `weaviatemultihop.py`

Performs iterative retrieval and reasoning to answer complex queries requiring multiple information pieces.

### 5. RAG with Feedback Loop
**Files**: `Ragwithfeedbackloop.py`, `Milvusragwithfeedbackloop.py`, `Qdrantragwithfeedbackloop.py`, `weaviateragwithfeedbackloop.py`

Incorporates user feedback to continuously refine retrieval and generation processes.

### 6. Plan-and-Solve RAG
**Files**: `Ragwithplanning.py`, `Milvusplansolverag.py`, `QdrantPlanandsolve.py`, `weaviateplanandsolve.py`

Breaks down complex queries into sub-tasks and executes them sequentially using planning algorithms.

### 7. RAG with Tool Use
**Files**: `Ragwithtooluse.py`, `Milvusragwithtooluse.py`, `QdrantRagwithtooluse.py`, `weaviateragwithtooluse.py`

Integrates external tools and APIs during the reasoning process for enhanced capabilities.

### 8. ReAct RAG
**Files**: `ReActrag.py`, `MilvusReACT.py`, `qdrantReact.py`, `weaviateselfrag.py` (wait, check)

Combines reasoning traces with action execution for more dynamic and interactive responses.

### 9. Self-RAG
**Files**: `Selfrag.py`, `Milvusselfrag.py`, `qdrantself.py`, `weaviateselfrag.py`

Implements self-reflective mechanisms where the system evaluates and refines its own outputs.

### 10. Multimodal RAG
**File**: `multimodal.py`

Extends RAG to handle multiple data modalities beyond text.

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Ruparani777/RAGalaxy---Advanced-RAG-Implementations--.git
   cd RAGalaxy---Advanced-RAG-Implementations--
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file with your API keys:
   ```
   GROQ_API_KEY=your_groq_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   MILVUS_URI=your_milvus_uri
   QDRANT_URL=your_qdrant_url
   WEAVIATE_URL=your_weaviate_url
   ```

## 📖 Usage

### Basic Example (Vanilla RAG with Pinecone)

```python
from vanilla import RagPipeline

# Initialize the pipeline
rag = RagPipeline()

# Add documents
rag.add_documents("path/to/your/documents/")

# Query
response = rag.query("What is the capital of France?")
print(response)
```

### Advanced Example (Memory-Augmented RAG)

```python
from memory_augementedrag import MemoryAugmentedRag

# Initialize with memory
rag = MemoryAugmentedRag()

# Conversational query with memory
response = rag.query_with_memory("Tell me about machine learning", user_id="user1")
```

## 🔧 Configuration

Each implementation supports configuration through:
- Environment variables for API keys
- Configurable parameters for chunk size, embedding models, and similarity thresholds
- Customizable LLM models and providers

## 📊 Performance Monitoring

All implementations include built-in latency tracking:
- Document processing time
- Embedding generation time
- Vector search time
- LLM inference time

## 🧪 Testing

Run the PDF probe utility to test document processing:
```bash
python pdf_probe.py
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [LangChain](https://www.langchain.com/)
- Powered by [Groq](https://groq.com/) for fast LLM inference
- Vector databases: [Pinecone](https://www.pinecone.io/), [Milvus](https://milvus.io/), [Qdrant](https://qdrant.tech/), [Weaviate](https://weaviate.io/)

## 📞 Contact

For questions or collaborations, please open an issue or reach out to the maintainers.

---

*Explore the universe of RAG implementations with RAGalaxy!* 🌌</content>
<parameter name="filePath">README.md