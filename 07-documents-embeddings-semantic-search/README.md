# Documents, Embeddings & Semantic Search

In this chapter, you'll learn the complete pipeline for working with documents in AI applications — from loading and preparing documents to enabling intelligent semantic search. You'll discover how to load content from various sources, split it into manageable chunks, convert text into numerical embeddings, and perform similarity searches that understand meaning rather than just matching keywords.

**Why learn this after agents?** You've already built agents that can use tools to solve problems. Now you'll learn how to create **retrieval tools** that give agents the power to search through your documents intelligently. This combination—agents with retrieval capabilities—enables **agentic RAG systems** where AI autonomously decides when and how to search your knowledge base to answer questions. You'll build this powerful pattern in [Building Agentic RAG Systems](../08-agentic-rag-systems/README.md).

## Prerequisites

- Completed [Getting Started with Agents](../05-agents/README.md)

## 🎯 Learning Objectives

By the end of this chapter, you'll be able to:

- ✅ Load documents from various sources (text, PDF, web)
- ✅ Split long documents into manageable chunks
- ✅ Understand chunking strategies and their trade-offs
- ✅ Work with document metadata
- ✅ Understand what embeddings are and how they work
- ✅ Create embeddings for text using AI models
- ✅ Store embeddings in vector databases
- ✅ Perform semantic similarity searches
- ✅ Build the foundation for RAG systems

---

## 📖 The Smart Library System Analogy

**Imagine you're building a modern, intelligent library system.**

### Part 1: Organizing the Library (Document Processing)

When someone donates a massive encyclopedia to your library, you can't:
- ❌ Hand readers the entire 2,000-page book
- ❌ Give them random pages
- ❌ Show them just individual words

Instead, you need to:
- Find the right sections (loading)
- Break it into manageable chapters (chunking)
- Label each piece with metadata (organization)
- Keep some overlap between sections so context isn't lost

### Part 2: The Smart Search System (Embeddings & Semantic Search)

Now imagine each book section gets a special "number tag" that represents its meaning:
- Section about "photosynthesis": `[plants: 0.9, biology: 0.8, energy: 0.7]`
- Section about "solar panels": `[plants: 0.1, technology: 0.9, energy: 0.8]`
- Section about "pasta recipes": `[plants: 0.2, food: 0.9, energy: 0.3]`

When someone asks "How do plants create energy?", the system:
1. Converts their question into numbers: `[plants: 0.9, biology: 0.7, energy: 0.8]`
2. Finds sections with similar numbers
3. Returns the photosynthesis section (perfect match!)

**This is how document processing and semantic search work together!**

---

## 📄 Part 1: Working with Documents

### Why Document Loaders?

LLMs need text input, but data comes in many formats: text files, PDFs, websites, JSON/CSV, and more. **Document loaders handle the complexity of reading different formats.**

---

### Example 1: Loading Text Files

**Code**: [`code/01_load_text.py`](./code/01_load_text.py)  
**Run**: `python 07-documents-embeddings-semantic-search/code/01_load_text.py`

**Example code:**

```python
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv

load_dotenv()

# Create sample data
sample_text = """
LangChain is a framework for building applications with large language models.

It provides tools for:
- Working with different AI providers
- Managing prompts and templates
- Processing and storing documents
- Building RAG systems
- Creating AI agents

The framework is designed to be modular and composable.
"""

with open("./data/sample.txt", "w") as f:
    f.write(sample_text.strip())

# Load the document
loader = TextLoader("./data/sample.txt")
docs = loader.load()

print(f"Loaded {len(docs)} document(s)")
print(f"Content: {docs[0].page_content}")
print(f"Metadata: {docs[0].metadata}")
```

### Expected Output

```
Loaded 1 document(s)
Content: LangChain is a framework for building applications with large language models.

It provides tools for:
- Working with different AI providers
- Managing prompts and templates
...

Metadata: {'source': './data/sample.txt'}
```

---

## ✂️ Splitting Documents

### Why Split Documents?

- **LLM context limits**: Models can only process ~4,000-128,000 tokens
- **Relevance**: Smaller chunks = more precise retrieval
- **Cost**: Smaller inputs = lower API costs

### Example 2: Text Splitting

**Code**: [`code/02_splitting.py`](./code/02_splitting.py)  
**Run**: `python 07-documents-embeddings-semantic-search/code/02_splitting.py`

**Example code:**

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
[Long article about AI and machine learning...]
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,      # Target size in characters
    chunk_overlap=50,    # Overlap between chunks (preserves context)
)

docs = splitter.create_documents([text])

print(f"Split into {len(docs)} chunks")

for i, doc in enumerate(docs):
    print(f"\nChunk {i + 1}:")
    print(doc.page_content)
    print(f"Length: {len(doc.page_content)} characters")
```

### Chunk Size Trade-offs

| Small Chunks (200-500 chars) | Large Chunks (1000-2000 chars) |
|------------------------------|--------------------------------|
| ✅ More precise | ✅ More context |
| ✅ Better for specific questions | ✅ Better for complex topics |
| ❌ May lose context | ❌ Less precise matching |

---

## 🔢 Part 2: Embeddings

### What Are Embeddings?

Embeddings convert text into numerical vectors that capture semantic meaning:
- Similar concepts → Similar vectors
- "king" - "man" + "woman" ≈ "queen"

### Example 3: Creating Embeddings

**Code**: [`code/03_embeddings.py`](./code/03_embeddings.py)  
**Run**: `python 07-documents-embeddings-semantic-search/code/03_embeddings.py`

**Example code:**

```python
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Embed a single text
text = "LangChain is a framework for building AI applications"
vector = embeddings.embed_query(text)

print(f"Text: {text}")
print(f"Vector dimensions: {len(vector)}")
print(f"First 5 values: {vector[:5]}")

# Embed multiple texts
texts = [
    "LangChain is a framework for building AI applications",
    "Python is a programming language",
    "Building AI apps with LangChain is easy",
]

vectors = embeddings.embed_documents(texts)

print(f"\nEmbedded {len(vectors)} texts")
for i, (text, vec) in enumerate(zip(texts, vectors)):
    print(f"  {i}: {text[:40]}... → {len(vec)} dimensions")
```

---

## 🗄️ Part 3: Vector Stores

### What Are Vector Stores?

Vector stores are databases optimized for storing and searching embeddings:
- Store: Add documents with their embeddings
- Search: Find similar documents using vector similarity

### Example 4: Using FAISS Vector Store

**Code**: [`code/04_vector_store.py`](./code/04_vector_store.py)  
**Run**: `python 07-documents-embeddings-semantic-search/code/04_vector_store.py`

**Example code:**

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# Create sample documents
docs = [
    Document(page_content="LangChain is a framework for building AI applications"),
    Document(page_content="Python is a popular programming language"),
    Document(page_content="Agents can use tools to solve complex problems"),
    Document(page_content="Vector databases store embeddings for fast similarity search"),
    Document(page_content="RAG combines retrieval with generation for accurate answers"),
]

# Create embeddings
embeddings = OpenAIEmbeddings()

# Create vector store from documents
vector_store = FAISS.from_documents(docs, embeddings)
print(f"Created vector store with {len(docs)} documents")

# Perform similarity search
query = "How do I build AI applications?"
results = vector_store.similarity_search(query, k=2)

print(f"\nQuery: {query}")
print(f"\nTop {len(results)} results:")
for i, doc in enumerate(results):
    print(f"  {i + 1}. {doc.page_content}")
```

### Expected Output

```
Created vector store with 5 documents

Query: How do I build AI applications?

Top 2 results:
  1. LangChain is a framework for building AI applications
  2. RAG combines retrieval with generation for accurate answers
```

---

## 🔍 Part 4: Semantic Search

### Example 5: Complete Semantic Search System

**Code**: [`code/05_semantic_search.py`](./code/05_semantic_search.py)  
**Run**: `python 07-documents-embeddings-semantic-search/code/05_semantic_search.py`

**Example code:**

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

def build_search_system(docs_path: str):
    """Build a complete semantic search system."""
    
    # 1. Load documents
    loader = TextLoader(docs_path)
    raw_docs = loader.load()
    print(f"Loaded {len(raw_docs)} document(s)")
    
    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    docs = splitter.split_documents(raw_docs)
    print(f"Split into {len(docs)} chunks")
    
    # 3. Create vector store
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(docs, embeddings)
    print("Created vector store")
    
    return vector_store

def search(vector_store, query: str, k: int = 3):
    """Perform semantic search."""
    results = vector_store.similarity_search(query, k=k)
    
    print(f"\n🔍 Query: {query}")
    print(f"📄 Top {len(results)} results:\n")
    
    for i, doc in enumerate(results):
        print(f"{i + 1}. {doc.page_content[:100]}...")
        if doc.metadata:
            print(f"   Source: {doc.metadata.get('source', 'unknown')}")
        print()

def main():
    # Build search system
    vector_store = build_search_system("./data/knowledge_base.txt")
    
    # Interactive search
    queries = [
        "How do I use LangChain?",
        "What are embeddings?",
        "How do agents work?",
    ]
    
    for query in queries:
        search(vector_store, query)

if __name__ == "__main__":
    main()
```

---

## 🏷️ Document Metadata

Metadata helps you track document source, filter by category, and understand context:

```python
from langchain_core.documents import Document

doc = Document(
    page_content="LangChain is a framework...",
    metadata={
        "source": "langchain-guide.md",
        "category": "tutorial",
        "date": "2024-01-15",
        "author": "Tech Team",
    },
)
```

---

## 🎓 Key Takeaways

- **Document loaders** read various file formats into Documents
- **Text splitters** break documents into manageable chunks
- **Embeddings** convert text into numerical vectors
- **Vector stores** enable fast similarity search
- **Semantic search** finds content by meaning, not keywords
- **Metadata** helps organize and filter documents

---

## 📦 Dependencies

```bash
pip install langchain langchain-openai langchain-community faiss-cpu python-dotenv
```

---

## 🗺️ Navigation

[← Previous: MCP](../06-mcp/README.md) | [Back to Main](../README.md) | [Next: Agentic RAG Systems →](../08-agentic-rag-systems/README.md)

---

## 💬 Questions?

[![Azure AI Foundry Discord](https://img.shields.io/badge/Discord-Azure_AI_Foundry_Community_Discord-blue?style=for-the-badge&logo=discord&color=5865f2&logoColor=fff)](https://aka.ms/foundry/discord)
