# Assignment: Documents, Embeddings & Semantic Search

## Overview

Practice building semantic search systems that understand meaning.

## Prerequisites

- Completed this [chapter](./README.md)
- Run all code examples in this chapter

---

## Challenge 1: Similarity Explorer 🔍

Build a tool that lets users explore how similarity scores change with different queries and documents.

### Requirements

- Create an interactive similarity explorer
- Allow users to add custom documents
- Calculate and display similarity scores between queries and documents
- Show how scores change with different phrasing

### Hints

```python
# Use OpenAIEmbeddings to create embeddings
from langchain_openai import OpenAIEmbeddings

# Use InMemoryVectorStore for storage and search
from langchain_core.vectorstores import InMemoryVectorStore

# Use similarity_search_with_score to get scores
results = vector_store.similarity_search_with_score(query, k=5)
```

---

## Challenge 2: Book Search System (Bonus) 📚

Build a semantic search system over a collection of book descriptions.

### Requirements

- Load at least 5 book descriptions with metadata (title, author, genre)
- Create embeddings and store in a vector database
- Implement semantic search that returns relevant books
- Include metadata in the results

### Hints

```python
# Use Document class with metadata
from langchain_core.documents import Document

doc = Document(
    page_content="Book description here...",
    metadata={"title": "Book Title", "author": "Author Name", "genre": "Fiction"}
)

# Access metadata in results
for doc in results:
    print(f"Title: {doc.metadata['title']}")
    print(f"Content: {doc.page_content}")
```

---

## Solutions

Solutions for all challenges are available in the [`solution/`](./solution/) folder.

- [`similarity_explorer.py`](./solution/similarity_explorer.py) - Challenge 1 solution
- [`book_search.py`](./solution/book_search.py) - Challenge 2 (Bonus) solution

---

## Next Steps

**[Building Agentic RAG Systems](../08-agentic-rag-systems/README.md)**
