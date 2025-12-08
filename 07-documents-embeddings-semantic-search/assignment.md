# Assignment: Documents, Embeddings & Semantic Search

## Overview

Practice building semantic search systems that understand meaning. You'll create tools that demonstrate how embeddings capture semantic relationships.

## Prerequisites

- Completed this [chapter](./README.md)
- Run all code examples in this chapter
- Environment variables configured

---

## Challenge 1: Similarity Explorer 🔍

Build a tool that lets users explore how similarity scores change with different queries and documents.

### Requirements

- Create a similarity explorer with a diverse set of documents
- Allow users to test different search queries
- Calculate and display similarity scores between queries and documents
- Show the top results ranked by similarity score
- Demonstrate how semantically similar queries return similar results

### Hints

1. Start with your imports and Azure OpenAI setup:

    ```python
    import os
    from dotenv import load_dotenv
    from langchain_core.documents import Document
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_openai import AzureOpenAIEmbeddings
    ```

2. Create documents covering different topics:

    ```python
    docs = [
        Document(page_content="Machine learning models can recognize patterns in data"),
        Document(page_content="The recipe calls for flour, eggs, and butter"),
        Document(page_content="Python is a popular programming language for AI"),
        Document(page_content="The sunset painted the sky in shades of orange"),
        # Add more diverse documents...
    ]
    ```

3. Build the vector store from your documents:

    ```python
    vector_store = InMemoryVectorStore.from_documents(docs, embeddings)
    ```

4. Use `similarity_search_with_score` to get similarity scores:

    ```python
    results = vector_store.similarity_search_with_score(query, k=5)
    for doc, score in results:
        print(f"Score: {score:.4f} - {doc.page_content}")
    ```

5. Test with different queries to see how similarity changes:

    ```python
    queries = [
        "How does AI learn?",
        "What ingredients do I need for baking?",
        "Beautiful evening colors",
    ]
    ```

> **💡 TIP:** Notice how the similarity scores reflect semantic meaning! "How does AI learn?" should score highest with the machine learning document, even though the words are different.

---

## Challenge 2: Book Search System (Bonus) 📚

Build a semantic search system over a collection of book descriptions.

### Requirements

- Create at least 5 book descriptions with metadata (title, author, genre)
- Create embeddings and store in a vector store
- Implement semantic search that returns relevant books
- Display book metadata (title, author, genre) in results
- Show how the search finds books based on themes and concepts

### Hints

1. Create documents with rich metadata:

    ```python
    from langchain_core.documents import Document

    books = [
        Document(
            page_content="A young wizard discovers his magical powers and battles dark forces at a school of magic",
            metadata={"title": "Harry Potter", "author": "J.K. Rowling", "genre": "Fantasy"}
        ),
        Document(
            page_content="A hobbit embarks on an epic quest through Middle-earth to destroy a powerful ring",
            metadata={"title": "The Lord of the Rings", "author": "J.R.R. Tolkien", "genre": "Fantasy"}
        ),
        # Add more books with diverse genres and themes...
    ]
    ```

2. Build the vector store and search:

    ```python
    book_store = InMemoryVectorStore.from_documents(books, embeddings)
    results = book_store.similarity_search("adventure stories for kids", k=3)
    ```

3. Display results with metadata:

    ```python
    for doc in results:
        print(f"📖 {doc.metadata['title']} by {doc.metadata['author']}")
        print(f"   Genre: {doc.metadata['genre']}")
        print(f"   {doc.page_content[:100]}...")
    ```

4. Try semantic queries that don't match exact words:

    ```python
    queries = [
        "stories about magic and wizards",
        "epic journey adventures",
        "books about the future",
        "mystery and detective stories",
    ]
    ```

> **💡 TIP:** The search should find relevant books even when your query uses different words than the descriptions! For example, "stories about magic" should find both Harry Potter and Lord of the Rings.

---

## Submission Checklist

Before submitting, verify:

- [ ] Similarity Explorer shows diverse documents across different topics
- [ ] Similarity scores are displayed and ranked properly
- [ ] Different queries return semantically appropriate results
- [ ] (Bonus) Book Search includes at least 5 books with metadata
- [ ] (Bonus) Book metadata displays correctly in search results

---

## Solutions

Solutions for all challenges are available in the [`solution/`](./solution/) folder.

- [`similarity_explorer.py`](./solution/similarity_explorer.py) - Challenge 1 solution
- [`book_search.py`](./solution/book_search.py) - Challenge 2 (Bonus) solution

---

## Need Help?

- Review the [chapter README](./README.md) for concepts
- Check the code examples in [`code/`](./code/)
- Look at sample implementations in [`samples/`](./samples/)
- Ask questions in [Discord](https://aka.ms/foundry/discord)

---

## Next Steps

**[Building Agentic RAG Systems](../08-agentic-rag-systems/README.md)**
