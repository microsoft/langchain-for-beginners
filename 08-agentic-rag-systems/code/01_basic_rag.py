"""
Basic RAG Example
Run: python 08-agentic-rag-systems/code/01_basic_rag.py

This example demonstrates a simple RAG (Retrieval-Augmented Generation) system.
"""

import os

from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

def main():
    # Sample documents about LangChain
    documents = [
        Document(page_content="LangChain is a framework for building applications with large language models."),
        Document(page_content="LangChain provides tools for prompt management, chains, and agents."),
        Document(page_content="RAG stands for Retrieval-Augmented Generation, combining search with LLM generation."),
        Document(page_content="Vector stores help with semantic search by storing document embeddings."),
    ]

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(
        model=os.getenv("AI_EMBEDDING_MODEL", "text-embedding-3-small"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )
    
    vector_store = FAISS.from_documents(documents, embeddings)
    retriever = vector_store.as_retriever()

    # Create the LLM
    model = ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )

    # Create RAG prompt
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Keep the answer concise.\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Create RAG chain
    question_answer_chain = create_stuff_documents_chain(model, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print("🔍 Basic RAG System Example\n")
    print("=" * 80)

    # Ask a question
    question = "What is RAG?"
    print(f"\n❓ Question: {question}")
    
    response = rag_chain.invoke({"input": question})
    
    print(f"\n✅ Answer: {response['answer']}")
    
    print("\n📚 Retrieved Context:")
    for i, doc in enumerate(response['context'], 1):
        print(f"   {i}. {doc.page_content}")

    print("\n" + "=" * 80)
    print("\n✅ RAG combines retrieval with generation!")
    print("💡 The AI answers based on retrieved documents")

if __name__ == "__main__":
    main()
