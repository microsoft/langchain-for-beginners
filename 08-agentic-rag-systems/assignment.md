# Assignment: Building Agentic RAG Systems

## Overview

Practice building intelligent RAG systems where agents make decisions about when and how to search documents.

## Prerequisites

- Completed this [chapter](./README.md)
- Run all code examples in this chapter

---

## Challenge 1: Personal Knowledge Base Q&A 🧠

Build an agentic RAG system over your own documents.

### Requirements

- Load and embed your personal documents
- Create a retrieval tool for the agent
- Agent should decide when to search vs. answer directly
- Handle multi-step reasoning over documents
- Provide accurate answers with context

### Hints

```python
# Use langgraph's create_react_agent for agentic RAG
from langgraph.prebuilt import create_react_agent

# Create a tool for searching your knowledge base
@tool
def search_my_notes(query: str) -> str:
    """Search my personal knowledge base for information."""
    results = vector_store.similarity_search(query, k=3)
    return "\n\n".join(
        f"[{doc.metadata['title']}]: {doc.page_content}"
        for doc in results
    )

# Create agent with your retrieval tool
agent = create_react_agent(model, tools=[search_my_notes])

# Invoke with a question - agent decides when to search
response = agent.invoke({
    "messages": [HumanMessage(content="Your question here")],
})
```

---

## Bonus Challenge: Conversational RAG 💬

Extend your RAG system to maintain conversation history.

### Requirements

- Maintain conversation context
- Handle follow-up questions
- Reference previous answers
- Update context as conversation evolves

### Hints

```python
# Maintain conversation history as a list of messages
conversation_history: list[HumanMessage | AIMessage] = []

# Add user message
conversation_history.append(HumanMessage(content=user_input))

# Invoke agent with full history for context
response = agent.invoke({
    "messages": list(conversation_history),
})

# Add assistant response to history
agent_message = response["messages"][-1]
conversation_history.append(AIMessage(content=agent_message.content))
```

---

## Solutions

Solutions for all challenges are available in the [`solution/`](./solution/) folder.

- [`knowledge_base_rag.py`](./solution/knowledge_base_rag.py) - Challenge 1 solution
- [`conversational_rag.py`](./solution/conversational_rag.py) - Bonus Challenge solution

---

## Congratulations! 🎉

You've completed the LangChain for Beginners course! You now have the knowledge to:
- Build conversational AI systems
- Create autonomous agents
- Implement semantic search
- Build production-ready RAG applications

Keep building and exploring!
