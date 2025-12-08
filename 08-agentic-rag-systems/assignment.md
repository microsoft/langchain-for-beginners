# Assignment: Building Agentic RAG Systems

## Overview

Practice building modern Agentic RAG systems where AI agents intelligently decide when to search your documents versus answering from general knowledge, combining retrieval with autonomous decision-making.

## Prerequisites

- Completed this [chapter](./README.md)
- Run all code examples in this chapter
- Understand agentic RAG architecture
- Familiar with agents from [Getting Started with Agents](../05-agents/README.md)

---

## Challenge: Personal Knowledge Base Q&A 📚

**Goal**: Build an agentic RAG system over your own documents where the agent decides when to search.

**Tasks**:

1. Create `knowledge_base_rag.py` in the `08-agentic-rag-systems/solution/` folder
2. Gather 5-10 documents about a topic you know well:
   - Personal notes
   - Blog articles you've written
   - Documentation you've created
   - Or use sample text about a hobby/interest
3. Build an agentic RAG system that:
   - Loads and chunks the documents into a vector store
   - Creates a retrieval tool for the agent
   - Uses `create_agent()` to build an autonomous agent
   - Agent decides when to search vs answer directly
4. Test with 5+ questions - mix of general knowledge and document-specific questions

**Success Criteria**:

- Loads documents successfully
- Agent answers general questions without searching
- Agent uses retrieval tool for document-specific questions
- Provides accurate answers with intelligent decision-making
- Handles questions not in the knowledge base gracefully

**Hints**:

```python
# 1. Import required modules
from langchain.agents import create_agent
from langchain_openai import AzureOpenAIEmbeddings, ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

# 2. Create AzureOpenAIEmbeddings and ChatOpenAI instances

# 3. Create an array of Document objects:
#    - Use your own content as page_content
#    - Add metadata (title, source, etc.)

# 4. Create an InMemoryVectorStore from your documents

# 5. Create a retrieval tool using @tool decorator:
#    - Define function that searches vector store
#    - Provide clear name and description
#    - Format results with source attribution

@tool
def search_my_notes(query: str) -> str:
    """Search my personal knowledge base for information."""
    results = vector_store.similarity_search(query, k=3)
    return "\n\n".join(
        f"[{doc.metadata['title']}]: {doc.page_content}"
        for doc in results
    )

# 6. Create agent with create_agent():
#    - Pass model and tools list
#    - Add system_prompt for context
#    - Agent will decide when to use retrieval tool

agent = create_agent(
    model,
    tools=[search_my_notes],
    system_prompt="You are a helpful assistant with access to my knowledge base..."
)

# 7. Test with questions that demonstrate agent decision-making:
#    - General knowledge (agent answers directly)
#    - Document-specific (agent searches)
#    - Questions not in docs (agent may search but won't find)

response = agent.invoke({
    "messages": [HumanMessage(content="Your question here")],
})
```

> [!TIP]
> **🤖 Get help from [GitHub Copilot](../docs/copilot.md):** If you need assistance with this challenge, open this file in your editor and use the [Challenge Tutor agent](../docs/copilot.md#challenge-tutor-agent) to get personalized help and explanations.

---

## Bonus Challenge: Conversational Agentic RAG 💬

**Goal**: Build an agentic RAG system that maintains conversation history.

**Tasks**:

1. Create `conversational_rag.py` in the `08-agentic-rag-systems/solution/` folder
2. Combine agentic RAG with conversation memory
3. Allow follow-up questions that reference previous context:

   ```text
   User: "What is Python?"
   Agent: "Python is..."
   User: "What are its main benefits?" ← Agent understands "its" refers to Python
   ```

4. Implement conversation history management
5. Add interactive CLI for multi-turn conversations
6. Add option to start new conversation

**Success Criteria**:

- Maintains conversation context across multiple turns
- Agent handles follow-up questions correctly
- Agent decides when to search based on conversation history
- Clear indication of conversation state
- Option to reset conversation

**Hints**:

```python
# 1. Create retrieval tool as in Challenge 1

# 2. Create agent with create_agent()

# 3. Initialize empty message history list
conversation_history: list[HumanMessage | AIMessage] = []

# 4. For each user question:
#    - Add new HumanMessage with user input to history
#    - Invoke agent with full message history
#    - Display agent's response
#    - Add agent's response to history
#    - Continue conversation loop

conversation_history.append(HumanMessage(content=user_input))

response = agent.invoke({
    "messages": list(conversation_history),
})

agent_message = response["messages"][-1]
conversation_history.append(AIMessage(content=agent_message.content))

# 5. Handle special commands:
#    - "exit" or "quit" to end conversation
#    - "reset" to clear history and start fresh

# 6. The agent will autonomously:
#    - Understand context from conversation history
#    - Decide when to search documents
#    - Answer follow-up questions intelligently
```

---

## Submission Checklist

Before completing this chapter:

- [ ] Challenge: Personal knowledge base agentic RAG works
- [ ] Agent demonstrates intelligent decision-making (searches when needed, answers directly when possible)
- [ ] Bonus: Conversational agentic RAG maintains context (optional)

---

## Solutions

Solutions available in [`solution/`](./solution/) folder. Try on your own first!

- [`knowledge_base_rag.py`](./solution/knowledge_base_rag.py) - Challenge solution
- [`conversational_rag.py`](./solution/conversational_rag.py) - Bonus Challenge solution

**Note**: The provided solutions use the modern agentic RAG approach with `create_agent()`. For comparison with traditional RAG patterns, see the code examples in the [`code/`](./code/) folder.

**Additional Examples**: Check out the [`samples/`](./samples/) folder for more examples including citation-based agentic RAG, multi-source agentic RAG, and hybrid search techniques!

---

## Need Help?

- **Agentic RAG basics**: Review `02_agentic_rag.py`
- **Agent fundamentals**: Review [Getting Started with Agents](../05-agents/README.md)
- **Retrieval tools**: Check `citation_rag.py` in samples
- **Any question**: Use the [Challenge Tutor agent](../docs/copilot.md#challenge-tutor-agent) in GitHub Copilot
- **Still stuck**: Join our [Discord community](https://aka.ms/foundry/discord)

---

## Next Steps

Congratulations! You've completed the LangChain for Beginners course! 🎉

You now know how to:

- ✅ Work with LLMs using prompts and message arrays
- ✅ Manage conversation history and state
- ✅ Create and use tools with function calling
- ✅ Build autonomous agents that make decisions
- ✅ Process documents and perform semantic search
- ✅ Build agentic RAG systems that intelligently retrieve information

### Continue Learning

- Explore [LangChain Python Documentation](https://python.langchain.com/)
- Build your own agentic applications
- Experiment with different vector stores (Pinecone, Chroma, Weaviate)
- Try extended agent patterns and multi-agent systems
- Integrate MCP servers for extended capabilities
- Deploy your agents to production

---

Excellent work building agentic RAG systems! 🎉

Keep building and exploring!
