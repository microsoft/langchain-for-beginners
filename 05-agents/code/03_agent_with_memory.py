"""
Example 3: Agent with Checkpointer for Memory

This example demonstrates using LangChain's checkpointer for agent memory.
The checkpointer allows the agent to maintain state across multiple invocations.

Use checkpointers for:
- Maintaining conversation history
- Enabling multi-turn conversations
- Persisting agent state between calls

Run: python 05-agents/code/03_agent_with_memory.py

🤖 Try asking GitHub Copilot Chat (https://github.com/features/copilot):
- "How does the checkpointer maintain conversation state?"
- "What's the difference between MemorySaver and other checkpointers?"
- "How can I persist agent memory to a database?"
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


class CalculatorInput(BaseModel):
    """Input for calculator."""

    expression: str = Field(description="The mathematical expression to evaluate")


class SearchInput(BaseModel):
    """Input for search."""

    query: str = Field(description="The search query")


@tool(args_schema=CalculatorInput)
def calculator(expression: str) -> str:
    """Perform mathematical calculations."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"The result is: {result}"
    except Exception as e:
        return f"Error: {e}"


@tool(args_schema=SearchInput)
def search(query: str) -> str:
    """Search for information."""
    # Simulate occasional failures for demonstration
    if "error" in query.lower():
        raise Exception("Search service temporarily unavailable")

    return f'Search results for "{query}": Found relevant information about {query}.'


def main():
    print("🔧 Agent with Memory (Checkpointer) Example\n")

    # Create the model
    model = ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )

    # Create a memory saver for conversation persistence
    memory = MemorySaver()

    # Create agent with checkpointer for memory
    agent = create_agent(
        model,
        tools=[calculator, search],
        checkpointer=memory,
    )

    # Configuration for this conversation thread
    config = {"configurable": {"thread_id": "user-123"}}

    print("Test 1: First calculation")
    print("─" * 60)
    query1 = "What is 25 * 8?"
    print(f"👤 User: {query1}\n")
    response1 = agent.invoke({"messages": [HumanMessage(content=query1)]}, config)
    last_message1 = response1["messages"][-1]
    print(f"🤖 Agent: {last_message1.content}\n\n")

    print("Test 2: Follow-up question (agent remembers context)")
    print("─" * 60)
    query2 = "Now multiply that result by 2"
    print(f"👤 User: {query2}\n")
    response2 = agent.invoke({"messages": [HumanMessage(content=query2)]}, config)
    last_message2 = response2["messages"][-1]
    print(f"🤖 Agent: {last_message2.content}\n\n")

    print("Test 3: Another follow-up")
    print("─" * 60)
    query3 = "What was my original calculation?"
    print(f"👤 User: {query3}\n")
    response3 = agent.invoke({"messages": [HumanMessage(content=query3)]}, config)
    last_message3 = response3["messages"][-1]
    print(f"🤖 Agent: {last_message3.content}\n\n")

    print("💡 Memory Benefits:")
    print("   • Maintains conversation context across turns")
    print("   • Agent remembers previous calculations and questions")
    print("   • Each thread_id has its own conversation history")
    print("   • Great for chat applications and multi-step workflows\n")

    print("✅ Production Use Cases:")
    print("   • Customer service chatbots")
    print("   • Personal assistants that remember preferences")
    print("   • Multi-step task completion")
    print("   • Interactive data analysis sessions")


if __name__ == "__main__":
    main()
