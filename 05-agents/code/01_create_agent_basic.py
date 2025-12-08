"""
Example 1: Using create_agent() (Recommended Approach)

This example demonstrates building an agent using create_agent(),
the recommended approach with LangChain.
For comparison, see samples/basic_agent_manual_loop.py which shows
manual ReAct loop implementation.

Key Benefits of create_agent():
- Handles the ReAct loop automatically
- Less boilerplate code
- Production-ready error handling built-in
- Cleaner, more maintainable

Run: python 05-agents/code/01_create_agent_basic.py

🤖 Try asking GitHub Copilot Chat (https://github.com/features/copilot):
- "What does create_agent() do under the hood?"
- "How does create_agent() handle iteration limits and prevent infinite loops?"
- "How can I add custom error handling to my agent?"
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


class CalculatorInput(BaseModel):
    """Input for calculator tool."""

    expression: str = Field(
        description="The mathematical expression to evaluate (e.g., '25 * 8')"
    )


@tool(args_schema=CalculatorInput)
def calculator(expression: str) -> str:
    """A calculator that can perform basic arithmetic operations.
    Use this when you need to calculate mathematical expressions."""
    try:
        # Use Python's eval with restricted builtins for safer evaluation
        allowed_names = {"abs": abs, "round": round, "min": min, "max": max, "pow": pow}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def main():
    print("🤖 Agent with create_agent() Example\n")

    # Create the model
    model = ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )

    # Create agent using create_agent() - that's it!
    agent = create_agent(model, tools=[calculator])

    # Use the agent
    query = "What is 125 * 8?"
    print(f"👤 User: {query}\n")

    # create_agent() returns a LangChain agent that expects messages array
    response = agent.invoke({"messages": [HumanMessage(content=query)]})

    # The response contains the full state, including all messages
    # Get the last message which is the agent's final answer
    last_message = response["messages"][-1]
    print(f"🤖 Agent: {last_message.content}\n")

    print("💡 Key Differences from Manual Loop:")
    print("   • create_agent() handles the ReAct loop automatically")
    print("   • Less code to write")
    print("   • Production-ready error handling built-in")
    print("   • Same result, simpler API\n")

    print("✅ Under the hood:")
    print(
        "   create_agent() implements the ReAct pattern (Thought → Action → Observation)"
    )
    print("   and handles all the boilerplate for you.")


if __name__ == "__main__":
    main()
