"""
Assignment Solution: Research Agent with create_agent()

Run: python 05-agents/solution/research_agent.py
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


class SearchInput(BaseModel):
    """Input for search tool."""

    query: str = Field(
        description="The search query, e.g., 'population of Tokyo' or 'capital of France'"
    )


class CalculatorInput(BaseModel):
    """Input for calculator tool."""

    expression: str = Field(
        description="The mathematical expression to evaluate, e.g., '14000000 * 2' or '(100 + 50) / 2'"
    )


@tool(args_schema=SearchInput)
def search(query: str) -> str:
    """Search for factual information on the web. Use this when you need to find facts,
    statistics, or general knowledge. Good for finding populations, capitals, distances,
    and other factual data."""
    # Simulated search results
    search_results = {
        "population of tokyo": "Tokyo has a population of approximately 14 million people in the city proper, and over 37 million in the greater metropolitan area.",
        "capital of france": "The capital of France is Paris.",
        "capital of japan": "The capital of Japan is Tokyo.",
        "population of new york": "New York City has a population of approximately 8.3 million people.",
        "distance london to paris": "The distance between London and Paris is approximately 343 kilometers.",
        "highest mountain": "Mount Everest is the highest mountain in the world at 8,849 meters (29,032 feet).",
    }

    query_lower = query.lower()

    # Find matching result
    for key, value in search_results.items():
        if key in query_lower or query_lower in key:
            return value

    return f'Search results for "{query}": No specific information found. This is a simulated search tool with limited data.'


@tool(args_schema=CalculatorInput)
def calculator(expression: str) -> str:
    """Perform mathematical calculations. Use this for arithmetic operations like
    addition, subtraction, multiplication, division, and more complex math expressions."""
    try:
        # Safe evaluation with restricted builtins
        allowed = {"abs": abs, "round": round, "min": min, "max": max, "pow": pow}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Error calculating {expression}: {e}"


def main():
    print("🔍 Research Agent using create_agent()\n")
    print("=" * 80 + "\n")

    # Create the model
    model = ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )

    # Create agent using create_agent() - handles ReAct loop automatically
    agent = create_agent(model, tools=[search, calculator])

    # Test queries
    queries = [
        "What is the population of Tokyo multiplied by 2?",
        "Search for the capital of France and tell me how many letters are in its name",
    ]

    for query in queries:
        print(f"👤 User: {query}\n")

        # Invoke the agent - it handles the ReAct loop internally
        response = agent.invoke({"messages": [HumanMessage(content=query)]})

        # Get the final answer (last message)
        last_message = response["messages"][-1]
        print(f"🤖 Agent: {last_message.content}\n")

        # Show which tools were used
        tool_calls = []
        for msg in response["messages"]:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                tool_calls.extend([tc["name"] for tc in msg.tool_calls])

        if tool_calls:
            unique_tools = list(set(tool_calls))
            print(f"📊 Tools used: {', '.join(unique_tools)}")
            print(f"   Total tool calls: {len(tool_calls)}\n")

        print("=" * 80 + "\n")

    print("💡 Key Concepts:")
    print("   • create_agent() handles the ReAct loop automatically")
    print("   • Agent decides which tools to use and when")
    print("   • Agent iterates until it has enough information")
    print("   • Much simpler than manual loop implementation")


if __name__ == "__main__":
    main()
