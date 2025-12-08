"""
Chapter 4 Example 4: Multiple Tools

Run: python 04-function-calling-tools/code/04_multiple_tools.py

🤖 Try asking GitHub Copilot Chat (https://github.com/features/copilot):
- "How does the LLM decide which tool to use for each query?"
- "Can I prioritize certain tools over others by adjusting their descriptions?"
"""

import os

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


class CalculatorInput(BaseModel):
    """Input for calculator."""

    expression: str = Field(description="Math expression to evaluate")


class SearchInput(BaseModel):
    """Input for search."""

    query: str = Field(description="Search query")


class WeatherInput(BaseModel):
    """Input for weather."""

    city: str = Field(description="City name")


@tool(args_schema=CalculatorInput)
def calculator(expression: str) -> str:
    """Perform mathematical calculations."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool(args_schema=SearchInput)
def search(query: str) -> str:
    """Search for factual information."""
    results = {
        "capital of france": "Paris",
        "population of tokyo": "14 million",
        "who created javascript": "Brendan Eich",
    }
    return results.get(query.lower(), "No results found")


@tool(args_schema=WeatherInput)
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Weather in {city}: 72°F, sunny"


def main():
    print("🎛️ Multiple Tools Demo\n")
    print("=" * 80 + "\n")

    model = ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )

    model_with_tools = model.bind_tools([calculator, search, get_weather])

    queries = [
        "What is 125 * 8?",
        "What's the capital of France?",
        "What's the weather in Tokyo?",
    ]

    for query in queries:
        print(f"\nQuery: \"{query}\"")

        response = model_with_tools.invoke(query)

        if response.tool_calls and len(response.tool_calls) > 0:
            tool_call = response.tool_calls[0]
            print(f"  ✓ Chose tool: {tool_call['name']}")
            print(f"  ✓ Args: {tool_call['args']}")
        else:
            print("  ✗ No tool call generated")

        print("─" * 80)

    print("\n" + "=" * 80 + "\n")
    print("💡 Key Takeaways:")
    print("   • LLMs automatically choose the right tool")
    print("   • Clear descriptions help with tool selection")
    print("   • Multiple tools enable complex capabilities")


if __name__ == "__main__":
    main()
