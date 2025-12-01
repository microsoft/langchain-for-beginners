"""
Chapter 5 Assignment Solution: Multi-Step Planning Agent

Run: python 05-agents/solution/planning_agent.py
"""

import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


# Tool 1: Search
class SearchInput(BaseModel):
    """Input for search."""

    query: str = Field(description="Search query")


@tool(args_schema=SearchInput)
def search(query: str) -> str:
    """Find factual information including populations, distances, capitals,
    and general knowledge. Use this first when you need facts."""
    search_results = {
        "population of tokyo": "Tokyo has a population of approximately 14 million",
        "population of new york": "New York City has a population of approximately 8.3 million",
        "distance london to paris": "The distance is approximately 343 kilometers",
        "capital of france": "Paris",
        "capital of japan": "Tokyo",
    }

    query_lower = query.lower()
    for key, value in search_results.items():
        if key in query_lower or query_lower in key:
            return value
    return f'No results found for "{query}"'


# Tool 2: Calculator
class CalculatorInput(BaseModel):
    """Input for calculator."""

    expression: str = Field(description="Math expression, e.g., '343 * 0.621371'")


@tool(args_schema=CalculatorInput)
def calculator(expression: str) -> str:
    """Perform mathematical calculations including arithmetic, percentages,
    and expressions. Use when you need to compute numbers."""
    try:
        allowed = {"abs": abs, "round": round, "min": min, "max": max, "pow": pow}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# Tool 3: Unit Converter
class UnitConverterInput(BaseModel):
    """Input for unit converter."""

    value: float = Field(description="The numeric value to convert")
    from_unit: str = Field(
        alias="from", description="Source unit, e.g., 'km', 'miles', 'USD'"
    )
    to_unit: str = Field(
        alias="to", description="Target unit, e.g., 'km', 'miles', 'EUR'"
    )


@tool(args_schema=UnitConverterInput)
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """Convert between units: kilometers to miles (and vice versa),
    USD to EUR (and vice versa). Use when you need to convert measurements or currencies."""
    conversions = {
        "km": {"miles": {"rate": 0.621371, "unit": "miles"}},
        "miles": {"km": {"rate": 1.60934, "unit": "kilometers"}},
        "usd": {"eur": {"rate": 0.92, "unit": "EUR"}},
        "eur": {"usd": {"rate": 1.09, "unit": "USD"}},
    }

    from_lower = from_unit.lower()
    to_lower = to_unit.lower()

    if from_lower not in conversions or to_lower not in conversions.get(from_lower, {}):
        return f"Error: Cannot convert from {from_unit} to {to_unit}. Available conversions: km↔miles, USD↔EUR"

    conversion = conversions[from_lower][to_lower]
    result = value * conversion["rate"]

    return f"{value} {from_unit} equals {result:.2f} {conversion['unit']}"


# Tool 4: Comparison
class ComparisonInput(BaseModel):
    """Input for comparison."""

    value1: float = Field(description="First value to compare")
    value2: float = Field(description="Second value to compare")
    operation: Literal["less", "greater", "equal", "difference"] = Field(
        description="Comparison operation to perform"
    )


@tool(args_schema=ComparisonInput)
def comparison_tool(
    value1: float, value2: float, operation: Literal["less", "greater", "equal", "difference"]
) -> str:
    """Compare two numeric values to determine if one is less than, greater than,
    equal to another, or calculate the difference.
    Use when you need to compare numbers or find differences."""
    if operation == "less":
        return (
            f"{value1} is less than {value2}"
            if value1 < value2
            else f"{value1} is not less than {value2}"
        )
    elif operation == "greater":
        return (
            f"{value1} is greater than {value2}"
            if value1 > value2
            else f"{value1} is not greater than {value2}"
        )
    elif operation == "equal":
        return (
            f"{value1} equals {value2}"
            if value1 == value2
            else f"{value1} does not equal {value2}"
        )
    elif operation == "difference":
        return f"The difference between {value1} and {value2} is {abs(value1 - value2)}"
    else:
        return f"Unknown operation: {operation}"


def main():
    print("🎯 Multi-Step Planning Agent using create_react_agent()\n")
    print("=" * 80 + "\n")

    # Create the model
    model = ChatOpenAI(model=os.environ.get("AI_MODEL", "gpt-4o-mini"))

    # Create agent using create_react_agent() - handles multi-tool selection automatically
    agent = create_react_agent(
        model, tools=[search, calculator, unit_converter, comparison_tool]
    )

    # Test queries requiring multiple steps
    queries = [
        "What's the distance from London to Paris in miles, and is that more or less than 500 miles?",
        "Find the city population of New York and Tokyo, calculate the difference, and tell me the result",
    ]

    for query in queries:
        print(f'\n🤔 Query: "{query}"\n')

        # Invoke the agent - it handles multi-step reasoning internally
        response = agent.invoke({"messages": [HumanMessage(content=query)]})

        # Get the final answer
        last_message = response["messages"][-1]
        print(f"\n✅ Agent: {last_message.content}\n")

        # Analyze which tools were used
        tool_calls = []
        for msg in response["messages"]:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                tool_calls.extend([tc["name"] for tc in msg.tool_calls])

        if tool_calls:
            unique_tools = list(set(tool_calls))
            print("─" * 80)
            print("📊 Agent Summary:")
            print(f"   • Tools used: {', '.join(unique_tools)}")
            print(f"   • Total tool calls: {len(tool_calls)}")
            print("   • Query solved successfully!")

        print("\n" + "=" * 80 + "\n")

    print("💡 Key Concepts:")
    print("   • create_react_agent() handles multi-step reasoning automatically")
    print("   • Agent chains multiple tools together")
    print("   • Each tool call builds on previous results")
    print("   • Clear descriptions help agent pick right tool")
    print("   • Complex queries are broken down into manageable steps")


if __name__ == "__main__":
    main()
