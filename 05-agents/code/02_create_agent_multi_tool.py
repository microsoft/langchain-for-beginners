"""
Example 2: create_agent() with Multiple Tools

This example demonstrates how create_agent() automatically selects
the correct tool for each query from a set of available tools.

The agent will:
- Use the calculator for math questions
- Use the weather tool for weather queries
- Use the search tool for general information

Run: python 05-agents/code/02_create_agent_multi_tool.py

🤖 Try asking GitHub Copilot Chat (https://github.com/features/copilot):
- "How does the agent decide which tool to use?"
- "Can I prioritize certain tools over others?"
- "How do tool descriptions affect tool selection?"
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
    """Input for calculator."""

    expression: str = Field(description="The mathematical expression to evaluate")


class WeatherInput(BaseModel):
    """Input for weather."""

    city: str = Field(description="The name of the city to get weather for")


class SearchInput(BaseModel):
    """Input for search."""

    query: str = Field(description="The search query")


@tool(args_schema=CalculatorInput)
def calculator(expression: str) -> str:
    """Perform mathematical calculations. Use this for arithmetic operations."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"The result is: {result}"
    except Exception as e:
        return f"Error: {e}"


@tool(args_schema=WeatherInput)
def get_weather(city: str) -> str:
    """Get current weather information for a specific city."""
    # Simulated weather data
    weather = {
        "Seattle": "62°F, cloudy with a chance of rain",
        "Paris": "18°C, sunny and pleasant",
        "Tokyo": "24°C, rainy with occasional thunder",
        "New York": "70°F, partly cloudy",
        "London": "15°C, foggy with light drizzle",
    }

    city_weather = weather.get(city)
    if city_weather:
        return f"Current weather in {city}: {city_weather}"
    return f"Weather data unavailable for {city}"


@tool(args_schema=SearchInput)
def search(query: str) -> str:
    """Search for information on the web. Use this for general knowledge questions."""
    # Simulated search results
    search_results = {
        "langchain": "LangChain is a framework for building applications with large language models (LLMs). It provides tools, agents, chains, and memory systems to create sophisticated AI applications.",
        "typescript": "TypeScript is a strongly typed programming language that builds on JavaScript, giving you better tooling at any scale.",
        "javascript frameworks": "Popular JavaScript frameworks include React, Vue, Angular, Svelte, and Next.js for building modern web applications.",
    }

    # Find best match (simplified)
    query_lower = query.lower()
    for key, value in search_results.items():
        if key in query_lower:
            return f'Search results for "{query}": {value}'

    return f'Search results for "{query}": Found information about web development, programming, and related topics.'


def main():
    print("🎛️  Multi-Tool Agent with create_agent()\n")

    # Create model
    model = ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )

    # Create agent with all three tools
    agent = create_agent(model, tools=[calculator, get_weather, search])

    # Test with different queries - agent selects the right tool automatically
    queries = [
        "What is 50 * 25?",
        "What's the weather in Tokyo?",
        "Tell me about LangChain",
    ]

    for query in queries:
        print(f"👤 User: {query}")
        response = agent.invoke({"messages": [HumanMessage(content=query)]})
        last_message = response["messages"][-1]
        print(f"🤖 Agent: {last_message.content}\n")

    print("💡 What just happened:")
    print("   • The agent automatically selected the right tool for each query")
    print("   • Calculator for math (50 * 25)")
    print("   • Weather tool for Tokyo weather")
    print("   • Search tool for LangChain information")
    print("   • All with the same agent instance!\n")

    print("✅ Production Pattern:")
    print("   This is how you build real-world agents:")
    print("   1. Define your tools")
    print("   2. Pass them to create_agent()")
    print("   3. Let the agent handle tool selection and execution")


if __name__ == "__main__":
    main()
