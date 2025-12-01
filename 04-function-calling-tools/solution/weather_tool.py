"""
Chapter 4 Assignment Solution: Weather Tool with Complete Execution Loop

Run: python 04-function-calling-tools/solution/weather_tool.py
"""

import os
from typing import Literal, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


class WeatherInput(BaseModel):
    """Input schema for weather tool."""

    city: str = Field(description="City name, e.g., 'Tokyo' or 'Paris'")
    units: Optional[Literal["celsius", "fahrenheit"]] = Field(
        default="fahrenheit",
        description="Temperature unit (default: fahrenheit)",
    )


@tool(args_schema=WeatherInput)
def get_weather(city: str, units: str = "fahrenheit") -> str:
    """Get current weather information for a city. Returns temperature and weather conditions.
    Use this when the user asks about weather, temperature, or conditions in a specific location."""
    # Simulated weather data for various cities
    weather_data = {
        "Tokyo": {"temp_f": 75, "temp_c": 24, "condition": "partly cloudy"},
        "Paris": {"temp_f": 64, "temp_c": 18, "condition": "sunny"},
        "London": {"temp_f": 59, "temp_c": 15, "condition": "rainy"},
        "New York": {"temp_f": 72, "temp_c": 22, "condition": "clear"},
        "Seattle": {"temp_f": 62, "temp_c": 17, "condition": "cloudy"},
        "Sydney": {"temp_f": 79, "temp_c": 26, "condition": "sunny"},
        "Mumbai": {"temp_f": 88, "temp_c": 31, "condition": "humid and hot"},
    }

    city_data = weather_data.get(city)

    if not city_data:
        available = ", ".join(weather_data.keys())
        return f"Weather data not available for {city}. Available cities: {available}"

    units = units or "fahrenheit"
    temp = city_data["temp_c"] if units == "celsius" else city_data["temp_f"]
    unit_symbol = "°C" if units == "celsius" else "°F"

    return f"Current weather in {city}: {temp}{unit_symbol}, {city_data['condition']}"


def main():
    print("⛅ Weather Tool - Complete Execution Loop\n")
    print("=" * 80 + "\n")

    model = ChatOpenAI(model=os.environ.get("AI_MODEL", "gpt-4o-mini"))

    model_with_tools = model.bind_tools([get_weather])

    # Test multiple queries
    queries = [
        "What's the weather in Tokyo?",
        "Tell me the temperature in Paris in celsius",
        "Is it raining in London?",
    ]

    for query in queries:
        print(f"User: {query}\n")

        # Step 1: Get tool call from LLM
        print("Step 1: LLM generates tool call...")
        response1 = model_with_tools.invoke([HumanMessage(content=query)])

        if not response1.tool_calls or len(response1.tool_calls) == 0:
            print("  No tool call generated - direct response")
            print(f"  Response: {response1.content}\n")
            print("─" * 80 + "\n")
            continue

        tool_call = response1.tool_calls[0]
        print(f"  Tool: {tool_call['name']}")
        print(f"  Args: {tool_call['args']}")
        print(f"  ID: {tool_call['id']}")

        # Step 2: Execute the tool
        print("\nStep 2: Executing tool...")
        tool_result = get_weather.invoke(tool_call["args"])
        print(f"  Result: {tool_result}")

        # Step 3: Send result back to LLM
        print("\nStep 3: Sending result back to LLM...")
        messages = [
            HumanMessage(content=query),
            AIMessage(
                content=str(response1.content),
                tool_calls=response1.tool_calls,
            ),
            ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"],
            ),
        ]

        final_response = model.invoke(messages)
        print(f"  Final answer: {final_response.content}\n")

        print("─" * 80 + "\n")

    print("=" * 80 + "\n")
    print("💡 Key Takeaways:")
    print("   • Tool uses Pydantic schema for type-safe parameters")
    print("   • Complete 3-step pattern: generate → execute → respond")
    print("   • Tool handles optional parameters (units)")
    print("   • LLM generates natural language from tool results")


if __name__ == "__main__":
    main()
