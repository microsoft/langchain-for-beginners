"""
Chapter 4 Assignment Solution: Multi-Tool Travel Assistant

Run: python 04-function-calling-tools/solution/travel_assistant.py
"""

import os
from datetime import datetime, timezone
from typing import Literal, Optional

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


# Tool 1: Currency Converter
class CurrencyInput(BaseModel):
    """Input for currency converter."""

    amount: float = Field(description="The amount to convert")
    from_currency: str = Field(
        description="Source currency code (e.g., 'USD', 'EUR', 'GBP')",
    )
    to_currency: str = Field(
        description="Target currency code (e.g., 'USD', 'EUR', 'GBP')",
    )


@tool(args_schema=CurrencyInput)
def currency_converter(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert amounts between different currencies (USD, EUR, GBP, JPY, AUD, CAD).
    Use this when the user wants to convert money from one currency to another
    or asks about exchange rates."""
    # Simulated exchange rates (relative to USD)
    rates = {
        "USD": 1.0,
        "EUR": 0.92,
        "GBP": 0.79,
        "JPY": 149.5,
        "AUD": 1.53,
        "CAD": 1.36,
    }

    from_rate = rates.get(from_currency.upper())
    to_rate = rates.get(to_currency.upper())

    if not from_rate:
        supported = ", ".join(rates.keys())
        return f"Error: Unknown currency '{from_currency}'. Supported currencies: {supported}"

    if not to_rate:
        supported = ", ".join(rates.keys())
        return f"Error: Unknown currency '{to_currency}'. Supported currencies: {supported}"

    # Convert to USD first, then to target currency
    amount_in_usd = amount / from_rate
    result = amount_in_usd * to_rate

    return f"{amount} {from_currency.upper()} equals approximately {result:.2f} {to_currency.upper()}"


# Tool 2: Distance Calculator
class DistanceInput(BaseModel):
    """Input for distance calculator."""

    from_city: str = Field(
        description="Starting city name, e.g., 'New York' or 'Paris'",
    )
    to_city: str = Field(
        description="Destination city name, e.g., 'London' or 'Tokyo'",
    )
    units: Optional[Literal["miles", "kilometers"]] = Field(
        default="kilometers",
        description="Distance unit (default: kilometers)",
    )


@tool(args_schema=DistanceInput)
def distance_calculator(
    from_city: str, to_city: str, units: str = "kilometers"
) -> str:
    """Calculate the distance between two cities in miles or kilometers.
    Use this when the user asks about distance between locations,
    how far apart cities are, or travel distances."""
    # Simulated distances between major cities (in kilometers)
    distances = {
        "New York": {"London": 5585, "Paris": 5837, "Tokyo": 10850, "Sydney": 15993},
        "London": {"New York": 5585, "Paris": 344, "Tokyo": 9562, "Sydney": 17015},
        "Paris": {"New York": 5837, "London": 344, "Tokyo": 9714, "Rome": 1430},
        "Tokyo": {"New York": 10850, "London": 9562, "Paris": 9714, "Sydney": 7823},
        "Sydney": {"New York": 15993, "London": 17015, "Tokyo": 7823, "Paris": 16965},
        "Rome": {"Paris": 1430, "London": 1434, "New York": 6896, "Tokyo": 9853},
    }

    if from_city not in distances:
        available = ", ".join(distances.keys())
        return f"Error: Unknown city '{from_city}'. Available cities: {available}"

    distance_km = distances[from_city].get(to_city)

    if not distance_km:
        available = ", ".join(distances[from_city].keys())
        return f"Error: Distance not available between {from_city} and {to_city}. Available destinations from {from_city}: {available}"

    units = units or "kilometers"
    if units == "miles":
        distance = int(distance_km * 0.621371)
        unit = "miles"
    else:
        distance = distance_km
        unit = "kilometers"

    return f"The distance from {from_city} to {to_city} is approximately {distance} {unit}"


# Tool 3: Time Zone Tool
class TimeZoneInput(BaseModel):
    """Input for time zone tool."""

    city: str = Field(description="City name to get time for, e.g., 'Tokyo' or 'New York'")


@tool(args_schema=TimeZoneInput)
def time_zone_tool(city: str) -> str:
    """Get the current time in a specific city and its time zone information.
    Use this when the user asks what time it is somewhere,
    about time zones, or time differences between locations."""
    # Simulated time zones (UTC offset in hours)
    time_zones = {
        "New York": {"offset": -5, "name": "EST"},
        "London": {"offset": 0, "name": "GMT"},
        "Paris": {"offset": 1, "name": "CET"},
        "Tokyo": {"offset": 9, "name": "JST"},
        "Sydney": {"offset": 10, "name": "AEST"},
        "Seattle": {"offset": -8, "name": "PST"},
        "Mumbai": {"offset": 5.5, "name": "IST"},
    }

    city_tz = time_zones.get(city)

    if not city_tz:
        available = ", ".join(time_zones.keys())
        return f"Error: Unknown city '{city}'. Available cities: {available}"

    # Get current UTC time
    now = datetime.now(timezone.utc)
    utc_hour = now.hour
    utc_minute = now.minute

    # Calculate city time
    city_hour = int((utc_hour + city_tz["offset"] + 24) % 24)
    formatted_time = f"{city_hour:02d}:{utc_minute:02d}"

    offset = city_tz["offset"]
    offset_str = f"+{offset}" if offset >= 0 else str(offset)

    return f"Current time in {city}: {formatted_time} {city_tz['name']} (UTC{offset_str})"


def main():
    print("🌍 Multi-Tool Travel Assistant\n")
    print("=" * 80 + "\n")

    model = ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )

    model_with_tools = model.bind_tools([
        currency_converter,
        distance_calculator,
        time_zone_tool,
    ])

    # Map tool names to functions
    tools_map = {
        "currency_converter": currency_converter,
        "distance_calculator": distance_calculator,
        "time_zone_tool": time_zone_tool,
    }

    # Test queries for each tool
    queries = [
        "Convert 100 USD to EUR",
        "What's the distance between New York and London?",
        "What time is it in Tokyo right now?",
        "How many miles from Paris to Rome?",
        "Convert 50 GBP to JPY",
    ]

    for query in queries:
        print(f"\nQuery: \"{query}\"")

        response = model_with_tools.invoke(query)

        if response.tool_calls and len(response.tool_calls) > 0:
            tool_call = response.tool_calls[0]
            print(f"  → LLM chose: {tool_call['name']}")
            print(f"  → Args: {tool_call['args']}")

            # Execute the tool
            tool_fn = tools_map.get(tool_call["name"])
            if tool_fn:
                tool_result = tool_fn.invoke(tool_call["args"])
                print(f"  → Result: {tool_result}")
            else:
                print("  → Unknown tool")
        else:
            print(f"  → Direct response: {response.content}")

        print("─" * 80)

    print("\n" + "=" * 80 + "\n")
    print("💡 Key Takeaways:")
    print("   • LLM automatically selects the right tool")
    print("   • Clear descriptions help tool selection")
    print("   • Each tool handles its domain (currency, distance, time)")
    print("   • Error handling provides helpful messages")


if __name__ == "__main__":
    main()
