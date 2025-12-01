"""
Chapter 4 Example 3: Complete Tool Execution Loop

Run: python 04-function-calling-tools/code/03_tool_execution.py

🤖 Try asking GitHub Copilot Chat (https://github.com/features/copilot):
- "Why do we need to send tool results back to the LLM in step 3?"
- "How would I handle errors that occur during tool execution?"
"""

import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


class WeatherInput(BaseModel):
    """Input for weather tool."""

    city: str = Field(description="City name")


@tool(args_schema=WeatherInput)
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    # Simulate API call
    temps = {"Seattle": 62, "Paris": 18, "Tokyo": 24, "London": 14, "Sydney": 26}
    temp = temps.get(city, 72)
    return f"Current temperature in {city}: {temp}°F, partly cloudy"


def main():
    print("🔄 Complete Tool Execution Loop\n")
    print("=" * 80 + "\n")

    model = ChatOpenAI(model=os.environ.get("AI_MODEL", "gpt-4o-mini"))

    model_with_tools = model.bind_tools([get_weather])

    query = "What's the weather in Seattle?"
    print(f"User: {query}\n")

    # ==========================================================================
    # STEP 1: LLM GENERATES TOOL CALL (Planning)
    # ==========================================================================
    # The LLM's role: Analyze the request and decide which tool to call
    # Important: The LLM does NOT execute anything - it just generates a plan!

    print("=== STEP 1: LLM GENERATES TOOL CALL ===")
    print("(The LLM's role: Planning - decides WHAT to do)\n")

    response1 = model_with_tools.invoke([HumanMessage(content=query)])

    if not response1.tool_calls or len(response1.tool_calls) == 0:
        print("No tool calls generated")
        return

    tool_call = response1.tool_calls[0]
    print("✅ LLM decided to call:", tool_call["name"])
    print("   With arguments:", tool_call["args"])
    print("   Tool call ID:", tool_call["id"])
    print("\n💡 Note: The LLM only DESCRIBED what to do - it didn't execute anything!\n")

    # ==========================================================================
    # STEP 2: YOUR CODE EXECUTES THE TOOL (Doing)
    # ==========================================================================
    # Your code's role: Actually execute the function and get real results
    # This is where the real work happens - API calls, database queries, etc.

    print("=== STEP 2: YOUR CODE EXECUTES THE TOOL ===")
    print("(Your code's role: Doing - actually performs the action)\n")

    tool_result = get_weather.invoke(tool_call["args"])
    print("✅ Tool executed successfully!")
    print("   Real result:", tool_result)
    print("\n💡 Note: This is where the actual API call/database query happens!\n")

    # ==========================================================================
    # STEP 3: SEND RESULTS BACK TO LLM (Communicating)
    # ==========================================================================
    # The LLM's role again: Receive results and formulate a natural response
    # The LLM converts raw data into human-friendly language

    print("=== STEP 3: SEND RESULTS BACK TO LLM ===")
    print("(The LLM's role: Communicating - converts data to natural language)\n")

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
    print("✅ LLM generated final response:")
    print("  ", final_response.content)

    print("\n" + "=" * 80 + "\n")
    print("🎓 Key Takeaways:")
    print("─" * 80)
    print("\n1. Three-Step Process:")
    print("   • Step 1: LLM generates tool call (Planning)")
    print("   • Step 2: Your code executes tool (Doing)")
    print("   • Step 3: LLM receives results (Communicating)")
    print("\n2. Separation of Concerns:")
    print("   • LLM handles: Understanding user intent + Natural language response")
    print("   • Your code handles: Actual execution + Security + Validation")
    print("\n3. Why This Matters:")
    print("   • Security: You control what actually gets executed")
    print("   • Flexibility: Switch implementations without retraining LLM")
    print("   • Reliability: Handle errors, retries, and edge cases")
    print("\n✅ The LLM never executes functions - it only describes what to do!")


if __name__ == "__main__":
    main()
