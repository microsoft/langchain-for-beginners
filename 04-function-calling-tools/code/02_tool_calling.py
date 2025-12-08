"""
Chapter 4 Example 2: Binding and Invoking Tools

Run: python 04-function-calling-tools/code/02_tool_calling.py

🤖 Try asking GitHub Copilot Chat (https://github.com/features/copilot):
- "What's in the response.tool_calls list and how does it differ from response.content?"
- "Why does the LLM return structured tool calls instead of executing the function?"
"""

import json
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


@tool(args_schema=CalculatorInput)
def calculator(expression: str) -> str:
    """Perform mathematical calculations."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def main():
    print("🔗 Tool Calling Demo\n")
    print("=" * 80 + "\n")

    # Create model and bind tools
    model = ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )

    model_with_tools = model.bind_tools([calculator])

    print("🤖 Asking: What is 25 * 17?\n")

    # Invoke with a question
    response = model_with_tools.invoke("What is 25 * 17?")

    print("Response content:", response.content)
    print("\nTool calls:", json.dumps(
        [
            {"name": tc["name"], "args": tc["args"], "id": tc["id"]}
            for tc in response.tool_calls
        ] if response.tool_calls else [],
        indent=2,
    ))

    if response.tool_calls and len(response.tool_calls) > 0:
        print("\n" + "─" * 80)
        print("\n✅ The LLM generated a tool call!")
        print("\nTool name:", response.tool_calls[0]["name"])
        print("Arguments:", response.tool_calls[0]["args"])
        print("Tool call ID:", response.tool_calls[0]["id"])

    print("\n" + "=" * 80 + "\n")
    print("💡 Key Takeaways:")
    print("   • Use bind_tools() to make tools available")
    print("   • LLM generates tool calls with arguments")
    print("   • Tool calls include name, args, and ID")
    print("   • Your code executes the actual function")


if __name__ == "__main__":
    main()
