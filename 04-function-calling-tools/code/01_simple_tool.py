"""
Chapter 4 Example 1: Simple Calculator Tool

Run: python 04-function-calling-tools/code/01_simple_tool.py

🤖 Try asking GitHub Copilot Chat (https://github.com/features/copilot):
- "Why do we need to sanitize the input expression before evaluating it?"
- "How does the Pydantic schema help with type safety in this calculator tool?"
"""

import json
import os

from dotenv import load_dotenv
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


# Define input schema with Pydantic
class CalculatorInput(BaseModel):
    """Input schema for calculator tool."""

    expression: str = Field(
        description="The mathematical expression to evaluate, e.g., '25 * 4'"
    )


# Define calculator tool using @tool decorator
@tool(args_schema=CalculatorInput)
def calculator(expression: str) -> str:
    """Useful for performing mathematical calculations. Use this when you need to compute numbers."""
    # Use Python's eval with restricted builtins for safer evaluation
    # Note: For production, use a proper math library like simpleeval or restrict further
    try:
        # Allow only safe mathematical operations
        allowed_names = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
        }
        # Evaluate the expression with restricted builtins
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"The result is: {result}"
    except Exception as error:
        return f"Error evaluating expression: {error}"


def main():
    print("🧮 Simple Calculator Tool Demo\n")
    print("=" * 80 + "\n")

    print("Tool Name:", calculator.name)
    print("Description:", calculator.description)
    print("\nSchema:", json.dumps(calculator.args_schema.model_json_schema(), indent=2))

    print("\n" + "=" * 80 + "\n")

    # Test the tool directly
    test_expressions = [
        "25 * 17",
        "(100 + 50) / 2",
        "pow(12, 2)",  # 12 squared
    ]

    for expr in test_expressions:
        print(f"\nExpression: {expr}")
        result = calculator.invoke({"expression": expr})
        print(f"Result: {result}")

    print("\n" + "=" * 80 + "\n")
    print("✅ Tool created successfully!")
    print("\n💡 Key Takeaways:")
    print("   • Tools are created with @tool decorator")
    print("   • Pydantic models define parameter schemas")
    print("   • Descriptions help LLMs understand when to use tools")


if __name__ == "__main__":
    main()
