"""
Chapter 6 Assignment Solution: Challenge 2 - Multi-Tool Agent with MCP

This solution demonstrates:
- Combining MCP tools (from Context7) with custom tools (calculator)
- Creating an agent that can choose between MCP and custom tools
- Testing queries that require different tool types
- Demonstrating the flexibility of the agent pattern

Run: python 06-mcp/solution/multi_tool_agent.py
"""

import asyncio
import math
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()


# Custom calculator tool schema
class CalculatorInput(BaseModel):
    """Input schema for the calculator tool."""

    expression: str = Field(
        description="The mathematical expression to evaluate, e.g., '125 * 8' or '50 + 25'"
    )


# Custom calculator tool (not from MCP)
@tool(args_schema=CalculatorInput)
def calculator(expression: str) -> str:
    """
    Perform mathematical calculations. Use this for arithmetic operations like
    addition, subtraction, multiplication, division, and more complex expressions.
    Examples: '125 * 8', '50 + 25', '(10 + 5) * 2'
    """
    print(f"   [Calculator] Evaluating: {expression}")
    try:
        # Create a safe namespace with math functions
        safe_namespace = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
            "floor": math.floor,
            "ceil": math.ceil,
            "pi": math.pi,
            "e": math.e,
        }

        result = eval(expression, {"__builtins__": {}}, safe_namespace)
        return f"The result is: {result}"
    except Exception as e:
        return f'Error calculating "{expression}": {e}'


async def main():
    print("🎛️  Chapter 6 Assignment Solution: Challenge 2")
    print("=" * 60)
    print()

    # Step 1: Connect to Context7 MCP server
    print("📡 Connecting to Context7 MCP server...")
    client = MultiServerMCPClient(
        {
            "context7": {
                "transport": "streamable_http",
                "url": "https://mcp.context7.com/mcp",
            },
        }
    )

    try:
        # Step 2: Get MCP tools
        print("🔍 Fetching MCP tools from Context7...")
        mcp_tools = await client.get_tools()

        # Step 3: Combine MCP tools with custom tools
        print("🔧 Combining MCP tools with custom calculator tool...\n")
        all_tools = [*mcp_tools, calculator]

        print("📋 Available Tools:")
        print("\n   From Context7 (MCP):")
        for tool_item in mcp_tools:
            print(f"   • {tool_item.name}: {tool_item.description}")
        print("\n   Custom Tools:")
        print("   • calculator: Mathematical calculations")
        print()

        # Step 4: Create model
        model = ChatOpenAI(
            model=os.getenv("AI_MODEL", "gpt-4o-mini"),
            base_url=os.getenv("AI_ENDPOINT"),
            api_key=os.getenv("AI_API_KEY"),
        )

        # Step 5: Create agent with all tools
        print("🤖 Creating multi-tool agent...\n")
        agent = create_agent(model, all_tools)

        # Step 6: Test with different queries
        queries = [
            "What is 125 * 8?",
            "How do I use React hooks?",
            "Calculate 50 + 25",
        ]

        for query in queries:
            print(f"👤 User: {query}")

            response = await agent.ainvoke({"messages": [("human", query)]})
            last_message = response["messages"][-1]

            print(f"🤖 Agent: {last_message.content}")
            print()
            print("-" * 60)
            print()

        print("✅ Challenge 2 Complete!")
        print()
        print("💡 What Just Happened:")
        print("   • Math queries → Agent used the custom calculator tool")
        print("   • Documentation queries → Agent used Context7 MCP tools")
        print("   • Agent autonomously selected the right tool for each task")
        print("   • Same agent instance handled both MCP and custom tools!")
        print()
        print("🎯 Key Pattern:")
        print("   all_tools = [*mcp_tools, custom_tool]")
        print("   agent = create_agent(model, all_tools)")
        print()
        print("   This pattern lets you mix MCP tools (from external servers)")
        print("   with your own custom tools in a single agent!")
        print()

    except Exception as e:
        print(f"❌ Error: {e}")
        raise

    finally:
        # Clean up
        print("🔌 MCP client connection closed")


if __name__ == "__main__":
    asyncio.run(main())
