"""
Chapter 6 Example 2: MCP with stdio Transport

This example shows how to use stdio transport to connect to an MCP server
running as a subprocess, communicating via standard input/output streams.

Comparison:
- Example 1: HTTP transport (network-based communication)
- Example 2: stdio transport (process-based communication)

Run: python 06-mcp/code/02_mcp_stdio_local.py
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

load_dotenv()

# Get the directory of this file for resolving server path
SCRIPT_DIR = Path(__file__).parent


async def main():
    print("🔧 Starting local MCP server via stdio...\n")

    # Path to the local calculator server
    server_path = SCRIPT_DIR / "servers" / "stdio_calculator_server.py"

    # Create MCP client with stdio transport - runs server as subprocess
    client = MultiServerMCPClient(
        {
            "localCalculator": {
                "transport": "stdio",
                "command": "python",
                "args": [str(server_path)],
            }
        }
    )

    try:
        # 1. Get tools from local MCP server
        print("📟 Connecting to stdio MCP server...")
        tools = await client.get_tools()

        print(f"✅ Connected! Retrieved {len(tools)} tools from local server:")
        for tool in tools:
            print(f"   • {tool.name}: {tool.description}")
        print()

        # 2. Create model
        model = ChatOpenAI(
            model=os.getenv("AI_MODEL"),
            base_url=os.getenv("AI_ENDPOINT"),
            api_key=os.getenv("AI_API_KEY"),
        )

        # 3. Create agent with stdio MCP tools
        agent = create_agent(model, tools)

        # 4. Test calculations
        print("🧮 Testing calculator tool...\n")

        math_query = "What is 15 * 23 + 100?"
        print(f"👤 User: {math_query}")

        math_response = await agent.ainvoke({"messages": [("human", math_query)]})
        math_result = math_response["messages"][-1]
        print(f"🤖 Agent: {math_result.content}\n")

        # 5. Test temperature conversion
        print("🌡️  Testing temperature conversion...\n")

        temp_query = "Convert 100 degrees Fahrenheit to Celsius"
        print(f"👤 User: {temp_query}")

        temp_response = await agent.ainvoke({"messages": [("human", temp_query)]})
        temp_result = temp_response["messages"][-1]
        print(f"🤖 Agent: {temp_result.content}\n")

        # 6. Test complex calculation
        print("🔢 Testing complex math...\n")

        complex_query = "Calculate the square root of 144 plus the sine of pi/2"
        print(f"👤 User: {complex_query}")

        complex_response = await agent.ainvoke({"messages": [("human", complex_query)]})
        complex_result = complex_response["messages"][-1]
        print(f"🤖 Agent: {complex_result.content}\n")

        print("💡 Key Concepts:")
        print("   • stdio transport runs MCP server as a subprocess")
        print("   • Communicates via standard input/output streams")
        print("   • Server runs as child process of the client")
        print("   • HTTP transport uses network-based communication")
        print("   • stdio transport uses process-based communication")
        print("   • Same agent code works with both transports!\n")

        print("📚 Transport Comparison:")
        print("   stdio:  Process communication via stdin/stdout")
        print("   HTTP:   Network communication via HTTP requests")
        print("   Choose based on your architecture needs!")

    except Exception as e:
        print(f"❌ Error with stdio MCP server: {e}")

    finally:
        # Note: Python MCP client handles cleanup automatically
        print("\n✅ MCP client connection closed")


if __name__ == "__main__":
    asyncio.run(main())
