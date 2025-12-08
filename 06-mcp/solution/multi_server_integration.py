"""
Assignment Solution: Challenge 3 - Multi-Server Integration

This solution demonstrates:
- Connecting to multiple MCP servers simultaneously
- Combining tools from different servers (Context7 + Local Calculator)
- Creating an agent that uses tools from all connected servers
- Agent intelligently selecting tools regardless of their source

Run: python 06-mcp/solution/multi_server_integration.py
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

load_dotenv()

# Get the directory of this file for resolving local server path
SCRIPT_DIR = Path(__file__).parent


async def main():
    print("🌐 Assignment Solution: Challenge 3")
    print("=" * 60)
    print()

    # Step 1: Connect to multiple MCP servers simultaneously
    print("📡 Connecting to multiple MCP servers...")
    print("   • Context7 (HTTP, remote documentation)")
    print("   • Local Calculator (stdio, subprocess)\n")

    # Path to the local calculator server
    server_path = SCRIPT_DIR.parent / "code" / "servers" / "stdio_calculator_server.py"

    client = MultiServerMCPClient(
        {
            # Server 1: Context7 for documentation (remote, HTTP transport)
            "context7": {
                "transport": "streamable_http",
                "url": "https://mcp.context7.com/mcp",
            },
            # Server 2: Local Calculator for math (local, stdio transport)
            "calculator": {
                "transport": "stdio",
                "command": "python",
                "args": [str(server_path)],
            },
        }
    )

    try:
        # Step 2: Get tools from ALL connected servers
        print("🔍 Fetching tools from all servers...\n")
        all_tools = await client.get_tools()

        # Step 3: Display available tools organized by server
        print("📋 Available Tools:\n")

        # Context7 tools (documentation-related)
        context7_tools = [
            t for t in all_tools if "library" in t.name or "resolve" in t.name
        ]
        print("   From context7:")
        for tool in context7_tools:
            print(f"   • {tool.name}")

        # Calculator tools (math-related)
        calc_tools = [
            t for t in all_tools if t.name in ("calculate", "convert_temperature")
        ]
        print("\n   From calculator:")
        for tool in calc_tools:
            print(f"   • {tool.name}")
        print()

        # Step 4: Create model
        model = ChatOpenAI(
            model=os.getenv("AI_MODEL", "gpt-5-mini"),
            base_url=os.getenv("AI_ENDPOINT"),
            api_key=os.getenv("AI_API_KEY"),
        )

        # Step 5: Create agent with tools from ALL servers
        print("🤖 Creating multi-server agent...\n")
        agent = create_agent(model, all_tools)  # Tools from multiple servers!

        # Step 6: Test queries that use different servers
        print("Testing agent with queries that use different servers:\n")
        print("-" * 60)
        print()

        # Test 1: Use calculator (stdio server)
        math_query = "What is 144 divided by 12?"
        print(f"👤 User: {math_query}")

        math_response = await agent.ainvoke({"messages": [("human", math_query)]})
        print(f"🤖 Agent: {math_response['messages'][-1].content}")
        print()
        print("-" * 60)
        print()

        # Test 2: Use Context7 (HTTP server)
        docs_query = "How do I use Python type hints?"
        print(f"👤 User: {docs_query}")

        docs_response = await agent.ainvoke({"messages": [("human", docs_query)]})
        print(f"🤖 Agent: {docs_response['messages'][-1].content}")
        print()
        print("-" * 60)
        print()

        # Test 3: Query that could use both servers
        combined_query = (
            "Calculate 50 * 2, then look up documentation about that number"
        )
        print(f"👤 User: {combined_query}")

        combined_response = await agent.ainvoke({"messages": [("human", combined_query)]})
        print(f"🤖 Agent: {combined_response['messages'][-1].content}")
        print()
        print("-" * 60)
        print()

        print("✅ Challenge 3 Complete!")
        print()
        print("💡 What Just Happened:")
        print("   • Connected to 2 MCP servers with DIFFERENT transports")
        print("   • Context7 uses HTTP (network-based communication)")
        print("   • Calculator uses stdio (process-based communication)")
        print("   • Agent received tools from BOTH servers seamlessly")
        print("   • Agent autonomously chose the right tool for each query")
        print("   • Same agent code worked with tools from different sources!")
        print()
        print("🎯 Key Pattern:")
        print("   client = MultiServerMCPClient({")
        print('     "server1": {"transport": "streamable_http", "url": "..."},')
        print('     "server2": {"transport": "stdio", "command": "...", "args": [...]}')
        print("   })")
        print("   tools = await client.get_tools()  # All tools!")
        print("   agent = create_agent(model, tools)")
        print()
        print("🚀 Scaling Up:")
        print("   You can connect to dozens of MCP servers:")
        print("   • GitHub for code repositories")
        print("   • Slack for team communication")
        print("   • Databases for data access")
        print("   • Internal tools specific to your organization")
        print("   • All available through one unified agent interface!")
        print()
        print("📖 MCP Registry:")
        print("   Find more MCP servers at: https://github.com/mcp")
        print()

    except Exception as e:
        print(f"❌ Error: {e}")
        print()
        print("💡 Troubleshooting:")
        print("   • Ensure you have internet connection (for Context7)")
        print("   • Check that the local calculator server path is correct")
        print("   • Verify .env file has AI_MODEL, AI_ENDPOINT, and AI_API_KEY")
        print("   • Try running the individual examples first (01, 02, 03)")
        raise

    finally:
        # Clean up - close ALL MCP connections
        print("🔌 All MCP server connections closed")


if __name__ == "__main__":
    asyncio.run(main())
