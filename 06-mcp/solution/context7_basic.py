"""
Chapter 6 Assignment Solution: Challenge 1 - Connect to Context7 MCP Server

This solution demonstrates:
- Connecting to the Context7 MCP server using HTTP transport
- Listing available tools from the server
- Creating an agent that uses MCP tools
- Querying documentation from Context7

Run: python 06-mcp/solution/context7_basic.py
"""

import asyncio
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

load_dotenv()


async def main():
    print("🔧 Chapter 6 Assignment Solution: Challenge 1")
    print("=" * 60)
    print()

    # Step 1: Create MCP client and connect to Context7
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
        # Step 2: Get available tools from Context7
        print("🔍 Fetching available tools...\n")
        tools = await client.get_tools()

        print("🔧 Available Tools from Context7:")
        for tool in tools:
            print(f"   • {tool.name}: {tool.description}")
        print()

        # Step 3: Create the AI model
        model = ChatOpenAI(
            model=os.getenv("AI_MODEL", "gpt-4o-mini"),
            base_url=os.getenv("AI_ENDPOINT"),
            api_key=os.getenv("AI_API_KEY"),
        )

        # Step 4: Create agent with MCP tools
        print("🤖 Creating agent with Context7 tools...\n")
        agent = create_agent(model, tools)

        # Step 5: Query for documentation
        queries = [
            "How do I use Express.js middleware?",
            "What are React hooks and how do I use useState?",
        ]

        for query in queries:
            print(f"👤 User: {query}")
            print()

            response = await agent.ainvoke({"messages": [("human", query)]})
            last_message = response["messages"][-1]

            print("🤖 Agent:")
            print(last_message.content)
            print()
            print("-" * 60)
            print()

        print("✅ Challenge 1 Complete!")
        print()
        print("💡 Key Concepts Demonstrated:")
        print("   • Connected to Context7 via HTTP transport")
        print("   • Retrieved and listed available MCP tools")
        print("   • Created an agent that uses external MCP tools")
        print("   • Queried documentation seamlessly through the agent")
        print("   • Same agent pattern as Chapter 5, different tool source!")
        print()

    except Exception as e:
        print(f"❌ Error: {e}")
        raise

    finally:
        # Step 6: Clean up - close MCP connection
        print("🔌 MCP client connection closed")


if __name__ == "__main__":
    asyncio.run(main())
