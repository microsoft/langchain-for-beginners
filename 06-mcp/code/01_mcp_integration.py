"""
Chapter 6 Example 1: Agent with MCP Server Integration

This example shows how to connect to Context7 MCP server - a documentation
provider that delivers current, version-specific docs directly to your agent.

Context7 provides these tools:
- resolve-library-id: Converts library names to Context7-compatible IDs
- get-library-docs: Retrieves documentation with optional topic filtering

Prerequisites:
1. Install langchain-mcp-adapters: pip install langchain-mcp-adapters
2. Optional: Get a Context7 API key for higher rate limits (https://context7.com)

Run: python 06-mcp/code/01_mcp_integration.py

🤖 Try asking GitHub Copilot Chat (https://github.com/features/copilot):
- "How does MultiServerMCPClient differ from manually creating tools?"
- "Can I connect to multiple MCP servers simultaneously?"
"""

import asyncio
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

load_dotenv()


async def main():
    print("🔌 MCP Integration Demo - Context7 Documentation Server\n")
    print("=" * 80 + "\n")

    # Context7 MCP Server - provides documentation for libraries
    # Remote (HTTP): https://mcp.context7.com/mcp (recommended)
    # Local (HTTP): http://localhost:3000/mcp (if running Context7 locally)
    mcp_server_url = os.getenv("MCP_SERVER_URL", "https://mcp.context7.com/mcp")

    print(f"📡 Connecting to MCP server at: {mcp_server_url}\n")

    # Create MCP client with HTTP transport to Context7
    client = MultiServerMCPClient(
        {
            "context7": {
                "transport": "streamable_http",
                "url": mcp_server_url,
                # Optional: Add Context7 API key for higher rate limits
                # "headers": {
                #     "Authorization": f"Bearer {os.getenv('CONTEXT7_API_KEY')}"
                # }
            },
        }
    )

    try:
        # 2. Get all available tools from Context7
        print("🔧 Fetching tools from Context7 MCP server...")
        tools = await client.get_tools()

        print(f"✅ Retrieved {len(tools)} tools from Context7:")
        for tool in tools:
            print(f"   • {tool.name}: {tool.description}")
        print()

        # 3. Create model
        model = ChatOpenAI(
            model=os.getenv("AI_MODEL"),
            base_url=os.getenv("AI_ENDPOINT"),
            api_key=os.getenv("AI_API_KEY"),
        )

        # 4. Create agent with MCP tools - uses same create_agent() pattern!
        print("🤖 Creating agent with MCP tools...\n")
        agent = create_agent(model, tools)  # Tools from MCP server - that's the only difference!

        # 5. Use the agent to get documentation
        query = "How do I use React useState hook? Get the latest documentation."
        print(f"👤 User: {query}\n")

        response = await agent.ainvoke({"messages": [("human", query)]})
        last_message = response["messages"][-1]

        print(f"🤖 Agent: {last_message.content}\n")

        print("=" * 80 + "\n")
        print("💡 Key Concepts:")
        print("   • MCP provides standardized access to external tools")
        print("   • MultiServerMCPClient connects to one or more MCP servers")
        print("   • HTTP transport works with remote servers like Context7")
        print("   • Tools from MCP servers work seamlessly with create_agent()")
        print("   • Same create_agent() pattern, different tool source!")
        print("   • No manual loop needed - create_agent() handles the ReAct pattern")

    except Exception as e:
        print(f"❌ Error connecting to Context7 MCP server: {e}")

    finally:
        # Note: In Python, the MultiServerMCPClient handles cleanup automatically
        # when using async context managers, but we log for clarity
        print("\n✅ MCP client connection closed")


if __name__ == "__main__":
    asyncio.run(main())
