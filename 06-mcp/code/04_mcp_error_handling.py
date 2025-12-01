"""
Chapter 6 Example 4: MCP Error Handling & Production Patterns

This example shows production-ready patterns for handling MCP failures:
- Built-in retry logic with LangChain's with_retry()
- Connection errors and timeouts
- Graceful degradation
- Fallback strategies

These patterns are essential for building reliable MCP integrations!

Run: python 06-mcp/code/04_mcp_error_handling.py
"""

import asyncio
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

load_dotenv()


async def create_mcp_client_safely(config: dict) -> MultiServerMCPClient | None:
    """Utility: Safe MCP client creation with error handling."""
    try:
        print("🔄 Attempting to connect to MCP server...")
        client = MultiServerMCPClient(config)

        # Test connection by getting tools (MCP client handles connection internally)
        tools = await client.get_tools()
        print(f"✅ Connected! Retrieved {len(tools)} tools")

        return client
    except Exception as e:
        print(f"❌ Failed to connect to MCP server: {e}")
        return None


async def check_mcp_health(client: MultiServerMCPClient) -> bool:
    """Check health of MCP server connection."""
    try:
        tools = await asyncio.wait_for(client.get_tools(), timeout=5.0)
        is_healthy = len(tools) > 0
        print("✅ MCP server is healthy" if is_healthy else "⚠️  MCP server returned no tools")
        return is_healthy
    except asyncio.TimeoutError:
        print("❌ MCP server is unhealthy: Health check timeout")
        return False
    except Exception as e:
        print(f"❌ MCP server is unhealthy: {e}")
        return False


async def main():
    print("🛡️  MCP Error Handling & Retry Patterns\n")

    # Pattern 1: Try primary server, fall back to alternative
    print("Pattern 1: Primary + Fallback Strategy\n")

    mcp_client: MultiServerMCPClient | None = None
    tools: list = []

    try:
        # Try Context7 (primary)
        print("📡 Trying primary server (Context7)...")
        mcp_client = await create_mcp_client_safely(
            {
                "context7": {
                    "transport": "streamable_http",
                    "url": "https://mcp.context7.com/mcp",
                }
            }
        )

        if not mcp_client:
            # If Context7 fails, you could fall back to alternative server
            print("\n📡 Primary failed, trying fallback server...")
            # This is where you'd try an alternative server
            # For demo, we'll continue without fallback
            raise RuntimeError("No MCP servers available")

        # Get tools with error handling
        try:
            print("\n🔧 Fetching tools from MCP server...")
            tools = await mcp_client.get_tools()

            print(f"✅ Retrieved {len(tools)} tools successfully\n")
            for tool in tools:
                print(f"   • {tool.name}")
        except Exception as e:
            print(f"❌ Failed to fetch tools: {e}")
            print("💡 Fallback: Using empty tools array")
            tools = []

        # Pattern 2: Create Model with Built-In Retry Logic
        print("\n\nPattern 2: Using LangChain's Built-In with_retry()\n")

        if not tools:
            print("⚠️  No tools available - agent will run without MCP tools")
            print("   This is graceful degradation - app continues to work!")

        # Create base model
        base_model = ChatOpenAI(
            model=os.getenv("AI_MODEL"),
            base_url=os.getenv("AI_ENDPOINT"),
            api_key=os.getenv("AI_API_KEY"),
        )

        # Use LangChain's built-in retry logic - automatically handles exponential backoff!
        model_with_retry = base_model.with_retry(stop_after_attempt=3)

        print("✅ Model configured with automatic retry (max 3 attempts)")
        print("   - LangChain handles exponential backoff automatically")
        print("   - No custom retry loops needed!")

        agent = create_agent(model_with_retry, tools)  # May be empty array if MCP failed

        # Pattern 3: Execute with timeout and error handling
        print("\n\nPattern 3: Query Execution with Timeout\n")

        query = "What is React? Get the latest documentation."
        print(f"👤 User: {query}")

        try:
            # Wrap agent execution with timeout
            timeout_seconds = 30  # 30 second timeout

            response = await asyncio.wait_for(
                agent.ainvoke({"messages": [("human", query)]}),
                timeout=timeout_seconds,
            )

            last_message = response["messages"][-1]
            print(f"🤖 Agent: {last_message.content}\n")

        except asyncio.TimeoutError:
            print("❌ Query failed: Query timeout")
            # Fallback response
            print("💡 Fallback: Providing cached/default response")
            print("🤖 Agent: I'm experiencing connectivity issues. Please try again later.")
        except Exception as e:
            print(f"❌ Query failed: {e}")
            print("💡 Fallback: Providing cached/default response")
            print("🤖 Agent: I'm experiencing connectivity issues. Please try again later.")

        # Pattern 4: Health checks
        print("\nPattern 4: MCP Server Health Check\n")

        is_healthy = await check_mcp_health(mcp_client)
        print(f"\n🏥 Health status: {'HEALTHY' if is_healthy else 'UNHEALTHY'}")

        # Best practices summary
        print("\n\n💡 Error Handling Best Practices:")
        print("   ✅ Use LangChain's with_retry() for automatic exponential backoff")
        print("   ✅ Implement fallback servers for high availability")
        print("   ✅ Set timeouts on all network operations")
        print("   ✅ Gracefully degrade when MCP is unavailable")
        print("   ✅ Implement health checks for monitoring")
        print("   ✅ Log errors for debugging and alerting")
        print("   ✅ Provide user-friendly error messages")

        print("\n🎯 Production Checklist:")
        print("   ✅ Use model.with_retry() for automatic retries")
        print("   □ Request timeouts")
        print("   □ Fallback strategies")
        print("   □ Health monitoring")
        print("   □ Error logging/metrics")
        print("   □ Graceful degradation")
        print("   □ Circuit breaker pattern (for advanced use)")

    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        print("💡 In production, this would trigger alerts and fallback to cached data")

    finally:
        # Always clean up connections
        if mcp_client:
            try:
                # Python MCP client handles cleanup automatically
                print("\n✅ MCP connection closed gracefully")
            except Exception as e:
                print(f"⚠️  Error closing MCP connection: {e}")

    print("\n🎓 Key Takeaways:")
    print("   • Use LangChain's with_retry() instead of custom retry loops")
    print("   • with_retry() provides production-tested exponential backoff")
    print("   • Always handle MCP connection failures gracefully")
    print("   • Implement timeouts to prevent hangs")
    print("   • Provide fallbacks for degraded operation")
    print("   • Monitor health and log errors")
    print("   • Clean up resources in finally blocks")


if __name__ == "__main__":
    asyncio.run(main())
