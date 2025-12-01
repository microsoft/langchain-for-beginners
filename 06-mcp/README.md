# MCP Client

In this chapter, you'll learn about the Model Context Protocol (MCP) and how to use MCP servers as tool sources for your LangChain agents. You'll discover how MCP provides a standardized way for AI applications to connect with external tools and services.

## Prerequisites

- Completed [Getting Started with Agents](../05-agents/README.md)

## 🎯 Learning Objectives

By the end of this chapter, you'll be able to:

- ✅ Understand what MCP (Model Context Protocol) is
- ✅ Use MCP servers as tool sources for your agents
- ✅ Connect to local and remote MCP servers
- ✅ Build agents that use MCP-provided tools
- ✅ Combine multiple MCP servers for rich tool ecosystems

---

## 📖 What is MCP?

**Model Context Protocol (MCP)** is an open protocol that standardizes how AI applications connect to external tools and data sources. Think of it as a **universal adapter** for AI tools.

### The Problem MCP Solves

Before MCP, every AI application had to implement its own integrations:
- Each app needed custom code for GitHub, Slack, databases, etc.
- Tool formats varied between providers
- Sharing tools between applications was difficult

### The MCP Solution

MCP provides a **standard protocol** that:
- ✅ Allows tools to be defined once and used anywhere
- ✅ Separates tool providers (servers) from tool consumers (clients)
- ✅ Works with any LLM or AI framework
- ✅ Supports local (stdio) and remote (HTTP) connections

---

## 🔌 MCP Architecture

```
┌─────────────────┐     MCP Protocol     ┌─────────────────┐
│                 │ ←─────────────────→  │                 │
│   MCP Client    │                      │   MCP Server    │
│  (Your Agent)   │                      │  (Tool Source)  │
│                 │                      │                 │
└─────────────────┘                      └─────────────────┘
         │                                        │
         │                                        │
    LangChain                              External APIs
    Application                           (GitHub, DB, etc.)
```

**MCP Client**: Your LangChain agent that consumes tools  
**MCP Server**: A service that exposes tools via the MCP protocol

---

## 🛠️ Using MCP Tools with LangChain

LangChain provides the `langchain-mcp-adapters` library to integrate MCP tools.

### Installation

```bash
pip install langchain-mcp-adapters
```

---

### Example 1: Connecting to an MCP Server

**Code**: [`code/01_mcp_basic.py`](./code/01_mcp_basic.py)  
**Run**: `python 06-mcp/code/01_mcp_basic.py`

**Example code:**

```python
import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_agent
from dotenv import load_dotenv
import os

load_dotenv()

async def main():
    # Define MCP server connection parameters
    server_params = {
        "url": "https://mcp.example.com/mcp",  # Replace with your MCP server
        "headers": {
            "Authorization": f"Bearer {os.getenv('MCP_API_KEY')}"
        }
    }

    # Connect to the MCP server
    async with streamablehttp_client(**server_params) as (read, write, _):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Load tools from the MCP server
            tools = await load_mcp_tools(session)

            print(f"Loaded {len(tools)} tools from MCP server:")
            for tool in tools:
                print(f"  - {tool.name}: {tool.description}")

            # Create an agent with the MCP tools
            agent = create_agent(os.getenv("AI_MODEL"), tools)

            # Use the agent
            response = await agent.ainvoke({
                "messages": [{"role": "user", "content": "What tools do you have available?"}]
            })
            print(f"\nAgent: {response['messages'][-1].content}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

### Example 2: Using MultiServerMCPClient

The `MultiServerMCPClient` allows connecting to multiple MCP servers at once.

**Code**: [`code/02_multi_server.py`](./code/02_multi_server.py)  
**Run**: `python 06-mcp/code/02_multi_server.py`

**Example code:**

```python
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

async def main():
    # Configure multiple MCP servers
    client = MultiServerMCPClient({
        "weather": {
            "transport": "streamable_http",
            "url": "https://weather-mcp.example.com/mcp",
        },
        "calculator": {
            "transport": "stdio",
            "command": "python",
            "args": ["-m", "mcp_calculator_server"],
        },
    })

    # Get all tools from all servers
    tools = await client.get_tools()

    print(f"Loaded {len(tools)} tools from MCP servers:")
    for tool in tools:
        print(f"  - {tool.name}")

    # Create agent with tools from multiple servers
    agent = create_agent(
        model=os.getenv("AI_MODEL"),
        tools=tools,
        system_prompt="You have access to weather and calculator tools.",
    )

    # Use the agent
    queries = [
        "What's the weather in Seattle?",
        "What is 42 * 17?",
    ]

    for query in queries:
        response = await agent.ainvoke({
            "messages": [HumanMessage(content=query)]
        })
        print(f"\nUser: {query}")
        print(f"Agent: {response['messages'][-1].content}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

### Example 3: Using a Local MCP Server (stdio)

You can also run MCP servers locally using stdio transport.

**Code**: [`code/03_local_mcp.py`](./code/03_local_mcp.py)  
**Run**: `python 06-mcp/code/03_local_mcp.py`

**Example code:**

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_agent
from dotenv import load_dotenv
import os

load_dotenv()

async def main():
    # Define local MCP server (using stdio)
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "my_mcp_server"],
        env={"PYTHONPATH": "."}
    )

    # Connect to local MCP server
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Load tools from local server
            tools = await load_mcp_tools(session)
            print(f"Loaded {len(tools)} tools from local MCP server")

            # Create and use agent
            agent = create_agent(os.getenv("AI_MODEL"), tools)

            response = await agent.ainvoke({
                "messages": [{"role": "user", "content": "Help me with a task"}]
            })
            print(f"Agent: {response['messages'][-1].content}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🌐 Popular MCP Servers

Here are some commonly used MCP servers you can connect to:

| Server | Description |
|--------|-------------|
| **GitHub MCP** | Access GitHub repositories, issues, and pull requests |
| **Slack MCP** | Send and read Slack messages |
| **PostgreSQL MCP** | Query and manipulate databases |
| **Filesystem MCP** | Read and write local files |
| **Google Drive MCP** | Access Google Drive files |

---

## 🔒 Security Considerations

When using MCP servers:

1. **Authentication**: Always use proper authentication (API keys, OAuth)
2. **Permissions**: Grant only the minimum required permissions
3. **Validation**: Validate tool inputs and outputs
4. **Sandboxing**: Consider running local servers in containers
5. **Audit logging**: Log all tool invocations for security audits

---

## 🎓 Key Takeaways

- **MCP standardizes** how AI apps connect to external tools
- **langchain-mcp-adapters** integrates MCP tools with LangChain
- **MultiServerMCPClient** connects to multiple servers at once
- **Transport options**: stdio (local) or HTTP (remote)
- **Tools are reusable** across different AI applications

---

## 📦 Dependencies

```bash
pip install langchain langchain-mcp-adapters mcp python-dotenv
```

---

## 🗺️ Navigation

[← Previous: Getting Started with Agents](../05-agents/README.md) | [Back to Main](../README.md) | [Next: Documents & Embeddings →](../07-documents-embeddings-semantic-search/README.md)

---

## 💬 Questions?

[![Azure AI Foundry Discord](https://img.shields.io/badge/Discord-Azure_AI_Foundry_Community_Discord-blue?style=for-the-badge&logo=discord&color=5865f2&logoColor=fff)](https://aka.ms/foundry/discord)
