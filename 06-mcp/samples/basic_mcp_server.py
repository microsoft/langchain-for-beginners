"""
Chapter 6 Sample: Simple MCP Server (HTTP Streaming - Stateful)

This example shows how to build a basic MCP server that exposes tools.
This is the server-side implementation - the counterpart to the client
code you've seen in the chapter examples.

Run: python 06-mcp/samples/basic_mcp_server.py

Then connect to it at: http://localhost:3000/mcp
"""

import math
import os
import sys

import uvicorn
from mcp.server.fastmcp import FastMCP

# Create MCP server with tools capability
mcp = FastMCP("my-calculator")


@mcp.tool()
def calculate(expression: str) -> str:
    """
    Perform mathematical calculations.
    
    Args:
        expression: Math expression to evaluate, e.g., '2 + 2', 'sqrt(16)'
    
    Returns:
        The result of the calculation.
    """
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
        return str(result)
    except Exception as e:
        raise ValueError(f"Invalid expression: {e}")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "3000"))
    
    print(f"🚀 MCP Calculator Server (HTTP Streaming - Stateful)")
    print(f"🌐 MCP endpoint: http://localhost:{port}/mcp")
    print(f"⌨️  Press Ctrl+C to stop the server")
    
    # Run with streamable HTTP transport
    mcp.run(transport="streamable-http", port=port)
