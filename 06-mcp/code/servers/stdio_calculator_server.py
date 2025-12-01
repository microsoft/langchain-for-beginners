"""
Simple MCP Server for stdio Transport (Local Development)

This server runs as a subprocess and communicates via stdio.
It's perfect for local development and testing.

This file is used by 02_mcp_stdio_local.py

Run: python 06-mcp/code/servers/stdio_calculator_server.py
"""

import math
import sys

from mcp.server.fastmcp import FastMCP

# Create MCP server
mcp = FastMCP("stdio-calculator")


@mcp.tool()
def calculate(expression: str) -> str:
    """
    Perform mathematical calculations using Python's math module.
    
    Args:
        expression: Math expression to evaluate (e.g., '2 + 2', 'sqrt(16)', 'sin(pi/2)')
    
    Returns:
        The result of the calculation as a string.
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
            # Math module functions
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "asin": math.asin,
            "acos": math.acos,
            "atan": math.atan,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
            "floor": math.floor,
            "ceil": math.ceil,
            "pi": math.pi,
            "e": math.e,
        }
        
        result = eval(expression, {"__builtins__": {}}, safe_namespace)
        return f"{expression} = {result}"
    except Exception as e:
        raise ValueError(f"Invalid expression: {e}")


@mcp.tool()
def convert_temperature(value: float, from_unit: str, to_unit: str) -> str:
    """
    Convert temperature between Celsius and Fahrenheit.
    
    Args:
        value: Temperature value to convert
        from_unit: Source unit ('celsius' or 'fahrenheit')
        to_unit: Target unit ('celsius' or 'fahrenheit')
    
    Returns:
        The converted temperature as a string.
    """
    from_unit = from_unit.lower()
    to_unit = to_unit.lower()
    
    if from_unit == to_unit:
        return f"{value}°{from_unit[0].upper()} = {value}°{to_unit[0].upper()}"
    
    if from_unit == "celsius" and to_unit == "fahrenheit":
        result = (value * 9 / 5) + 32
    elif from_unit == "fahrenheit" and to_unit == "celsius":
        result = (value - 32) * 5 / 9
    else:
        raise ValueError(f"Invalid conversion: {from_unit} to {to_unit}")
    
    return f"{value}°{from_unit[0].upper()} = {result:.2f}°{to_unit[0].upper()}"


if __name__ == "__main__":
    # Log to stderr (stdout is used for MCP communication)
    print("📟 stdio MCP Calculator Server running...", file=sys.stderr)
    print("🔧 Tools: calculate, convert_temperature", file=sys.stderr)
    
    # Run with stdio transport
    mcp.run(transport="stdio")
