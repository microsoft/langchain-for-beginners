# Function Calling & Tools

In this chapter, you'll learn how to extend AI capabilities beyond text generation by enabling function calling and tools. You'll discover how LLMs can invoke functions with structured arguments, create type-safe tools using Pydantic schemas, and build systems where AI can trigger real-world actions like API calls, database queries, or calculations.

**This is a foundational chapter for building AI agents.** Tools are the building blocks that give agents their capabilities—without tools, agents are just text generators. In [Getting Started with Agents](../05-agents/README.md), you'll see how agents use the tools you create here to autonomously make decisions and solve multi-step problems.

## Prerequisites

- Completed [Prompts, Messages, and Structured Outputs](../03-prompts-messages-outputs/README.md)

## 🎯 Learning Objectives

By the end of this chapter, you'll be able to:

- ✅ Understand what function calling is and why it matters
- ✅ Create tools with Pydantic schemas for type safety
- ✅ Bind tools to chat models
- ✅ Invoke tools and handle responses
- ✅ Build systems with multiple tools
- ✅ Apply best practices for tool design

---

## 📖 The Restaurant Staff Analogy

**Imagine you're a restaurant manager coordinating your team.**

When a customer orders "I'd like the daily special with no onions, a side salad, and sparkling water," you don't do everything yourself. Instead:

1. **You understand the request** (what they want)
2. **You delegate to specialists**:
   - 👨‍🍳 Chef: "Make the daily special, no onions" (function: prepare_meal)
   - 🥗 Salad station: "Prepare a side salad" (function: make_salad)
   - 🍷 Bar: "Serve sparkling water" (function: serve_beverage)
3. **Each specialist confirms** what they're doing
4. **You coordinate the response** back to the customer

**Function calling in AI works exactly the same way!**

The LLM:
- **Understands** the user's request
- **Generates structured function calls** with proper arguments
- **Returns** the function details (but doesn't execute them)
- **Processes** the function results to form a response

---

## 🛠️ Creating Tools with the @tool Decorator

In LangChain Python, tools are created using the `@tool` decorator with type hints and docstrings.

### Example 1: Simple Calculator Tool

**Code**: [`code/01_simple_tool.py`](./code/01_simple_tool.py)  
**Run**: `python 04-function-calling-tools/code/01_simple_tool.py`

**Example code:**

```python
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

@tool
def calculator(expression: str) -> str:
    """Useful for performing mathematical calculations.
    
    Args:
        expression: The mathematical expression to evaluate, e.g., '25 * 4'
    """
    try:
        # Use eval safely for simple math (in production, use a proper math library)
        result = eval(expression, {"__builtins__": {}}, {})
        return f"The result is: {result}"
    except Exception as e:
        return f"Error evaluating expression: {e}"

def main():
    print("Tool created:", calculator.name)
    print("Description:", calculator.description)
    
    # Test the tool directly
    result = calculator.invoke({"expression": "25 * 4"})
    print("Result:", result)

if __name__ == "__main__":
    main()
```

### Expected Output

```
Tool created: calculator
Description: Useful for performing mathematical calculations.
Result: The result is: 100
```

---

## 🔗 Binding Tools to Models

Use `bind_tools()` to make tools available to the LLM.

### Example 2: Binding and Invoking Tools

**Code**: [`code/02_tool_calling.py`](./code/02_tool_calling.py)  
**Run**: `python 04-function-calling-tools/code/02_tool_calling.py`

**Example code:**

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from dotenv import load_dotenv
import os

load_dotenv()

@tool
def calculator(expression: str) -> str:
    """Perform mathematical calculations."""
    result = eval(expression, {"__builtins__": {}}, {})
    return str(result)

def main():
    model = ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY")
    )

    # Bind tools to the model
    model_with_tools = model.bind_tools([calculator])

    # Invoke with a question
    response = model_with_tools.invoke("What is 25 * 17?")

    print("Response:", response)
    print("\nTool calls:", response.tool_calls)

if __name__ == "__main__":
    main()
```

### Expected Output

```
🤖 Asking: What is 25 * 17?

Tool calls: [
  {
    "name": "calculator",
    "args": {"expression": "25 * 17"},
    "id": "call_abc123"
  }
]
```

---

## 🔄 Complete Tool Call Loop

### Example 3: Complete Tool Execution

**Code**: [`code/03_tool_execution.py`](./code/03_tool_execution.py)  
**Run**: `python 04-function-calling-tools/code/03_tool_execution.py`

**Example code:**

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from dotenv import load_dotenv
import os

load_dotenv()

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    # Simulated weather data
    temps = {"Seattle": 62, "Paris": 18, "Tokyo": 24}
    temp = temps.get(city, 72)
    return f"Current temperature in {city}: {temp}°F"

def main():
    model = ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY")
    )

    model_with_tools = model.bind_tools([get_weather])

    # Step 1: Get tool call from LLM
    query = "What's the weather in Seattle?"
    response1 = model_with_tools.invoke(query)
    print("Step 1 - Tool call:", response1.tool_calls[0])

    # Step 2: Execute the tool
    tool_call = response1.tool_calls[0]
    tool_result = get_weather.invoke(tool_call["args"])
    print("Step 2 - Tool result:", tool_result)

    # Step 3: Send result back to LLM
    messages = [
        HumanMessage(content=query),
        AIMessage(content="", tool_calls=response1.tool_calls),
        ToolMessage(content=tool_result, tool_call_id=tool_call["id"]),
    ]

    final_response = model.invoke(messages)
    print("Step 3 - Final answer:", final_response.content)

if __name__ == "__main__":
    main()
```

### Expected Output

```
Step 1 - Tool call: {'name': 'get_weather', 'args': {'city': 'Seattle'}, 'id': 'call_xyz'}
Step 2 - Tool result: Current temperature in Seattle: 62°F
Step 3 - Final answer: The current temperature in Seattle is 62°F.
```

---

## 🎛️ Multiple Tools

### Example 4: Multi-Tool System

**Code**: [`code/04_multiple_tools.py`](./code/04_multiple_tools.py)  
**Run**: `python 04-function-calling-tools/code/04_multiple_tools.py`

**Example code:**

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from dotenv import load_dotenv
import os

load_dotenv()

@tool
def calculator(expression: str) -> str:
    """Perform mathematical calculations."""
    return str(eval(expression, {"__builtins__": {}}, {}))

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    temps = {"Seattle": 62, "Paris": 18, "Tokyo": 24}
    return f"Weather in {city}: {temps.get(city, 72)}°F"

@tool
def search(query: str) -> str:
    """Search for information about a topic."""
    return f"Search results for '{query}': LangChain is a framework for building AI apps."

def main():
    model = ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY")
    )

    # Bind multiple tools
    model_with_tools = model.bind_tools([calculator, get_weather, search])

    queries = [
        "What is 50 * 25?",              # → Uses calculator
        "What's the weather in Tokyo?",  # → Uses get_weather
        "Tell me about LangChain",       # → Uses search
    ]

    for query in queries:
        response = model_with_tools.invoke(query)
        if response.tool_calls:
            print(f"Query: {query}")
            print(f"  Tool: {response.tool_calls[0]['name']}")
            print()

if __name__ == "__main__":
    main()
```

---

## 🎓 Key Takeaways

- **Tools extend AI capabilities** beyond text generation
- **The @tool decorator** creates tools from Python functions
- **bind_tools()** connects tools to models
- **Tool calls are descriptions** - your code executes them
- **Complete the loop** - send results back to get natural responses

---

## 🗺️ Navigation

[← Previous: Prompts & Outputs](../03-prompts-messages-outputs/README.md) | [Back to Main](../README.md) | [Next: Getting Started with Agents →](../05-agents/README.md)

---

## 💬 Questions?

[![Azure AI Foundry Discord](https://img.shields.io/badge/Discord-Azure_AI_Foundry_Community_Discord-blue?style=for-the-badge&logo=discord&color=5865f2&logoColor=fff)](https://aka.ms/foundry/discord)
