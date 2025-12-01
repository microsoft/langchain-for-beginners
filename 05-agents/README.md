# Getting Started with Agents

In this chapter, you'll learn to build AI agents that can reason about problems, select appropriate tools, and work iteratively towards solutions. You'll understand the ReAct (Reasoning + Acting) pattern by implementing agent loops step-by-step, and discover how agents autonomously choose tools to accomplish complex tasks. These skills enable you to build autonomous AI systems that can handle complex, multi-step tasks.

## Prerequisites

- Completed [Function Calling & Tools](../04-function-calling-tools/README.md)

## 🎯 Learning Objectives

By the end of this chapter, you'll be able to:

- ✅ Understand what AI agents are and how they work
- ✅ Implement the ReAct (Reasoning + Acting) pattern
- ✅ Build agent loops that iterate until solving a problem
- ✅ Give agents multiple tools and let them choose the right one
- ✅ Use create_agent() for production-ready agent systems
- ✅ Build multi-step, autonomous AI systems

---

## 📖 The Manager with Specialists Analogy

**Imagine you're a project manager with a team of specialists:**

- 📊 Data Analyst - can query databases
- 🔍 Researcher - can search the web
- 🧮 Accountant - can do calculations
- ✉️ Assistant - can send emails

When someone asks: *"What's our revenue growth this quarter compared to last year?"*

You (the manager) don't do everything yourself. You:
1. **Reason**: "I need data from the database and calculations"
2. **Act**: Ask the Data Analyst for revenue data
3. **Observe**: Review the data received
4. **Reason**: "Now I need to calculate the percentage change"
5. **Act**: Ask the Accountant to do the math
6. **Observe**: Get the calculated result
7. **Reason**: "Now I have the answer"
8. **Respond**: Give the final answer

**AI Agents work the same way!**

They:
- **Think** about what needs to be done (Reasoning)
- **Choose** the right tool (Decision Making)
- **Use** the tool (Acting)
- **Evaluate** the result (Observation)
- **Repeat** until they have the answer
- **Respond** to the user

---

## 🤖 What Are Agents?

### Standard LLM (No Agency or Tools)

```
User: "What's the current weather in Paris?"
LLM: "I cannot access real-time weather data. I can only provide general information..."
```

### Agent with Tools

```
User: "What's the current weather in Paris?"
Agent: [Thinks] "I need to use the weather tool"
Agent: [Uses] get_weather(city="Paris")
Agent: [Observes] "18°C, partly cloudy"
Agent: [Responds] "It's currently 18°C and partly cloudy in Paris"
```

---

## 🧠 The ReAct Pattern

ReAct = **Rea**soning + **Act**ing

Agents follow this iterative loop:

```
1. Thought: What should I do next?
2. Action: Use a specific tool
3. Observation: What did the tool return?
4. (Repeat 1-3 as needed)
5. Final Answer: Respond to the user
```

**Example**:
```
User: "Calculate 25 * 17, then tell me if it's a prime number"

Thought 1: I need to calculate 25 * 17
Action 1: calculator(expression="25 * 17")
Observation 1: 425

Thought 2: I need to check if 425 is prime
Action 2: is_prime(number=425)
Observation 2: False (divisible by 5)

Final Answer: "25 * 17 equals 425, which is not a prime number
because it's divisible by 5."
```

---

## 🚀 Building Agents with create_agent()

LangChain Python provides `create_agent()` from `langchain.agents` - a high-level API that handles the ReAct loop automatically. This is the **recommended approach** for building production agents.

**What create_agent() does for you**:
- ✅ Manages the ReAct loop (Thought → Action → Observation → Repeat)
- ✅ Handles message history automatically
- ✅ Implements iteration limits to prevent infinite loops
- ✅ Provides production-ready error handling
- ✅ Returns clean, structured responses

---

### Example 1: Basic Agent with create_agent()

**Code**: [`code/01_basic_agent.py`](./code/01_basic_agent.py)  
**Run**: `python 05-agents/code/01_basic_agent.py`

**Example code:**

```python
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

# Define a calculator tool for the agent
@tool
def calculator(expression: str) -> str:
    """A calculator that can perform basic arithmetic operations.
    
    Args:
        expression: The mathematical expression to evaluate
    """
    result = eval(expression, {"__builtins__": {}}, {})
    return str(result)

def main():
    # Create agent using create_agent() - that's it!
    agent = create_agent(
        model=os.getenv("AI_MODEL"),
        tools=[calculator],
        system_prompt="You are a helpful math assistant.",
    )

    # Use the agent with messages array
    query = "What is 125 * 8?"
    response = agent.invoke({
        "messages": [HumanMessage(content=query)]
    })

    # Get the final answer from the last message
    last_message = response["messages"][-1]
    print(f"Agent: {last_message.content}")

if __name__ == "__main__":
    main()
```

### Expected Output

```
🤖 Agent with create_agent() Example

👤 User: What is 125 * 8?

🤖 Agent: 125 × 8 = 1000

✅ Under the hood:
   create_agent() implements the ReAct pattern (Thought → Action → Observation)
   and handles all the boilerplate for you.
```

### How It Works

**What's happening behind the scenes**:
1. **Agent receives query**: "What is 125 * 8?"
2. **Reasons**: Determines it needs the calculator tool
3. **Acts**: Executes `calculator(expression="125 * 8")`
4. **Observes**: Gets result "1000"
5. **Responds**: Formats natural language response

---

### Example 2: create_agent() with Multiple Tools

Let's give an agent multiple tools and observe how it autonomously selects the right one.

**Code**: [`code/02_multi_tool_agent.py`](./code/02_multi_tool_agent.py)  
**Run**: `python 05-agents/code/02_multi_tool_agent.py`

**Example code:**

```python
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

@tool
def calculator(expression: str) -> str:
    """Perform mathematical calculations."""
    result = eval(expression, {"__builtins__": {}}, {})
    return str(result)

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    temps = {"Seattle": 62, "Paris": 18, "Tokyo": 24}
    temp = temps.get(city, 72)
    return f"Current weather in {city}: {temp}°F"

@tool
def search(query: str) -> str:
    """Search for information about a topic."""
    return f"LangChain is a Python framework for building AI applications with LLMs."

def main():
    # Create agent with all three tools
    agent = create_agent(
        model=os.getenv("AI_MODEL"),
        tools=[calculator, get_weather, search],
        system_prompt="You are a helpful assistant with access to multiple tools.",
    )

    # Agent automatically picks the correct tool for each query
    queries = [
        "What is 50 * 25?",              # → Uses calculator
        "What's the weather in Tokyo?",  # → Uses get_weather
        "Tell me about LangChain",       # → Uses search
    ]

    for query in queries:
        response = agent.invoke({
            "messages": [HumanMessage(content=query)]
        })
        last_message = response["messages"][-1]
        print(f"User: {query}")
        print(f"Agent: {last_message.content}\n")

if __name__ == "__main__":
    main()
```

### Expected Output

```
🎛️  Multi-Tool Agent with create_agent()

👤 User: What is 50 * 25?
🤖 Agent: 50 multiplied by 25 equals 1250.

👤 User: What's the weather in Tokyo?
🤖 Agent: Current weather in Tokyo: 24°F

👤 User: Tell me about LangChain
🤖 Agent: LangChain is a Python framework for building applications with large
language models (LLMs).

💡 What just happened:
   • The agent automatically selected the right tool for each query
   • Calculator for math (50 * 25)
   • Weather tool for Tokyo weather
   • Search tool for LangChain information
   • All with the same agent instance!
```

---

### Example 3: Manual ReAct Loop (Understanding the Pattern)

To understand what `create_agent()` does under the hood, let's implement the ReAct loop manually.

**Code**: [`code/03_manual_react.py`](./code/03_manual_react.py)  
**Run**: `python 05-agents/code/03_manual_react.py`

**Example code:**

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from dotenv import load_dotenv
import os

load_dotenv()

@tool
def calculator(expression: str) -> str:
    """Perform mathematical calculations."""
    result = eval(expression, {"__builtins__": {}}, {})
    return str(result)

@tool
def is_prime(number: int) -> str:
    """Check if a number is prime."""
    if number < 2:
        return "False"
    for i in range(2, int(number ** 0.5) + 1):
        if number % i == 0:
            return f"False (divisible by {i})"
    return "True"

def run_react_loop(query: str, tools: list, max_iterations: int = 5):
    """Manually implement the ReAct loop."""
    
    model = ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY")
    )
    
    # Create tool lookup
    tools_by_name = {t.name: t for t in tools}
    
    # Bind tools to model
    model_with_tools = model.bind_tools(tools)
    
    # Initialize messages
    messages = [HumanMessage(content=query)]
    
    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")
        
        # Step 1: Call the model
        response = model_with_tools.invoke(messages)
        messages.append(response)
        
        # Step 2: Check if there are tool calls
        if not response.tool_calls:
            print("No more tool calls - Final answer ready")
            return response.content
        
        # Step 3: Execute each tool call
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            print(f"Action: {tool_name}({tool_args})")
            
            # Execute the tool
            tool_result = tools_by_name[tool_name].invoke(tool_args)
            print(f"Observation: {tool_result}")
            
            # Add tool result to messages
            messages.append(
                ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
            )
    
    return "Max iterations reached"

def main():
    tools = [calculator, is_prime]
    
    query = "Calculate 25 * 17, then tell me if the result is a prime number"
    print(f"Query: {query}")
    
    result = run_react_loop(query, tools)
    print(f"\n🤖 Final Answer: {result}")

if __name__ == "__main__":
    main()
```

### Expected Output

```
Query: Calculate 25 * 17, then tell me if the result is a prime number

--- Iteration 1 ---
Action: calculator({'expression': '25 * 17'})
Observation: 425

--- Iteration 2 ---
Action: is_prime({'number': 425})
Observation: False (divisible by 5)

--- Iteration 3 ---
No more tool calls - Final answer ready

🤖 Final Answer: 25 * 17 equals 425, which is not a prime number 
because it is divisible by 5.
```

---

## 🔧 Tool Selection Logic

The agent uses tool **names** and **descriptions** to match queries to tools:

| User Query | Tool Selected | Why |
|-----------|---------------|-----|
| "What is 50 * 25?" | calculator | Matches "mathematical calculations" |
| "Weather in Tokyo?" | get_weather | Matches "weather for a city" |
| "Tell me about X" | search | Matches "search for information" |

**Tips for better tool selection**:
1. Use **descriptive names** - `get_weather` not `tool1`
2. Write **clear descriptions** - explain what the tool does
3. Document **parameters** - use docstrings with Args sections
4. Be **specific** - more detail helps the LLM choose correctly

---

## 🎓 Key Takeaways

- **Agents combine LLMs with tools** for autonomous problem-solving
- **ReAct pattern**: Reason → Act → Observe → Repeat
- **create_agent()** handles the loop automatically
- **Tool descriptions** guide the agent's tool selection
- **Multiple tools** let agents handle diverse tasks
- **Iteration limits** prevent infinite loops

---

## 📦 Dependencies

Make sure you have the required packages:

```bash
pip install langchain langchain-openai python-dotenv
```

---

## 🗺️ Navigation

[← Previous: Function Calling & Tools](../04-function-calling-tools/README.md) | [Back to Main](../README.md) | [Next: MCP Client →](../06-mcp-client/README.md)

---

## 💬 Questions?

[![Azure AI Foundry Discord](https://img.shields.io/badge/Discord-Azure_AI_Foundry_Community_Discord-blue?style=for-the-badge&logo=discord&color=5865f2&logoColor=fff)](https://aka.ms/foundry/discord)
