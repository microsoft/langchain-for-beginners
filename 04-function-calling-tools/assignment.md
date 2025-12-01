# Assignment: Function Calling & Tooling

## Overview

Practice creating type-safe tools with Pydantic schemas, implementing the complete tool execution pattern, and building multi-tool systems that extend AI capabilities.

## Prerequisites

- Completed this [chapter](./README.md)
- Run all code examples in this chapter
- Understand tool creation, binding, and execution
- Completed the Prompts, Messages & Outputs chapter

---

## Challenge: Weather Tool with Complete Execution Loop ⛅

**Goal**: Build a weather tool and implement the complete 3-step execution pattern (generate → execute → respond).

**Tasks**:
1. Create `weather_tool.py` in the `04-function-calling-tools/solution/` folder
2. Build a weather tool with Pydantic schema that accepts:
   - `city` (string, required) - The city name
   - `units` (Literal["celsius", "fahrenheit"], optional, default: "fahrenheit") - Temperature unit
3. Implement the tool to return simulated weather data for at least 5 cities
4. Implement the complete 3-step execution pattern:
   - **Step 1**: Get tool call from LLM
   - **Step 2**: Execute the tool
   - **Step 3**: Send result back to LLM for final response
5. Test with multiple queries using different cities and units

**Example Queries**:
- "What's the weather in Tokyo?"
- "Tell me the temperature in Paris in celsius"
- "Is it raining in London?"

**Success Criteria**:
- Tool uses proper Pydantic schema with `Field(description=...)` for parameters
- Handles both celsius and fahrenheit units
- Implements all 3 steps of tool execution
- LLM generates natural language responses based on tool results
- Clear console output showing each step

**Hints**:
```python
# 1. Import required modules
import os
from typing import Literal, Optional
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# 2. Create input schema with Pydantic BaseModel
#    - city: str with Field(description="...")
#    - units: Optional[Literal["celsius", "fahrenheit"]] with default

# 3. Create a weather tool using the @tool decorator:
#    - Use args_schema parameter to specify the Pydantic model
#    - Implement function to return simulated weather data
#    - Add detailed docstring as the tool description

# 4. Bind the tool to the model using model.bind_tools()

# 5. Implement the 3-step execution pattern:
#    Step 1: Invoke model with user query, check for tool_calls
#    Step 2: Execute the tool with tool.invoke(tool_call["args"])
#    Step 3: Create messages list with HumanMessage, AIMessage, and ToolMessage
#            Then invoke model again for final natural language response
```

> [!TIP]
> **🤖 Get help from [GitHub Copilot](../docs/copilot.md):** If you need assistance with this challenge, open this file in your editor and [use the Challenge Tutor agent](../docs/copilot.md#challenge-tutor-agent) to get personalized help and explanations.

---

## Bonus Challenge: Multi-Tool Travel Assistant 🌍

**Goal**: Build a system with multiple tools where the LLM automatically selects the appropriate tool for travel-related queries.

**Tasks**:
1. Create `travel_assistant.py`
2. Build three specialized tools:
   - **Currency Converter**: Convert amounts between currencies (USD, EUR, GBP, JPY)
   - **Distance Calculator**: Calculate distance between two cities in miles or kilometers
   - **Time Zone Tool**: Get current time in a city and calculate time difference
3. Each tool should have:
   - Clear, descriptive name
   - Detailed docstring explaining when to use it
   - Proper Pydantic schema with parameter descriptions
4. Bind all three tools to the model
5. Test with queries that require different tools:
   - "Convert 100 USD to EUR"
   - "What's the distance between New York and London?"
   - "What time is it in Tokyo right now?"
   - "If it's 3pm in Seattle, what time is it in Paris?"

**Success Criteria**:
- All three tools work correctly
- LLM automatically chooses the right tool for each query
- Tool descriptions are clear enough to guide LLM selection
- Returns accurate simulated results
- Handles edge cases (invalid currencies, unknown cities)

**Hints**:
```python
# 1. Import required modules
import os
from typing import Literal, Optional
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# 2. Create Pydantic input schemas for each tool

# 3. Create three tools using @tool decorator:
#    Currency Converter - with amount, from_currency, to_currency parameters
#    Distance Calculator - with from_city, to_city, and units parameters
#    Time Zone Tool - with city parameter
#    Make sure each has:
#    - Clear, descriptive docstring
#    - Proper Pydantic schema with Field(description="...") on parameters
#    - Simulated implementation returning appropriate data

# 4. Bind all three tools to the model with model.bind_tools([...])

# 5. Test with different queries and observe which tool the LLM selects

# 6. Create a tools_map dict to look up tool functions by name
#    Execute with: tool_fn.invoke(tool_call["args"])
```

**Additional Feature** (Optional):
Add error handling that returns helpful error messages when:
- Invalid currency code provided
- Unknown city name
- Invalid input format

**Example Output**:
```
Query: "Convert 50 EUR to JPY"
→ LLM chose: currency_converter
→ Args: { "amount": 50, "from": "EUR", "to": "JPY" }
→ Result: "50 EUR equals approximately 8,100 JPY"

Query: "What's the distance from Paris to Rome?"
→ LLM chose: distance_calculator
→ Args: { "from": "Paris", "to": "Rome", "units": "kilometers" }
→ Result: "The distance from Paris to Rome is approximately 1,430 kilometers"
```

---

## Submission Checklist

Before continuing, make sure you've completed:

- [ ] Challenge: Weather tool with complete 3-step execution
- [ ] Bonus: Multi-tool travel assistant (optional)

---

## Solutions

Solutions for all challenges are available in the [`solution/`](./solution/) folder. Try to complete the challenges on your own first!

---

## Need Help?

- **Tool creation**: Review Example 1 in [`code/01_simple_tool.py`](./code/01_simple_tool.py)
- **Execution pattern**: Check Example 3 in [`code/03_tool_execution.py`](./code/03_tool_execution.py)
- **Multiple tools**: See Example 4 in [`code/04_multiple_tools.py`](./code/04_multiple_tools.py)
- **Pydantic schemas**: Review the [Pydantic section](./README.md#🛠️-creating-tools-with-pydantic) in the README
- **Any question**: Use the [Challenge Tutor agent](../docs/copilot.md#challenge-tutor-agent) in GitHub Copilot
- **Still stuck**: Join our [Discord community](https://aka.ms/foundry/discord)

---

## Next Steps

Once you've completed these challenges, you're ready for:

**[Getting Started with Agents](../05-agents/README.md)**

Great work mastering function calling and tooling! 🚀
