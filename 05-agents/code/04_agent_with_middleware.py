"""
Example 4: create_agent() with Middleware

This example shows how to use **middleware** with create_agent()
for production scenarios like dynamic model selection based on
conversation complexity and graceful error handling.

Middleware can intercept and modify agent behavior at various stages:
- before_model: Before each LLM call
- after_model: After each LLM response
- wrap_model_call: Around each LLM call (for retries, caching)
- wrap_tool_call: Around each tool call (for error handling)

Run: python 05-agents/code/04_agent_with_middleware.py

🤖 Try asking GitHub Copilot Chat (https://github.com/features/copilot):
- "How would I add request logging middleware?"
- "Can middleware modify tool arguments before execution?"
- "What's the difference between before_model and wrap_model_call?"
"""

import os
from typing import Any, Callable

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


# Define tool schemas
class CalculatorInput(BaseModel):
    """Input for calculator."""

    expression: str = Field(description="The mathematical expression to evaluate")


class SearchInput(BaseModel):
    """Input for search."""

    query: str = Field(description="The search query")


# Define tools
@tool(args_schema=CalculatorInput)
def calculator(expression: str) -> str:
    """Perform mathematical calculations."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"The result is: {result}"
    except Exception as e:
        return f"Error: {e}"


@tool(args_schema=SearchInput)
def search(query: str) -> str:
    """Search for information. May fail occasionally for demonstration."""
    # Simulate occasional failures for demonstration
    if "error" in query.lower():
        raise Exception("Search service temporarily unavailable")

    return f'Search results for "{query}": Found relevant information about {query}.'


# Middleware 1: Dynamic Model Selection
# Switches to a more capable model for complex conversations
class DynamicModelMiddleware(AgentMiddleware):
    """Switch to a more capable model for complex conversations."""

    def __init__(self, messages_threshold: int = 10):
        super().__init__()
        self.messages_threshold = messages_threshold

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        message_count = len(request.state["messages"])
        print(f"  [Middleware] Message count: {message_count}")

        # For complex conversations (>threshold messages), could use more capable model
        if message_count > self.messages_threshold:
            print("  [Middleware] 🔄 Would switch to more capable model")
            # In production: request = request.override(model=advanced_model)

        print("  [Middleware] ✓ Using current model")
        return handler(request)


# Middleware 2: Tool Error Handler
# Catches tool failures and provides helpful fallback messages
class ToolErrorMiddleware(AgentMiddleware):
    """Catch tool failures and provide graceful fallbacks."""

    def wrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], ToolMessage],
    ) -> ToolMessage:
        try:
            return handler(request)
        except Exception as e:
            tool_name = request.tool_call.get("name", "unknown")
            print(f"  [Middleware] ⚠️  Tool '{tool_name}' failed: {e}")
            print("  [Middleware] 🔄 Returning fallback message")

            # Return graceful fallback instead of crashing
            return ToolMessage(
                content=f"I encountered an error while using the {tool_name} tool. "
                f"Let me try a different approach to answer your question.",
                tool_call_id=request.tool_call.get("id", ""),
            )


def main():
    print("🔧 Agent with Middleware Example\n")

    # Create the model
    model = ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )

    # Create agent with middleware
    agent = create_agent(
        model,
        tools=[calculator, search],
        middleware=[
            DynamicModelMiddleware(messages_threshold=10),
            ToolErrorMiddleware(),
        ],
    )

    # Test 1: Simple calculation (middleware logs but doesn't change behavior)
    print("Test 1: Simple calculation")
    print("─" * 60)
    query1 = "What is 25 * 8?"
    print(f"👤 User: {query1}\n")
    response1 = agent.invoke({"messages": [HumanMessage(content=query1)]})
    last_message1 = response1["messages"][-1]
    print(f"\n🤖 Agent: {last_message1.content}\n\n")

    # Test 2: Search with error handling (triggers error middleware)
    print("Test 2: Search with error handling")
    print("─" * 60)
    query2 = "Search for information about error handling"
    print(f"👤 User: {query2}\n")
    response2 = agent.invoke({"messages": [HumanMessage(content=query2)]})
    last_message2 = response2["messages"][-1]
    print(f"\n🤖 Agent: {last_message2.content}\n\n")

    print("💡 Middleware Benefits:")
    print("   • Dynamic model selection → Cost optimization")
    print("   • Error handling → Graceful degradation")
    print("   • Logging → Easy debugging")
    print("   • Flexibility → Customize behavior without changing tools\n")

    print("✅ Production Use Cases:")
    print("   • Switch to cheaper models for simple queries")
    print("   • Automatic retries with exponential backoff")
    print("   • Request/response logging for monitoring")
    print("   • User context injection (auth, permissions)")
    print("   • Rate limiting and quota management")


if __name__ == "__main__":
    main()
