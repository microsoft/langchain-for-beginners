"""
Chapter 5 Example: Basic Agent with Manual Loop

NOTE: This example demonstrates the agent pattern using a manual loop implementation.
Compare this to the simplified create_agent() approach in the main code examples.
In production, you would use LangChain's built-in agent implementation.

Run: python 05-agents/samples/basic_agent_manual_loop.py

🤖 Try asking GitHub Copilot Chat (https://github.com/features/copilot):
- "How does an agent differ from a simple chain?"
- "Why does the agent loop have a maximum iteration limit?"
- "What happens if the agent can't answer the question?"
"""

import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


class CalculatorInput(BaseModel):
    """Input for calculator."""

    expression: str = Field(description="Math expression")


@tool(args_schema=CalculatorInput)
def calculator(expression: str) -> str:
    """Perform mathematical calculations."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def main():
    print("🤖 Basic Agent Demo (Manual Loop)\n")
    print("=" * 80 + "\n")

    model = ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )

    model_with_tools = model.bind_tools([calculator])

    query = "What is 125 * 8?"
    print(f"User: {query}\n")

    # Agent loop simulation
    messages = [HumanMessage(content=query)]
    iteration = 1
    max_iterations = 3

    while iteration <= max_iterations:
        print(f"Iteration {iteration}:")

        response = model_with_tools.invoke(messages)

        if not response.tool_calls or len(response.tool_calls) == 0:
            print(f"  Final Answer: {response.content}\n")
            break

        # Tool call found
        tool_call = response.tool_calls[0]
        print(f"  Thought: I should use the {tool_call['name']} tool")
        print(f"  Action: {tool_call['name']}({tool_call['args']})")

        # Execute tool
        tool_result = calculator.invoke(tool_call["args"])
        print(f"  Observation: {tool_result}\n")

        # Add to conversation history
        messages.append(
            AIMessage(
                content=str(response.content),
                tool_calls=response.tool_calls,
            )
        )
        messages.append(
            ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"],
            )
        )

        iteration += 1

    print("=" * 80 + "\n")
    print("💡 Key Concepts:")
    print("   • Agent follows ReAct pattern: Reason → Act → Observe")
    print("   • Tools extend agent capabilities")
    print("   • Agent iterates until it has an answer")


if __name__ == "__main__":
    main()
