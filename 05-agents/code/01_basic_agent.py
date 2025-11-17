"""
Basic ReAct Agent Example
Run: python 05-agents/code/01_basic_agent.py

This example demonstrates how to create a simple ReAct agent.
"""

import os

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

def main():
    model = ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
        temperature=0,
    )

    tools = [multiply, add]

    # Get the ReAct prompt
    prompt = hub.pull("hwchase17/react")

    # Create the agent
    agent = create_react_agent(model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    print("🤖 ReAct Agent Example\n")
    print("=" * 80)

    # Test the agent with a math problem
    result = agent_executor.invoke({
        "input": "What is 25 multiplied by 4, plus 10?"
    })

    print(f"\n✅ Final Answer: {result['output']}")
    print("\n" + "=" * 80)
    print("\n💡 The agent reasoned through the problem and used tools!")

if __name__ == "__main__":
    main()
