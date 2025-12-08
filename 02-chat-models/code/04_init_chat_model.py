"""
Provider-Agnostic Model Initialization
Run: python 02-chat-models/code/04_init_chat_model.py

IMPORTANT: init_chat_model() works best with standard provider APIs.
For GitHub Models or Azure OpenAI (used in this course), use ChatOpenAI directly.

This example demonstrates init_chat_model() concepts, but the course uses
ChatOpenAI because it properly handles custom endpoints like GitHub Models.

🤖 Try asking GitHub Copilot Chat (https://github.com/features/copilot):
- "What are the advantages of init_chat_model over using ChatOpenAI directly?"
- "How would I switch from OpenAI to Anthropic using init_chat_model?"
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()


def standard_openai_example():
    print("\n=== init_chat_model() with Standard OpenAI ===\n")

    # NOTE: This requires a standard OpenAI API key (not GitHub Models)
    # Uncomment and add OPENAI_API_KEY to your .env to test:
    """
    model = init_chat_model(
        "gpt-5-mini",
        model_provider="openai",
        temperature=0.7,
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.getenv("AI_ENDPOINT")
    )

    response = model.invoke([
        HumanMessage(content="What is LangChain in one sentence?")
    ])

    print("Response:", response.content)
    """

    print("This example requires a standard OpenAI API key.")
    print("For GitHub Models/Azure, use ChatOpenAI instead (see below).\n")


def switching_providers():
    print("\n=== Switching Between Providers ===\n")

    # This is where init_chat_model() shines - switching providers with similar code:
    """
    # OpenAI
    openai_model = init_chat_model(
        "gpt-5-mini",
        model_provider="openai",
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    # Anthropic
    anthropic_model = init_chat_model(
        "claude-3-5-sonnet-20241022",
        model_provider="anthropic",
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )

    # Google
    google_model = init_chat_model(
        "gemini-pro",
        model_provider="google-genai",
        api_key=os.environ.get("GOOGLE_API_KEY"),
    )
    """

    print("init_chat_model() excels at switching between different providers")
    print("(OpenAI, Anthropic, Google, etc.) with similar code structure.\n")


def course_recommendation():
    print("\n=== Recommended Approach for This Course ===\n")

    # For GitHub Models and Azure OpenAI, use ChatOpenAI directly:
    model = ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY")
    )

    response = model.invoke([HumanMessage(content="What is LangChain in one sentence?")])

    print("✅ Using ChatOpenAI (recommended for this course)")
    print(f"Response: {response.content}")
    print("\nWhy ChatOpenAI?")
    print("- Properly handles GitHub Models and Azure OpenAI endpoints")
    print("- More explicit and easier to understand for learning")
    print("- Works seamlessly with custom base_url configuration")


def main():
    print("🔌 Provider-Agnostic Initialization Concepts\n")
    print("=" * 60)

    try:
        standard_openai_example()
        switching_providers()
        course_recommendation()

        print("\n" + "=" * 60)
        print("\n📚 Key Takeaway:")
        print("- init_chat_model() is great for switching between provider types")
        print("- For this course (GitHub Models/Azure), ChatOpenAI is recommended")
        print("- Both approaches are valid - choose based on your needs\n")
    except Exception as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
