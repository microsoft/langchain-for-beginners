"""
Provider-Agnostic Model Initialization
Run: python 02-chat-models/code/04_init_chat_model.py

This example demonstrates init_chat_model() with LangChain Azure AI - the recommended
approach for Azure AI Foundry and GitHub Models.

ENDPOINT FORMAT:
- GitHub Models: https://models.inference.ai.azure.com (works as-is)
- Azure AI Foundry: https://your-resource.services.ai.azure.com/models
  (NOT the /openai/v1 endpoint - this script auto-converts if needed)

🤖 Try asking GitHub Copilot Chat (https://github.com/features/copilot):
- "What are the advantages of init_chat_model over using ChatOpenAI directly?"
- "How would I switch from Azure AI to Anthropic using init_chat_model?"
"""

import os
import warnings

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

# Filter out the ExperimentalWarning from langchain-azure-ai
warnings.filterwarnings("ignore", message=".*AzureAIChatCompletionsModel is currently in preview.*")


def get_azure_ai_endpoint():
    """
    Get the Azure AI endpoint, converting from OpenAI v1 format if needed.

    LangChain Azure AI expects the /models endpoint format, not /openai/v1.
    This function handles the conversion automatically.

    Returns:
        str: The properly formatted Azure AI endpoint
    """
    endpoint = os.getenv("AI_ENDPOINT", "")

    # Convert Azure OpenAI v1 endpoint to Azure AI models endpoint
    # e.g., https://resource.openai.azure.com/openai/v1 -> https://resource.openai.azure.com/models
    if endpoint.endswith("/openai/v1"):
        endpoint = endpoint.replace("/openai/v1", "/models")
        print(f"📝 Note: Converted endpoint from /openai/v1 to /models format")
    elif endpoint.endswith("/openai/v1/"):
        endpoint = endpoint.replace("/openai/v1/", "/models")
        print(f"📝 Note: Converted endpoint from /openai/v1/ to /models format")

    return endpoint


def azure_ai_example():
    """Demonstrate init_chat_model() with LangChain Azure AI."""
    print("\n=== init_chat_model() with Azure AI ===\n")

    # Get and convert endpoint if needed
    endpoint = get_azure_ai_endpoint()

    # Set environment variables required by LangChain Azure AI
    os.environ["AZURE_AI_ENDPOINT"] = endpoint
    os.environ["AZURE_AI_CREDENTIAL"] = os.getenv("AI_API_KEY", "")

    model_name = os.getenv("AI_MODEL", "gpt-5-mini")

    print(f"🔗 Using endpoint: {endpoint}")
    print(f"🤖 Using model: {model_name}\n")

    # Initialize model using the azure_ai provider prefix
    # Format: "azure_ai:<model_name>"
    model = init_chat_model(f"azure_ai:{model_name}")

    response = model.invoke([
        HumanMessage(content="What is LangChain in one sentence?")
    ])

    print("✅ Response:", response.content)


def switching_providers_concept():
    """Show how init_chat_model() enables easy provider switching."""
    print("\n=== Provider Switching Concepts ===\n")

    # This is where init_chat_model() shines - switching providers with similar code:
    print("init_chat_model() makes switching between providers simple:\n")

    print("  # Azure AI (recommended for this course)")
    print('  model = init_chat_model("azure_ai:gpt-5-mini")')
    print()
    print("  # Standard OpenAI")
    print('  model = init_chat_model("openai:gpt-5-mini")')
    print()
    print("  # Anthropic")
    print('  model = init_chat_model("anthropic:claude-3-5-sonnet-20241022")')
    print()
    print("  # Google")
    print('  model = init_chat_model("google-genai:gemini-pro")')
    print()
    print("💡 Same interface, different providers - just change the model string!")


def main():
    print("🔌 Provider-Agnostic Initialization with LangChain Azure AI\n")
    print("=" * 60)

    try:
        azure_ai_example()
        switching_providers_concept()

        print("\n" + "=" * 60)
        print("\n📚 Key Takeaways:")
        print("  - Use 'azure_ai:<model>' format for Azure AI Foundry and GitHub Models")
        print("  - Set AZURE_AI_ENDPOINT and AZURE_AI_CREDENTIAL environment variables")
        print("  - Azure AI expects /models endpoint (not /openai/v1)")
        print("  - GitHub Models endpoint works as-is")
        print("  - init_chat_model() provides a unified interface across providers")
        print()
    except ImportError as error:
        print(f"⚠️  Missing dependency: {error}")
        print("\n💡 Install LangChain Azure AI with:")
        print("   pip install langchain-azure-ai")
    except Exception as error:
        print(f"❌ Error: {error}")
        print("\n💡 Make sure your .env file has:")
        print("   - AI_ENDPOINT set to your Azure AI or GitHub Models endpoint")
        print("   - AI_API_KEY set to your API key")
        print("\n📝 Endpoint format:")
        print("   - GitHub Models: https://models.inference.ai.azure.com")
        print("   - Azure AI Foundry: https://your-resource.openai.azure.com/models")


if __name__ == "__main__":
    main()
