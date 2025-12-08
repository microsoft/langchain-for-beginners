"""
Model Parameters
Run: python 02-chat-models/code/03_parameters.py

🤖 Try asking GitHub Copilot Chat (https://github.com/features/copilot):
- "What temperature value should I use for a customer service chatbot?"
- "How do I add the max_tokens parameter to limit response length?"
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()


def temperature_comparison():
    model_name = os.environ.get("AI_MODEL", "gpt-5-mini")
    print(f"🌡️  Temperature Comparison for {model_name}\n")
    print("=" * 80)

    prompt = "Write a creative opening line for a sci-fi story about time travel."
    is_ci = os.environ.get("CI") == "true"
    temperatures = [0, 1] if is_ci else [0, 1, 2]  # Reduce temperatures in CI mode
    tries = 1 if is_ci else 2  # Reduce tries in CI mode

    for temp in temperatures:
        print(f"\nTemperature: {temp}")
        print("-" * 80)

        model = ChatOpenAI(
            model=model_name,
            temperature=temp,
            base_url=os.getenv("AI_ENDPOINT"),
            api_key=os.getenv("AI_API_KEY"),
        )

        try:
            for i in range(1, tries + 1):
                response = model.invoke(prompt)
                print(f"  Try {i}: {response.content}")
        except Exception as error:
            # Some models may not support certain temperature values
            error_msg = str(error)
            if "temperature" in error_msg.lower():
                print(f"  ⚠️  This model doesn't support temperature={temp}. Skipping...")
                print(f"  💡 Error: {error_msg}")
            else:
                # Re-raise unexpected errors
                raise

    print("\n💡 General Temperature Guidelines:")
    print("   - Lower values (0-0.3): More deterministic, consistent responses")
    print("   - Medium values (0.7-1.0): Balanced creativity and consistency")
    print("   - Higher values (1.5-2.0): More creative and varied responses")
    print("\n⚠️  Note: Model support varies - some models only support specific values")


def max_tokens_example():
    print("\n\n📏 Max Tokens Limit\n")
    print("=" * 80)

    prompt = "Write a detailed explanation of machine learning in 5 paragraphs."

    is_ci = os.environ.get("CI") == "true"
    token_limits = [150] if is_ci else [50, 150, 500]  # Reduce in CI mode

    for max_tokens in token_limits:
        print(f"\nMax Tokens: {max_tokens}")
        print("-" * 80)

        model = ChatOpenAI(
            model=os.environ.get("AI_MODEL", "gpt-5-mini"),
            max_tokens=max_tokens,
        )

        try:
            response = model.invoke(prompt)
            print(response.content)
            print(f"\n(Character count: {len(str(response.content))})")
        except Exception as error:
            error_msg = str(error)
            if "max_tokens" in error_msg.lower():
                print(f"  ⚠️  This model doesn't support max_tokens={max_tokens}. Skipping...")
                print(f"  💡 Error: {error_msg}")
            else:
                raise

    print("\n💡 Observations:")
    print("   - Lower max tokens = shorter responses")
    print("   - Response may be cut off if limit is too low")
    print("   - Use max tokens to control costs and response length")


def main():
    print("🎛️  Model Parameters Tutorial\n")

    temperature_comparison()
    max_tokens_example()

    print("\n\n✅ Summary:")
    print("   - Lower temperatures: Consistent, factual responses")
    print("   - Higher temperatures: Creative, varied responses")
    print("   - max_tokens: Control response length and costs")
    print("   - Always check your model's supported parameter ranges")


if __name__ == "__main__":
    main()
