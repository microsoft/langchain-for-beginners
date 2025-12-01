"""
Challenge 3 Solution: Temperature Experiment
Run: python 02-chat-models/solution/temperature_lab.py
"""

import os
import time

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

prompt = "Write a catchy tagline for a coffee shop."
is_ci = os.environ.get("CI") == "true"
temperatures = [0, 1] if is_ci else [0, 0.5, 1, 1.5, 2]  # Reduce in CI mode
trials_per_temp = 1 if is_ci else 3  # Reduce trials in CI mode


def temperature_experiment():
    print("🌡️  Temperature Experiment\n")
    print(f'Prompt: "{prompt}"\n')
    print("=" * 80)

    for temp in temperatures:
        print(f"\n🌡️ Temperature: {temp}")
        print("-" * 80)

        model = ChatOpenAI(
            model=os.environ.get("AI_MODEL", "gpt-4o-mini"),
            temperature=temp,
        )

        responses = []

        for trial in range(1, trials_per_temp + 1):
            response = model.invoke(prompt)
            content = str(response.content)
            responses.append(content)
            print(f'Try {trial}: "{content}"')

            # Small delay to avoid rate limits (skip in CI for faster execution)
            if not is_ci:
                time.sleep(0.5)

        # Check for uniqueness
        unique_responses = set(responses)
        print(f"\n📊 Unique responses: {len(unique_responses)}/{trials_per_temp}")

    print("\n" + "=" * 80)
    print("📊 Analysis\n")
    print("Temperature 0.0:")
    print("  ✅ Consistent and deterministic")
    print("  ✅ Best for: Code generation, factual Q&A, translations")
    print("  ❌ Not ideal for: Creative writing, brainstorming\n")

    print("Temperature 0.5-1.0:")
    print("  ✅ Balanced between consistency and creativity")
    print("  ✅ Best for: General conversation, helpful suggestions")
    print("  ℹ️  Default for most applications\n")

    print("Temperature 1.5-2.0:")
    print("  ✅ Highly creative and varied")
    print("  ✅ Best for: Creative writing, unique ideas, brainstorming")
    print("  ❌ Not ideal for: Factual information, code\n")

    print("=" * 80)


if __name__ == "__main__":
    temperature_experiment()
