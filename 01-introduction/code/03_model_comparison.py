"""
Lesson 01 - Model Comparison in LangChain
This example shows how to compare different AI models.
"""

import os
import time

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def compare_models():
    print("🔬 Comparing AI Models\n")

    prompt = "Explain recursion in programming in one sentence."
    models = ["gpt-4o", "gpt-4o-mini"]

    for model_name in models:
        print(f"\n📊 Testing: {model_name}")
        print("─" * 50)

        model = ChatOpenAI(
            model=model_name,
            base_url=os.getenv("AI_ENDPOINT"),
            api_key=os.getenv("AI_API_KEY"),
        )

        start_time = time.time()
        response = model.invoke(prompt)
        duration = (time.time() - start_time) * 1000

        print(f"Response: {response.content}")
        print(f"⏱️  Time: {duration:.0f}ms")

    print("\n✅ Comparison complete!")
    print("\n💡 Key Observations:")
    print("   - gpt-4o is more capable and detailed")
    print("   - gpt-4o-mini is faster and uses fewer resources")
    print("   - Choose based on your needs: speed vs. capability")


if __name__ == "__main__":
    compare_models()
