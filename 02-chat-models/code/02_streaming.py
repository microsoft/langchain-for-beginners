"""
Streaming Responses
Run: python 02-chat-models/code/02_streaming.py

🤖 Try asking GitHub Copilot Chat (https://github.com/features/copilot):
- "How does the 'for...in' loop work with the stream?"
- "Can I collect all chunks into a single string while streaming?"
"""

import os
import sys
import time

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()


def non_streaming_example():
    print("📝 Non-Streaming (traditional way):\n")

    model = ChatOpenAI(model=os.environ.get("AI_MODEL", "gpt-4o-mini"))

    start_time = time.time()
    response = model.invoke("Explain how the internet works in 2 paragraphs.")
    end_time = time.time()

    print(response.content)
    elapsed_ms = (end_time - start_time) * 1000
    print(f"\n⏱️  Received after: {elapsed_ms:.0f}ms\n")


def streaming_example():
    print("\n" + "=" * 80)
    print("⚡ Streaming (appears immediately):\n")

    model = ChatOpenAI(model=os.environ.get("AI_MODEL", "gpt-4o-mini"))

    start_time = time.time()
    first_chunk_time = 0

    # Stream the response chunk by chunk
    stream = model.stream("Explain how the internet works in 2 paragraphs.")

    for chunk in stream:
        if first_chunk_time == 0:
            first_chunk_time = time.time()
        # Write each chunk as it arrives (no newline)
        sys.stdout.write(str(chunk.content))
        sys.stdout.flush()

    end_time = time.time()

    print("\n")
    print(f"⏱️  First chunk arrived: {(first_chunk_time - start_time) * 1000:.0f}ms")
    print(f"⏱️  Stream completed: {(end_time - start_time) * 1000:.0f}ms")
    print("\n✅ Notice how streaming feels more responsive!")


def main():
    print("🎯 Comparing Streaming vs Non-Streaming\n")
    print("=" * 80)

    non_streaming_example()
    streaming_example()

    print("\n💡 Key Insight:")
    print("   Streaming provides immediate feedback, making your app feel faster!")


if __name__ == "__main__":
    main()
