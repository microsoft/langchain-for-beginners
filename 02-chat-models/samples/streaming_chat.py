"""
Streaming Chat Interface
Run: python 02-chat-models/samples/streaming_chat.py
"""

import os
import sys
import time

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

model = ChatOpenAI(model=os.environ.get("AI_MODEL", "gpt-4o-mini"))


def streaming_chat():
    """Interactive streaming chat interface."""
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye!")
            break

        if user_input.lower() in ("quit", "exit"):
            print("\n👋 Goodbye!")
            break

        if not user_input:
            continue

        try:
            print("\n🤖 Typing...\n")

            start_time = time.time()
            first_chunk_time = 0
            full_response = ""

            # Clear the "Typing..." line and print bot prefix
            sys.stdout.write("\r🤖 Chatbot: ")
            sys.stdout.flush()

            stream = model.stream(user_input)

            for chunk in stream:
                if first_chunk_time == 0:
                    first_chunk_time = time.time()
                content = str(chunk.content)
                sys.stdout.write(content)
                sys.stdout.flush()
                full_response += content

            end_time = time.time()

            print("\n")
            print(f"⚡ First chunk: {(first_chunk_time - start_time) * 1000:.0f}ms")
            print(f"⏱️  Full response: {(end_time - start_time) * 1000:.0f}ms")

            # Exit in CI mode after one interaction
            if os.environ.get("CI") == "true":
                break

        except Exception as error:
            print(f"\n❌ Error: {error}")


def main():
    print("⚡ Streaming Chat Interface")
    print('Type your question and watch the response stream! (Type "quit" to exit)\n')

    streaming_chat()


if __name__ == "__main__":
    main()
