# Chat Models & Basic Interactions

In this chapter, you'll learn the art of having natural conversations with AI models. You'll learn how to maintain conversation context across multiple exchanges, stream responses in real-time for better user experience, and handle errors gracefully with retry logic. You'll also explore key parameters like temperature to control AI creativity and understand token usage for cost optimization.

## Prerequisites

- Completed [Introduction to LangChain](../01-introduction/README.md)

## 🎯 Learning Objectives

By the end of this chapter, you'll be able to:

- ✅ Have multi-turn conversations with AI
- ✅ Stream responses for better user experience
- ✅ Handle errors gracefully
- ✅ Control model behavior with parameters
- ✅ Understand token usage

---

## 📖 The Knowledgeable Friend Analogy

**Imagine you're having coffee with a knowledgeable friend.**

When you talk to them:
- 💬 **You have a back-and-forth conversation** (not just one question)
- 🧠 **They remember what you said earlier** (conversation context)
- 🗣️ **They speak as they think** (streaming responses)
- 😊 **They adjust their tone** based on your preferences (model parameters)
- ⚠️ **Sometimes they need clarification** (error handling)

**Chat models work the same way!**

Unlike simple one-off questions, chat models excel at:
- Multi-turn conversations
- Maintaining context
- Streaming responses in real-time
- Adapting their behavior

This chapter teaches you how to have natural, interactive conversations with AI.

---

## 💬 Multi-Turn Conversations

Previously, we sent single messages. But real conversations have multiple exchanges.

### How Conversation History Works

Chat models don't actually "remember" previous messages. Instead, you send the entire conversation history with each new message.

**Think of it like this**: Every time you send a message, you're showing the AI the entire conversation thread so far.

---

### Message Types in LangChain

LangChain provides three core message types for building conversations:

| Type | Purpose | Example |
|------|---------|---------|
| **SystemMessage** | Set AI behavior and personality | `SystemMessage(content="You are a helpful coding tutor")` |
| **HumanMessage** | User input and questions | `HumanMessage(content="What is TypeScript?")` |
| **AIMessage** | AI responses with metadata | Returned by `model.invoke()` with `content`, `usage_metadata`, `id` |

---

### Example 1: Multi-Turn Conversation

Let's see how to maintain conversation context using a `messages` list with `SystemMessage`, `HumanMessage`, and `AIMessage`.

**Code**: [`code/01_multi_turn.py`](./code/01_multi_turn.py)  
**Run**: `python 02-chat-models/code/01_multi_turn.py`

**Example code:**

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import os

load_dotenv()

def main():
    print("💬 Multi-Turn Conversation Example\n")

    model = ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )

    # Start with system message and first question
    messages = [
        SystemMessage(content="You are a helpful coding tutor who gives clear, concise explanations."),
        HumanMessage(content="What is Python?"),
    ]

    print("👤 User: What is Python?")

    # First exchange
    response1 = model.invoke(messages)
    print("\n🤖 AI:", response1.content)
    messages.append(AIMessage(content=response1.content))

    # Second exchange - AI remembers the context
    print("\n👤 User: Can you show me a simple example?")
    messages.append(HumanMessage(content="Can you show me a simple example?"))

    response2 = model.invoke(messages)
    print("\n🤖 AI:", response2.content)

    # Third exchange - AI still remembers everything
    print("\n👤 User: What are the benefits compared to other languages?")
    messages.append(AIMessage(content=response2.content))
    messages.append(HumanMessage(content="What are the benefits compared to other languages?"))

    response3 = model.invoke(messages)
    print("\n🤖 AI:", response3.content)

    print("\n\n✅ Notice how the AI maintains context throughout the conversation!")
    print(f"📊 Total messages in history: {len(messages)}")

if __name__ == "__main__":
    main()
```

### Expected Output

```
💬 Multi-Turn Conversation Example

👤 User: What is Python?

🤖 AI: [Detailed explanation of Python]

👤 User: Can you show me a simple example?

🤖 AI: [Python code example with explanation]

👤 User: What are the benefits compared to other languages?

🤖 AI: [Explanation of Python benefits]

✅ Notice how the AI maintains context throughout the conversation!
📊 Total messages in history: 6
```

### How It Works

**Key Points**:
1. **Messages list holds the entire conversation** - We create a list that stores all messages (system, human, and AI)
2. **Each response is added to the history** - After getting a response, we append it to the messages list
3. **The AI can reference earlier messages** - When we ask "Can you show me a simple example?", the AI knows we're talking about Python from the first exchange
4. **Full history is sent each time** - With every `invoke()` call, we send the complete conversation history

---

## ⚡ Streaming Responses

When you ask a complex question, waiting for the entire response can feel slow. Streaming sends the response word-by-word as it's generated.

**Like watching a friend think out loud** instead of waiting for them to finish their entire thought.

### Example 2: Streaming

Let's see how to use `.stream()` instead of `.invoke()` to display responses as they're generated.

**Code**: [`code/02_streaming.py`](./code/02_streaming.py)  
**Run**: `python 02-chat-models/code/02_streaming.py`

**Example code:**

```python
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

def main():
    model = ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY")
    )

    print("🤖 AI (streaming): ")

    # Stream the response
    for chunk in model.stream("Explain how the internet works in 3 paragraphs."):
        print(chunk.content, end="", flush=True)

    print("\n\n✅ Stream complete!")

if __name__ == "__main__":
    main()
```

### Expected Output

When you run this example, you'll see the response appear word-by-word:

```
🤖 AI (streaming):
The internet is a global network of interconnected computers that communicate using standardized protocols, primarily TCP/IP...

✅ Stream complete!
```

### How It Works

**What's happening**:
1. We call `model.stream()` instead of `model.invoke()`
2. This returns an iterator that yields chunks as they're generated
3. We loop through each chunk with a for loop
4. Each chunk contains a piece of the response (usually a few words)
5. We use `flush=True` to display text immediately

**Benefits of Streaming**:
- Better user experience (immediate feedback)
- Users can start reading while AI generates the rest
- Perceived performance improvement even if total time is the same

---

## 🎛️ Model Parameters

You can control how the AI responds by adjusting parameters.

### Key Parameters

#### Temperature (0.0 - 2.0)

Temperature controls randomness and creativity:

- **0.0 = Deterministic**: Same question → Same answer (use for code, factual answers)
- **1.0 = Balanced** (default): Mix of consistency and variety
- **2.0 = Creative**: More random and creative responses (use for creative writing)

### Example 3: Model Parameters

Let's see how to control creativity by adjusting the `temperature` parameter.

**Code**: [`code/03_parameters.py`](./code/03_parameters.py)  
**Run**: `python 02-chat-models/code/03_parameters.py`

**Example code:**

```python
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

def temperature_comparison():
    prompt = "Write a creative opening line for a sci-fi story about time travel."
    temperatures = [0, 1, 2]

    for temp in temperatures:
        print(f"\n🌡️ Temperature: {temp}")
        print("-" * 80)

        model = ChatOpenAI(
            model=os.getenv("AI_MODEL"),
            base_url=os.getenv("AI_ENDPOINT"),
            api_key=os.getenv("AI_API_KEY"),
            temperature=temp,
        )

        try:
            for i in range(1, 3):
                response = model.invoke(prompt)
                print(f"  Try {i}: {response.content}")
        except Exception as e:
            print(f"  ⚠️  This model may not support temperature={temp}")
            print(f"  💡 Error: {e}")

    print("\n💡 General Temperature Guidelines:")
    print("   - Lower values (0-0.3): More deterministic, consistent responses")
    print("   - Medium values (0.7-1.0): Balanced creativity and consistency")
    print("   - Higher values (1.5-2.0): More creative and varied responses")

if __name__ == "__main__":
    temperature_comparison()
```

### Expected Output

```
🌡️ Temperature: 0
────────────────────────────────────────────────────────────────────────────────
  Try 1: "In the year 2157, humanity had finally broken free..."
  Try 2: "In the year 2157, humanity had finally broken free..."

🌡️ Temperature: 1
────────────────────────────────────────────────────────────────────────────────
  Try 1: "The stars whispered secrets through the observation deck..."
  Try 2: "Time folded like origami in Dr. Chen's laboratory..."

💡 General Temperature Guidelines:
   - Lower values (0-0.3): More deterministic, consistent responses
   - Medium values (0.7-1.0): Balanced creativity and consistency
   - Higher values (1.5-2.0): More creative and varied responses
```

---

## 🎓 Key Takeaways

- **Conversation history is your responsibility** - Chat models don't have memory; you must send the full history
- **Streaming improves UX** - Users see progress immediately
- **Temperature controls creativity** - Lower for consistency, higher for variety
- **Token limits matter** - Be mindful of context length for cost and performance

---

## 🏆 Assignment

Ready to practice? Complete the challenges in [assignment.md](./assignment.md)!

---

## 🗺️ Navigation

[← Previous: Introduction](../01-introduction/README.md) | [Back to Main](../README.md) | [Next: Prompts, Messages & Outputs →](../03-prompts-messages-outputs/README.md)

---

## 💬 Questions?

If you get stuck or have any questions about building AI apps, join:

[![Azure AI Foundry Discord](https://img.shields.io/badge/Discord-Azure_AI_Foundry_Community_Discord-blue?style=for-the-badge&logo=discord&color=5865f2&logoColor=fff)](https://aka.ms/foundry/discord)
