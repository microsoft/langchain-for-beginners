# Prompts, Messages, and Structured Outputs

In this chapter, you'll learn the three essential techniques for working with LLMs in LangChain: **messages**, **prompt templates**, and **structured outputs**. Understanding these techniques is key, because modern LangChain applications choose different approaches depending on the use case. Messages provide dynamic construction for flexible workflows like agents, templates provide reusable prompts with variable substitution, and structured outputs ensure type-safe data extraction.

This chapter prepares you for both **agents** (using messages) and **RAG systems** (using templates).

## Prerequisites

- Completed [Chat Models & Basic Interactions](../02-chat-models/README.md)

## 🎯 Learning Objectives

By the end of this chapter, you'll be able to:

- ✅ Understand when to use messages vs templates
- ✅ Construct message arrays for agent workflows
- ✅ Create reusable prompt templates for RAG systems
- ✅ Use variables and dynamic content in prompts
- ✅ Implement few-shot prompting (teaching by example)
- ✅ Generate structured outputs with Pydantic models
- ✅ Choose the right approach for your use case

---

## 🎯 Decision Framework: Messages vs Templates

**Choose the right approach for your use case**:

| Approach | Use For | Chapter |
|----------|---------|---------|
| **Messages** | Agents, dynamic workflows, multi-step reasoning, tool integration | [Getting Started with Agents](../05-agents/README.md) |
| **Templates** | Reusable prompts, variable substitution, consistency, RAG systems | [Documents, Embeddings & Semantic Search](../07-documents-embeddings-semantic-search/README.md) |

**Both approaches are valuable**: Messages for dynamic workflows, templates for reusability and consistency.

---

## PART 1: Message-Based Prompting

Message arrays are the foundation of agent systems in LangChain. When you work with agents, you'll use message arrays as input and output.

### Example 1: Messages vs Templates

This foundational example compares both approaches side-by-side.

**Code**: [`code/01_messages_vs_templates.py`](./code/01_messages_vs_templates.py)  
**Run**: `python 03-prompts-messages-outputs/code/01_messages_vs_templates.py`

**Example code:**

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

def main():
    model = ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )

    # APPROACH 1: Messages
    print("APPROACH 1: Message Arrays\n")

    messages = [
        SystemMessage(content="You are a helpful translator."),
        HumanMessage(content="Translate 'Hello, world!' to French"),
    ]

    message_response = model.invoke(messages)
    print("Response:", message_response.content)

    # APPROACH 2: Templates
    print("\nAPPROACH 2: Templates\n")

    template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful translator."),
        ("human", "Translate '{text}' to {language}"),
    ])

    template_chain = template | model
    template_response = template_chain.invoke({
        "text": "Hello, world!",
        "language": "French",
    })

    print("Response:", template_response.content)

if __name__ == "__main__":
    main()
```

### How It Works

**Message Arrays**:
- Direct construction using `SystemMessage()` and `HumanMessage()`
- Passed directly to `model.invoke(messages)`
- No templating or variable substitution
- Used by agents in LangChain

**Templates**:
- Created with `ChatPromptTemplate.from_messages()`
- Uses variables like `{text}` and `{language}`
- Piped to model using `|` operator: `template | model`
- Valuable for reusability and consistency

---

### Example 2: Dynamic Message Construction

Learn how to build message arrays programmatically and use few-shot prompting.

**Code**: [`code/02_message_construction.py`](./code/02_message_construction.py)  
**Run**: `python 03-prompts-messages-outputs/code/02_message_construction.py`

**Example code:**

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from dotenv import load_dotenv
import os
from typing import List

load_dotenv()

def create_conversation(role: str, examples: List[dict], new_question: str) -> List[BaseMessage]:
    """Build message arrays programmatically."""
    messages: List[BaseMessage] = [SystemMessage(content=f"You are a {role}.")]
    
    # Add few-shot examples
    for example in examples:
        messages.append(HumanMessage(content=example["question"]))
        messages.append(AIMessage(content=example["answer"]))
    
    messages.append(HumanMessage(content=new_question))
    return messages

def main():
    model = ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )

    # Few-Shot Learning with Messages
    emoji_messages = create_conversation(
        "emoji translator",
        [
            {"question": "happy", "answer": "😊"},
            {"question": "sad", "answer": "😢"},
            {"question": "excited", "answer": "🎉"},
        ],
        "surprised",
    )

    print(f"Messages constructed: {len(emoji_messages)}")
    response = model.invoke(emoji_messages)
    print(f"AI Response: {response.content}")  # Expected: 😮

if __name__ == "__main__":
    main()
```

---

## PART 2: Template-Based Prompting

Templates allow you to create reusable, maintainable prompts with variables.

### Example 3: Basic Templates

**Code**: [`code/03_basic_template.py`](./code/03_basic_template.py)  
**Run**: `python 03-prompts-messages-outputs/code/03_basic_template.py`

**Example code:**

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

def main():
    # Create a reusable template
    template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
        ("human", "{text}"),
    ])

    model = ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )

    # Use the template multiple times with different values
    chain = template | model

    result1 = chain.invoke({
        "input_language": "English",
        "output_language": "French",
        "text": "Hello, how are you?",
    })

    print("French:", result1.content)

    result2 = chain.invoke({
        "input_language": "English",
        "output_language": "Spanish",
        "text": "Hello, how are you?",
    })

    print("Spanish:", result2.content)

if __name__ == "__main__":
    main()
```

---

## PART 3: Structured Outputs with Pydantic

Use Pydantic models to get type-safe, structured data from LLMs.

### Example 4: Structured Output

**Code**: [`code/04_structured_output.py`](./code/04_structured_output.py)  
**Run**: `python 03-prompts-messages-outputs/code/04_structured_output.py`

**Example code:**

```python
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()

# Define your output schema with Pydantic
class MovieReview(BaseModel):
    """Schema for a movie review analysis."""
    title: str = Field(description="The movie title")
    sentiment: str = Field(description="Overall sentiment: positive, negative, or neutral")
    rating: int = Field(description="Rating from 1-10")
    summary: str = Field(description="Brief summary of the review")

def main():
    model = ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )

    # Create a structured output model
    structured_model = model.with_structured_output(MovieReview)

    review_text = """
    I just watched "The Matrix" and it was absolutely mind-blowing! 
    The special effects were revolutionary and the story about reality 
    and perception really made me think. Keanu Reeves was perfect as Neo.
    A must-watch for any sci-fi fan!
    """

    result = structured_model.invoke(f"Analyze this movie review:\n{review_text}")

    print(f"Title: {result.title}")
    print(f"Sentiment: {result.sentiment}")
    print(f"Rating: {result.rating}/10")
    print(f"Summary: {result.summary}")

if __name__ == "__main__":
    main()
```

### Expected Output

```
Title: The Matrix
Sentiment: positive
Rating: 9/10
Summary: The reviewer found the movie mind-blowing with revolutionary special effects and thought-provoking themes about reality.
```

---

## 🎓 Key Takeaways

- **Messages** are best for dynamic, agent-like workflows
- **Templates** are best for reusable prompts with variables
- **Pydantic** enables type-safe structured outputs
- Choose based on your use case: flexibility vs. reusability

---

## 🗺️ Navigation

[← Previous: Chat Models](../02-chat-models/README.md) | [Back to Main](../README.md) | [Next: Function Calling & Tools →](../04-function-calling-tools/README.md)

---

## 💬 Questions?

[![Azure AI Foundry Discord](https://img.shields.io/badge/Discord-Azure_AI_Foundry_Community_Discord-blue?style=for-the-badge&logo=discord&color=5865f2&logoColor=fff)](https://aka.ms/foundry/discord)
