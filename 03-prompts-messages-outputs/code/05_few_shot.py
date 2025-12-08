"""
Few-Shot Prompting
Run: python 03-prompts-messages-outputs/code/05_few_shot.py

🤖 Try asking GitHub Copilot Chat (https://github.com/features/copilot):
- "How many examples should I provide for effective few-shot prompting?"
- "Can I dynamically select which examples to include based on the input?"
"""

import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()


def emotion_to_emoji_example():
    print("1️⃣  Example: Emotion to Emoji Converter\n")

    model = ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )

    examples = [
        {"input": "happy", "output": "😊"},
        {"input": "sad", "output": "😢"},
        {"input": "excited", "output": "🎉"},
        {"input": "angry", "output": "😠"},
    ]

    # Create example template
    example_template = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}"),
    ])

    # Create few-shot template
    few_shot_template = FewShotChatMessagePromptTemplate(
        example_prompt=example_template,
        examples=examples,
    )

    # Combine with the final question
    final_template = ChatPromptTemplate.from_messages([
        ("system", "Convert emotions to emojis based on these examples:"),
        few_shot_template,
        ("human", "{input}"),
    ])

    chain = final_template | model
    test_emotions = ["surprised", "confused", "tired", "proud"]

    for emotion in test_emotions:
        result = chain.invoke({"input": emotion})
        print(f"{emotion} → {result.content}")


def code_comment_example():
    print("\n" + "=" * 80)
    print("\n2️⃣  Example: Code Comment Generator\n")

    model = ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )

    # Examples of code → comment pairs
    examples = [
        {
            "code": "total = sum(numbers)",
            "comment": "# Calculates the sum of all numbers in the list",
        },
        {
            "code": "users = [u for u in data if u.active]",
            "comment": "# Filters the data list to only include active users",
        },
        {
            "code": "db.save(record)",
            "comment": "# Saves the record to the database",
        },
    ]

    example_template = ChatPromptTemplate.from_messages([
        ("human", "Code: {code}"),
        ("ai", "{comment}"),
    ])

    few_shot_template = FewShotChatMessagePromptTemplate(
        example_prompt=example_template,
        examples=examples,
    )

    final_template = ChatPromptTemplate.from_messages([
        ("system", "Generate clear, concise comments for code based on these examples:"),
        few_shot_template,
        ("human", "Code: {code}"),
    ])

    chain = final_template | model
    test_code = [
        "sorted_items = sorted(items, key=lambda x: x.price)",
        "if user.role == 'admin': return True",
    ]

    for code in test_code:
        result = chain.invoke({"code": code})
        print(f"Code: {code}")
        print(f"{result.content}\n")


def main():
    print("💡 Few-Shot Prompting Examples\n")
    print("=" * 80)

    emotion_to_emoji_example()
    code_comment_example()

    print("=" * 80)
    print("\n✅ Few-shot prompting teaches AI by example")
    print("💡 More reliable than just instructions alone")
    print("🎯 Great for teaching specific formats or styles")


if __name__ == "__main__":
    main()
