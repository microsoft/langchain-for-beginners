"""
Dynamic Prompt Builder

Run: python 03-prompts-messages-outputs/samples/prompt_builder.py
"""

import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

model = ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )

# Modular components for composing prompts
role_prompts = {
    "Teacher": "You are a patient teacher who explains concepts clearly to students.",
    "Expert": "You are a domain expert with deep technical knowledge.",
    "Friend": "You are a friendly peer having a casual conversation.",
    "Professional": "You are a professional consultant providing formal advice.",
}

style_prompts = {
    "Concise": "Keep your explanation brief and to the point (2-3 sentences).",
    "Detailed": "Provide a comprehensive, detailed explanation with examples.",
    "Creative": "Use analogies, metaphors, and creative explanations.",
    "Technical": "Use precise technical terminology and scientific accuracy.",
}

format_prompts = {
    "Bullet points": "Format your response as bullet points.",
    "Paragraph": "Format your response as flowing paragraphs.",
    "Step-by-step": "Format your response as numbered steps.",
    "Q&A": "Format your response as questions and answers.",
}


def build_template(role: str, style: str, format_type: str) -> ChatPromptTemplate:
    system_message = f"""{role_prompts[role]}
{style_prompts[style]}
{format_prompts[format_type]}"""

    return ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "{question}"),
    ])


def answer_question(question: str, role: str, style: str, format_type: str):
    print("📝 Configuration:")
    print(f"   Role: {role}")
    print(f"   Style: {style}")
    print(f"   Format: {format_type}")
    print("─" * 80)

    template = build_template(role, style, format_type)
    chain = template | model

    result = chain.invoke({"question": question})

    print(result.content)
    print("─" * 80 + "\n")


def main():
    print("🏗️  Dynamic Prompt Builder\n")
    print("=" * 80 + "\n")

    test_question = "How does photosynthesis work?"

    print(f'Question: "{test_question}"\n')
    print("=" * 80 + "\n")

    # Test Combination 1: Teacher + Detailed + Step-by-step
    print("📚 Combination 1: Teacher + Detailed + Step-by-step\n")
    answer_question(test_question, "Teacher", "Detailed", "Step-by-step")

    # Test Combination 2: Expert + Technical + Bullet points
    print("🔬 Combination 2: Expert + Technical + Bullet points\n")
    answer_question(test_question, "Expert", "Technical", "Bullet points")

    # Test Combination 3: Friend + Concise + Paragraph
    print("💬 Combination 3: Friend + Concise + Paragraph\n")
    answer_question(test_question, "Friend", "Concise", "Paragraph")

    # Test Combination 4: Professional + Creative + Q&A
    print("💼 Combination 4: Professional + Creative + Q&A\n")
    answer_question(test_question, "Professional", "Creative", "Q&A")

    print("=" * 80)
    print("\n✅ Dynamic prompt builder demonstration complete!")
    print("💡 Notice how the same question gets very different responses")
    print("   based on the combination of role, style, and format!")


if __name__ == "__main__":
    main()
