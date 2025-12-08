"""
Multi-Language Translation System

Run: python 03-prompts-messages-outputs/samples/translator.py
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

# Translation template with formality support
translation_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a professional translator. Translate text to {target_language} with {formality} formality.
Maintain the original meaning while adapting to cultural context.""",
    ),
    (
        "human",
        """Translate this text to {target_language} ({formality} tone):

{text}

Provide only the translation, no explanations.""",
    ),
])

languages = {
    "1": "Spanish",
    "2": "French",
    "3": "German",
    "4": "Japanese",
    "5": "Italian",
}

formality_levels = {
    "1": "casual",
    "2": "formal",
}


def translate_text(target_language: str, formality: str, text: str):
    print("\n🔄 Translating...\n")

    chain = translation_template | model

    result = chain.invoke({
        "target_language": target_language,
        "formality": formality,
        "text": text,
    })

    print("─" * 80)
    print(f"Source (English): {text}")
    print(f"Target Language: {target_language}")
    print(f"Formality: {formality}")
    print("─" * 80)
    print(f"Translation: {result.content}")
    print("─" * 80 + "\n")


def main():
    print("🌍 Multi-Language Translation System\n")
    print("=" * 80 + "\n")

    # Check if running in CI mode
    if os.environ.get("CI") == "true":
        print("Running in CI mode - testing with sample data\n")

        # Test translation
        translate_text("Spanish", "formal", "Good morning. How can I assist you today?")
        translate_text("French", "casual", "Thanks for your help! See you later.")

        print("✅ Translation system working correctly!")
        return

    # Interactive mode
    try:
        print("Select target language:")
        for key, lang in languages.items():
            print(f"  {key}. {lang}")
        lang_choice = input("\nEnter choice (1-5): ")
        target_language = languages.get(lang_choice)

        if not target_language:
            print("❌ Invalid language choice")
            return

        print("\nSelect formality level:")
        for key, level in formality_levels.items():
            print(f"  {key}. {level}")
        formality_choice = input("\nEnter choice (1-2): ")
        formality = formality_levels.get(formality_choice)

        if not formality:
            print("❌ Invalid formality choice")
            return

        text = input("\nEnter text to translate: ")

        if not text.strip():
            print("❌ No text provided")
            return

        translate_text(target_language, formality, text)

        print("✅ Translation complete!")
    except (EOFError, KeyboardInterrupt):
        print("\n\n👋 Goodbye!")


if __name__ == "__main__":
    main()
