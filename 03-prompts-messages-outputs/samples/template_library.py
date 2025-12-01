"""
Prompt Template Library

Run: python 03-prompts-messages-outputs/samples/template_library.py
"""

import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

model = ChatOpenAI(model=os.environ.get("AI_MODEL", "gpt-4o-mini"))

# Template Library
templates = {
    "code_explainer": {
        "name": "Code Explainer",
        "description": "Explains code snippets in plain English",
        "variables": ["code", "language"],
        "template": ChatPromptTemplate.from_messages([
            ("system", "You are a programming instructor. Explain code clearly to beginners."),
            (
                "human",
                """Explain this {language} code:

```{language}
{code}
```

Describe what it does, how it works, and any key concepts.""",
            ),
        ]),
    },
    "summarizer": {
        "name": "Text Summarizer",
        "description": "Creates concise summaries of long text",
        "variables": ["text", "length"],
        "template": ChatPromptTemplate.from_messages([
            ("system", "You are a professional summarizer. Create clear, {length} summaries."),
            ("human", "Summarize this text:\n\n{text}"),
        ]),
    },
    "creative_writer": {
        "name": "Creative Writing Prompt",
        "description": "Generates creative writing based on prompts",
        "variables": ["genre", "theme", "length"],
        "template": ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a creative writer. Write {length} {genre} stories that are engaging and well-crafted.",
            ),
            ("human", "Write a story about: {theme}"),
        ]),
    },
    "data_formatter": {
        "name": "Data Formatter",
        "description": "Formats data into specific structures",
        "variables": ["data", "format"],
        "template": ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a data formatting expert. Convert data to {format} format with proper structure.",
            ),
            ("human", "Format this data:\n\n{data}"),
        ]),
    },
    "question_answerer": {
        "name": "Question Answerer",
        "description": "Answers questions with specific expertise",
        "variables": ["question", "expertise"],
        "template": ChatPromptTemplate.from_messages([
            ("system", "You are an expert in {expertise}. Provide accurate, detailed answers."),
            ("human", "{question}"),
        ]),
    },
}


def list_templates():
    print("\n📚 Available Templates:\n")
    for i, (key, template) in enumerate(templates.items(), 1):
        print(f"{i}. {template['name']}")
        print(f"   {template['description']}")
        print(f"   Variables: {', '.join(template['variables'])}\n")


def execute_template(template_key: str, variables: dict[str, str]):
    template_info = templates[template_key]

    print("\n🔄 Processing...\n")
    print("─" * 80)

    chain = template_info["template"] | model
    result = chain.invoke(variables)

    print(result.content)
    print("─" * 80 + "\n")


def main():
    print("📚 Prompt Template Library\n")
    print("=" * 80)

    # Check if running in CI mode
    if os.environ.get("CI") == "true":
        print("\nRunning in CI mode - testing templates\n")

        # Test Code Explainer
        print("Testing Code Explainer Template:")
        execute_template("code_explainer", {
            "code": "total = sum(numbers)",
            "language": "Python",
        })

        # Test Summarizer
        print("Testing Summarizer Template:")
        execute_template("summarizer", {
            "text": "Artificial intelligence is transforming the world. Machine learning enables computers to learn from data without explicit programming. Deep learning uses neural networks to solve complex problems.",
            "length": "brief",
        })

        print("✅ Template library working correctly!")
        return

    # Interactive mode
    list_templates()

    template_keys = list(templates.keys())
    try:
        choice = input("Select template (1-5): ")
        template_index = int(choice) - 1

        if template_index < 0 or template_index >= len(template_keys):
            print("❌ Invalid choice")
            return

        template_key = template_keys[template_index]
        template_info = templates[template_key]

        print(f"\n✅ Selected: {template_info['name']}\n")

        # Collect variables
        variables = {}
        for variable in template_info["variables"]:
            value = input(f"Enter {variable}: ")
            variables[variable] = value

        execute_template(template_key, variables)

        print("✅ Complete!")
    except (EOFError, KeyboardInterrupt):
        print("\n\n👋 Goodbye!")


if __name__ == "__main__":
    main()
