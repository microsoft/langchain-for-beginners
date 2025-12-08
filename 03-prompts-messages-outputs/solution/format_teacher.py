"""
Challenge 2 Solution: Few-Shot Format Teacher
Run: python 03-prompts-messages-outputs/solution/format_teacher.py
"""

import json
import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

model = ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )

# Teaching examples
examples = [
    {
        "input": "Premium wireless headphones with noise cancellation, $199",
        "output": json.dumps(
            {
                "name": "Premium Wireless Headphones",
                "price": "$199.00",
                "category": "Electronics",
                "highlight": "Noise cancellation",
            },
            indent=2,
        ),
    },
    {
        "input": "Organic cotton t-shirt in blue, comfortable fit, $29.99",
        "output": json.dumps(
            {
                "name": "Organic Cotton T-Shirt",
                "price": "$29.99",
                "category": "Clothing",
                "highlight": "Organic cotton, comfortable fit",
            },
            indent=2,
        ),
    },
    {
        "input": "Gaming laptop with RTX 4070, 32GB RAM, $1,499",
        "output": json.dumps(
            {
                "name": "Gaming Laptop",
                "price": "$1,499.00",
                "category": "Computers",
                "highlight": "RTX 4070, 32GB RAM",
            },
            indent=2,
        ),
    },
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

# Final template
final_template = ChatPromptTemplate.from_messages([
    (
        "system",
        "Convert product descriptions into JSON format. Follow the examples exactly. "
        "Output ONLY valid JSON, no additional text.",
    ),
    few_shot_template,
    ("human", "{input}"),
])


def convert_product(description: str):
    print(f"\n📦 Input: {description}")
    print("─" * 80)

    chain = final_template | model
    result = chain.invoke({"input": description})

    try:
        # Parse to validate JSON
        parsed = json.loads(str(result.content))
        print("✅ Valid JSON output:")
        print(json.dumps(parsed, indent=2))

        # Validate structure
        required_fields = ["name", "price", "category", "highlight"]
        has_all_fields = all(field in parsed for field in required_fields)

        if has_all_fields:
            print("\n✅ All required fields present")
        else:
            print("\n⚠️  Warning: Missing some required fields")
    except json.JSONDecodeError:
        print("❌ Invalid JSON output:")
        print(result.content)


def main():
    print("🎓 Few-Shot Format Teacher\n")
    print("=" * 80)

    test_products = [
        "Stainless steel water bottle, keeps drinks cold for 24 hours, $24.99",
        "Leather messenger bag with laptop compartment, handcrafted, $149",
        "Smart watch with heart rate monitor and GPS, waterproof, $299.99",
        "Ergonomic office chair with lumbar support, adjustable height, $399",
    ]

    for product in test_products:
        convert_product(product)

    print("\n" + "=" * 80)
    print("\n✅ Format successfully taught using few-shot examples!")
    print("💡 Few-shot learning is great for consistent structured output")


if __name__ == "__main__":
    main()
