"""
Complex Structured Data with Pydantic Schemas
Run: python 03-prompts-messages-outputs/code/08_pydantic_schemas.py

🤖 Try asking GitHub Copilot Chat (https://github.com/features/copilot):
- "How do I add validation constraints like min/max to Pydantic model fields?"
- "How would I handle arrays of nested objects in a schema?"
"""

import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


# Define a complex nested schema
class Headquarters(BaseModel):
    """Company headquarters location."""

    city: str = Field(description="City name")
    country: str = Field(description="Country name")


class Company(BaseModel):
    """Information about a company."""

    name: str = Field(description="Company name")
    founded: int = Field(description="Year the company was founded")
    headquarters: Headquarters = Field(description="Company headquarters location")
    products: list[str] = Field(description="List of main products or services")
    employee_count: int = Field(description="Approximate number of employees")
    is_public: bool = Field(description="Whether the company is publicly traded")


def main():
    print("🏢 Complex Structured Output Example\n")

    model = ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )

    # Create structured model
    structured_model = model.with_structured_output(Company)

    # Create a prompt template
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            "Extract company information from the text. If information is not available, "
            "make reasonable estimates based on common knowledge.",
        ),
        ("human", "{text}"),
    ])

    # Combine template with structured output
    chain = template | structured_model

    print("🧪 Extracting data from company descriptions:\n")
    print("=" * 80)

    # Test 1: Microsoft
    print("\n1️⃣  Microsoft:\n")
    company_info1 = """
    Microsoft was founded in 1975 and is headquartered in Redmond, Washington.
    The company is publicly traded and has over 220,000 employees worldwide.
    Their main products include Windows, Office, Azure, and Xbox.
    """

    result1 = chain.invoke({"text": company_info1})

    print("✅ Extracted Company Data:")
    print(result1.model_dump_json(indent=2))
    print("\n📊 Type-safe access:")
    print(f"   {result1.name} ({'Public' if result1.is_public else 'Private'})")
    print(f"   Founded: {result1.founded}")
    print(f"   Location: {result1.headquarters.city}, {result1.headquarters.country}")
    print(f"   Products: {', '.join(result1.products)}")
    print(f"   Employees: {result1.employee_count:,}")

    # Test 2: Adobe
    print("\n" + "=" * 80)
    print("\n2️⃣  Adobe:\n")
    company_info2 = """
    Adobe was founded in 1982 and is headquartered in San Jose, California.
    The company is publicly traded and has approximately 30,000 employees.
    Their main products include Photoshop, Illustrator, Acrobat, and Creative Cloud.
    """

    result2 = chain.invoke({"text": company_info2})

    print("✅ Extracted Company Data:")
    print(result2.model_dump_json(indent=2))

    # Test 3: Netflix
    print("\n" + "=" * 80)
    print("\n3️⃣  Netflix:\n")
    company_info3 = """
    Netflix started in 1997 in Los Gatos, California.
    It's a publicly traded streaming service with about 12,800 employees.
    Main offerings include streaming video, original content, and DVD rentals (discontinued).
    """

    result3 = chain.invoke({"text": company_info3})

    print("✅ Extracted Company Data:")
    print(result3.model_dump_json(indent=2))

    print("\n" + "=" * 80)
    print("\n💡 Complex Schema Features:")
    print("   - ✅ Nested objects (headquarters with city/country)")
    print("   - ✅ Arrays (products list)")
    print("   - ✅ Multiple data types (strings, numbers, booleans)")
    print("   - ✅ Validation (ensures correct types)")
    print("   - ✅ Descriptions guide the AI's extraction")
    print("\n🎯 Use Cases:")
    print("   - Data extraction from documents")
    print("   - Form filling from natural language")
    print("   - Structured database inserts")
    print("   - API response formatting")
    print("   - Classification with predefined categories")


if __name__ == "__main__":
    main()
