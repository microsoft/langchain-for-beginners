"""
Structured Output with Pydantic Example
Run: python 03-prompts-messages-outputs/code/02_structured_output.py

This example demonstrates how to extract structured data using Pydantic models.
"""

import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Define the structure we want to extract
class Product(BaseModel):
    """Information about a product."""
    name: str = Field(description="The product name")
    price: float = Field(description="The price in dollars")
    category: str = Field(description="The product category")
    in_stock: bool = Field(description="Whether the product is in stock")

def main():
    # Initialize the model with structured output
    model = ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
        temperature=0,
    )

    # Bind the Pydantic model to get structured output
    structured_model = model.with_structured_output(Product)

    # Create a prompt template
    template = ChatPromptTemplate.from_messages([
        ("system", "Extract product information from the user's description."),
        ("human", "{description}")
    ])

    # Create the chain
    chain = template | structured_model

    print("🏷️ Structured Output Extraction\n")
    print("=" * 80)

    # Example 1: Extract product info
    description1 = "Premium wireless headphones with noise cancellation, only $199, available now"
    result1 = chain.invoke({"description": description1})
    
    print(f"\n📝 Input: {description1}")
    print(f"\n✅ Extracted:")
    print(f"   Name: {result1.name}")
    print(f"   Price: ${result1.price}")
    print(f"   Category: {result1.category}")
    print(f"   In Stock: {result1.in_stock}")

    # Example 2: Extract another product
    description2 = "Organic cotton t-shirt in blue, comfortable fit, $29.99, currently out of stock"
    result2 = chain.invoke({"description": description2})
    
    print(f"\n📝 Input: {description2}")
    print(f"\n✅ Extracted:")
    print(f"   Name: {result2.name}")
    print(f"   Price: ${result2.price}")
    print(f"   Category: {result2.category}")
    print(f"   In Stock: {result2.in_stock}")

    print("\n" + "=" * 80)
    print("\n✅ Structured outputs provide type-safe data!")
    print("💡 Perfect for building reliable applications")

if __name__ == "__main__":
    main()
