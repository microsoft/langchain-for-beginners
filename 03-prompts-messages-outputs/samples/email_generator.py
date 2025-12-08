"""
Email Response Generator

Run: python 03-prompts-messages-outputs/samples/email_generator.py
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

# Create reusable email template
email_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a customer service representative for {company_name}.
Write professional emails with a {tone} tone.
Always be helpful and provide clear next steps.""",
    ),
    (
        "human",
        """Generate an email response for this scenario:
Customer: {customer_name}
Issue Type: {issue_type}
Details: {details}

The email should be {tone} and address their concern appropriately.""",
    ),
])


def generate_email(
    company_name: str,
    customer_name: str,
    issue_type: str,
    details: str,
    tone: str,
):
    print("📧 Generating email...\n")
    print(f"Company: {company_name}")
    print(f"Customer: {customer_name}")
    print(f"Issue: {issue_type}")
    print(f"Tone: {tone}\n")
    print("─" * 80)

    chain = email_template | model

    result = chain.invoke({
        "company_name": company_name,
        "customer_name": customer_name,
        "issue_type": issue_type,
        "details": details,
        "tone": tone,
    })

    print(result.content)
    print("─" * 80 + "\n")


def main():
    print("📧 Email Response Generator\n")
    print("=" * 80 + "\n")

    # Scenario 1: Refund request - apologetic
    generate_email(
        "TechGadgets Inc.",
        "Sarah Johnson",
        "Refund Request",
        "Product arrived damaged and customer wants a full refund",
        "apologetic and empathetic",
    )

    # Scenario 2: Technical support - friendly
    generate_email(
        "CloudHost Solutions",
        "Mike Chen",
        "Technical Support",
        "Customer can't connect to their database and needs help troubleshooting",
        "friendly and helpful",
    )

    # Scenario 3: Exchange request - formal
    generate_email(
        "Fashion Forward",
        "Dr. Emily Roberts",
        "Exchange Request",
        "Wrong size received, needs to exchange for a different size",
        "formal and professional",
    )

    print("=" * 80)
    print("\n✅ All emails generated successfully!")
    print("💡 Same template, different contexts and tones!")


if __name__ == "__main__":
    main()
