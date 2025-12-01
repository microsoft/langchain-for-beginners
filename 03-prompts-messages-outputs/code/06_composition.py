"""
Prompt Composition
Run: python 03-prompts-messages-outputs/code/06_composition.py

🤖 Try asking GitHub Copilot Chat (https://github.com/features/copilot):
- "How does template.partial() work and when would I use it?"
- "What's the benefit of composing prompts vs using one large template?"
"""

import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()


def educator_example():
    print("1️⃣  Example: Composable Educator Prompts\n")

    model = ChatOpenAI(model=os.environ.get("AI_MODEL", "gpt-4o-mini"))

    # Reusable prompt pieces
    system_role = "You are an expert {domain} educator."
    teaching_context = (
        "Teaching level: {level}\nAudience: {audience}\nGoal: Clear, accurate explanations"
    )
    task_instruction = "Explain {topic} in simple terms with an example."

    # Compose them together
    template = ChatPromptTemplate.from_messages([
        ("system", system_role + "\n\n" + teaching_context),
        ("human", task_instruction),
    ])

    chain = template | model

    # Scenario 1: Teaching beginners
    print("📚 Beginner Level:\n")
    result1 = chain.invoke({
        "domain": "programming",
        "level": "beginner",
        "audience": "high school students with no coding experience",
        "topic": "variables",
    })
    print(result1.content)

    # Scenario 2: Teaching intermediate learners
    print("\n" + "=" * 80)
    print("\n📚 Intermediate Level:\n")
    result2 = chain.invoke({
        "domain": "programming",
        "level": "intermediate",
        "audience": "college students who know basic programming",
        "topic": "closures in Python",
    })
    print(result2.content)


def customer_service_example():
    print("\n" + "=" * 80)
    print("\n2️⃣  Example: Customer Service Templates\n")

    model = ChatOpenAI(model=os.environ.get("AI_MODEL", "gpt-4o-mini"))

    # Composable pieces for customer service
    brand_voice = "You represent {company_name}, known for {brand_personality}."
    service_policy = "Policy: {policy}\nPriority: {priority}"
    response_guidelines = "Always: {guidelines}"

    template = ChatPromptTemplate.from_messages([
        ("system", f"{brand_voice}\n\n{service_policy}\n\n{response_guidelines}"),
        ("human", "Customer issue: {issue}"),
    ])

    chain = template | model

    # Different company scenarios using same template
    result = chain.invoke({
        "company_name": "TechGadgets Inc.",
        "brand_personality": "being helpful, friendly, and technically knowledgeable",
        "policy": "30-day returns, free shipping on orders over $50",
        "priority": "Customer satisfaction and quick resolution",
        "guidelines": "Be empathetic, provide clear steps, offer alternatives",
        "issue": "Customer received wrong item and needs replacement urgently",
    })

    print(result.content)


def partial_template_example():
    print("\n" + "=" * 80)
    print("\n3️⃣  Example: Partial Templates (Pre-fill Some Variables)\n")

    model = ChatOpenAI(model=os.environ.get("AI_MODEL", "gpt-4o-mini"))

    # Create a template with many variables
    template = ChatPromptTemplate.from_messages([
        ("system", "You are a {role} at {company} specializing in {specialty}."),
        ("human", "{task}"),
    ])

    # Create a partial template with some values pre-filled
    partial_template = template.partial(
        role="Technical Writer",
        company="DevDocs Pro",
    )

    chain = partial_template | model

    # Now only need to provide remaining variables
    print("Pre-filled: role = Technical Writer, company = DevDocs Pro\n")

    result1 = chain.invoke({
        "specialty": "API documentation",
        "task": "Write a brief intro paragraph for a REST API guide",
    })

    print("API Documentation task:")
    print(result1.content)

    print("\n---\n")

    result2 = chain.invoke({
        "specialty": "user guides",
        "task": "Write a getting started section for a mobile app",
    })

    print("User Guide task:")
    print(result2.content)


def main():
    print("🔗 Prompt Composition Examples\n")
    print("=" * 80)

    educator_example()
    customer_service_example()
    partial_template_example()

    print("\n" + "=" * 80)
    print("\n✅ Benefits of Composition:")
    print("   - Reuse prompt pieces across different scenarios")
    print("   - Maintain consistency in your brand voice")
    print("   - Easy to update - change once, affects all uses")
    print("   - Partial templates reduce repetitive variable passing")


if __name__ == "__main__":
    main()
