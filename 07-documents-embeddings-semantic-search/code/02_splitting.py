"""
Text Splitting

Run: python 07-documents-embeddings-semantic-search/code/02_splitting.py

🤖 Try asking GitHub Copilot Chat (https://github.com/features/copilot):
- "How do I determine the optimal chunk size for my documents?"
- "Can I split on specific delimiters like headings or paragraphs?"
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def main():
    print("✂️  Text Splitting Example\n")

    long_text = """
Artificial Intelligence and Machine Learning

Artificial Intelligence (AI) is transforming how we interact with technology.
From virtual assistants to recommendation systems, AI is becoming an integral
part of our daily lives.

Machine Learning Basics

Machine learning is a subset of AI that enables systems to learn and improve
from experience without being explicitly programmed. It focuses on developing
computer programs that can access data and use it to learn for themselves.

The process of learning begins with observations or data, such as examples,
direct experience, or instruction, in order to look for patterns in data and
make better decisions in the future.

Types of Machine Learning

1. Supervised Learning: The algorithm learns from labeled training data
2. Unsupervised Learning: The algorithm finds patterns in unlabeled data
3. Reinforcement Learning: The algorithm learns through trial and error

Deep Learning

Deep learning is a subset of machine learning that uses neural networks with
multiple layers. These networks can learn increasingly complex patterns as
data passes through each layer.

Applications include image recognition, natural language processing, speech
recognition, and autonomous vehicles. The field continues to evolve rapidly
with new architectures and techniques emerging regularly.
""".strip()

    print("Original text length:", len(long_text), "characters\n")

    # Create splitter with specific configuration
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
    )

    docs = splitter.create_documents([long_text])

    print(f"✂️  Split into {len(docs)} chunks\n")
    print("=" * 80)

    # Display each chunk
    for i, doc in enumerate(docs):
        print(f"\n📄 Chunk {i + 1}/{len(docs)}")
        print("─" * 80)
        print(doc.page_content)
        print(f"\n📏 Length: {len(doc.page_content)} characters")

    print("\n" + "=" * 80)
    print("\n💡 Key Observations:")
    print(f"   - Original: {len(long_text)} characters")
    print(f"   - Chunks: {len(docs)}")
    print(f"   - Average chunk size: {round(len(long_text) / len(docs))} characters")
    print("   - Overlap: 50 characters ensures context is preserved")


if __name__ == "__main__":
    main()
