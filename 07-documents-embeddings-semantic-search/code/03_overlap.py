"""
Comparing Chunk Overlap

Run: python 07-documents-embeddings-semantic-search/code/03_overlap.py

🤖 Try asking GitHub Copilot Chat (https://github.com/features/copilot):
- "What percentage overlap is recommended for different chunk sizes?"
- "Can too much overlap cause duplicate information in search results?"
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def main():
    print("🔄 Chunk Overlap Comparison\n")

    text = """
The mitochondria is often called the powerhouse of the cell. This organelle
is responsible for producing ATP, the energy currency that powers cellular
processes. Mitochondria have their own DNA, separate from the cell's nuclear
DNA, which supports the theory that they were once independent organisms.
Through a process called cellular respiration, mitochondria convert nutrients
into usable energy. This process involves several complex steps including
glycolysis, the Krebs cycle, and the electron transport chain.
""".strip()

    print("Original text:")
    print("─" * 80)
    print(text)
    print("\n" + "=" * 80)

    # Splitter with NO overlap
    print("\n1️⃣  Splitting with NO overlap:\n")

    no_overlap = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=0,
    )

    chunks1 = no_overlap.create_documents([text])

    for i, doc in enumerate(chunks1):
        print(f'Chunk {i + 1}: "{doc.page_content}"')
        print()

    print("⚠️  Notice: Context may be lost between chunks!\n")

    # Splitter WITH overlap
    print("=" * 80)
    print("\n2️⃣  Splitting WITH overlap (30 characters):\n")

    with_overlap = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=30,
    )

    chunks2 = with_overlap.create_documents([text])

    for i, doc in enumerate(chunks2):
        print(f'Chunk {i + 1}: "{doc.page_content}"')

        # Highlight overlap if not the first chunk
        if i > 0:
            overlap = doc.page_content[:30]
            print(f'   ↪️  Overlaps with: "{overlap}..."')
        print()

    print("✅ Notice: Overlapping text preserves context!\n")

    print("=" * 80)
    print("\n📊 Comparison:")
    print(f"   Without overlap: {len(chunks1)} chunks")
    print(f"   With overlap: {len(chunks2)} chunks")
    print("\n💡 Recommendation: Use 10-20% overlap for most use cases")


if __name__ == "__main__":
    main()
