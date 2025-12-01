"""
Token Usage Tracker
Run: python 02-chat-models/samples/token_tracker.py
"""

import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()


@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float


@dataclass
class CallRecord:
    call_number: int
    query: str
    usage: TokenUsage


class TokenTracker:
    # Pricing per 1M tokens (approximate for gpt-4o-mini)
    INPUT_COST_PER_MILLION = 0.15  # $0.15 per 1M input tokens
    OUTPUT_COST_PER_MILLION = 0.60  # $0.60 per 1M output tokens
    WARNING_THRESHOLD = 10000  # Warn at 10k tokens

    def __init__(self):
        self.calls: list[CallRecord] = []
        self.call_count: int = 0

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        input_cost = (input_tokens / 1_000_000) * self.INPUT_COST_PER_MILLION
        output_cost = (output_tokens / 1_000_000) * self.OUTPUT_COST_PER_MILLION
        return input_cost + output_cost

    def track_call(self, model: ChatOpenAI, query: str) -> str:
        self.call_count += 1

        print(f"\n🔄 Call #{self.call_count}: Processing...")

        response = model.invoke(query)

        # Get token usage from response metadata
        usage = response.usage_metadata

        if not usage:
            print("⚠️  Token usage data not available")
            return str(response.content)

        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
        cost = self.calculate_cost(prompt_tokens, completion_tokens)

        call_record = CallRecord(
            call_number=self.call_count,
            query=query,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost=cost,
            ),
        )

        self.calls.append(call_record)

        # Display call info
        print("─" * 50)
        print(f"📝 Input: \"{query[:40]}...\"")
        print(f"  Input tokens: {prompt_tokens}")
        print(f"  Output tokens: {completion_tokens}")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Cost: ${cost:.6f}")

        # Check warning threshold
        total_session_tokens = self.get_total_tokens()
        if total_session_tokens > self.WARNING_THRESHOLD:
            print(f"\n⚠️  WARNING: Session total ({total_session_tokens} tokens) exceeds threshold!")

        return str(response.content)

    def get_total_tokens(self) -> int:
        return sum(call.usage.total_tokens for call in self.calls)

    def get_total_cost(self) -> float:
        return sum(call.usage.cost for call in self.calls)

    def display_report(self):
        print("\n" + "=" * 60)
        print("📊 TOKEN USAGE REPORT")
        print("=" * 60 + "\n")

        for call in self.calls:
            print(f"Call #{call.call_number}")
            print(f"  Query: \"{call.query[:50]}...\"")
            print(f"  Input: {call.usage.prompt_tokens} tokens")
            print(f"  Output: {call.usage.completion_tokens} tokens")
            print(f"  Total: {call.usage.total_tokens} tokens")
            print(f"  Cost: ${call.usage.cost:.6f}")
            print()

        print("=" * 60)
        print("SESSION SUMMARY")
        print("=" * 60)
        print(f"Total Calls: {len(self.calls)}")
        print(f"Total Tokens: {self.get_total_tokens():,}")
        print(f"Total Cost: ${self.get_total_cost():.6f}")
        avg_tokens = self.get_total_tokens() // len(self.calls) if self.calls else 0
        avg_cost = self.get_total_cost() / len(self.calls) if self.calls else 0
        print(f"Average Tokens/Call: {avg_tokens}")
        print(f"Average Cost/Call: ${avg_cost:.6f}")

        # Breakdown
        total_input = sum(call.usage.prompt_tokens for call in self.calls)
        total_output = sum(call.usage.completion_tokens for call in self.calls)
        total = self.get_total_tokens()

        print()
        print("Token Breakdown:")
        if total > 0:
            print(f"  Input: {total_input:,} ({total_input / total * 100:.1f}%)")
            print(f"  Output: {total_output:,} ({total_output / total * 100:.1f}%)")

        print()
        print("Cost Breakdown:")
        print(f"  Input: ${self.calculate_cost(total_input, 0):.6f}")
        print(f"  Output: ${self.calculate_cost(0, total_output):.6f}")

        print("=" * 60)

    def export_csv(self) -> str:
        csv = "Call,Query,InputTokens,OutputTokens,TotalTokens,Cost\n"

        for call in self.calls:
            escaped_query = call.query.replace('"', '""')
            csv += f'{call.call_number},"{escaped_query}",{call.usage.prompt_tokens},{call.usage.completion_tokens},{call.usage.total_tokens},{call.usage.cost:.6f}\n'

        return csv


def main():
    print("📊 Token Usage Tracker\n")
    print("=" * 60 + "\n")

    model = ChatOpenAI(model=os.environ.get("AI_MODEL", "gpt-4o-mini"))

    tracker = TokenTracker()

    queries = [
        "What is Python?",
        "Explain async/await in Python in detail",
        "Write a short example of a Python HTTP server",
        "What are the benefits of using type hints in Python?",
        "Explain the difference between SQL and NoSQL databases",
    ]

    print("🚀 Running test queries...")

    for query in queries:
        tracker.track_call(model, query)

    # Display final report
    tracker.display_report()

    # Show CSV export
    print("\n📄 CSV Export Preview:")
    print(tracker.export_csv())

    print("💡 Token Tracking Features:")
    print("   ✓ Tracks tokens per call (input, output, total)")
    print("   ✓ Calculates costs based on current pricing")
    print("   ✓ Cumulative session tracking")
    print("   ✓ Warning system for high usage")
    print("   ✓ Detailed reports and breakdowns")
    print("   ✓ CSV export capability")
    print()

    print("💰 Cost Optimization Tips:")
    print("   • Use gpt-4o-mini for simple tasks")
    print("   • Keep prompts concise")
    print("   • Use streaming for better UX without extra cost")
    print("   • Cache responses when possible")
    print("   • Monitor usage regularly")


if __name__ == "__main__":
    main()
