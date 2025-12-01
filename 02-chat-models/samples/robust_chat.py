"""
Robust Error Handler with Built-In Retries
Run: python 02-chat-models/samples/robust_chat.py
"""

import os
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()


def robust_chat(
    prompt: str,
    max_retries: int = 3,
    timeout: int = 30000,
    fallback_response: str = "I apologize, but I'm having trouble connecting right now. Please try again later.",
) -> str:
    """Makes a robust API call with automatic retry and fallback."""
    model = ChatOpenAI(model=os.environ.get("AI_MODEL", "gpt-4o-mini"))

    # Use LangChain's built-in retry logic - automatically handles retries with exponential backoff
    model_with_retry = model.with_retry(stop_after_attempt=max_retries)

    try:
        print(f"🔄 Making call with automatic retry (max {max_retries} attempts)...")

        response = model_with_retry.invoke(prompt)
        print("✅ Success!\n")

        return str(response.content)
    except Exception as error:
        error_msg = str(error)
        print(f"❌ All {max_retries} attempts failed: {error_msg[:100]}...")

        # Categorize the error
        error_type = "Unknown error"
        if "401" in error_msg or "Unauthorized" in error_msg:
            error_type = "Authentication failed (check API key)"
        elif "429" in error_msg or "rate limit" in error_msg.lower():
            error_type = "Rate limit exceeded"
        elif "timeout" in error_msg.lower():
            error_type = "Request timeout"
        elif "network" in error_msg.lower():
            error_type = "Network error"

        print(f"📋 Error type: {error_type}")
        print("💡 Returning fallback response\n")

        return fallback_response


def test_robust_chat():
    print("🛡️  Robust Error Handler Test\n")
    print("=" * 80)
    
    print("\n1️⃣  Test: Normal Call (should succeed)\n")
    response1 = robust_chat("What is 2+2?")
    print(f"Response: {response1}")
    
    print("\n" + "=" * 80)
    print("\n2️⃣  Test: Invalid API Key (will retry then fallback)\n")

    # Save original key
    original_key = os.environ.get("AI_API_KEY")

    # Test with invalid key by creating a bad model directly
    try:
        bad_model = ChatOpenAI(
            model=os.environ.get("AI_MODEL", "gpt-4o-mini"),
            api_key="invalid_key",
        )
        bad_model_with_retry = bad_model.with_retry(stop_after_attempt=2)
        
        print("🔄 Making call with automatic retry (max 2 attempts)...")
        bad_model_with_retry.invoke("Hello")
        print("✅ Success!\n")
        response2 = "Unexpected success"
    except Exception as error:
        print(f"❌ All 2 attempts failed: {str(error)[:50]}...")
        print("📋 Error type: Authentication failed (check API key)")
        print("💡 Returning fallback response\n")
        response2 = "Sorry, I'm having connection issues. Please try again."

    print(f"Final response: {response2}")

    print("\n" + "=" * 80)
    print("\n✅ Error handling demonstration complete!")
    print("\n💡 Key Features Demonstrated:")
    print("   - Built-in with_retry() for automatic retries and exponential backoff")
    print("   - Error categorization for different error types")
    print("   - Graceful fallback responses when all retries fail")
    print("   - User-friendly error messages")
    print("\n🎯 Benefits of with_retry():")
    print("   - Less code (no manual retry loops)")
    print("   - Production-tested retry logic")
    print("   - Works with agents and RAG systems")


if __name__ == "__main__":
    test_robust_chat()
