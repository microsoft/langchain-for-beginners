"""
Error Handling with Built-In Retries
Run: python 02-chat-models/code/05_error_handling.py

🤖 Try asking GitHub Copilot Chat (https://github.com/features/copilot):
- "How does with_retry() implement exponential backoff?"
- "Can I customize the retry delay and max attempts with with_retry()?"
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()


def robust_call(prompt: str, max_retries: int = 3) -> str:
    """Makes an API call with automatic retry logic using LangChain's built-in with_retry()"""
    model = ChatOpenAI(model=os.environ.get("AI_MODEL", "gpt-4o-mini"))

    # Use LangChain's built-in retry logic - automatically handles retries with exponential backoff
    model_with_retry = model.with_retry(stop_after_attempt=max_retries)

    print(f"🔄 Making call with automatic retry (max {max_retries} attempts)...")

    response = model_with_retry.invoke(prompt)
    print("✅ Success!")

    return str(response.content)


def error_examples():
    """Demonstrates different error scenarios"""
    print("🛡️  Error Handling Examples\n")
    print("=" * 80)

    # Example 1: Invalid API key - actually demonstrate the error!
    print("\n1️⃣  Example: Invalid API Key\n")
    try:
        bad_model = ChatOpenAI(
            model=os.environ.get("AI_MODEL", "gpt-4o-mini"),
            api_key="invalid_key_12345",  # Intentionally invalid
        )

        print("🔄 Attempting call with invalid API key...")
        bad_model.invoke("Hello")
        print("✅ Call succeeded (unexpected!)")
    except Exception as error:
        error_msg = str(error)[:100] + "..." if len(str(error)) > 100 else str(error)
        print(f"❌ Caught error: {error_msg}")
        print("💡 Solution: Check your API key in .env file\n")

    # Example 2: Normal with_retry() usage (no failures)
    print("\n2️⃣  Example: Using with_retry() with Valid Credentials\n")
    try:
        print("🔄 Making call with with_retry() (should succeed on first try)...")
        response = robust_call("What is 5+5?")
        print(f"🤖 Response: {response}")
        print("💡 No retries needed when everything works correctly!\n")
    except Exception as error:
        print(f"❌ All retries failed: {error}")

    # Example 3: Error categorization
    print("\n3️⃣  Example: Categorizing Different Error Types\n")

    # Save original key
    original_key = os.environ.get("AI_API_KEY")

    try:
        # Test with invalid key
        bad_model = ChatOpenAI(
            model=os.environ.get("AI_MODEL", "gpt-4o-mini"),
            api_key="sk-invalid12345",
        )

        print("🔄 Testing error categorization with invalid credentials...")
        bad_model.invoke("Hello")
    except Exception as error:
        error_msg = str(error).lower()
        
        # Categorize the error
        error_type = "Unknown error"
        solution = "Check the error message for details"

        if "401" in error_msg or "unauthorized" in error_msg or "invalid" in error_msg:
            error_type = "Authentication Error (401)"
            solution = "Verify your API key is correct"
        elif "429" in error_msg or "rate limit" in error_msg:
            error_type = "Rate Limit Error (429)"
            solution = "Use with_retry() to handle rate limits automatically"
        elif "timeout" in error_msg:
            error_type = "Timeout Error"
            solution = "Increase timeout or use with_retry()"

        print(f"📋 Error type detected: {error_type}")
        print(f"💡 Solution: {solution}")


def show_best_practices():
    """Best practices for error handling"""
    print("\n\n📋 Error Handling Best Practices\n")
    print("=" * 80)

    print("""
1. ✅ Always wrap API calls in try-except
   try:
       response = model.invoke(prompt)
   except Exception as error:
       print(f"Error: {error}")

2. ✅ Use built-in retry logic with with_retry()
   model_with_retry = model.with_retry(stop_after_attempt=3)
   # Automatically handles exponential backoff!

3. ✅ Handle specific error types
   if "429" in str(error):
       # Rate limit - with_retry() handles this automatically
   elif "401" in str(error):
       # Auth error - check API key

4. ✅ Log errors for debugging
   import logging
   logging.error(f"API Error: {error}")

5. ✅ Provide helpful error messages to users
   "Sorry, I'm having trouble connecting. Please try again."

6. ✅ Have fallback behavior
   if api_call_fails:
       return cached_response or default_response

7. ✅ Monitor error rates in production
   Track failed requests to identify issues early
""")


def main():
    error_examples()
    show_best_practices()

    print("\n✅ Remember: Good error handling makes your app reliable!")


if __name__ == "__main__":
    main()
