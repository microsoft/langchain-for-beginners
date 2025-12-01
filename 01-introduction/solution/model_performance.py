"""
Lesson 01 - Solution: Model Performance Comparison
This solution compares performance between different OpenAI models.
"""

import os
import time

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Models to compare
models_to_test = ["gpt-4o", "gpt-4o-mini"]

# Test prompt
prompt = "Explain the concept of machine learning in 2-3 sentences."


def test_model(model_name: str) -> dict:
    """Test a model and return performance metrics."""
    model = ChatOpenAI(model=model_name)
    
    # Measure response time
    start_time = time.time()
    response = model.invoke(prompt)
    end_time = time.time()
    
    elapsed_ms = (end_time - start_time) * 1000
    response_length = len(response.content)
    
    return {
        "model": model_name,
        "response": response.content,
        "time_ms": elapsed_ms,
        "response_length": response_length,
    }


def main():
    """Run the model performance comparison."""
    print("Model Performance Comparison")
    print("=" * 50)
    print(f"Prompt: {prompt}\n")
    
    results = []
    
    for model_name in models_to_test:
        print(f"Testing {model_name}...")
        result = test_model(model_name)
        results.append(result)
    
    # Display results
    print("\n" + "=" * 50)
    print("Results:")
    print("=" * 50)
    
    for result in results:
        print(f"\n--- {result['model']} ---")
        print(f"Response: {result['response']}")
        print(f"Time: {result['time_ms']:.2f}ms")
        print(f"Response length: {result['response_length']} characters")
    
    # Compare
    print("\n" + "=" * 50)
    print("Comparison Summary:")
    print("=" * 50)
    
    fastest = min(results, key=lambda x: x["time_ms"])
    print(f"Fastest: {fastest['model']} ({fastest['time_ms']:.2f}ms)")
    
    longest_response = max(results, key=lambda x: x["response_length"])
    print(f"Longest response: {longest_response['model']} ({longest_response['response_length']} chars)")


if __name__ == "__main__":
    main()
