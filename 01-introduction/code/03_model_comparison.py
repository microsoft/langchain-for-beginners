"""
Lesson 01 - Model Comparison
This example demonstrates how to compare responses from different models.
"""

import os
import time

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# List of models to compare
models = ["gpt-4o-mini", "gpt-4o"]

# The prompt to test
prompt = "Explain quantum computing in one sentence."

# Compare each model
for model_name in models:
    print(f"\n--- {model_name} ---")
    
    # Create a ChatOpenAI instance for this model
    model = ChatOpenAI(model=model_name)
    
    # Measure the time it takes to get a response
    start_time = time.time()
    response = model.invoke(prompt)
    end_time = time.time()
    
    # Calculate elapsed time in milliseconds
    elapsed_ms = (end_time - start_time) * 1000
    
    # Print the results
    print(f"Response: {response.content}")
    print(f"Time: {elapsed_ms:.2f}ms")
