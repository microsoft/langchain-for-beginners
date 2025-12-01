# Assignment: LLM Foundations

## Challenge 1: Experiment with System Prompts

Create a program that tests how different system prompts affect the LLM's responses.

### Requirements

1. Create at least 3 different system prompts that give the model distinct "personalities"
2. Ask the same question to each personality
3. Compare and document how the responses differ

### Hints

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

model = ChatOpenAI(model="gpt-4o-mini")

# Example personalities
personalities = [
    "You are a helpful assistant that speaks like a pirate.",
    "You are a formal business analyst.",
    "You are an enthusiastic teacher.",
]

for personality in personalities:
    messages = [
        SystemMessage(content=personality),
        HumanMessage(content="Your question here"),
    ]
    response = model.invoke(messages)
    print(response.content)
```

## Challenge 2: Model Performance Comparison

Build a program that compares the performance of different models.

### Requirements

1. Test at least 2 different models (e.g., "gpt-4o" vs "gpt-4o-mini")
2. Measure and compare response times
3. Compare response lengths
4. Document your findings about cost vs. performance tradeoffs

### Hints

```python
import time
from langchain_openai import ChatOpenAI

models = ["gpt-4o", "gpt-4o-mini"]
prompt = "Explain machine learning in 2-3 sentences."

for model_name in models:
    model = ChatOpenAI(model=model_name)
    
    start_time = time.time()
    response = model.invoke(prompt)
    elapsed_ms = (time.time() - start_time) * 1000
    
    print(f"Model: {model_name}")
    print(f"Time: {elapsed_ms:.2f}ms")
    print(f"Response length: {len(response.content)} chars")
```

## Solutions

Check the `solution/` folder for example implementations:
- `personality_test.py` - Solution for Challenge 1
- `model_performance.py` - Solution for Challenge 2
