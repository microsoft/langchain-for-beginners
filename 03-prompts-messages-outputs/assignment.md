# Assignment: Prompts, Messages, and Structured Outputs

## Overview

Practice creating reusable, maintainable prompts using templates, few-shot learning, and structured output techniques. This assignment focuses on **templates and structured outputs** (Parts 2 and 3 of the chapter).

## Prerequisites

- Completed this [chapter](./README.md)
- Run all code examples (including structured outputs examples) in this chapter
- Understand template syntax and composition

---

## Challenge: Few-Shot Format Teacher 🎓

**Goal**: Use few-shot prompting to teach the AI a custom output format.

**Tasks**:
1. Create `format_teacher.py`
2. Teach the AI to convert product descriptions into a specific JSON format:
   ```json
   {
     "name": "Product name",
     "price": "$XX.XX",
     "category": "Category",
     "highlight": "Key feature"
   }
   ```
3. Provide 3-4 example conversions
4. Test with new product descriptions
5. Parse and validate the JSON output

**Teaching Examples** (provide these as few-shot examples):
- Input: "Premium wireless headphones with noise cancellation, $199"
- Input: "Organic cotton t-shirt in blue, comfortable fit, $29.99"
- Input: "Gaming laptop with RTX 4070, 32GB RAM, $1,499"

**Success Criteria**:
- AI consistently outputs valid JSON
- Format matches your examples
- Works with various product descriptions

**Hints**:
```python
# 1. Import required modules
import json
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_openai import ChatOpenAI

# 2. Load environment variables and create model with temperature 0

# 3. Define your teaching examples list with input/output pairs
#    - Each example should show a product description as input
#    - And the corresponding JSON format as output (use json.dumps for formatting)

# 4. Create an example template using ChatPromptTemplate.from_messages
#    with ("human", "{input}") and ("ai", "{output}")

# 5. Create a FewShotChatMessagePromptTemplate with your examples

# 6. Build a final prompt that includes the few-shot template

# 7. Test with new product descriptions and parse the JSON output with json.loads()
```

> [!TIP]
> **🤖 Get help from [GitHub Copilot](../docs/copilot.md):** If you need assistance with this challenge, open this file in your editor and [use the Challenge Tutor agent](../docs/copilot.md#challenge-tutor-agent) to get personalized help and explanations.

---

## Bonus Challenge: Product Data Extractor with Structured Outputs 🏷️

**Goal**: Build a system that extracts product information into validated, typed data structures.

**Tasks**:
1. Create `product_extractor.py`
2. Define a Pydantic model for product information:
   ```python
   class Product(BaseModel):
       name: str
       price: float
       category: Literal["Electronics", "Clothing", "Food", "Books", "Home"]
       in_stock: bool
       rating: float  # 1-5
       features: list[str]
   ```
3. Use `with_structured_output()` to extract product data
4. Test with product descriptions in various formats:
   - Formal product listings
   - Casual marketplace descriptions
   - Mixed content (reviews + specifications)
5. Validate that all outputs match your schema
6. Handle edge cases (missing information)

**Example Inputs**:
- "MacBook Pro 16-inch with M3 chip, $2,499. Currently in stock. Users rate it 4.8/5. Features: Liquid Retina display, 18-hour battery, 1TB SSD"
- "Cozy wool sweater, blue color, medium size. $89, available now! Customers love it - 4.5 stars. Hand-washable, made in Ireland"
- "The Great Gatsby by F. Scott Fitzgerald. Classic novel, paperback edition for $12.99. In stock. Rated 4.9 stars. 180 pages, published 1925"

**Success Criteria**:
- All outputs are properly typed
- Schema validation works correctly
- Handles various input formats
- Correctly categorizes products
- Gracefully handles missing data

**Hints**:
```python
# 1. Import required modules
import os
from typing import Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# 2. Load environment variables and create the ChatOpenAI model

# 3. Define a Pydantic model with all required fields:
#    - name: str with Field(description="...")
#    - price: float
#    - category: Literal["Electronics", "Clothing", "Food", "Books", "Home"]
#    - in_stock: bool
#    - rating: float with Field(ge=1, le=5)
#    - features: list[str]
#    Use Field(description="...") to add descriptions for each field

# 4. Create a structured output model using model.with_structured_output(Product)

# 5. Create a prompt template asking to extract product information

# 6. Create a chain by piping template | structured_model

# 7. Test with various product descriptions and handle edge cases
#    Access fields using result.name, result.price, etc.
#    Use result.model_dump_json(indent=2) for formatted JSON output
```

---

## Submission Checklist

Before continuing, make sure you've completed:

- [ ] Challenge: Few-shot format teacher with JSON output
- [ ] Bonus: Product data extractor with structured outputs (optional)

---

## Solutions

Solutions for all challenges are available in the [`solution/`](./solution/) folder. Try to complete the challenges on your own first!

**Additional Examples**: Check out the [`samples/`](./samples/) folder for more examples including email generation, translation systems, dynamic prompt builders, and template libraries!

> **💡 Note**: This assignment focuses on templates (Part 2 of this chapter). For message-based patterns (Part 1), practice building agents later in the course!

---

## Need Help?

- **Template syntax**: Review examples in [`code/`](./code/)
- **Few-shot issues**: Check Example 5 ([`code/05_few_shot.py`](./code/05_few_shot.py))
- **Composition**: Review Example 6 ([`code/06_composition.py`](./code/06_composition.py))
- **Any question**: Use the [Challenge Tutor agent](../docs/copilot.md#challenge-tutor-agent) in GitHub Copilot
- **Still stuck**: Join our [Discord community](https://aka.ms/foundry/discord)

---

## Next Steps

Once you've completed these challenges, you're ready for:

**[Function Calling & Tools](../04-function-calling-tools/README.md)**

Excellent progress! You're mastering prompt engineering! 🚀
