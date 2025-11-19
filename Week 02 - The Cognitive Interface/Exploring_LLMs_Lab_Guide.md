# Lab Guide: Exploring LLMs - Multi-Provider Integration & Evaluation

## Overview
In this lab, you will:
- Connect to 6 different LLM providers (3 cloud, 3 local)
- Build a model-agnostic interface for comparing responses
- Implement an LLM-as-Judge evaluation pattern
- Learn when to use cloud vs. local inference

**Time Required:** 90-120 minutes  
**Prerequisites:** Python 3.12+, VS Code, uv environment manager

---

## Part 0: Pre-Lab Setup

### Install Required Services

#### 1. Ollama (Local LLM Runner)
- Download from: https://ollama.com/
- After installation, run in terminal:
  ```bash
  ollama pull llama3.2
  ```
- Verify it's running:
  ```bash
  ollama list  # Should show llama3.2
  ```

#### 2. LM Studio (Local LLM with GUI)
- Download from: https://lmstudio.ai/
- Open LM Studio → Discover tab
- Search and download: `mistral-7b-instruct` (Q4_K_M variant)
- Go to Local Server tab → Load the model → Start Server
- Verify it's running at: http://localhost:1234

#### 3. API Keys (Cloud Providers)
Create free accounts and get API keys from:
- **OpenAI:** https://platform.openai.com/api-keys
- **Anthropic:** https://console.anthropic.com/
- **Google AI Studio:** https://aistudio.google.com/app/apikey
- **Groq:** https://console.groq.com/keys
- **Hugging Face:** https://huggingface.co/settings/tokens

---

## Part 1: Environment Configuration

### Step 1: Create Your .env File
In your project root (same folder as the notebook), create `.env`:

```
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxx
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxx
GOOGLE_API_KEY=AIzaSyxxxxxxxxxxxxxxxx
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxx
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

⚠️ **Security Note:** Never commit .env files to Git! Add `.env` to your `.gitignore`.

---

### Step 2: Install Python Dependencies
In your activated uv environment:
```bash
uv pip install openai anthropic python-dotenv requests ipython
```

---

### Step 3: Test Your Setup
Run this code block in your notebook:

```python
import os
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

def is_service_running(url):
    try:
        response = requests.get(url, timeout=5)
        return response.status_code == 200
    except:
        return False

# Check services
print("Ollama:", "✅" if is_service_running('http://localhost:11434') else "❌")
print("LM Studio:", "✅" if is_service_running('http://localhost:1234') else "❌")

# Check API keys (don't print actual values!)
keys = {
    "OpenAI": os.getenv('OPENAI_API_KEY'),
    "Anthropic": os.getenv('ANTHROPIC_API_KEY'),
    "Google": os.getenv('GOOGLE_API_KEY'),
    "Groq": os.getenv('GROQ_API_KEY'),
    "HuggingFace": os.getenv('HF_TOKEN'),
}

for name, value in keys.items():
    print(f"{name}:", "✅" if value else "❌")
```

**Expected Output:**
```
Ollama: ✅
LM Studio: ✅
OpenAI: ✅
Anthropic: ✅
Google: ✅
Groq: ✅
HuggingFace: ✅
```

**Troubleshooting:**
- ❌ for Ollama/LM Studio → Make sure they're running
- ❌ for API keys → Check your .env file spelling and location

---

## Part 2: Multi-Provider LLM Integration

### Conceptual Framework
You're building a **model-agnostic interface**—one code pattern that works across all providers. This is crucial for agentic systems where you might need to:
- Route simple tasks to free local models
- Escalate complex reasoning to expensive cloud models
- A/B test which model performs best for your use case

---

### The Test Question
We'll ask each LLM the same question:

```python
import json
from openai import OpenAI
from anthropic import Anthropic
from IPython.display import Markdown, display

# Initialize tracking
llms = []
responses = []

request = "You are an AI tasked with writing a single, one-sentence instruction for a human to prevent a paradox in a time-travel scenario. What is that instruction?"
messages = [{"role": "user", "content": request}]
```

**Why this question?** It tests logical reasoning, conciseness, and creativity—all important for agent tasks.

---

### Provider 1: OpenAI (GPT-4o-mini)

```python
# Cloud inference - fast, high-quality, costs ~$0.15/1M tokens
openai = OpenAI()
model = "gpt-4o-mini"

response = openai.chat.completions.create(model=model, messages=messages)
result = response.choices[0].message.content

display(Markdown(result))
llms.append(model)
responses.append(result)
```

**Concept:** The `OpenAI()` client automatically reads `OPENAI_API_KEY` from your environment.

---

### Provider 2: Anthropic (Claude Sonnet)

```python
# Cloud inference - excellent reasoning, costs ~$3/1M tokens
model = "claude-3-7-sonnet-latest"

claude = Anthropic()
response = claude.messages.create(
    model=model,
    messages=messages,
    max_tokens=1000  # ⚠️ Required for Anthropic
)
result = response.content[0].text

display(Markdown(result))
llms.append(model)
responses.append(result)
```

**Key Difference:** Anthropic's API uses `response.content[0].text` instead of `response.choices[0].message.content`.

---

### Provider 3: Google Gemini

```python
# Cloud inference - free tier, good for prototyping
gemini = OpenAI(
    api_key=os.getenv('GOOGLE_API_KEY'),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
model = "gemini-2.0-flash"

response = gemini.chat.completions.create(model=model, messages=messages)
result = response.choices[0].message.content

display(Markdown(result))
llms.append(model)
responses.append(result)
```

**Concept:** Google provides an **OpenAI-compatible endpoint**—same client, just change the `base_url`.

---

### Provider 4: Groq (Hosted Llama)

```python
# Cloud inference - ultra-fast, free tier, great for development
groq = OpenAI(
    api_key=os.getenv('GROQ_API_KEY'),
    base_url="https://api.groq.com/openai/v1"
)
model = "llama-3.3-70b-versatile"

response = groq.chat.completions.create(model=model, messages=messages)
result = response.choices[0].message.content

display(Markdown(result))
llms.append(model)
responses.append(result)
```

**Why Groq?** They use LPUs (Language Processing Units) for blazing-fast inference—up to 10x faster than GPUs.

---

### Provider 5: Ollama (Local Llama)

```python
# Local inference - free, private, works offline
ollama = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
model = "llama3.2"

response = ollama.chat.completions.create(model=model, messages=messages)
result = response.choices[0].message.content

display(Markdown(result))
llms.append(model)
responses.append(result)
```

**Trade-offs:**
- ✅ Completely free after download
- ✅ Your data never leaves your machine
- ❌ Slower than cloud (unless you have a GPU)
- ❌ Smaller model → lower quality than GPT-4

---

### Provider 6: LM Studio (Local Mistral)

```python
# Local inference - GUI-based, good for experimentation
mistral = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
model = "mistral_instruct_7b"

response = mistral.chat.completions.create(model=model, messages=messages)
result = response.choices[0].message.content

display(Markdown(result))
llms.append(model)
responses.append(result)
```

**When to use LM Studio:**
- Testing prompts before burning API credits
- Demos at conferences (works without internet)
- Processing sensitive data (HIPAA compliance, etc.)

---

## Part 3: LLM-as-Judge Evaluation

### Concept: Using AI to Evaluate AI
Instead of manually reading 6 responses, we'll use an LLM to:
1. Compare them on specific criteria
2. Rank them from best to worst
3. Return structured JSON we can parse

This pattern is used in production for:
- **Multi-agent systems:** A supervisor agent grades worker outputs
- **Automated testing:** Check if responses meet quality standards
- **A/B testing:** Determine which prompt engineering technique works best

---

### Step 1: Prepare the Evaluation Prompt

```python
# Format all responses into a single text block
text_prep = ""
for index, response in enumerate(responses):
    text_prep += f"# Response from llm {index+1}\n\n"
    text_prep += response + "\n\n"

# Build the evaluation prompt
evaluator = f"""You are evaluating responses from {len(llms)} LLMs.
Each model has been given this question:

{request}

Your job is to evaluate each response for clarity and strength of argument, and rank them in order of best to worst.
Respond with JSON, and only JSON, with the following format:
{{"results": ["best llm number", "second best llm number", "third best llm number", ...]}}

Here are the responses from each llm:

{text_prep}

Now, please respond with the ranked order of the llms using JSON, nothing else. Do not include markdown formatting or code blocks."""
```

**Prompt Engineering Techniques Used:**
1. ✅ Clear role assignment ("You are evaluating...")
2. ✅ Specific criteria ("clarity and strength of argument")
3. ✅ Structured output enforcement (JSON schema)
4. ✅ Explicit constraints ("nothing else", "no markdown")

---

### Step 2: Execute the Evaluation

```python
evaluator_messages = [{"role": "user", "content": evaluator}]

openai = OpenAI()
response = openai.chat.completions.create(
    model="o3-mini",  # Reasoning-optimized model
    messages=evaluator_messages,
)
results = response.choices[0].message.content
print(results)
```

**Why o3-mini?** It's specifically trained for reasoning tasks like comparative analysis.

**Expected Output:**
```json
{"results": ["1", "2", "6", "4", "5", "3"]}
```

---

### Step 3: Parse and Display Rankings

```python
# Parse JSON
results_dict = json.loads(results)
ranks = results_dict["results"]

# Display ranked results
print("\n🏆 Final Rankings 🏆\n")
for index, result in enumerate(ranks):
    llm = llms[int(result)-1]
    print(f"Rank {index+1}: {llm}")
```

**Expected Output:**
```
🏆 Final Rankings 🏆

Rank 1: gpt-4o-mini
Rank 2: claude-3-7-sonnet-latest
Rank 3: mistral_instruct_7b
Rank 4: llama-3.3-70b-versatile
Rank 5: llama3.2
Rank 6: gemini-2.0-flash
```

---

## Part 4: Analysis Questions

### Question 1: Cost vs. Performance
Calculate the approximate cost for your experiment:
- OpenAI GPT-4o-mini: ~$0.15/1M tokens
- Anthropic Claude: ~$3/1M tokens
- Groq: Free tier (but rate-limited)
- Local models: Free (but uses electricity/compute)

**Exercise:** If you ran this experiment 1,000 times per day, which provider would be most cost-effective?

---

### Question 2: When to Use Each Provider
Match these scenarios to the best provider:

1. Processing 10 million customer support tickets/month
2. Analyzing confidential medical records
3. Building a prototype demo for investors
4. Complex strategic planning (budget isn't a concern)
5. Code generation during development

**Answers:**
1. Local models (cost) or Groq (speed + cost)
2. Local models (privacy/compliance)
3. Groq or Gemini (free tier)
4. Claude Opus or GPT-4 (quality)
5. Local models (iterate without burning credits)

---

### Question 3: Evaluation Bias
Run the experiment again with a different evaluator model (e.g., Claude instead of o3-mini). Do the rankings change?

```python
# Try this:
claude = Anthropic()
response = claude.messages.create(
    model="claude-3-7-sonnet-latest",
    messages=evaluator_messages,
    max_tokens=1000
)
results = response.content[0].text
# Then parse and compare rankings
```

**Discussion:** Does the judge's "preference" matter? In production, how would you mitigate evaluation bias?

---

## Part 5: Optional Extensions

### Extension 1: Add Token Counting
Track how many tokens each model used:

```python
tokens_used = response.usage.total_tokens
print(f"{model}: {tokens_used} tokens")
```

**Goal:** Find the most token-efficient model.

---

### Extension 2: Implement Error Handling
Wrap API calls in try-except blocks:

```python
try:
    response = openai.chat.completions.create(model=model, messages=messages)
    result = response.choices[0].message.content
except Exception as e:
    print(f"❌ Error with {model}: {e}")
    result = "ERROR: Unable to generate response"
    
llms.append(model)
responses.append(result)
```

**Why it matters:** In production, APIs fail. Agents need graceful degradation.

---

### Extension 3: Multi-Criteria Evaluation
Instead of a single ranking, evaluate on 3 dimensions:

```json
{
  "rankings": [
    {"llm": "1", "clarity": 9, "creativity": 7, "accuracy": 8},
    {"llm": "2", "clarity": 8, "creativity": 8, "accuracy": 9}
  ]
}
```

Modify the evaluator prompt to return this structure.

---

### Extension 4: Pydantic Validation
Enforce structured outputs with type checking:

```python
from pydantic import BaseModel

class EvaluationResult(BaseModel):
    results: list[str]

# This will raise an error if JSON doesn't match schema
results_obj = EvaluationResult.model_validate_json(results)
print(results_obj.results)
```

**Preview:** This is what you'll learn in depth in Week 3 (Structured Output).

---

## Submission Checklist

Before submitting, ensure:
- [ ] All 6 providers successfully return responses
- [ ] API keys are in .env (not hardcoded in the notebook)
- [ ] Evaluation runs without errors
- [ ] You've answered the analysis questions
- [ ] (Optional) You've completed at least one extension

**Deliverable:** Export your notebook as `06_Exploring_LLMs_YourName.ipynb` and upload to the course portal.

---

## Troubleshooting Guide

### Problem: "Connection refused" for Ollama
**Solution:**
```bash
# Start Ollama service
ollama serve
# In a new terminal:
ollama list  # Verify it's running
```

---

### Problem: "Invalid API key"
**Solution:**
```python
# Debug: Print what's being loaded
print("Loaded key:", os.getenv('OPENAI_API_KEY')[:10])  # Only first 10 chars
```
Make sure:
- .env file is in the correct directory
- No extra spaces in the .env file
- Key hasn't expired (regenerate in the provider's console)

---

### Problem: "JSONDecodeError" when parsing results
**Solution:** The LLM might have wrapped JSON in markdown:
```python
# Clean the results before parsing
results = results.strip()
if results.startswith("```"):
    results = results.split("```")[1]  # Extract content between ```
    if results.startswith("json"):
        results = results[4:]  # Remove "json" prefix
        
results_dict = json.loads(results)
```

---

### Problem: Rate limits on free tiers
**Solution:** Add delays between API calls:
```python
import time
time.sleep(2)  # Wait 2 seconds between requests
```

---

## Key Concepts Summary

| Concept | Definition | Why It Matters for Agents |
|---------|-----------|---------------------------|
| **Model-Agnostic Interface** | Code that works across providers | Agents can dynamically switch models based on task complexity/cost |
| **LLM-as-Judge** | Using an LLM to evaluate other LLM outputs | Enables automated quality control in multi-agent systems |
| **OpenAI-Compatible API** | Standardized interface many providers implement | Write once, deploy anywhere (Ollama, LM Studio, Groq all support it) |
| **Structured Output** | Forcing LLMs to return JSON/XML | Agents need reliable data formats to make decisions |
| **Cloud vs. Local Inference** | Trade-offs between speed, cost, privacy | Production systems often use hybrid approaches |

---

## Next Lab Preview

In **Week 3: Async Programming & Structured Output**, you'll learn:
- Running multiple LLM calls in parallel (asyncio)
- Forcing strict schemas with Pydantic
- Handling concurrent agent workflows

This lab laid the foundation—you now know *how* to call LLMs. Next, you'll learn to do it *efficiently* and *reliably*.

---

## Resources

### Documentation
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Anthropic Claude API](https://docs.anthropic.com/)
- [Ollama GitHub](https://github.com/ollama/ollama)
- [LM Studio Docs](https://lmstudio.ai/docs)

### Further Reading
- [LLM-as-Judge Best Practices](https://huggingface.co/blog/llm-judge)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)

### Community
- Discord: [Agentic AI Course Server](#) (ask questions!)
- GitHub: [Course Repository](https://github.com/gitmystuff/AgenticAI)

---

**Lab Complete! 🎉**

You've successfully built a multi-provider LLM interface and implemented automated evaluation. These skills are foundational for the agentic systems you'll build later in the course.
