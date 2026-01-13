# Module 1: Prompt Engineering Fundamentals

## Introduction

Prompt engineering is the art and science of crafting inputs that effectively communicate with Large Language Models (LLMs) to produce desired outputs. As LLMs become increasingly central to AI applications, mastering prompt engineering has become an essential skill for developers, researchers, and AI practitioners.

This module covers the foundational concepts you need to understand before diving into advanced techniques. We'll explore how LLMs process prompts, fundamental prompt structures, common pitfalls, and evaluation methods.

## Table of Contents

1. [What is Prompt Engineering?](#what-is-prompt-engineering)
2. [How LLMs Process Prompts](#how-llms-process-prompts)
3. [Basic Prompt Structure](#basic-prompt-structure)
4. [Prompt Clarity Principles](#prompt-clarity-principles)
5. [Temperature and Sampling Parameters](#temperature-and-sampling-parameters)
6. [Common Mistakes and Solutions](#common-mistakes-and-solutions)
7. [Prompt Evaluation Metrics](#prompt-evaluation-metrics)
8. [Model Comparison Basics](#model-comparison-basics)
9. [Hands-On Examples](#hands-on-examples)
10. [Exercises](#exercises)

---

## What is Prompt Engineering?

### Definition

Prompt engineering is the practice of designing and refining text inputs (prompts) to maximize the quality, relevance, and reliability of outputs from language models. Unlike traditional programming where you write explicit instructions, prompt engineering involves crafting natural language that guides the model toward the desired behavior.

### Why It Matters

- **Efficiency**: Well-crafted prompts can dramatically improve output quality without changing the underlying model
- **Cost-Effectiveness**: Better prompts reduce the need for expensive model fine-tuning
- **Flexibility**: Prompt engineering allows quick iteration and adaptation to new tasks
- **Consistency**: Systematic prompting leads to more reliable and reproducible results

### Key Concepts

**Direct Prompting**: Explicit instructions that tell the model exactly what to do.

```
Translate the following text to French: "Hello, how are you?"
```

**Indirect Prompting**: Implicit guidance through examples, context, or framing.

```
Here are some English phrases and their French translations:
"Good morning" → "Bonjour"
"Thank you" → "Merci"
"Hello, how are you?" →
```

---

## How LLMs Process Prompts

### Tokenization

Before processing, text is broken down into **tokens**—subword units that models can understand.

#### Token Basics

- Tokens are not always words (can be word parts, characters, or common phrases)
- Different models use different tokenizers (GPT uses Byte-Pair Encoding)
- Token count affects both cost and context window limits

#### Example Tokenization

```python
# Using tiktoken (OpenAI's tokenizer)
import tiktoken

encoding = tiktoken.encoding_for_model("gpt-4")
text = "Hello, world! How are you?"
tokens = encoding.encode(text)

print(f"Text: {text}")
print(f"Tokens: {tokens}")
print(f"Token count: {len(tokens)}")
print(f"Decoded: {encoding.decode(tokens)}")
```

Output:
```
Text: Hello, world! How are you?
Tokens: [9906, 11, 1917, 0, 527, 499, 30]
Token count: 7
Decoded: Hello, world! How are you?
```

#### Token Optimization Tips

1. **Shorter prompts = lower cost**: Every token costs money
2. **Be aware of token limits**: Context windows have maximum token counts
3. **Common words are often fewer tokens**: "the" = 1 token, "supercalifragilisticexpialidocious" = multiple tokens

### Attention Mechanism

LLMs use **attention mechanisms** to understand relationships between different parts of the input.

#### How Attention Works

1. **Token Embedding**: Each token is converted to a vector representation
2. **Position Encoding**: Position information is added
3. **Self-Attention**: The model computes attention scores between all token pairs
4. **Context Building**: Each token's representation incorporates information from relevant tokens

#### Implications for Prompting

- **Order Matters**: Information at the beginning and end often receives more attention
- **Relevance**: Important information should be placed prominently
- **Context Length**: Longer prompts allow richer context but may dilute focus

### Context Windows

A **context window** is the maximum number of tokens a model can process in a single request.

#### Context Window Sizes (as of 2024)

| Model | Context Window | Notes |
|-------|---------------|-------|
| GPT-4 Turbo | 128K tokens | ~300 pages of text |
| Claude 3 Opus | 200K tokens | ~500 pages |
| GPT-3.5 Turbo | 16K tokens | ~40 pages |
| Llama 3 70B | 8K tokens | ~20 pages |

#### Managing Context Windows

**Problem**: You have a 50-page document but model only supports 128K tokens.

**Solutions**:
1. **Truncation**: Take the most relevant sections
2. **Summarization**: Summarize sections before including
3. **Chunking**: Process in smaller chunks and combine results
4. **Retrieval-Augmented Generation (RAG)**: We'll cover this in Module 6

### Model Architecture Overview

Understanding the basics helps you craft better prompts:

**Transformer Architecture**:
- **Encoder**: Processes input (used in models like BERT)
- **Decoder**: Generates output (used in GPT models)
- **Encoder-Decoder**: Both (used in T5, some translation models)

**Key Insight**: GPT models are **autoregressive**—they generate text token by token, left to right. This means:
- Earlier tokens influence later ones
- The model can't "see ahead"
- Prompts guide the entire generation process

---

## Basic Prompt Structure

Effective prompts have a clear structure, though the exact format varies by task.

### Core Components

1. **Instruction**: What should the model do?
2. **Context**: What information does it need?
3. **Input**: The actual data to process
4. **Output Format**: How should results be structured?

### Simple Structure

```
[Instruction] [Context] [Input] [Output Format]
```

### Detailed Breakdown

#### 1. Instruction

The explicit command or question.

**Good Examples**:
```
Translate the following text to Spanish.
Summarize the key points from the article below.
Extract all email addresses from the following text.
```

**Bad Examples**:
```
Translate maybe?
Something about translation?
```

#### 2. Context

Background information that helps the model understand the task.

**Example with Context**:
```
You are a professional translator specializing in technical documentation. 
Translate the following text to Spanish, maintaining technical accuracy:
```

**Without Context**:
```
Translate to Spanish:
```

#### 3. Input

The actual data to be processed.

```
Input text: "The API rate limit is 1000 requests per hour."
```

#### 4. Output Format

Specifies how results should be structured.

**Examples**:
```
Respond in JSON format with keys: "translation" and "confidence"
Use bullet points for each key insight
Return a single sentence summary
```

### Complete Example

```
Instruction: You are an expert data analyst.

Context: The following dataset contains customer feedback from Q1 2024.

Input: 
Customer ID: 12345
Feedback: "Great service but delivery was slow"
Rating: 4/5

Output Format: Extract the sentiment (positive/negative/neutral) and 
the main topic. Return as JSON: {"sentiment": "...", "topic": "..."}
```

### System vs User Messages

Many modern APIs support **system messages** (special context that persists across the conversation) and **user messages** (individual requests).

**OpenAI API Example**:
```python
import openai

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that explains concepts simply."},
        {"role": "user", "content": "What is machine learning?"}
    ]
)
```

**Benefits of System Messages**:
- Set persistent persona or behavior
- Define constraints that apply to all interactions
- More efficient than repeating context in every prompt

---

## Prompt Clarity Principles

Clear, unambiguous prompts lead to better outputs. Follow these principles:

### 1. Be Specific

**Vague**:
```
Tell me about dogs.
```

**Specific**:
```
Explain three key differences between dog breeds in terms of exercise requirements, 
focusing on working dogs versus toy breeds. Write 2-3 sentences for each difference.
```

### 2. Use Examples

Examples clarify intent better than descriptions alone.

**Without Examples**:
```
Format dates consistently.
```

**With Examples**:
```
Format all dates as YYYY-MM-DD. Examples:
- "March 15, 2024" → "2024-03-15"
- "12/25/23" → "2023-12-25"
- "Jan 1" → "2024-01-01" (assume current year)
```

### 3. Define Constraints

Explicitly state what should and shouldn't be included.

**Good Constraint Definition**:
```
Summarize the article in exactly 50 words. Do not include:
- Personal opinions
- External information not in the article
- Direct quotes longer than 5 words

Focus on: main argument, supporting evidence, conclusion.
```

### 4. Use Structure

Break complex tasks into clear steps.

**Unstructured**:
```
Analyze this code, find bugs, suggest fixes, and explain everything.
```

**Structured**:
```
Analyze the following code in three steps:

Step 1: Identify all syntax errors
Step 2: List potential logical bugs
Step 3: Suggest fixes with brief explanations

For each issue, provide: line number, description, suggested fix.
```

### 5. Specify Format

Always indicate desired output format.

**Format Options**:
- JSON/XML for structured data
- Markdown for formatted text
- Bullet points for lists
- Tables for comparisons
- Code blocks for code

**Example**:
```
List the top 5 programming languages by popularity in 2024.
Format as a markdown table with columns: Rank, Language, Primary Use Case.
```

### 6. Set the Tone and Style

**Academic Tone**:
```
Write a formal academic analysis of the following research paper...
```

**Casual Tone**:
```
In simple terms, explain what this research paper says...
```

### 7. Provide Negative Examples

Sometimes showing what NOT to do helps.

```
Generate product descriptions. 

Good example: "The ergonomic design reduces wrist strain during extended use."

Bad example: "This is a really cool product that you should definitely buy!"
(too promotional, lacks specific benefits)
```

---

## Temperature and Sampling Parameters

These parameters control the randomness and creativity of model outputs.

### Temperature

Controls randomness in token selection.

- **Range**: 0.0 to 2.0 (typically)
- **Low (0.0-0.3)**: Deterministic, focused, repetitive
- **Medium (0.4-0.7)**: Balanced creativity and coherence
- **High (0.8-2.0)**: Creative, diverse, less predictable

#### When to Use Different Temperatures

**Low Temperature (0.0-0.3)**:
- Factual responses
- Data extraction
- Code generation
- When consistency is critical

**Medium Temperature (0.4-0.7)**:
- Creative writing
- Brainstorming
- General conversation
- Most general-purpose tasks

**High Temperature (0.8-2.0)**:
- Creative fiction
- Idea generation
- When diversity is more important than accuracy

#### Code Example

```python
import openai

prompt = "Write a haiku about programming."

# Low temperature - more deterministic
response_low = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.2
)

# High temperature - more creative
response_high = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}],
    temperature=1.2
)

print("Low temp:", response_low.choices[0].message.content)
print("High temp:", response_high.choices[0].message.content)
```

### Top-p (Nucleus Sampling)

Controls diversity via probability mass.

- **Range**: 0.0 to 1.0
- **How it works**: Considers tokens whose cumulative probability mass reaches the threshold
- **0.1**: Very focused (considers top 10% probability mass)
- **0.9**: Very diverse (considers top 90% probability mass)

**Relationship with Temperature**:
- Often used together
- Temperature adjusts all probabilities
- Top-p dynamically adjusts the candidate set

### Top-k Sampling

Limits choices to the k most likely tokens.

- **Range**: 1 to vocabulary size
- **Effect**: Prevents very unlikely tokens from being selected
- **Common values**: 40-50 for most tasks

### Max Tokens / Max Length

Controls maximum output length.

**Best Practices**:
- Set appropriate limits to control costs
- For completions, set max_tokens high enough
- For chat, let it run until stop sequence or natural end

### Stop Sequences

Tokens that signal the model to stop generating.

**Common Use Cases**:
- Multi-turn conversations: `["User:", "Human:", "\n\n\n"]`
- Code generation: `["```", "\n\n\n\n"]`
- List generation: `["Item 11:", "11."]`

**Example**:
```python
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "List 5 colors:"}],
    stop=["6.", "Color 6", "\n\n"]  # Stop before item 6
)
```

### Frequency and Presence Penalties

Control repetition.

- **Frequency Penalty**: Reduces probability of tokens that have appeared frequently
- **Presence Penalty**: Reduces probability of tokens that have appeared at all

**Use Cases**:
- Creative writing (avoid repetition)
- Long-form generation
- When outputs become too repetitive

---

## Common Mistakes and Solutions

### 1. Ambiguity

**Problem**: Prompts that could be interpreted multiple ways.

**Bad Example**:
```
Fix the code.
```

**Solution**: Be explicit about what needs fixing.

**Good Example**:
```
Review the following Python code for:
1. Syntax errors (highlight with line numbers)
2. Potential runtime errors
3. Style issues (PEP 8 violations)

For each issue, suggest a specific fix.
```

### 2. Over-Prompting

**Problem**: Including too much unnecessary information.

**Bad Example**:
```
In the context of the history of computing, which dates back to ancient abacus 
devices and evolved through mechanical calculators, electronic computers, and 
modern digital systems, considering that programming languages have developed 
from machine code to high-level languages like Python, and acknowledging that 
syntax errors are common issues, please identify any syntax errors in this code.
```

**Solution**: Include only relevant context.

**Good Example**:
```
Identify syntax errors in the following Python code and suggest fixes:
[code here]
```

### 3. Under-Prompting

**Problem**: Not providing enough context or constraints.

**Bad Example**:
```
Summarize this.
```

**Solution**: Specify format, length, focus areas.

**Good Example**:
```
Summarize the following article in 3-5 bullet points, focusing on:
- Main argument or thesis
- Key supporting evidence
- Conclusion or implications

Article: [content]
```

### 4. Assuming Model Knowledge

**Problem**: Assuming the model knows about recent events or your specific context.

**Bad Example**:
```
What happened in the meeting yesterday?
```

**Solution**: Provide necessary context.

**Good Example**:
```
Based on the following meeting notes from yesterday's team standup, summarize 
the key action items:

Meeting notes:
- Sarah: Working on authentication bug
- Mike: Implementing new dashboard feature
- Lisa: Blocked on API rate limits
```

### 5. Ignoring Output Format

**Problem**: Not specifying desired format leads to inconsistent outputs.

**Bad Example**:
```
Extract names and emails.
```

**Solution**: Specify exact format.

**Good Example**:
```
Extract all names and email addresses from the following text. Return as JSON:
{
  "contacts": [
    {"name": "...", "email": "..."}
  ]
}
```

### 6. Neglecting Edge Cases

**Problem**: Prompts work for common cases but fail on edge cases.

**Bad Example**:
```
Extract the first date mentioned.
```

**Solution**: Handle edge cases explicitly.

**Good Example**:
```
Extract the first date mentioned in the following text. 
- If no date found, return null
- Handle various formats: "March 15, 2024", "2024-03-15", "15/03/24"
- Return in ISO format: YYYY-MM-DD
```

### 7. Forgetting Context Limits

**Problem**: Including too much context, hitting token limits.

**Solution**: 
- Summarize long documents
- Extract only relevant sections
- Use chunking strategies
- Implement retrieval-augmented generation

### 8. Not Iterating

**Problem**: Using first prompt draft without refinement.

**Solution**: 
- Test with multiple examples
- Measure performance
- Refine based on failures
- Keep a prompt library

---

## Prompt Evaluation Metrics

How do you know if your prompt is good? Use these metrics:

### 1. Relevance

Does the output address the task?

**Evaluation**:
- Manual review
- Automated keyword/entity matching
- Semantic similarity (embedding-based)

**Example**:
```
Prompt: "Summarize the article about climate change"
Output: "Climate change refers to long-term shifts in global temperatures..."
✓ Relevant

Output: "Here's a recipe for chocolate cake..."
✗ Not relevant
```

### 2. Accuracy

Is the output factually correct?

**Evaluation**:
- Fact-checking against ground truth
- Cross-referencing with source material
- Expert review

**Challenges**:
- Models may hallucinate
- Need verification sources
- Context-dependent

### 3. Consistency

Does the prompt produce similar outputs for similar inputs?

**Evaluation**:
- Run prompt multiple times with same input
- Measure output variance
- Check for contradictions

**Example**:
```
Input: "What is 2+2?"
Run 10 times, all outputs should be "4" or similar
```

### 4. Completeness

Does the output include all required elements?

**Checklist Evaluation**:
```
Required elements:
✓ Main topic covered
✓ Key points included
✓ Examples provided (if requested)
✓ Conclusion present
```

### 5. Format Compliance

Does the output match the specified format?

**Evaluation**:
- JSON schema validation
- Regex pattern matching
- Structure parsing

**Example**:
```python
import json

def validate_json_output(output, expected_schema):
    try:
        data = json.loads(output)
        # Validate against schema
        return validate_schema(data, expected_schema)
    except json.JSONDecodeError:
        return False
```

### 6. Efficiency

Cost and latency metrics.

**Metrics**:
- **Tokens used**: Lower is better (cost)
- **Response time**: Faster is better (latency)
- **Success rate**: Higher is better (reliability)

### 7. User Satisfaction

Subjective quality from end users.

**Methods**:
- Surveys and ratings
- A/B testing
- User feedback
- Task completion rates

### Evaluation Framework Example

```python
class PromptEvaluator:
    def __init__(self):
        self.metrics = {}
    
    def evaluate(self, prompt, test_cases):
        results = {
            'relevance': [],
            'accuracy': [],
            'consistency': [],
            'completeness': [],
            'format_compliance': [],
            'efficiency': {'avg_tokens': 0, 'avg_time': 0}
        }
        
        for test_case in test_cases:
            # Run prompt with test case
            output = run_prompt(prompt, test_case['input'])
            
            # Evaluate each metric
            results['relevance'].append(
                self.check_relevance(output, test_case['expected'])
            )
            results['accuracy'].append(
                self.check_accuracy(output, test_case['ground_truth'])
            )
            # ... other metrics
        
        return self.compute_scores(results)
```

---

## Model Comparison Basics

Different models have different strengths. Understanding these helps you choose the right model and craft model-specific prompts.

### Major Models Overview

#### GPT-4 (OpenAI)

**Strengths**:
- Strong reasoning capabilities
- Good at following complex instructions
- Large context window (128K tokens)
- Code generation

**Best For**:
- Complex reasoning tasks
- Multi-step problem solving
- Code generation and review
- Analysis tasks

**Prompt Tips**:
- Can handle very long, detailed instructions
- Responds well to structured formats
- Benefits from few-shot examples

#### Claude 3 (Anthropic)

**Strengths**:
- Very large context window (200K tokens)
- Strong safety features
- Good at long-form content
- Less likely to refuse requests

**Best For**:
- Long document analysis
- Content creation
- Tasks requiring large context
- Safety-critical applications

**Prompt Tips**:
- Can reference very long documents
- Good at maintaining context across long conversations
- Works well with system messages

#### Gemini (Google)

**Strengths**:
- Multimodal (text, images, audio)
- Good performance/price ratio
- Fast inference
- Strong at factual knowledge

**Best For**:
- Multimodal tasks
- Cost-sensitive applications
- Fast responses needed
- Google ecosystem integration

**Prompt Tips**:
- Can include images in prompts
- Good for visual question answering
- Efficient for simple tasks

#### Llama 3 (Meta)

**Strengths**:
- Open source
- Good for fine-tuning
- Cost-effective (self-hosted)
- Growing capabilities

**Best For**:
- Custom applications
- Fine-tuned models
- Cost-sensitive use cases
- Privacy-sensitive applications

**Prompt Tips**:
- May need more explicit instructions
- Benefits from clear formatting
- Works well with prompt templates

### Model Selection Framework

Choose based on:

1. **Task Type**: Reasoning, generation, analysis, etc.
2. **Context Requirements**: How much input context needed?
3. **Output Quality**: How critical is quality vs. speed?
4. **Cost Constraints**: What's the budget?
5. **Latency Requirements**: Real-time vs. batch?
6. **Multimodal Needs**: Text-only or images/video too?

### Comparison Table

| Model | Context | Cost | Speed | Reasoning | Code | Multimodal |
|-------|---------|------|-------|-----------|------|------------|
| GPT-4 Turbo | 128K | High | Medium | Excellent | Excellent | Text only |
| Claude 3 Opus | 200K | High | Medium | Excellent | Good | Text only |
| Gemini Pro | 32K | Medium | Fast | Good | Good | Yes |
| Llama 3 70B | 8K | Low* | Medium | Good | Good | Text only |

*When self-hosted

---

## Hands-On Examples

### Example 1: Simple Classification

**Task**: Classify customer feedback as positive, negative, or neutral.

**Basic Prompt**:
```
Classify this feedback: "The product arrived damaged and customer service was unhelpful."
```

**Improved Prompt**:
```
You are a customer sentiment analyst. Classify the following customer feedback 
as "positive", "negative", or "neutral".

Consider:
- Positive: Satisfaction, praise, recommendation
- Negative: Complaints, dissatisfaction, problems
- Neutral: Facts without emotional tone

Feedback: "The product arrived damaged and customer service was unhelpful."

Respond with only one word: positive, negative, or neutral.
```

**Why Better?**:
- Defines role and context
- Provides criteria for each class
- Specifies output format
- More reliable results

### Example 2: Data Extraction

**Task**: Extract structured data from unstructured text.

**Basic Prompt**:
```
Extract information from: "John Doe, age 35, works as a software engineer at Google."
```

**Improved Prompt**:
```
Extract the following information from the text below:
- Name (first and last)
- Age (as integer)
- Job title
- Company name

If any information is missing, use null.

Text: "John Doe, age 35, works as a software engineer at Google."

Return as JSON:
{
  "name": "...",
  "age": ...,
  "job_title": "...",
  "company": "..."
}
```

**Python Implementation**:
```python
import openai
import json

def extract_person_info(text):
    prompt = f"""Extract the following information from the text below:
- Name (first and last)
- Age (as integer)
- Job title
- Company name

If any information is missing, use null.

Text: "{text}"

Return as JSON:
{{
  "name": "...",
  "age": ...,
  "job_title": "...",
  "company": "..."
}}"""
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0  # Deterministic for extraction
    )
    
    return json.loads(response.choices[0].message.content)

# Test
text = "John Doe, age 35, works as a software engineer at Google."
result = extract_person_info(text)
print(json.dumps(result, indent=2))
```

### Example 3: Multi-Step Task

**Task**: Analyze code quality and suggest improvements.

**Prompt**:
```
Analyze the following Python function for code quality issues. Perform the analysis in three steps:

Step 1: Identify syntax and runtime errors
Step 2: Find style issues (PEP 8 violations, naming, etc.)
Step 3: Suggest improvements for readability and performance

For each issue found:
- Provide line number (if applicable)
- Describe the issue
- Suggest a fix

Function:
```python
def calcTotal(items):
    total=0
    for i in items:
        total+=i
    return total
```

Format your response as:
**Step 1: Errors**
- [issue description]

**Step 2: Style Issues**
- [issue description]

**Step 3: Improvements**
- [suggestion]
```

### Example 4: Creative Task with Constraints

**Task**: Generate product descriptions with specific tone and format.

**Prompt**:
```
Write a product description for a wireless mouse targeting professional users.

Requirements:
- Length: 50-75 words
- Tone: Professional, technical, confident (not promotional)
- Include: Key features, use case, benefit
- Avoid: Exclamation marks, superlatives ("best", "amazing"), promotional language
- Format: 2-3 short paragraphs

Example of good tone:
"The ergonomic design reduces wrist strain during extended coding sessions. 
Connectivity options include both Bluetooth and USB receiver for flexibility across devices."

Now write a description for: Wireless Mouse Pro X1
Features: Ergonomic design, silent clicks, precision tracking, long battery life, USB-C charging
```

### Example 5: Using System Messages (OpenAI API)

**Task**: Create a consistent assistant persona.

**Python Code**:
```python
import openai

# System message sets persistent behavior
system_message = """You are a helpful programming tutor. Your teaching style is:
- Patient and encouraging
- Uses simple analogies for complex concepts
- Provides code examples
- Asks clarifying questions before giving answers
- Never gives away complete solutions immediately"""

def tutor_session(user_question):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_question}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content

# The system message persists, so the tutor persona is maintained
print(tutor_session("What is a recursive function?"))
print(tutor_session("Can you explain closures?"))  # Still acts as tutor
```

### Example 6: Handling Long Context

**Problem**: You need to analyze a long document but want focused responses.

**Strategy**: Provide document + specific question.

**Prompt Structure**:
```
You have access to the following document (2000 words). Answer the question 
below by referencing specific sections.

Document:
[Full document content here]

Question: What are the three main recommendations for improving team productivity?

Instructions:
1. Identify relevant sections (provide paragraph numbers)
2. Extract key points
3. Synthesize into three recommendations
4. Cite specific quotes where helpful

Format:
**Recommendation 1:** [text]
*Source: Paragraph X*

**Recommendation 2:** [text]
*Source: Paragraph Y*

**Recommendation 3:** [text]
*Source: Paragraph Z*
```

### Example 7: Error Handling and Retries

**Practical Implementation**:
```python
import openai
import json
import time

def extract_with_retry(text, max_retries=3):
    prompt = f"""Extract name, email, and phone from: "{text}"
    
Return as JSON: {{"name": "...", "email": "...", "phone": "..."}}
If any field is missing, use null."""
    
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            
            output = response.choices[0].message.content
            
            # Try to parse JSON
            result = json.loads(output)
            
            # Validate structure
            if all(key in result for key in ["name", "email", "phone"]):
                return result
            else:
                raise ValueError("Missing required fields")
                
        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                # Add explicit JSON instruction
                prompt += "\n\nIMPORTANT: Return ONLY valid JSON, no other text."
                time.sleep(1)  # Brief delay
                continue
            else:
                raise
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            else:
                raise
    
    return None

# Usage
text = "Contact John at john@example.com or call 555-1234"
result = extract_with_retry(text)
print(result)
```

---

## Exercises

### Exercise 1: Basic Prompt Structure

**Task**: Create a prompt that converts temperatures from Celsius to Fahrenheit.

**Requirements**:
- Use all four core components (instruction, context, input, output format)
- Handle edge cases (negative numbers, decimals)
- Specify output precision

**Starting Point**:
```
[Your prompt here]
```

**Test Cases**:
- 0°C → 32°F
- 100°C → 212°F
- -40°C → -40°F
- 37.5°C → 99.5°F

### Exercise 2: Clarifying Ambiguous Prompts

**Original (Ambiguous)**:
```
Fix this code.
```

**Task**: Rewrite this prompt to be specific and actionable. Include:
- What type of issues to find
- How to report them
- What fixes to suggest

**Code to Analyze**:
```python
def calculate_average(numbers):
    sum = 0
    for n in numbers:
        sum = sum + n
    average = sum / len(numbers)
    return average
```

### Exercise 3: Temperature Selection

**Scenario**: You're building three different applications. Choose appropriate temperature settings:

1. **Code review bot**: Finds bugs and suggests fixes
2. **Creative writing assistant**: Generates story ideas
3. **Customer support chatbot**: Provides factual responses

**Task**: For each, choose a temperature (0.0-2.0) and explain your choice.

### Exercise 4: Output Format Specification

**Task**: Create a prompt that extracts meeting action items from notes.

**Requirements**:
- Extract: task description, assignee, due date, priority
- Handle missing information
- Return as structured JSON
- Include validation rules

**Sample Input**:
```
Team Meeting - March 15, 2024

Action Items:
- Sarah: Fix login bug by Friday (high priority)
- Mike: Review PR #123 (no specific date)
- Lisa: Schedule demo with stakeholders
```

### Exercise 5: System Message Design

**Task**: Design a system message for a technical documentation assistant.

**Requirements**:
- Persona: Expert technical writer
- Style: Clear, concise, structured
- Behavior: Asks clarifying questions when needed
- Specialization: Software documentation

### Exercise 6: Error Handling Prompt

**Task**: Create a prompt for data validation that handles errors gracefully.

**Scenario**: Extract addresses from text, but text may be malformed.

**Requirements**:
- Extract: street, city, state, zip
- Handle missing fields
- Detect invalid formats
- Return structured error messages when validation fails

**Test Cases**:
```
1. "123 Main St, New York, NY 10001"
2. "New York" (incomplete)
3. "123 Main St, NY" (missing city)
```

### Exercise 7: Token Optimization

**Task**: Optimize this prompt to use fewer tokens while maintaining quality.

**Original**:
```
I want you to help me understand something. Specifically, I need you to explain 
to me in a clear and comprehensive manner the concept of recursion in computer 
programming. Please provide a detailed explanation that covers what recursion is, 
how it works, why it's useful, and include some examples to illustrate the concept. 
I'm a beginner, so please make sure the explanation is accessible and easy to 
understand. Thank you!
```

**Goal**: Reduce tokens by 40-50% while keeping the same intent.

### Exercise 8: Multi-Turn Conversation

**Task**: Design a prompt system for a cooking assistant that maintains context.

**Requirements**:
- First message: "I want to make pasta"
- Follow-up: "What ingredients do I need?"
- Follow-up: "I don't have tomatoes, what can I substitute?"

**Challenge**: Ensure the assistant remembers it's about pasta and adapts the recipe.

### Exercise 9: Model-Specific Prompting

**Task**: Adapt the same task for different models.

**Base Task**: Summarize a 10-page document.

**Adapt for**:
1. GPT-4 (128K context)
2. GPT-3.5 (16K context - may need chunking)

**Hint**: Consider context window limitations.

### Exercise 10: Evaluation Framework

**Task**: Design an evaluation for a prompt that extracts product information.

**Prompt**: Extract product name, price, and description from e-commerce product pages.

**Create**:
1. 5 test cases (various formats)
2. Ground truth for each
3. Evaluation checklist
4. Success criteria

### Exercise 11: Few-Shot Learning Setup

**Task**: Create a few-shot prompt for email classification (spam/important/normal).

**Provide**:
- 3 examples of each category
- Clear formatting
- Instructions for classification

**Example Format**:
```
Email: "You've won $1,000,000!"
Category: spam

Email: "Your package will arrive tomorrow"
Category: important

[Continue pattern...]
```

### Exercise 12: Constraint Specification

**Task**: Generate social media post captions with constraints.

**Requirements**:
- Length: 50-100 characters
- Tone: Professional but friendly
- Include: One hashtag, one emoji
- Exclude: Controversial topics, promotional language
- Focus: Tech industry updates

**Create prompt with all constraints clearly defined**.

### Exercise 13: Context Window Management

**Task**: Design a strategy for analyzing a 50-page document with a model that has 16K token context.

**Consider**:
- How to chunk the document
- What information to preserve across chunks
- How to synthesize results

### Exercise 14: Iterative Refinement

**Task**: Start with a basic prompt and refine it through 3 iterations.

**Initial Prompt**:
```
Translate this to French: "Hello, how are you?"
```

**Iterations**:
1. Add context about formality level
2. Add output format specification
3. Add error handling instructions

### Exercise 15: Complete Application

**Task**: Build a complete prompt-based application.

**Application**: Customer review analyzer

**Features**:
1. Sentiment analysis (positive/negative/neutral)
2. Extract key topics mentioned
3. Identify specific complaints or praises
4. Generate summary report

**Deliverables**:
- System message
- Main prompt template
- Output format specification
- Error handling strategy
- Evaluation plan

---

## Summary

### Key Takeaways

1. **Prompt Structure**: Use clear instruction, context, input, and output format
2. **Clarity Principles**: Be specific, use examples, define constraints
3. **Parameters Matter**: Temperature, top-p, and other settings affect output quality
4. **Common Mistakes**: Avoid ambiguity, over-prompting, under-prompting
5. **Evaluation**: Measure relevance, accuracy, consistency, and completeness
6. **Model Selection**: Choose based on task, context, cost, and quality needs
7. **Iteration**: Good prompts require testing and refinement

### Next Steps

- **Module 2**: Learn prompt design patterns (zero-shot, few-shot, role prompting)
- Practice with the exercises above
- Experiment with different models and parameters
- Build a prompt library with your successful prompts

### Additional Resources

- OpenAI Prompt Engineering Guide
- Anthropic Prompt Library
- LangChain Documentation
- Prompt Engineering Papers (arXiv)

---

**End of Module 1**
