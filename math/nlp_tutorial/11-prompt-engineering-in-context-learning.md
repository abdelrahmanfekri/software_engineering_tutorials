# NLP Tutorial Module 11: Prompt Engineering and In-Context Learning (PhD Level)

## Learning Objectives
By the end of this module, you will be able to:
- Master theoretical foundations of in-context learning and emergent abilities
- Design and optimize prompts for various tasks systematically
- Understand and implement chain-of-thought reasoning
- Apply advanced prompting techniques (self-consistency, tree-of-thought, ReAct)
- Build instruction-following models through instruction tuning
- Implement and analyze few-shot learning mechanisms
- Design evaluation frameworks for prompt effectiveness
- Understand the theoretical limits and capabilities of in-context learning

## Theoretical Foundations of In-Context Learning

### What is In-Context Learning?

In-context learning (ICL) is the ability of language models to perform tasks by conditioning on examples within the input prompt, without any gradient updates.

Formally, given a task $\mathcal{T}$ with examples $\{(x_i, y_i)\}_{i=1}^k$ and query $x_q$:

$$P(y_q | x_q, \{(x_i, y_i)\}_{i=1}^k) = P_{\text{LM}}(y_q | [\text{prompt}; x_1, y_1; ...; x_k, y_k; x_q])$$

### Theoretical Understanding

#### 1. Bayesian Perspective

From a Bayesian view, ICL implements approximate Bayesian inference:

$$P(y|x, \mathcal{D}) \approx \int P(y|x, \theta) P(\theta|\mathcal{D}) d\theta$$

where the model implicitly learns a prior over functions $P(\theta)$ during pre-training.

#### 2. Meta-Learning View

ICL can be understood as implicit meta-learning where the model learns to:
1. Infer the task from examples
2. Adapt its predictions accordingly

This is related to MAML (Model-Agnostic Meta-Learning):

$$\theta^* = \theta - \alpha \nabla_\theta \mathcal{L}_{\text{task}}(\theta)$$

but implemented in forward pass rather than through gradient descent.

#### 3. Information-Theoretic View

The effectiveness of ICL relates to mutual information between examples and query:

$$I(Y_q; \{(X_i, Y_i)\}_{i=1}^k | X_q)$$

### Implementation and Analysis

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class PromptExample:
    """Single example in a prompt"""
    input_text: str
    output_text: str
    metadata: Optional[Dict] = None

class InContextLearningAnalyzer:
    """Analyze in-context learning behavior"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def measure_icl_performance(self, task_examples: List[Tuple[str, str]], 
                                test_examples: List[Tuple[str, str]],
                                num_shots: List[int] = [0, 1, 2, 4, 8, 16]):
        """Measure how performance scales with number of examples"""
        results = {}
        
        for k in num_shots:
            if k > len(task_examples):
                continue
            
            # Sample k examples
            demo_examples = task_examples[:k]
            
            # Test on all test examples
            accuracies = []
            for test_input, test_output in test_examples:
                prompt = self._create_prompt(demo_examples, test_input)
                prediction = self._get_prediction(prompt)
                accuracy = self._compute_accuracy(prediction, test_output)
                accuracies.append(accuracy)
            
            results[k] = {
                'mean_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies)
            }
        
        return results
    
    def analyze_example_order_sensitivity(self, examples: List[Tuple[str, str]], 
                                         test_query: str, 
                                         num_permutations: int = 10):
        """Analyze sensitivity to example ordering"""
        predictions = []
        
        for _ in range(num_permutations):
            # Random permutation
            perm_examples = np.random.permutation(examples).tolist()
            prompt = self._create_prompt(perm_examples, test_query)
            prediction = self._get_prediction(prompt)
            predictions.append(prediction)
        
        # Measure consistency
        unique_predictions = len(set(predictions))
        consistency_score = 1.0 - (unique_predictions - 1) / len(predictions)
        
        return {
            'predictions': predictions,
            'consistency_score': consistency_score,
            'unique_predictions': unique_predictions
        }
    
    def analyze_example_selection(self, train_examples: List[Tuple[str, str]],
                                  test_query: str,
                                  selection_strategies: List[str] = ['random', 'similar', 'diverse']):
        """Compare different example selection strategies"""
        results = {}
        
        for strategy in selection_strategies:
            if strategy == 'random':
                selected = self._random_selection(train_examples, k=4)
            elif strategy == 'similar':
                selected = self._similar_selection(train_examples, test_query, k=4)
            elif strategy == 'diverse':
                selected = self._diverse_selection(train_examples, k=4)
            
            prompt = self._create_prompt(selected, test_query)
            prediction = self._get_prediction(prompt)
            
            results[strategy] = {
                'selected_examples': selected,
                'prediction': prediction
            }
        
        return results
    
    def _create_prompt(self, examples: List[Tuple[str, str]], query: str) -> str:
        """Create few-shot prompt"""
        prompt = ""
        for input_text, output_text in examples:
            prompt += f"Input: {input_text}\nOutput: {output_text}\n\n"
        prompt += f"Input: {query}\nOutput:"
        return prompt
    
    def _get_prediction(self, prompt: str) -> str:
        """Get model prediction"""
        inputs = self.tokenizer(prompt, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=100)
        prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return prediction.split("Output:")[-1].strip()
    
    def _compute_accuracy(self, prediction: str, ground_truth: str) -> float:
        """Compute accuracy"""
        return 1.0 if prediction.strip().lower() == ground_truth.strip().lower() else 0.0
    
    def _random_selection(self, examples: List[Tuple[str, str]], k: int) -> List[Tuple[str, str]]:
        """Random example selection"""
        return np.random.choice(examples, size=min(k, len(examples)), replace=False).tolist()
    
    def _similar_selection(self, examples: List[Tuple[str, str]], query: str, k: int) -> List[Tuple[str, str]]:
        """Select most similar examples to query"""
        # Compute embeddings
        query_embedding = self._get_embedding(query)
        example_embeddings = [self._get_embedding(ex[0]) for ex in examples]
        
        # Compute similarities
        similarities = [self._cosine_similarity(query_embedding, ex_emb) 
                       for ex_emb in example_embeddings]
        
        # Select top-k
        top_indices = np.argsort(similarities)[-k:]
        return [examples[i] for i in top_indices]
    
    def _diverse_selection(self, examples: List[Tuple[str, str]], k: int) -> List[Tuple[str, str]]:
        """Select diverse set of examples"""
        if k >= len(examples):
            return examples
        
        selected = [examples[0]]  # Start with first example
        remaining = examples[1:]
        
        for _ in range(k - 1):
            # Find example most dissimilar to already selected
            min_similarity = float('inf')
            best_idx = 0
            
            for idx, candidate in enumerate(remaining):
                max_sim = max([self._text_similarity(candidate[0], sel[0]) 
                              for sel in selected])
                if max_sim < min_similarity:
                    min_similarity = max_sim
                    best_idx = idx
            
            selected.append(remaining[best_idx])
            remaining.pop(best_idx)
        
        return selected
    
    def _get_embedding(self, text: str) -> torch.Tensor:
        """Get text embedding"""
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[-1].mean(dim=1).squeeze()
    
    def _cosine_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """Compute cosine similarity"""
        return F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Compute text similarity"""
        emb1 = self._get_embedding(text1)
        emb2 = self._get_embedding(text2)
        return self._cosine_similarity(emb1, emb2)
```

## Prompt Engineering Principles

### 1. Task Specification

Clear task specification is crucial. Components:
- **Task description**: What the model should do
- **Input format**: How inputs are structured
- **Output format**: Expected output format
- **Constraints**: Any limitations or requirements

```python
class PromptTemplate:
    """Template for structured prompts"""
    
    def __init__(self, task_description: str, 
                 input_format: str,
                 output_format: str,
                 constraints: Optional[List[str]] = None):
        self.task_description = task_description
        self.input_format = input_format
        self.output_format = output_format
        self.constraints = constraints or []
    
    def create_prompt(self, input_data: str, examples: Optional[List[Tuple[str, str]]] = None) -> str:
        """Create prompt from template"""
        prompt = f"Task: {self.task_description}\n\n"
        
        if self.constraints:
            prompt += "Constraints:\n"
            for constraint in self.constraints:
                prompt += f"- {constraint}\n"
            prompt += "\n"
        
        prompt += f"Input Format: {self.input_format}\n"
        prompt += f"Output Format: {self.output_format}\n\n"
        
        if examples:
            prompt += "Examples:\n"
            for i, (ex_input, ex_output) in enumerate(examples, 1):
                prompt += f"\nExample {i}:\n"
                prompt += f"Input: {ex_input}\n"
                prompt += f"Output: {ex_output}\n"
            prompt += "\n"
        
        prompt += f"Now complete this:\nInput: {input_data}\nOutput:"
        
        return prompt
    
    def validate_output(self, output: str) -> Tuple[bool, str]:
        """Validate output against format"""
        # Implementation depends on output format
        return True, ""

# Example usage
sentiment_template = PromptTemplate(
    task_description="Classify the sentiment of movie reviews",
    input_format="A movie review text",
    output_format="One of: positive, negative, neutral",
    constraints=[
        "Consider the overall tone and sentiment",
        "Focus on the reviewer's opinion, not plot description"
    ]
)

examples = [
    ("The movie was amazing!", "positive"),
    ("Waste of time", "negative"),
    ("It was okay", "neutral")
]

prompt = sentiment_template.create_prompt("Best film I've seen this year!", examples)
print(prompt)
```

### 2. Instruction Following

Instruction tuning improves zero-shot performance:

```python
class InstructionGenerator:
    """Generate instructions for tasks"""
    
    def __init__(self):
        self.instruction_templates = {
            'classification': [
                "Classify the following {item} into one of these categories: {categories}",
                "Determine which category this {item} belongs to: {categories}",
                "Label this {item} as one of: {categories}"
            ],
            'generation': [
                "Generate a {output_type} based on the following {input_type}",
                "Create a {output_type} that {requirement}",
                "Write a {output_type} following these guidelines: {guidelines}"
            ],
            'transformation': [
                "Transform the following {input_type} into {output_type}",
                "Convert this {input_type} to {output_type}",
                "Rewrite the {input_type} as a {output_type}"
            ]
        }
    
    def generate_instruction(self, task_type: str, **kwargs) -> str:
        """Generate instruction for task"""
        templates = self.instruction_templates.get(task_type, [])
        if not templates:
            raise ValueError(f"Unknown task type: {task_type}")
        
        # Select random template and fill in
        template = np.random.choice(templates)
        instruction = template.format(**kwargs)
        
        return instruction
    
    def create_instruction_prompt(self, instruction: str, input_text: str, 
                                 examples: Optional[List[Dict]] = None) -> str:
        """Create instruction-following prompt"""
        prompt = f"{instruction}\n\n"
        
        if examples:
            for i, example in enumerate(examples, 1):
                prompt += f"Example {i}:\n"
                prompt += f"Input: {example['input']}\n"
                prompt += f"Output: {example['output']}\n\n"
        
        prompt += f"Input: {input_text}\nOutput:"
        
        return prompt

# Example usage
instruction_gen = InstructionGenerator()

instruction = instruction_gen.generate_instruction(
    'classification',
    item='text',
    categories='positive, negative, neutral'
)

prompt = instruction_gen.create_instruction_prompt(
    instruction,
    "This movie is fantastic!",
    examples=[
        {'input': 'Great film!', 'output': 'positive'},
        {'input': 'Boring movie', 'output': 'negative'}
    ]
)
```

## Chain-of-Thought (CoT) Reasoning

### Theory

Chain-of-Thought prompting elicits step-by-step reasoning:

$$P(y|x) = \sum_{r} P(y|r, x) P(r|x)$$

where $r$ represents the reasoning chain.

### Implementation

```python
class ChainOfThoughtPrompter:
    """Chain-of-Thought prompting system"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def create_cot_prompt(self, question: str, examples: List[Dict]) -> str:
        """Create CoT prompt with reasoning examples"""
        prompt = "Let's solve these problems step by step.\n\n"
        
        for i, example in enumerate(examples, 1):
            prompt += f"Q{i}: {example['question']}\n"
            prompt += f"A{i}: Let's think step by step.\n"
            prompt += f"{example['reasoning']}\n"
            prompt += f"Therefore, the answer is {example['answer']}.\n\n"
        
        prompt += f"Q: {question}\n"
        prompt += "A: Let's think step by step.\n"
        
        return prompt
    
    def generate_with_cot(self, question: str, examples: List[Dict],
                         max_length: int = 500) -> Dict[str, str]:
        """Generate answer with chain-of-thought reasoning"""
        prompt = self.create_cot_prompt(question, examples)
        
        # Generate reasoning and answer
        inputs = self.tokenizer(prompt, return_tensors='pt')
        outputs = self.model.generate(**inputs, max_length=max_length, 
                                     temperature=0.7, do_sample=True)
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse reasoning and answer
        response_text = response.split("Let's think step by step.")[-1].strip()
        
        if "Therefore, the answer is" in response_text:
            reasoning, answer = response_text.split("Therefore, the answer is")
            answer = answer.strip().rstrip('.')
        else:
            reasoning = response_text
            answer = "Unable to determine"
        
        return {
            'reasoning': reasoning.strip(),
            'answer': answer.strip(),
            'full_response': response
        }
    
    def zero_shot_cot(self, question: str) -> Dict[str, str]:
        """Zero-shot Chain-of-Thought (just add 'Let's think step by step')"""
        prompt = f"Q: {question}\nA: Let's think step by step."
        
        inputs = self.tokenizer(prompt, return_tensors='pt')
        outputs = self.model.generate(**inputs, max_length=500)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer from reasoning
        lines = response.split('\n')
        reasoning_lines = []
        answer = "Unable to determine"
        
        for line in lines:
            if line.strip():
                reasoning_lines.append(line)
                if "answer is" in line.lower():
                    answer = line.split("answer is")[-1].strip().rstrip('.')
        
        return {
            'reasoning': '\n'.join(reasoning_lines),
            'answer': answer
        }

# Example usage
cot_prompter = ChainOfThoughtPrompter(model, tokenizer)

# Few-shot CoT examples
examples = [
    {
        'question': 'Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?',
        'reasoning': 'Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11.',
        'answer': '11'
    },
    {
        'question': 'The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?',
        'reasoning': 'The cafeteria had 23 apples originally. They used 20 to make lunch. So they had 23 - 20 = 3. They bought 6 more apples, so they have 3 + 6 = 9.',
        'answer': '9'
    }
]

question = "Tom has 10 marbles. He gives 3 to his friend and then finds 5 more. How many marbles does Tom have now?"

result = cot_prompter.generate_with_cot(question, examples)
print(f"Reasoning: {result['reasoning']}")
print(f"Answer: {result['answer']}")

# Zero-shot CoT
zero_shot_result = cot_prompter.zero_shot_cot(question)
print(f"\nZero-shot reasoning: {zero_shot_result['reasoning']}")
```

### Self-Consistency

Sample multiple reasoning paths and take majority vote:

```python
class SelfConsistencyDecoder:
    """Self-consistency for improved CoT reasoning"""
    
    def __init__(self, cot_prompter, num_samples: int = 5):
        self.cot_prompter = cot_prompter
        self.num_samples = num_samples
    
    def decode_with_self_consistency(self, question: str, 
                                    examples: List[Dict]) -> Dict[str, any]:
        """Generate multiple reasoning paths and aggregate"""
        reasoning_paths = []
        answers = []
        
        for _ in range(self.num_samples):
            result = self.cot_prompter.generate_with_cot(question, examples)
            reasoning_paths.append(result['reasoning'])
            answers.append(result['answer'])
        
        # Majority vote
        answer_counts = {}
        for answer in answers:
            answer_counts[answer] = answer_counts.get(answer, 0) + 1
        
        final_answer = max(answer_counts.items(), key=lambda x: x[1])[0]
        confidence = answer_counts[final_answer] / self.num_samples
        
        return {
            'answer': final_answer,
            'confidence': confidence,
            'all_answers': answers,
            'all_reasoning': reasoning_paths,
            'answer_distribution': answer_counts
        }

# Example usage
self_consistency = SelfConsistencyDecoder(cot_prompter, num_samples=10)
result = self_consistency.decode_with_self_consistency(question, examples)

print(f"Final Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Answer Distribution: {result['answer_distribution']}")
```

## Advanced Prompting Techniques

### 1. Tree-of-Thought (ToT)

Explore multiple reasoning paths in tree structure:

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ThoughtNode:
    """Node in tree-of-thought"""
    content: str
    parent: Optional['ThoughtNode'] = None
    children: List['ThoughtNode'] = None
    value: float = 0.0
    visits: int = 0
    
    def __post_init__(self):
        if self.children is None:
            self.children = []

class TreeOfThoughtReasoner:
    """Tree-of-Thought reasoning with search"""
    
    def __init__(self, model, tokenizer, branching_factor: int = 3, max_depth: int = 5):
        self.model = model
        self.tokenizer = tokenizer
        self.branching_factor = branching_factor
        self.max_depth = max_depth
    
    def solve_problem(self, problem: str, search_strategy: str = 'bfs') -> Dict:
        """Solve problem using ToT"""
        root = ThoughtNode(content=f"Problem: {problem}")
        
        if search_strategy == 'bfs':
            solution = self._breadth_first_search(root)
        elif search_strategy == 'dfs':
            solution = self._depth_first_search(root)
        elif search_strategy == 'best_first':
            solution = self._best_first_search(root)
        else:
            raise ValueError(f"Unknown search strategy: {search_strategy}")
        
        return {
            'solution': solution,
            'reasoning_path': self._extract_path(solution) if solution else None
        }
    
    def _generate_children(self, node: ThoughtNode, num_children: int) -> List[ThoughtNode]:
        """Generate child thoughts"""
        prompt = self._create_generation_prompt(node)
        
        children = []
        for _ in range(num_children):
            # Generate next thought
            inputs = self.tokenizer(prompt, return_tensors='pt')
            outputs = self.model.generate(**inputs, max_length=200, 
                                        temperature=0.8, do_sample=True)
            thought_content = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Create child node
            child = ThoughtNode(content=thought_content, parent=node)
            
            # Evaluate thought
            child.value = self._evaluate_thought(child, node.content)
            
            children.append(child)
            node.children.append(child)
        
        return children
    
    def _evaluate_thought(self, thought: ThoughtNode, problem: str) -> float:
        """Evaluate quality of a thought"""
        eval_prompt = f"""
        Problem: {problem}
        
        Reasoning step: {thought.content}
        
        Rate this reasoning step on a scale of 1-10 based on:
        - Logical correctness
        - Relevance to problem
        - Progress toward solution
        
        Rating (1-10):"""
        
        inputs = self.tokenizer(eval_prompt, return_tensors='pt')
        outputs = self.model.generate(**inputs, max_length=10, temperature=0.1)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        try:
            rating = float(response.split(':')[-1].strip())
            return min(10.0, max(1.0, rating)) / 10.0
        except:
            return 0.5
    
    def _breadth_first_search(self, root: ThoughtNode) -> Optional[ThoughtNode]:
        """BFS to find solution"""
        queue = [root]
        depth = 0
        
        while queue and depth < self.max_depth:
            level_size = len(queue)
            
            for _ in range(level_size):
                node = queue.pop(0)
                
                # Check if this is a solution
                if self._is_solution(node):
                    return node
                
                # Generate and add children
                if depth < self.max_depth - 1:
                    children = self._generate_children(node, self.branching_factor)
                    queue.extend(children)
            
            depth += 1
        
        # Return best node found
        return self._find_best_node(root)
    
    def _best_first_search(self, root: ThoughtNode) -> Optional[ThoughtNode]:
        """Best-first search using value estimates"""
        frontier = [root]
        best_solution = None
        best_value = 0.0
        
        iteration = 0
        max_iterations = 50
        
        while frontier and iteration < max_iterations:
            # Sort by value
            frontier.sort(key=lambda n: n.value, reverse=True)
            
            # Expand best node
            node = frontier.pop(0)
            
            # Check if solution
            if self._is_solution(node):
                if node.value > best_value:
                    best_solution = node
                    best_value = node.value
                continue
            
            # Generate children
            if node.depth < self.max_depth:
                children = self._generate_children(node, self.branching_factor)
                # Update children depth
                for child in children:
                    child.depth = node.depth + 1
                frontier.extend(children)
            
            iteration += 1
        
        return best_solution or self._find_best_node(root)
    
    def _is_solution(self, node: ThoughtNode) -> bool:
        """Check if node represents a solution"""
        # Check if thought contains solution indicators
        content = node.content.lower()
        return any(indicator in content for indicator in 
                  ['therefore', 'final answer', 'solution is', 'the answer is'])
    
    def _find_best_node(self, root: ThoughtNode) -> ThoughtNode:
        """Find highest-valued node in tree"""
        best_node = root
        best_value = root.value
        
        def traverse(node):
            nonlocal best_node, best_value
            if node.value > best_value:
                best_node = node
                best_value = node.value
            for child in node.children:
                traverse(child)
        
        traverse(root)
        return best_node
    
    def _extract_path(self, node: ThoughtNode) -> List[str]:
        """Extract reasoning path from node to root"""
        path = []
        current = node
        while current is not None:
            path.insert(0, current.content)
            current = current.parent
        return path
    
    def _create_generation_prompt(self, node: ThoughtNode) -> str:
        """Create prompt for generating next thought"""
        path = self._extract_path(node)
        prompt = "Problem solving process:\n\n"
        for i, thought in enumerate(path, 1):
            prompt += f"Step {i}: {thought}\n"
        prompt += "\nNext step:"
        return prompt
    
    def _depth_first_search(self, root: ThoughtNode) -> Optional[ThoughtNode]:
        """DFS to find solution"""
        def dfs(node, depth):
            if depth >= self.max_depth or self._is_solution(node):
                return node
            
            children = self._generate_children(node, self.branching_factor)
            
            best_solution = None
            best_value = 0.0
            
            for child in children:
                child.depth = depth + 1
                result = dfs(child, depth + 1)
                if result and result.value > best_value:
                    best_solution = result
                    best_value = result.value
            
            return best_solution or node
        
        return dfs(root, 0)
```

### 2. ReAct: Reasoning and Acting

Interleave reasoning traces with actions:

```python
class ReActAgent:
    """ReAct: Reasoning and Acting agent"""
    
    def __init__(self, model, tokenizer, available_tools: Dict[str, callable]):
        self.model = model
        self.tokenizer = tokenizer
        self.tools = available_tools
        self.max_iterations = 10
    
    def solve_task(self, task: str) -> Dict:
        """Solve task using ReAct approach"""
        history = []
        thought = ""
        action = ""
        observation = ""
        
        for iteration in range(self.max_iterations):
            # Generate thought
            thought = self._generate_thought(task, history)
            history.append(('thought', thought))
            
            # Check if task is complete
            if self._is_complete(thought):
                answer = self._extract_answer(thought)
                return {
                    'answer': answer,
                    'history': history,
                    'iterations': iteration + 1
                }
            
            # Generate action
            action = self._generate_action(thought, history)
            history.append(('action', action))
            
            # Execute action
            observation = self._execute_action(action)
            history.append(('observation', observation))
        
        return {
            'answer': "Max iterations reached without solution",
            'history': history,
            'iterations': self.max_iterations
        }
    
    def _generate_thought(self, task: str, history: List[Tuple[str, str]]) -> str:
        """Generate reasoning thought"""
        prompt = f"Task: {task}\n\n"
        
        # Add history
        for i, (step_type, content) in enumerate(history):
            prompt += f"{step_type.capitalize()} {i+1}: {content}\n"
        
        prompt += "\nThought:"
        
        inputs = self.tokenizer(prompt, return_tensors='pt')
        outputs = self.model.generate(**inputs, max_length=200)
        thought = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return thought.split("Thought:")[-1].strip()
    
    def _generate_action(self, thought: str, history: List[Tuple[str, str]]) -> str:
        """Generate action based on thought"""
        prompt = f"Based on the thought: {thought}\n\n"
        prompt += f"Available actions: {list(self.tools.keys())}\n\n"
        prompt += "Action:"
        
        inputs = self.tokenizer(prompt, return_tensors='pt')
        outputs = self.model.generate(**inputs, max_length=100)
        action = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return action.split("Action:")[-1].strip()
    
    def _execute_action(self, action: str) -> str:
        """Execute action using available tools"""
        # Parse action
        if '[' in action and ']' in action:
            tool_name = action.split('[')[0].strip()
            args = action.split('[')[1].split(']')[0]
        else:
            tool_name = action.strip()
            args = ""
        
        # Execute tool
        if tool_name in self.tools:
            try:
                result = self.tools[tool_name](args)
                return f"Result: {result}"
            except Exception as e:
                return f"Error: {str(e)}"
        else:
            return f"Unknown tool: {tool_name}"
    
    def _is_complete(self, thought: str) -> bool:
        """Check if task is complete"""
        indicators = ['final answer', 'therefore', 'in conclusion', 'the answer is']
        return any(indicator in thought.lower() for indicator in indicators)
    
    def _extract_answer(self, thought: str) -> str:
        """Extract answer from final thought"""
        for indicator in ['final answer is', 'therefore', 'the answer is']:
            if indicator in thought.lower():
                return thought.split(indicator)[-1].strip()
        return thought

# Example tools
def wikipedia_search(query: str) -> str:
    """Mock Wikipedia search"""
    return f"Wikipedia result for '{query}': [Mock content]"

def calculator(expression: str) -> str:
    """Calculate mathematical expression"""
    try:
        result = eval(expression)
        return str(result)
    except:
        return "Invalid expression"

def lookup(entity: str) -> str:
    """Look up information about entity"""
    return f"Information about {entity}: [Mock data]"

# Initialize agent
tools = {
    'Search': wikipedia_search,
    'Calculate': calculator,
    'Lookup': lookup
}

react_agent = ReActAgent(model, tokenizer, tools)

# Solve task
task = "What is the population of France multiplied by 2?"
result = react_agent.solve_task(task)

print("Answer:", result['answer'])
print("\nReasoning trace:")
for step_type, content in result['history']:
    print(f"{step_type.upper()}: {content}")
```

### 3. Program-Aided Language Models (PAL)

Generate and execute code for reasoning:

```python
class ProgramAidedLM:
    """Program-Aided Language Model"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def solve_with_code(self, problem: str, examples: Optional[List[Dict]] = None) -> Dict:
        """Solve problem by generating and executing code"""
        # Generate code
        code = self._generate_code(problem, examples)
        
        # Execute code
        try:
            result = self._execute_code(code)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        return {
            'code': code,
            'result': result,
            'success': success,
            'error': error
        }
    
    def _generate_code(self, problem: str, examples: Optional[List[Dict]]) -> str:
        """Generate Python code to solve problem"""
        prompt = "# Generate Python code to solve this problem\n\n"
        
        if examples:
            for example in examples:
                prompt += f"# Problem: {example['problem']}\n"
                prompt += f"{example['code']}\n\n"
        
        prompt += f"# Problem: {problem}\n"
        prompt += "def solve():\n    "
        
        inputs = self.tokenizer(prompt, return_tensors='pt')
        outputs = self.model.generate(**inputs, max_length=500, 
                                     temperature=0.2, do_sample=True)
        
        code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the function definition
        if "def solve():" in code:
            code = "def solve():" + code.split("def solve():")[-1]
        
        # Add call to solve function
        code += "\n\nresult = solve()\nprint(result)"
        
        return code
    
    def _execute_code(self, code: str) -> any:
        """Execute generated code safely"""
        # Create restricted execution environment
        namespace = {
            '__builtins__': {
                'range': range,
                'len': len,
                'sum': sum,
                'max': max,
                'min': min,
                'abs': abs,
                'round': round,
                'int': int,
                'float': float,
                'str': str,
                'list': list,
                'dict': dict,
                'print': print
            }
        }
        
        # Execute code
        exec(code, namespace)
        
        # Return result
        return namespace.get('result')

# Example usage
pal = ProgramAidedLM(model, tokenizer)

examples = [
    {
        'problem': 'Calculate the sum of all even numbers from 1 to 10',
        'code': '''def solve():
    return sum(x for x in range(1, 11) if x % 2 == 0)'''
    }
]

problem = "Calculate the sum of squares of all odd numbers from 1 to 20"
result = pal.solve_with_code(problem, examples)

print("Generated code:")
print(result['code'])
print(f"\nResult: {result['result']}")
```

## Instruction Tuning

### Supervised Fine-Tuning on Instructions

```python
class InstructionTuner:
    """Fine-tune model on instruction-following tasks"""
    
    def __init__(self, model, tokenizer, learning_rate: float = 5e-5):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
    def prepare_instruction_data(self, instructions: List[Dict]) -> List[Dict]:
        """Prepare instruction data for training"""
        prepared_data = []
        
        for item in instructions:
            prompt = self._format_instruction(item)
            target = item['output']
            
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            targets = self.tokenizer(
                target,
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            prepared_data.append({
                'input_ids': inputs['input_ids'],
                'attention_mask': inputs['attention_mask'],
                'labels': targets['input_ids']
            })
        
        return prepared_data
    
    def _format_instruction(self, item: Dict) -> str:
        """Format instruction item into prompt"""
        prompt = ""
        
        if 'instruction' in item:
            prompt += f"Instruction: {item['instruction']}\n\n"
        
        if 'input' in item and item['input']:
            prompt += f"Input: {item['input']}\n\n"
        
        prompt += "Output:"
        
        return prompt
    
    def train(self, train_data: List[Dict], epochs: int = 3):
        """Train on instruction data"""
        prepared_data = self.prepare_instruction_data(train_data)
        
        for epoch in range(epochs):
            total_loss = 0
            
            for batch in prepared_data:
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                loss = outputs.loss
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(prepared_data)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# Example instruction data
instruction_data = [
    {
        'instruction': 'Classify the sentiment of the following text',
        'input': 'This movie was fantastic!',
        'output': 'positive'
    },
    {
        'instruction': 'Translate the following English sentence to French',
        'input': 'Hello, how are you?',
        'output': 'Bonjour, comment allez-vous?'
    },
    {
        'instruction': 'Summarize the following text in one sentence',
        'input': 'Artificial intelligence is transforming how we work...',
        'output': 'AI is changing work and society.'
    }
]

tuner = InstructionTuner(model, tokenizer)
tuner.train(instruction_data, epochs=3)
```

## Prompt Optimization

### Automatic Prompt Engineering

```python
class AutomaticPromptEngineer:
    """Automatically optimize prompts"""
    
    def __init__(self, model, tokenizer, eval_dataset: List[Dict]):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        
    def optimize_prompt(self, initial_prompt: str, num_iterations: int = 10) -> str:
        """Optimize prompt through iterative refinement"""
        current_prompt = initial_prompt
        best_prompt = initial_prompt
        best_score = self._evaluate_prompt(initial_prompt)
        
        for iteration in range(num_iterations):
            # Generate prompt variations
            variations = self._generate_variations(current_prompt)
            
            # Evaluate each variation
            for variation in variations:
                score = self._evaluate_prompt(variation)
                
                if score > best_score:
                    best_score = score
                    best_prompt = variation
            
            current_prompt = best_prompt
            print(f"Iteration {iteration+1}: Best score = {best_score:.3f}")
        
        return best_prompt
    
    def _evaluate_prompt(self, prompt: str) -> float:
        """Evaluate prompt on eval dataset"""
        correct = 0
        total = len(self.eval_dataset)
        
        for example in self.eval_dataset:
            full_prompt = prompt.format(input=example['input'])
            prediction = self._get_prediction(full_prompt)
            
            if prediction.strip().lower() == example['output'].strip().lower():
                correct += 1
        
        return correct / total
    
    def _generate_variations(self, prompt: str) -> List[str]:
        """Generate variations of prompt"""
        # Strategy 1: Rephrase instruction
        rephrase_prompt = f"Rephrase the following instruction to make it clearer:\n{prompt}\n\nRephrased:"
        variation1 = self._get_prediction(rephrase_prompt)
        
        # Strategy 2: Add constraints
        variation2 = prompt + " Provide a concise and accurate answer."
        
        # Strategy 3: Add examples
        variation3 = "Here are some examples:\n[Example 1]\n[Example 2]\n\n" + prompt
        
        # Strategy 4: Change format
        variation4 = prompt.replace(":", ":\n")
        
        return [variation1, variation2, variation3, variation4]
    
    def _get_prediction(self, prompt: str) -> str:
        """Get model prediction"""
        inputs = self.tokenizer(prompt, return_tensors='pt')
        outputs = self.model.generate(**inputs, max_length=100)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
eval_data = [
    {'input': 'Great movie!', 'output': 'positive'},
    {'input': 'Terrible film', 'output': 'negative'},
    {'input': 'It was okay', 'output': 'neutral'}
]

ape = AutomaticPromptEngineer(model, tokenizer, eval_data)
initial_prompt = "Classify the sentiment: {input}"
optimized_prompt = ape.optimize_prompt(initial_prompt, num_iterations=5)

print(f"Optimized prompt: {optimized_prompt}")
```

## Evaluation and Analysis

### Prompt Evaluation Framework

```python
class PromptEvaluator:
    """Comprehensive prompt evaluation"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def evaluate_comprehensive(self, prompt_template: str, 
                              test_data: List[Dict]) -> Dict:
        """Comprehensive evaluation of prompt"""
        metrics = {
            'accuracy': self._evaluate_accuracy(prompt_template, test_data),
            'consistency': self._evaluate_consistency(prompt_template, test_data),
            'robustness': self._evaluate_robustness(prompt_template, test_data),
            'efficiency': self._evaluate_efficiency(prompt_template, test_data)
        }
        
        return metrics
    
    def _evaluate_accuracy(self, prompt_template: str, test_data: List[Dict]) -> float:
        """Evaluate prediction accuracy"""
        correct = 0
        for example in test_data:
            prompt = prompt_template.format(**example['input'])
            prediction = self._get_prediction(prompt)
            if prediction == example['output']:
                correct += 1
        
        return correct / len(test_data)
    
    def _evaluate_consistency(self, prompt_template: str, test_data: List[Dict], 
                            num_runs: int = 5) -> float:
        """Evaluate consistency across multiple runs"""
        consistency_scores = []
        
        for example in test_data:
            prompt = prompt_template.format(**example['input'])
            predictions = [self._get_prediction(prompt) for _ in range(num_runs)]
            
            # Consistency = fraction of runs with most common answer
            most_common = max(set(predictions), key=predictions.count)
            consistency = predictions.count(most_common) / num_runs
            consistency_scores.append(consistency)
        
        return np.mean(consistency_scores)
    
    def _evaluate_robustness(self, prompt_template: str, test_data: List[Dict]) -> float:
        """Evaluate robustness to perturbations"""
        robustness_scores = []
        
        for example in test_data:
            original_prompt = prompt_template.format(**example['input'])
            original_prediction = self._get_prediction(original_prompt)
            
            # Create perturbations
            perturbations = self._create_perturbations(original_prompt)
            
            # Check if predictions remain same
            consistent = sum(1 for p in perturbations 
                           if self._get_prediction(p) == original_prediction)
            
            robustness_scores.append(consistent / len(perturbations))
        
        return np.mean(robustness_scores)
    
    def _evaluate_efficiency(self, prompt_template: str, test_data: List[Dict]) -> Dict:
        """Evaluate efficiency metrics"""
        import time
        
        total_time = 0
        total_tokens = 0
        
        for example in test_data:
            prompt = prompt_template.format(**example['input'])
            
            # Measure time
            start_time = time.time()
            prediction = self._get_prediction(prompt)
            elapsed_time = time.time() - start_time
            
            total_time += elapsed_time
            
            # Count tokens
            tokens = self.tokenizer.encode(prompt + prediction)
            total_tokens += len(tokens)
        
        return {
            'avg_time': total_time / len(test_data),
            'avg_tokens': total_tokens / len(test_data),
            'tokens_per_second': total_tokens / total_time
        }
    
    def _create_perturbations(self, prompt: str) -> List[str]:
        """Create perturbations of prompt"""
        perturbations = []
        
        # Case changes
        perturbations.append(prompt.lower())
        perturbations.append(prompt.upper())
        
        # Whitespace changes
        perturbations.append(prompt.replace(' ', '  '))
        
        # Punctuation changes
        perturbations.append(prompt.replace('.', ''))
        
        return perturbations
    
    def _get_prediction(self, prompt: str) -> str:
        """Get model prediction"""
        inputs = self.tokenizer(prompt, return_tensors='pt')
        outputs = self.model.generate(**inputs, max_length=100)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Practical Exercises

### Exercise 1: Prompt Engineering Competition
Design optimal prompts for various tasks:
- Classification
- Generation
- Reasoning
- Translation
Compare performance systematically

### Exercise 2: Implement Advanced Reasoning
Implement and compare:
- Chain-of-Thought
- Tree-of-Thought
- Self-Consistency
- ReAct
Analyze when each method works best

### Exercise 3: Instruction Tuning
Build instruction-tuned model:
- Collect diverse instruction dataset
- Fine-tune model
- Evaluate zero-shot performance
- Compare with base model

### Exercise 4: Analyze ICL Behavior
Conduct thorough analysis:
- Example selection strategies
- Order sensitivity
- Scaling with number of examples
- Task complexity effects

## Research Directions

### Open Questions:
1. **Theoretical Understanding**: Why does ICL work?
2. **Optimal Prompting**: How to systematically find best prompts?
3. **Reasoning Limits**: What can/cannot be achieved through prompting?
4. **Instruction Generalization**: How to improve zero-shot generalization?
5. **Multimodal Prompting**: Extending to vision, audio, etc.

## Key Papers

1. Brown et al. (2020) - GPT-3 and Few-Shot Learning
2. Wei et al. (2022) - Chain-of-Thought Prompting
3. Wang et al. (2023) - Self-Consistency
4. Yao et al. (2023) - Tree of Thought
5. Yao et al. (2023) - ReAct
6. Gao et al. (2021) - PAL
7. Wei et al. (2022) - Emergent Abilities
8. Ouyang et al. (2022) - InstructGPT
9. Mishra et al. (2022) - CrossTask Generalization
10. Zhou et al. (2023) - Automatic Prompt Engineering

## Key Takeaways

- In-context learning enables task adaptation without gradient updates
- Prompt design significantly impacts model performance
- Chain-of-thought reasoning improves complex problem-solving
- Advanced techniques (ToT, ReAct) enable more sophisticated reasoning
- Instruction tuning improves zero-shot task generalization
- Systematic evaluation is crucial for prompt engineering
- Theoretical understanding of ICL is still developing

## Next Steps

In the next module, we'll explore model evaluation and alignment, including RLHF, constitutional AI, and advanced techniques for building safe and helpful AI systems.

