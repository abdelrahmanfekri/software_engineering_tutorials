# NLP Tutorial Module 15: Advanced Reasoning - Tree-of-Thought and Self-Reflection

## Learning Objectives
By the end of this module, you will be able to:
- Understand and implement Tree-of-Thought (ToT) reasoning
- Build self-reflection and self-correction mechanisms
- Apply advanced reasoning techniques to complex problems
- Implement iterative refinement and multi-step reasoning
- Use verification and validation techniques for reasoning
- Build reasoning systems with human feedback integration
- Evaluate and improve reasoning quality

## Introduction to Advanced Reasoning in LLMs

Advanced reasoning techniques enable language models to tackle complex problems through structured thinking, self-reflection, and iterative refinement. These methods move beyond simple prompt-response patterns to sophisticated problem-solving approaches.

### Key Advanced Reasoning Techniques

1. **Tree-of-Thought (ToT)**: Explore multiple reasoning paths
2. **Self-Reflection**: Model evaluates and improves its own reasoning
3. **Iterative Refinement**: Multiple passes of reasoning and correction
4. **Verification**: Cross-checking and validation of reasoning steps
5. **Multi-Agent Reasoning**: Multiple reasoning agents collaborating

## Tree-of-Thought (ToT) Implementation

### Core ToT Architecture

Tree-of-Thought extends Chain-of-Thought by exploring multiple reasoning branches and selecting the best path.

```python
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Optional
import json
import random
from dataclasses import dataclass
from abc import ABC, abstractmethod
import asyncio
from concurrent.futures import ThreadPoolExecutor

@dataclass
class Thought:
    """Represents a single thought in the reasoning tree"""
    content: str
    parent: Optional['Thought'] = None
    children: List['Thought'] = None
    value: float = 0.0
    confidence: float = 0.0
    depth: int = 0
    
    def __post_init__(self):
        if self.children is None:
            self.children = []

class TreeOfThoughtReasoner:
    """Tree-of-Thought reasoning system"""
    
    def __init__(self, llm, evaluator, max_depth=5, branching_factor=3, max_iterations=10):
        self.llm = llm
        self.evaluator = evaluator
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.max_iterations = max_iterations
        
    def generate_thoughts(self, problem: str, parent_thought: Optional[Thought] = None, num_thoughts: int = None) -> List[Thought]:
        """Generate multiple thoughts from a given state"""
        if num_thoughts is None:
            num_thoughts = self.branching_factor
            
        # Build context from problem and parent thought
        context = problem
        if parent_thought:
            context += f"\nPrevious reasoning: {parent_thought.content}"
        
        # Generate multiple thoughts
        thoughts = []
        for i in range(num_thoughts):
            prompt = f"""
            Problem: {context}
            
            Generate a reasoning step that moves closer to solving this problem. 
            Think step by step and provide a clear, logical reasoning step.
            """
            
            response = self.llm.generate(prompt, max_tokens=200, temperature=0.7)
            thought_content = response.strip()
            
            thought = Thought(
                content=thought_content,
                parent=parent_thought,
                depth=(parent_thought.depth + 1) if parent_thought else 0
            )
            thoughts.append(thought)
            
            if parent_thought:
                parent_thought.children.append(thought)
        
        return thoughts
    
    def evaluate_thought(self, thought: Thought, problem: str) -> float:
        """Evaluate the quality of a thought"""
        prompt = f"""
        Problem: {problem}
        
        Reasoning step: {thought.content}
        
        Rate this reasoning step on a scale of 1-10 based on:
        1. Logical correctness
        2. Relevance to the problem
        3. Progress toward solution
        4. Clarity and coherence
        
        Provide only a number between 1 and 10.
        """
        
        response = self.llm.generate(prompt, max_tokens=10, temperature=0.1)
        
        try:
            score = float(response.strip())
            return max(1.0, min(10.0, score)) / 10.0  # Normalize to 0-1
        except:
            return 0.5  # Default score if parsing fails
    
    def is_final_answer(self, thought: Thought, problem: str) -> bool:
        """Check if a thought represents a final answer"""
        prompt = f"""
        Problem: {problem}
        
        Reasoning: {thought.content}
        
        Does this reasoning step contain a complete answer to the problem?
        Answer with "YES" if it's a complete answer, "NO" otherwise.
        """
        
        response = self.llm.generate(prompt, max_tokens=5, temperature=0.1)
        return "YES" in response.upper()
    
    def search_tree(self, problem: str) -> Tuple[Thought, float]:
        """Search the reasoning tree using best-first search"""
        # Initialize root
        root = Thought(content="Starting to think about the problem", depth=0)
        root.value = self.evaluate_thought(root, problem)
        
        # Priority queue for best-first search
        frontier = [root]
        best_solution = None
        best_score = 0.0
        
        iteration = 0
        while frontier and iteration < self.max_iterations:
            iteration += 1
            
            # Sort by value (best thoughts first)
            frontier.sort(key=lambda t: t.value, reverse=True)
            
            # Take top thoughts for expansion
            top_thoughts = frontier[:self.branching_factor]
            new_frontier = []
            
            for thought in top_thoughts:
                # Check if this is a final answer
                if self.is_final_answer(thought, problem):
                    if thought.value > best_score:
                        best_solution = thought
                        best_score = thought.value
                    continue
                
                # Don't expand if at max depth
                if thought.depth >= self.max_depth:
                    continue
                
                # Generate children
                children = self.generate_thoughts(problem, thought)
                
                # Evaluate children
                for child in children:
                    child.value = self.evaluate_thought(child, problem)
                    
                    # Add to frontier if not at max depth
                    if child.depth < self.max_depth:
                        new_frontier.append(child)
            
            # Update frontier
            frontier = new_frontier + [t for t in frontier[self.branching_factor:] if t.depth < self.max_depth]
        
        return best_solution, best_score
    
    def solve_problem(self, problem: str) -> Dict[str, Any]:
        """Solve a problem using Tree-of-Thought reasoning"""
        best_solution, score = self.search_tree(problem)
        
        # Extract reasoning path
        reasoning_path = []
        if best_solution:
            current = best_solution
            while current:
                reasoning_path.insert(0, current.content)
                current = current.parent
        
        return {
            'solution': best_solution.content if best_solution else "No solution found",
            'score': score,
            'reasoning_path': reasoning_path,
            'confidence': score
        }

# Example usage
class MockLLM:
    """Mock LLM for demonstration"""
    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> str:
        # Simple mock responses for demonstration
        if "reasoning step" in prompt.lower():
            return random.choice([
                "Let me break this down into smaller parts.",
                "I need to consider the constraints carefully.",
                "This reminds me of a similar problem I've seen.",
                "Let me think about this systematically."
            ])
        elif "rate this reasoning" in prompt.lower():
            return str(random.randint(6, 9))
        elif "complete answer" in prompt.lower():
            return random.choice(["YES", "NO"])
        else:
            return "This is a mock response."

def demonstrate_tree_of_thought():
    """Demonstrate Tree-of-Thought reasoning"""
    llm = MockLLM()
    evaluator = MockLLM()  # Same mock for simplicity
    
    reasoner = TreeOfThoughtReasoner(llm, evaluator, max_depth=3, branching_factor=2)
    
    problem = "How can we reduce traffic congestion in major cities?"
    
    result = reasoner.solve_problem(problem)
    
    print("Tree-of-Thought Reasoning Result:")
    print(f"Problem: {problem}")
    print(f"Solution: {result['solution']}")
    print(f"Score: {result['score']:.2f}")
    print(f"Reasoning Path:")
    for i, step in enumerate(result['reasoning_path'], 1):
        print(f"  {i}. {step}")
    
    return result

# Run demonstration
tot_result = demonstrate_tree_of_thought()
```

## Self-Reflection and Self-Correction

### Self-Reflection System

```python
class SelfReflectionReasoner:
    """Reasoning system with self-reflection capabilities"""
    
    def __init__(self, llm, reflection_prompts=None):
        self.llm = llm
        self.reflection_prompts = reflection_prompts or self._default_reflection_prompts()
    
    def _default_reflection_prompts(self):
        """Default reflection prompts for different aspects"""
        return {
            'logic': "Is the reasoning logically sound? Are there any logical fallacies?",
            'completeness': "Is the reasoning complete? Are there missing steps or considerations?",
            'accuracy': "Are the facts and assumptions accurate? Are there any errors?",
            'relevance': "Is the reasoning relevant to the problem? Are there tangential points?",
            'clarity': "Is the reasoning clear and well-structured? Could it be improved?"
        }
    
    def reflect_on_reasoning(self, problem: str, reasoning: str, aspect: str = 'comprehensive') -> Dict[str, Any]:
        """Reflect on a piece of reasoning"""
        if aspect == 'comprehensive':
            # Comprehensive reflection
            reflection_results = {}
            for aspect_name, prompt in self.reflection_prompts.items():
                reflection_results[aspect_name] = self._reflect_on_aspect(problem, reasoning, prompt)
            return reflection_results
        else:
            # Specific aspect reflection
            prompt = self.reflection_prompts.get(aspect, self.reflection_prompts['logic'])
            return {aspect: self._reflect_on_aspect(problem, reasoning, prompt)}
    
    def _reflect_on_aspect(self, problem: str, reasoning: str, prompt: str) -> Dict[str, Any]:
        """Reflect on a specific aspect of reasoning"""
        full_prompt = f"""
        Problem: {problem}
        
        Reasoning: {reasoning}
        
        {prompt}
        
        Please provide:
        1. Your assessment (1-10 scale)
        2. Specific issues or strengths
        3. Suggestions for improvement (if any)
        """
        
        response = self.llm.generate(full_prompt, max_tokens=300, temperature=0.3)
        
        # Parse response (simplified)
        lines = response.strip().split('\n')
        assessment = 5  # Default
        issues = []
        suggestions = []
        
        for line in lines:
            if 'assessment' in line.lower() or any(char.isdigit() for char in line[:10]):
                try:
                    assessment = float([char for char in line if char.isdigit() or char == '.'][0])
                except:
                    pass
            elif 'issue' in line.lower() or 'problem' in line.lower():
                issues.append(line.strip())
            elif 'suggest' in line.lower() or 'improve' in line.lower():
                suggestions.append(line.strip())
        
        return {
            'assessment': max(1, min(10, assessment)) / 10.0,
            'issues': issues,
            'suggestions': suggestions,
            'raw_response': response
        }
    
    def iterative_reasoning(self, problem: str, max_iterations: int = 3) -> Dict[str, Any]:
        """Perform iterative reasoning with self-reflection"""
        current_reasoning = "Let me start thinking about this problem."
        reasoning_history = []
        
        for iteration in range(max_iterations):
            # Generate reasoning
            reasoning_prompt = f"""
            Problem: {problem}
            
            Previous reasoning: {current_reasoning}
            
            Provide the next step in your reasoning. If this is your first attempt, start from the beginning.
            Think step by step and be thorough.
            """
            
            new_reasoning = self.llm.generate(reasoning_prompt, max_tokens=400, temperature=0.5)
            
            # Reflect on the reasoning
            reflection = self.reflect_on_reasoning(problem, new_reasoning, 'comprehensive')
            
            # Calculate overall quality
            quality_scores = [reflection[aspect]['assessment'] for aspect in reflection]
            overall_quality = sum(quality_scores) / len(quality_scores)
            
            reasoning_history.append({
                'iteration': iteration + 1,
                'reasoning': new_reasoning,
                'reflection': reflection,
                'quality': overall_quality
            })
            
            # Decide whether to continue or stop
            if overall_quality > 0.8 and iteration > 0:
                break
            
            current_reasoning = new_reasoning
        
        # Select best reasoning
        best_reasoning = max(reasoning_history, key=lambda x: x['quality'])
        
        return {
            'final_reasoning': best_reasoning['reasoning'],
            'final_quality': best_reasoning['quality'],
            'reasoning_history': reasoning_history,
            'iterations_used': len(reasoning_history)
        }

# Example usage
def demonstrate_self_reflection():
    """Demonstrate self-reflection reasoning"""
    llm = MockLLM()
    reasoner = SelfReflectionReasoner(llm)
    
    problem = "What are the main causes of climate change and how can they be addressed?"
    
    result = reasoner.iterative_reasoning(problem, max_iterations=2)
    
    print("Self-Reflection Reasoning Result:")
    print(f"Problem: {problem}")
    print(f"Final Quality Score: {result['final_quality']:.2f}")
    print(f"Final Reasoning: {result['final_reasoning']}")
    print(f"Iterations Used: {result['iterations_used']}")
    
    return result

self_reflection_result = demonstrate_self_reflection()
```

## Multi-Agent Reasoning System

```python
class ReasoningAgent:
    """Individual reasoning agent"""
    
    def __init__(self, llm, role: str, expertise: str):
        self.llm = llm
        self.role = role
        self.expertise = expertise
        self.reasoning_history = []
    
    def reason(self, problem: str, context: str = "") -> str:
        """Generate reasoning from this agent's perspective"""
        prompt = f"""
        You are a {self.role} with expertise in {self.expertise}.
        
        Problem: {problem}
        Context: {context}
        
        Provide your reasoning and analysis from your specialized perspective.
        Be thorough and consider your area of expertise.
        """
        
        reasoning = self.llm.generate(prompt, max_tokens=300, temperature=0.6)
        self.reasoning_history.append(reasoning)
        return reasoning
    
    def critique(self, other_reasoning: str, problem: str) -> Dict[str, Any]:
        """Critique reasoning from another agent"""
        prompt = f"""
        You are a {self.role} with expertise in {self.expertise}.
        
        Problem: {problem}
        Reasoning to critique: {other_reasoning}
        
        Provide your critique focusing on:
        1. Strengths from your perspective
        2. Weaknesses or gaps
        3. Suggestions for improvement
        4. Overall assessment (1-10)
        """
        
        critique = self.llm.generate(prompt, max_tokens=250, temperature=0.4)
        
        # Simple parsing (in practice, use more sophisticated parsing)
        lines = critique.split('\n')
        assessment = 5
        for line in lines:
            if any(char.isdigit() for char in line[:10]):
                try:
                    assessment = float([char for char in line if char.isdigit() or char == '.'][0])
                except:
                    pass
        
        return {
            'critique': critique,
            'assessment': max(1, min(10, assessment)) / 10.0
        }

class MultiAgentReasoningSystem:
    """System with multiple reasoning agents"""
    
    def __init__(self, llm):
        self.llm = llm
        self.agents = []
        self.discussion_history = []
    
    def add_agent(self, role: str, expertise: str):
        """Add a reasoning agent"""
        agent = ReasoningAgent(self.llm, role, expertise)
        self.agents.append(agent)
        return agent
    
    def collaborative_reasoning(self, problem: str, rounds: int = 3) -> Dict[str, Any]:
        """Perform collaborative reasoning with multiple agents"""
        # Initial reasoning from each agent
        initial_reasonings = {}
        for agent in self.agents:
            reasoning = agent.reason(problem)
            initial_reasonings[agent.role] = reasoning
        
        self.discussion_history.append({
            'round': 0,
            'type': 'initial',
            'reasonings': initial_reasonings
        })
        
        current_reasonings = initial_reasonings.copy()
        
        # Collaborative rounds
        for round_num in range(1, rounds + 1):
            round_critiques = {}
            round_updates = {}
            
            # Each agent critiques others
            for agent in self.agents:
                critiques = {}
                for other_role, other_reasoning in current_reasonings.items():
                    if other_role != agent.role:
                        critique = agent.critique(other_reasoning, problem)
                        critiques[other_role] = critique
                
                round_critiques[agent.role] = critiques
            
            # Each agent updates their reasoning based on critiques
            for agent in self.agents:
                context = f"Previous reasoning: {current_reasonings[agent.role]}\n"
                context += "Critiques received:\n"
                for critic_role, critique in round_critiques.items():
                    if agent.role in critique:
                        context += f"- {critic_role}: {critique[agent.role]['critique']}\n"
                
                updated_reasoning = agent.reason(problem, context)
                round_updates[agent.role] = updated_reasoning
            
            current_reasonings = round_updates
            
            self.discussion_history.append({
                'round': round_num,
                'type': 'collaborative',
                'critiques': round_critiques,
                'updated_reasonings': round_updates
            })
        
        # Synthesize final reasoning
        synthesis_prompt = f"""
        Problem: {problem}
        
        Final reasonings from different perspectives:
        """
        for role, reasoning in current_reasonings.items():
            synthesis_prompt += f"\n{role}: {reasoning}"
        
        synthesis_prompt += """
        
        Synthesize these perspectives into a comprehensive solution.
        """
        
        final_synthesis = self.llm.generate(synthesis_prompt, max_tokens=500, temperature=0.5)
        
        return {
            'final_synthesis': final_synthesis,
            'individual_reasonings': current_reasonings,
            'discussion_history': self.discussion_history,
            'rounds_conducted': rounds
        }

# Example usage
def demonstrate_multi_agent_reasoning():
    """Demonstrate multi-agent reasoning"""
    llm = MockLLM()
    system = MultiAgentReasoningSystem(llm)
    
    # Add different types of agents
    system.add_agent("Environmental Scientist", "climate and environmental systems")
    system.add_agent("Economist", "economic policies and market mechanisms")
    system.add_agent("Policy Analyst", "government policies and regulations")
    system.add_agent("Technology Expert", "technological solutions and innovations")
    
    problem = "How can we transition to renewable energy while maintaining economic growth?"
    
    result = system.collaborative_reasoning(problem, rounds=2)
    
    print("Multi-Agent Reasoning Result:")
    print(f"Problem: {problem}")
    print(f"Final Synthesis: {result['final_synthesis']}")
    print(f"Rounds Conducted: {result['rounds_conducted']}")
    
    return result

multi_agent_result = demonstrate_multi_agent_reasoning()
```

## Verification and Validation System

```python
class ReasoningVerifier:
    """System for verifying and validating reasoning"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def verify_factual_claims(self, reasoning: str) -> Dict[str, Any]:
        """Verify factual claims in reasoning"""
        prompt = f"""
        Reasoning: {reasoning}
        
        Extract all factual claims from this reasoning and assess their accuracy.
        For each claim, provide:
        1. The claim
        2. Accuracy assessment (True/False/Uncertain)
        3. Reasoning for the assessment
        
        Format as a structured list.
        """
        
        response = self.llm.generate(prompt, max_tokens=400, temperature=0.2)
        
        # Parse response (simplified)
        claims = []
        lines = response.split('\n')
        current_claim = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith('1.') or 'claim' in line.lower():
                if current_claim:
                    claims.append(current_claim)
                current_claim = {'claim': line}
            elif line.startswith('2.') or 'accuracy' in line.lower():
                current_claim['accuracy'] = line
            elif line.startswith('3.') or 'reasoning' in line.lower():
                current_claim['reasoning'] = line
        
        if current_claim:
            claims.append(current_claim)
        
        return {'claims': claims, 'raw_response': response}
    
    def check_logical_consistency(self, reasoning: str) -> Dict[str, Any]:
        """Check logical consistency of reasoning"""
        prompt = f"""
        Reasoning: {reasoning}
        
        Check this reasoning for logical consistency. Look for:
        1. Contradictions
        2. Logical fallacies
        3. Inconsistent assumptions
        4. Unsupported conclusions
        
        Provide a detailed analysis.
        """
        
        response = self.llm.generate(prompt, max_tokens=300, temperature=0.2)
        
        return {'analysis': response}
    
    def validate_solution_completeness(self, problem: str, reasoning: str) -> Dict[str, Any]:
        """Validate if reasoning addresses the complete problem"""
        prompt = f"""
        Problem: {problem}
        Reasoning: {reasoning}
        
        Assess whether this reasoning completely addresses the problem:
        1. Are all parts of the problem addressed?
        2. Are there missing considerations?
        3. Is the solution comprehensive?
        4. What additional steps might be needed?
        
        Provide a detailed assessment.
        """
        
        response = self.llm.generate(prompt, max_tokens=300, temperature=0.2)
        
        return {'assessment': response}
    
    def comprehensive_verification(self, problem: str, reasoning: str) -> Dict[str, Any]:
        """Perform comprehensive verification of reasoning"""
        factual_verification = self.verify_factual_claims(reasoning)
        logical_check = self.check_logical_consistency(reasoning)
        completeness_check = self.validate_solution_completeness(problem, reasoning)
        
        # Calculate overall verification score
        score_components = []
        
        # Factual accuracy score (simplified)
        if factual_verification['claims']:
            accuracy_scores = []
            for claim in factual_verification['claims']:
                if 'accuracy' in claim:
                    if 'true' in claim['accuracy'].lower():
                        accuracy_scores.append(1.0)
                    elif 'false' in claim['accuracy'].lower():
                        accuracy_scores.append(0.0)
                    else:
                        accuracy_scores.append(0.5)
            factual_score = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0.5
        else:
            factual_score = 0.5
        
        score_components.append(factual_score)
        
        # Logical consistency score (simplified)
        logical_text = logical_check['analysis'].lower()
        if 'consistent' in logical_text and 'no contradictions' in logical_text:
            logical_score = 1.0
        elif 'contradiction' in logical_text or 'fallacy' in logical_text:
            logical_score = 0.3
        else:
            logical_score = 0.7
        
        score_components.append(logical_score)
        
        # Completeness score (simplified)
        completeness_text = completeness_check['assessment'].lower()
        if 'complete' in completeness_text and 'comprehensive' in completeness_text:
            completeness_score = 1.0
        elif 'missing' in completeness_text or 'incomplete' in completeness_text:
            completeness_score = 0.4
        else:
            completeness_score = 0.7
        
        score_components.append(completeness_score)
        
        overall_score = sum(score_components) / len(score_components)
        
        return {
            'factual_verification': factual_verification,
            'logical_consistency': logical_check,
            'completeness_validation': completeness_check,
            'overall_score': overall_score,
            'component_scores': {
                'factual_accuracy': factual_score,
                'logical_consistency': logical_score,
                'completeness': completeness_score
            }
        }

# Example usage
def demonstrate_verification():
    """Demonstrate reasoning verification"""
    llm = MockLLM()
    verifier = ReasoningVerifier(llm)
    
    problem = "What are the main causes of air pollution in urban areas?"
    reasoning = """
    Air pollution in urban areas is primarily caused by:
    1. Vehicle emissions from cars and trucks
    2. Industrial emissions from factories
    3. Construction activities and dust
    4. Energy production from coal plants
    
    These sources release pollutants like PM2.5, NOx, and SO2 into the atmosphere.
    """
    
    verification = verifier.comprehensive_verification(problem, reasoning)
    
    print("Reasoning Verification Result:")
    print(f"Overall Score: {verification['overall_score']:.2f}")
    print(f"Component Scores:")
    for component, score in verification['component_scores'].items():
        print(f"  {component}: {score:.2f}")
    
    return verification

verification_result = demonstrate_verification()
```

## Advanced Reasoning Orchestrator

```python
class AdvancedReasoningOrchestrator:
    """Orchestrates multiple advanced reasoning techniques"""
    
    def __init__(self, llm):
        self.llm = llm
        self.tot_reasoner = TreeOfThoughtReasoner(llm, llm)
        self.self_reflection_reasoner = SelfReflectionReasoner(llm)
        self.multi_agent_system = MultiAgentReasoningSystem(llm)
        self.verifier = ReasoningVerifier(llm)
    
    def comprehensive_reasoning(self, problem: str, method: str = 'adaptive') -> Dict[str, Any]:
        """Perform comprehensive reasoning using multiple methods"""
        
        if method == 'adaptive':
            # Choose method based on problem complexity
            complexity = self._assess_problem_complexity(problem)
            
            if complexity == 'simple':
                method = 'self_reflection'
            elif complexity == 'medium':
                method = 'tree_of_thought'
            else:
                method = 'multi_agent'
        
        # Execute chosen method
        if method == 'tree_of_thought':
            result = self.tot_reasoner.solve_problem(problem)
            reasoning = result['solution']
        elif method == 'self_reflection':
            result = self.self_reflection_reasoner.iterative_reasoning(problem)
            reasoning = result['final_reasoning']
        elif method == 'multi_agent':
            result = self.multi_agent_system.collaborative_reasoning(problem)
            reasoning = result['final_synthesis']
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Verify the reasoning
        verification = self.verifier.comprehensive_verification(problem, reasoning)
        
        # If verification score is low, try alternative method
        if verification['overall_score'] < 0.6:
            alternative_method = self._get_alternative_method(method)
            alt_result = self.comprehensive_reasoning(problem, alternative_method)
            
            # Choose better result
            if alt_result['verification']['overall_score'] > verification['overall_score']:
                return alt_result
        
        return {
            'method_used': method,
            'reasoning': reasoning,
            'verification': verification,
            'original_result': result
        }
    
    def _assess_problem_complexity(self, problem: str) -> str:
        """Assess the complexity of a problem"""
        prompt = f"""
        Problem: {problem}
        
        Assess the complexity of this problem:
        - Simple: Can be solved with basic reasoning
        - Medium: Requires multi-step reasoning or exploration
        - Complex: Requires multiple perspectives or specialized knowledge
        
        Answer with: Simple, Medium, or Complex
        """
        
        response = self.llm.generate(prompt, max_tokens=10, temperature=0.1)
        
        if 'complex' in response.lower():
            return 'complex'
        elif 'medium' in response.lower():
            return 'medium'
        else:
            return 'simple'
    
    def _get_alternative_method(self, current_method: str) -> str:
        """Get an alternative reasoning method"""
        alternatives = {
            'tree_of_thought': 'self_reflection',
            'self_reflection': 'multi_agent',
            'multi_agent': 'tree_of_thought'
        }
        return alternatives.get(current_method, 'self_reflection')

# Example usage
def demonstrate_advanced_reasoning():
    """Demonstrate advanced reasoning orchestrator"""
    llm = MockLLM()
    orchestrator = AdvancedReasoningOrchestrator(llm)
    
    problem = "How can we design a sustainable transportation system for a growing city?"
    
    result = orchestrator.comprehensive_reasoning(problem, method='adaptive')
    
    print("Advanced Reasoning Result:")
    print(f"Problem: {problem}")
    print(f"Method Used: {result['method_used']}")
    print(f"Reasoning: {result['reasoning']}")
    print(f"Verification Score: {result['verification']['overall_score']:.2f}")
    
    return result

advanced_reasoning_result = demonstrate_advanced_reasoning()
```

## Practical Exercises

### Exercise 1: Tree-of-Thought Implementation
Implement a complete Tree-of-Thought system:
- Build the tree structure with proper evaluation
- Implement different search strategies (breadth-first, depth-first, best-first)
- Add pruning mechanisms for efficiency

### Exercise 2: Self-Reflection System
Create a comprehensive self-reflection system:
- Implement multiple reflection dimensions
- Add confidence scoring mechanisms
- Build iterative improvement loops

### Exercise 3: Multi-Agent Reasoning
Design a multi-agent reasoning system:
- Create specialized agents for different domains
- Implement agent communication protocols
- Add consensus mechanisms

## Assessment Questions

1. **What is the main advantage of Tree-of-Thought over Chain-of-Thought?**
   - Explores multiple reasoning paths
   - Better handling of complex problems
   - Can backtrack and revise reasoning
   - All of the above

2. **How does self-reflection improve reasoning quality?**
   - Identifies errors and gaps
   - Provides iterative improvement
   - Increases confidence in solutions
   - All of the above

3. **What role does verification play in advanced reasoning?**
   - Ensures factual accuracy
   - Checks logical consistency
   - Validates solution completeness
   - All of the above

## Key Takeaways

- Advanced reasoning techniques enable more sophisticated problem-solving
- Tree-of-Thought allows exploration of multiple reasoning paths
- Self-reflection provides iterative improvement and error correction
- Multi-agent systems leverage diverse perspectives and expertise
- Verification ensures reasoning quality and reliability
- Adaptive reasoning systems can choose appropriate methods for different problems
- These techniques are crucial for building reliable AI reasoning systems

## Next Steps

This completes our comprehensive NLP tutorial series. You now have the knowledge and tools to:
- Understand and implement traditional NLP models
- Work with modern neural architectures including Transformers
- Build and train large language models
- Apply advanced reasoning techniques
- Create sophisticated NLP applications

Continue practicing with these concepts and stay updated with the rapidly evolving field of NLP and AI reasoning!
