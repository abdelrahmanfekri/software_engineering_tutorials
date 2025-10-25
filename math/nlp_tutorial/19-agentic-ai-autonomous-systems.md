# NLP Tutorial Module 19: Agentic AI and Autonomous Systems

## Learning Objectives
By the end of this module, you will be able to:
- Understand the principles of agentic AI systems
- Implement autonomous agents using LLMs
- Build tool-using agents with function calling
- Design multi-agent systems and orchestration
- Implement planning and reasoning capabilities
- Create self-improving and self-correcting agents
- Apply agent frameworks (LangChain, AutoGPT, BabyAGI)

## Introduction to Agentic AI

Agentic AI refers to autonomous systems that can perceive their environment, make decisions, plan actions, use tools, and work towards achieving specified goals with minimal human intervention.

### Key Characteristics of Agents

1. **Autonomy**: Operate independently without constant human guidance
2. **Reactivity**: Respond to environmental changes
3. **Proactivity**: Take initiative to achieve goals
4. **Social Ability**: Interact with other agents and humans
5. **Goal-Oriented**: Work towards specific objectives
6. **Learning**: Improve performance over time

### Agent Architecture Components

```
┌─────────────────────────────────────┐
│         Agent Architecture          │
├─────────────────────────────────────┤
│  Perception  → Reasoning → Action   │
│      ↓            ↓           ↓     │
│   Memory ←  Planning  →  Tools      │
└─────────────────────────────────────┘
```

## Basic Agent Implementation

### Simple ReAct Agent (Reasoning + Acting)

```python
import torch
import numpy as np
from typing import List, Dict, Callable, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
import re

class AgentAction(Enum):
    """Types of agent actions"""
    THINK = "think"
    OBSERVE = "observe"
    ACT = "act"
    FINISH = "finish"

@dataclass
class AgentStep:
    """Represents a single step in agent execution"""
    action_type: AgentAction
    content: str
    observation: Optional[str] = None
    success: bool = True

class Tool:
    """Base class for agent tools"""
    def __init__(self, name: str, description: str, function: Callable):
        self.name = name
        self.description = description
        self.function = function
    
    def execute(self, *args, **kwargs) -> str:
        """Execute the tool"""
        try:
            result = self.function(*args, **kwargs)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"

class SimpleReActAgent:
    """
    ReAct Agent: Reasoning + Acting
    Interleaves reasoning and action steps to solve tasks
    """
    def __init__(self, llm_model, tools: List[Tool], max_iterations: int = 10):
        """
        Initialize ReAct agent
        
        Args:
            llm_model: Language model for reasoning
            tools: List of available tools
            max_iterations: Maximum reasoning-action cycles
        """
        self.llm = llm_model
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = max_iterations
        self.memory = []  # Store conversation history
        self.steps = []   # Store execution steps
    
    def get_tool_descriptions(self) -> str:
        """Format tool descriptions for prompt"""
        descriptions = []
        for name, tool in self.tools.items():
            descriptions.append(f"- {name}: {tool.description}")
        return "\n".join(descriptions)
    
    def parse_action(self, response: str) -> tuple:
        """Parse LLM response to extract action and input"""
        # Look for patterns like:
        # Action: tool_name
        # Action Input: input_text
        
        action_match = re.search(r"Action:\s*(.+?)(?:\n|$)", response)
        input_match = re.search(r"Action Input:\s*(.+?)(?:\n|$)", response, re.DOTALL)
        
        if action_match:
            action = action_match.group(1).strip()
            action_input = input_match.group(1).strip() if input_match else ""
            return action, action_input
        
        return None, None
    
    def run(self, task: str) -> Dict[str, Any]:
        """
        Execute agent on task
        
        Args:
            task: Task description
            
        Returns:
            Dictionary with final answer and execution trace
        """
        self.steps = []
        self.memory = []
        
        # Initial prompt
        system_prompt = f"""You are an autonomous agent that can use tools to accomplish tasks.
        
Available Tools:
{self.get_tool_descriptions()}

Your response format:
Thought: [your reasoning about what to do next]
Action: [tool name or "Finish"]
Action Input: [input to the tool]

When you have the final answer, use:
Action: Finish
Action Input: [your final answer]

Task: {task}
"""
        
        self.memory.append({"role": "system", "content": system_prompt})
        
        for iteration in range(self.max_iterations):
            # Get agent's reasoning and action
            response = self._query_llm(self.memory)
            
            # Parse action
            action, action_input = self.parse_action(response)
            
            if action == "Finish":
                # Agent has finished
                self.steps.append(AgentStep(
                    action_type=AgentAction.FINISH,
                    content=response,
                    observation=action_input
                ))
                return {
                    "success": True,
                    "answer": action_input,
                    "steps": self.steps,
                    "iterations": iteration + 1
                }
            
            elif action in self.tools:
                # Execute tool
                tool = self.tools[action]
                try:
                    observation = tool.execute(action_input)
                    success = True
                except Exception as e:
                    observation = f"Error executing tool: {str(e)}"
                    success = False
                
                # Record step
                self.steps.append(AgentStep(
                    action_type=AgentAction.ACT,
                    content=response,
                    observation=observation,
                    success=success
                ))
                
                # Update memory
                self.memory.append({"role": "assistant", "content": response})
                self.memory.append({"role": "user", "content": f"Observation: {observation}"})
            
            else:
                # Invalid action
                error_msg = f"Invalid action: {action}. Available tools: {list(self.tools.keys())}"
                self.memory.append({"role": "assistant", "content": response})
                self.memory.append({"role": "user", "content": error_msg})
                
                self.steps.append(AgentStep(
                    action_type=AgentAction.ACT,
                    content=response,
                    observation=error_msg,
                    success=False
                ))
        
        # Max iterations reached
        return {
            "success": False,
            "answer": "Maximum iterations reached without solution",
            "steps": self.steps,
            "iterations": self.max_iterations
        }
    
    def _query_llm(self, messages: List[Dict]) -> str:
        """Query language model (placeholder - implement with actual LLM)"""
        # This is a simplified version - in practice, use OpenAI API, HuggingFace, etc.
        # For demonstration, return mock response
        return """Thought: I need to search for information to answer this question.
Action: search
Action Input: relevant search query"""

# Example tools
def calculator(expression: str) -> float:
    """Evaluate mathematical expressions"""
    try:
        # Safe eval for basic math
        allowed_chars = set('0123456789+-*/()., ')
        if not all(c in allowed_chars for c in expression):
            raise ValueError("Invalid characters in expression")
        result = eval(expression)
        return result
    except Exception as e:
        raise ValueError(f"Cannot evaluate: {str(e)}")

def search(query: str) -> str:
    """Mock search function"""
    # In practice, integrate with real search API
    knowledge_base = {
        "python": "Python is a high-level programming language.",
        "machine learning": "Machine learning is a subset of AI.",
        "transformers": "Transformers are neural network architectures."
    }
    
    query_lower = query.lower()
    for key, value in knowledge_base.items():
        if key in query_lower:
            return value
    
    return "No results found"

def wikipedia_lookup(topic: str) -> str:
    """Mock Wikipedia lookup"""
    return f"Wikipedia information about {topic}: [mock content]"

# Create tools
tools = [
    Tool("calculator", "Perform mathematical calculations", calculator),
    Tool("search", "Search for information", search),
    Tool("wikipedia", "Look up topics on Wikipedia", wikipedia_lookup)
]

# Create agent
# agent = SimpleReActAgent(llm_model=None, tools=tools, max_iterations=10)
# result = agent.run("What is 25 * 4 + 10?")
# print(result)
```

## Advanced Agent with Memory and Planning

### Agent with Short-term and Long-term Memory

```python
from collections import deque
import chromadb
from datetime import datetime

class MemorySystem:
    """Advanced memory system for agents"""
    
    def __init__(self, short_term_capacity: int = 10):
        """
        Initialize memory system
        
        Args:
            short_term_capacity: Maximum items in short-term memory
        """
        # Short-term memory (working memory)
        self.short_term = deque(maxlen=short_term_capacity)
        
        # Long-term memory (vector database)
        self.client = chromadb.Client()
        self.long_term = self.client.create_collection("agent_memory")
        
        # Episodic memory (important events/experiences)
        self.episodic = []
        
        # Semantic memory (learned facts and knowledge)
        self.semantic = {}
    
    def add_to_short_term(self, item: Dict):
        """Add item to short-term memory"""
        item['timestamp'] = datetime.now().isoformat()
        self.short_term.append(item)
    
    def add_to_long_term(self, item: Dict, embedding: np.ndarray):
        """Add item to long-term memory with embedding"""
        self.long_term.add(
            embeddings=[embedding.tolist()],
            documents=[json.dumps(item)],
            ids=[f"mem_{datetime.now().timestamp()}"]
        )
    
    def add_episodic(self, event: Dict):
        """Add important event to episodic memory"""
        event['timestamp'] = datetime.now().isoformat()
        self.episodic.append(event)
    
    def add_semantic(self, key: str, value: Any):
        """Add learned fact to semantic memory"""
        self.semantic[key] = value
    
    def recall_similar(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Retrieve similar memories from long-term storage"""
        results = self.long_term.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k
        )
        
        memories = []
        for doc in results['documents'][0]:
            memories.append(json.loads(doc))
        
        return memories
    
    def get_recent_short_term(self, n: int = 5) -> List[Dict]:
        """Get n most recent items from short-term memory"""
        return list(self.short_term)[-n:]
    
    def consolidate(self, threshold: int = 5):
        """
        Consolidate important short-term memories to long-term
        (Simplified version)
        """
        # In practice, use importance scoring
        if len(self.short_term) >= threshold:
            # Move important items to long-term
            pass

class PlanningAgent:
    """Agent with planning and reasoning capabilities"""
    
    def __init__(self, llm_model, tools: List[Tool]):
        """Initialize planning agent"""
        self.llm = llm_model
        self.tools = {tool.name: tool for tool in tools}
        self.memory = MemorySystem()
        self.current_plan = []
        self.current_goal = None
    
    def create_plan(self, goal: str) -> List[Dict]:
        """
        Create high-level plan to achieve goal
        
        Args:
            goal: Goal description
            
        Returns:
            List of planned steps
        """
        # Prompt LLM to create plan
        planning_prompt = f"""Create a step-by-step plan to achieve the following goal:
Goal: {goal}

Available tools: {list(self.tools.keys())}

Provide a numbered list of steps, where each step describes:
1. What to do
2. Which tool to use (if applicable)
3. Expected outcome

Plan:"""
        
        # Get plan from LLM (simplified)
        plan_text = self._query_llm(planning_prompt)
        
        # Parse plan into structured format
        plan = self._parse_plan(plan_text)
        
        self.current_plan = plan
        self.current_goal = goal
        
        return plan
    
    def _parse_plan(self, plan_text: str) -> List[Dict]:
        """Parse plan text into structured steps"""
        steps = []
        lines = plan_text.split('\n')
        
        for line in lines:
            if line.strip() and line[0].isdigit():
                # Extract step information
                step = {
                    'description': line.strip(),
                    'status': 'pending',
                    'result': None
                }
                steps.append(step)
        
        return steps
    
    def execute_plan(self) -> Dict:
        """Execute the current plan"""
        if not self.current_plan:
            return {"success": False, "error": "No plan to execute"}
        
        results = []
        
        for i, step in enumerate(self.current_plan):
            print(f"\nExecuting step {i+1}: {step['description']}")
            
            # Update step status
            step['status'] = 'in_progress'
            
            # Execute step
            result = self._execute_step(step)
            
            # Update step
            step['status'] = 'completed' if result['success'] else 'failed'
            step['result'] = result
            
            # Store in memory
            self.memory.add_to_short_term({
                'type': 'plan_step',
                'step_number': i + 1,
                'description': step['description'],
                'result': result
            })
            
            results.append(result)
            
            # If step failed, try to adapt
            if not result['success']:
                adapted = self._adapt_plan(i, result['error'])
                if not adapted:
                    return {
                        "success": False,
                        "error": f"Failed at step {i+1}",
                        "completed_steps": results
                    }
        
        return {
            "success": True,
            "goal": self.current_goal,
            "completed_steps": results
        }
    
    def _execute_step(self, step: Dict) -> Dict:
        """Execute a single plan step"""
        # Determine which tool to use
        # In practice, use LLM to decide
        
        description = step['description'].lower()
        
        for tool_name, tool in self.tools.items():
            if tool_name in description:
                try:
                    # Extract input from description
                    # Simplified - in practice, use LLM
                    result = tool.execute(description)
                    return {
                        "success": True,
                        "tool": tool_name,
                        "output": result
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e)
                    }
        
        # No specific tool - use LLM reasoning
        return {
            "success": True,
            "tool": "reasoning",
            "output": "Step completed through reasoning"
        }
    
    def _adapt_plan(self, failed_step_idx: int, error: str) -> bool:
        """Adapt plan when a step fails"""
        print(f"Adapting plan after failure at step {failed_step_idx + 1}")
        
        # Ask LLM to revise plan
        adaptation_prompt = f"""The plan failed at step {failed_step_idx + 1}.
Error: {error}

Original plan:
{self._format_plan()}

Suggest an alternative approach for this step and adjust subsequent steps if needed.
"""
        
        # Get adapted plan (simplified)
        # In practice, integrate with actual LLM
        
        # For now, return False (couldn't adapt)
        return False
    
    def _format_plan(self) -> str:
        """Format current plan as text"""
        lines = []
        for i, step in enumerate(self.current_plan, 1):
            lines.append(f"{i}. {step['description']} [{step['status']}]")
        return "\n".join(lines)
    
    def _query_llm(self, prompt: str) -> str:
        """Query language model"""
        # Placeholder - implement with actual LLM
        return "1. Search for information\n2. Analyze results\n3. Formulate answer"

# Example usage
planning_agent = PlanningAgent(llm_model=None, tools=tools)
plan = planning_agent.create_plan("Find information about Python and calculate 5 * 10")
print("Plan created:")
for i, step in enumerate(plan, 1):
    print(f"{i}. {step['description']}")
```

## Multi-Agent Systems

### Multi-Agent Collaboration Framework

```python
class Agent:
    """Base agent class"""
    
    def __init__(self, name: str, role: str, capabilities: List[str]):
        self.name = name
        self.role = role
        self.capabilities = capabilities
        self.inbox = deque()
        self.outbox = deque()
    
    def receive_message(self, message: Dict):
        """Receive message from another agent"""
        self.inbox.append(message)
    
    def send_message(self, recipient: str, content: Dict):
        """Send message to another agent"""
        self.outbox.append({
            'from': self.name,
            'to': recipient,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
    
    def process_task(self, task: Dict) -> Dict:
        """Process assigned task"""
        # Override in subclasses
        raise NotImplementedError

class ResearchAgent(Agent):
    """Agent specialized in research and information gathering"""
    
    def __init__(self, name: str):
        super().__init__(name, "researcher", ["search", "analyze", "summarize"])
        self.knowledge_base = {}
    
    def process_task(self, task: Dict) -> Dict:
        """Research a topic"""
        topic = task.get('topic', '')
        
        # Simulate research
        research_result = {
            'topic': topic,
            'findings': f"Research findings about {topic}",
            'sources': ['source1', 'source2'],
            'confidence': 0.85
        }
        
        # Store in knowledge base
        self.knowledge_base[topic] = research_result
        
        return {
            'success': True,
            'result': research_result,
            'agent': self.name
        }

class AnalystAgent(Agent):
    """Agent specialized in analysis and reasoning"""
    
    def __init__(self, name: str):
        super().__init__(name, "analyst", ["analyze", "reason", "evaluate"])
    
    def process_task(self, task: Dict) -> Dict:
        """Analyze information"""
        data = task.get('data', {})
        
        # Simulate analysis
        analysis = {
            'summary': f"Analysis of provided data",
            'insights': ["Insight 1", "Insight 2"],
            'recommendations': ["Recommendation 1"]
        }
        
        return {
            'success': True,
            'result': analysis,
            'agent': self.name
        }

class ExecutorAgent(Agent):
    """Agent specialized in task execution"""
    
    def __init__(self, name: str):
        super().__init__(name, "executor", ["execute", "implement", "validate"])
        self.tools = {}
    
    def process_task(self, task: Dict) -> Dict:
        """Execute action"""
        action = task.get('action', '')
        params = task.get('parameters', {})
        
        # Simulate execution
        result = {
            'action': action,
            'status': 'completed',
            'output': f"Executed {action} with params {params}"
        }
        
        return {
            'success': True,
            'result': result,
            'agent': self.name
        }

class MultiAgentOrchestrator:
    """Orchestrates multiple agents to accomplish complex tasks"""
    
    def __init__(self):
        self.agents = {}
        self.task_queue = deque()
        self.results = []
    
    def register_agent(self, agent: Agent):
        """Register an agent with the orchestrator"""
        self.agents[agent.name] = agent
        print(f"Registered agent: {agent.name} ({agent.role})")
    
    def assign_task(self, task: Dict) -> str:
        """Assign task to most suitable agent"""
        task_type = task.get('type', '')
        
        # Find agent with matching capabilities
        for agent in self.agents.values():
            if any(cap in task_type for cap in agent.capabilities):
                return agent.name
        
        # Default to first agent
        return list(self.agents.keys())[0] if self.agents else None
    
    def execute_workflow(self, workflow: List[Dict]) -> List[Dict]:
        """Execute a workflow of tasks across multiple agents"""
        results = []
        
        for task in workflow:
            # Assign task to appropriate agent
            agent_name = task.get('assigned_to') or self.assign_task(task)
            
            if agent_name not in self.agents:
                results.append({
                    'success': False,
                    'error': f'Agent {agent_name} not found'
                })
                continue
            
            agent = self.agents[agent_name]
            
            print(f"\nAssigning task to {agent_name}: {task.get('description', 'No description')}")
            
            # Execute task
            result = agent.process_task(task)
            results.append(result)
            
            # Handle inter-agent communication
            if 'notify' in task:
                for recipient_name in task['notify']:
                    if recipient_name in self.agents:
                        agent.send_message(recipient_name, {
                            'type': 'task_result',
                            'result': result
                        })
                        self.agents[recipient_name].receive_message(
                            agent.outbox[-1]
                        )
        
        return results
    
    def collaborative_solve(self, problem: str) -> Dict:
        """Have multiple agents collaborate to solve a problem"""
        # Create workflow
        workflow = [
            {
                'type': 'search',
                'description': f'Research: {problem}',
                'topic': problem,
                'assigned_to': 'researcher_1',
                'notify': ['analyst_1']
            },
            {
                'type': 'analyze',
                'description': 'Analyze research findings',
                'assigned_to': 'analyst_1',
                'notify': ['executor_1']
            },
            {
                'type': 'execute',
                'description': 'Generate final solution',
                'assigned_to': 'executor_1',
                'action': 'synthesize_solution'
            }
        ]
        
        # Execute workflow
        results = self.execute_workflow(workflow)
        
        return {
            'problem': problem,
            'workflow_results': results,
            'success': all(r.get('success', False) for r in results)
        }

# Create multi-agent system
orchestrator = MultiAgentOrchestrator()

# Register agents
researcher = ResearchAgent("researcher_1")
analyst = AnalystAgent("analyst_1")
executor = ExecutorAgent("executor_1")

orchestrator.register_agent(researcher)
orchestrator.register_agent(analyst)
orchestrator.register_agent(executor)

# Solve problem collaboratively
result = orchestrator.collaborative_solve("How to build a scalable AI system")
print("\n" + "="*50)
print("Collaborative Solution:")
print(json.dumps(result, indent=2))
```

## Self-Improving Agent with Reflection

### Agent with Self-Reflection and Learning

```python
class SelfImprovingAgent:
    """Agent that learns from experience and improves over time"""
    
    def __init__(self, llm_model, tools: List[Tool]):
        self.llm = llm_model
        self.tools = {tool.name: tool for tool in tools}
        self.memory = MemorySystem()
        self.performance_history = []
        self.learned_strategies = {}
    
    def execute_with_reflection(self, task: str, max_attempts: int = 3) -> Dict:
        """Execute task with self-reflection and retry capability"""
        
        for attempt in range(max_attempts):
            print(f"\n{'='*50}")
            print(f"Attempt {attempt + 1} of {max_attempts}")
            print(f"{'='*50}")
            
            # Execute task
            result = self._execute_task(task)
            
            # Reflect on performance
            reflection = self._reflect_on_performance(task, result, attempt)
            
            # Store experience
            self.memory.add_episodic({
                'task': task,
                'attempt': attempt + 1,
                'result': result,
                'reflection': reflection
            })
            
            # Check if successful
            if result.get('success', False):
                # Learn from success
                self._learn_from_success(task, result, reflection)
                return result
            
            # If failed, adapt strategy
            if attempt < max_attempts - 1:
                adaptation = self._adapt_strategy(task, result, reflection)
                print(f"\nAdapting strategy: {adaptation}")
        
        # All attempts failed
        self._learn_from_failure(task, result, reflection)
        return {
            'success': False,
            'attempts': max_attempts,
            'final_result': result
        }
    
    def _execute_task(self, task: str) -> Dict:
        """Execute task using current strategy"""
        # Check if we've learned a strategy for similar tasks
        strategy = self._retrieve_relevant_strategy(task)
        
        if strategy:
            print(f"Using learned strategy: {strategy['name']}")
        
        # Execute (simplified)
        steps = []
        success = np.random.rand() > 0.3  # Simulate success/failure
        
        return {
            'success': success,
            'steps': steps,
            'strategy_used': strategy['name'] if strategy else 'default'
        }
    
    def _reflect_on_performance(self, task: str, result: Dict, attempt: int) -> Dict:
        """Reflect on task execution and performance"""
        reflection = {
            'task': task,
            'success': result.get('success', False),
            'attempt': attempt + 1,
            'strengths': [],
            'weaknesses': [],
            'improvements': []
        }
        
        # Analyze what went well and what didn't
        if result.get('success'):
            reflection['strengths'].append("Task completed successfully")
            # Analyze efficient steps
        else:
            reflection['weaknesses'].append("Task execution failed")
            # Analyze failure points
        
        # Generate improvement suggestions
        # In practice, use LLM for deep reflection
        reflection['improvements'] = self._generate_improvements(task, result)
        
        print(f"\nReflection:")
        print(f"  Success: {reflection['success']}")
        print(f"  Strengths: {reflection['strengths']}")
        print(f"  Weaknesses: {reflection['weaknesses']}")
        print(f"  Improvements: {reflection['improvements']}")
        
        return reflection
    
    def _generate_improvements(self, task: str, result: Dict) -> List[str]:
        """Generate improvement suggestions"""
        improvements = []
        
        if not result.get('success'):
            improvements.append("Break task into smaller subtasks")
            improvements.append("Verify intermediate results")
            improvements.append("Use different tool combination")
        
        return improvements
    
    def _learn_from_success(self, task: str, result: Dict, reflection: Dict):
        """Learn successful strategy"""
        strategy_name = f"strategy_{len(self.learned_strategies) + 1}"
        
        strategy = {
            'name': strategy_name,
            'task_pattern': self._extract_task_pattern(task),
            'approach': result.get('steps', []),
            'success_rate': 1.0,
            'usage_count': 1
        }
        
        self.learned_strategies[strategy_name] = strategy
        self.memory.add_semantic(strategy_name, strategy)
        
        print(f"\nLearned new strategy: {strategy_name}")
    
    def _learn_from_failure(self, task: str, result: Dict, reflection: Dict):
        """Learn from failure to avoid similar mistakes"""
        failure_pattern = {
            'task_pattern': self._extract_task_pattern(task),
            'failure_mode': result.get('error', 'Unknown'),
            'lessons': reflection.get('improvements', [])
        }
        
        self.memory.add_semantic(f"failure_{len(self.memory.semantic)}", failure_pattern)
        print(f"\nRecorded failure pattern for future reference")
    
    def _adapt_strategy(self, task: str, result: Dict, reflection: Dict) -> str:
        """Adapt strategy based on reflection"""
        improvements = reflection.get('improvements', [])
        
        if improvements:
            return improvements[0]  # Try first improvement
        
        return "Try alternative approach"
    
    def _retrieve_relevant_strategy(self, task: str) -> Optional[Dict]:
        """Retrieve learned strategy relevant to current task"""
        task_pattern = self._extract_task_pattern(task)
        
        # Find matching strategy
        for strategy in self.learned_strategies.values():
            if strategy['task_pattern'] == task_pattern:
                return strategy
        
        return None
    
    def _extract_task_pattern(self, task: str) -> str:
        """Extract general pattern from task"""
        # Simplified - in practice, use NLP to extract key patterns
        keywords = ['calculate', 'search', 'analyze', 'summarize', 'generate']
        
        task_lower = task.lower()
        for keyword in keywords:
            if keyword in task_lower:
                return keyword
        
        return 'general'
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics and learning progress"""
        total_tasks = len(self.memory.episodic)
        successful = sum(1 for e in self.memory.episodic if e.get('result', {}).get('success', False))
        
        return {
            'total_tasks': total_tasks,
            'successful_tasks': successful,
            'success_rate': successful / total_tasks if total_tasks > 0 else 0,
            'learned_strategies': len(self.learned_strategies),
            'memory_items': len(self.memory.semantic)
        }

# Example usage
improving_agent = SelfImprovingAgent(llm_model=None, tools=tools)

# Execute tasks with learning
tasks = [
    "Calculate the sum of 15 and 27",
    "Search for information about neural networks",
    "Calculate the product of 8 and 9"
]

for task in tasks:
    result = improving_agent.execute_with_reflection(task, max_attempts=3)
    print(f"\nFinal result: {result['success']}")

# Check performance
metrics = improving_agent.get_performance_metrics()
print("\n" + "="*50)
print("Performance Metrics:")
print(json.dumps(metrics, indent=2))
```

## LangChain Integration

### Building Agents with LangChain

```python
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI

class LangChainAgentSystem:
    """Agent system using LangChain framework"""
    
    def __init__(self, api_key: str = None):
        """Initialize LangChain agent system"""
        # Initialize LLM
        self.llm = OpenAI(temperature=0, openai_api_key=api_key) if api_key else None
        
        # Define tools
        self.tools = self._create_tools()
        
        # Create memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    
    def _create_tools(self) -> List[Tool]:
        """Create LangChain tools"""
        tools = [
            Tool(
                name="Calculator",
                func=self._calculator,
                description="Useful for mathematical calculations. Input should be a math expression."
            ),
            Tool(
                name="Search",
                func=self._search,
                description="Useful for finding information. Input should be a search query."
            ),
            Tool(
                name="DataAnalyzer",
                func=self._analyze_data,
                description="Analyze data and provide insights. Input should be data description."
            )
        ]
        return tools
    
    def _calculator(self, expression: str) -> str:
        """Calculator tool"""
        try:
            result = eval(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _search(self, query: str) -> str:
        """Search tool (mock)"""
        return f"Search results for '{query}': [Mock search results]"
    
    def _analyze_data(self, data_description: str) -> str:
        """Data analysis tool (mock)"""
        return f"Analysis of '{data_description}': [Mock analysis]"
    
    def create_agent(self) -> AgentExecutor:
        """Create LangChain agent"""
        if not self.llm:
            raise ValueError("LLM not initialized. Provide API key.")
        
        # Create agent
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self._get_prompt_template()
        )
        
        # Create executor
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True
        )
        
        return agent_executor
    
    def _get_prompt_template(self) -> PromptTemplate:
        """Get prompt template for agent"""
        template = """You are a helpful AI assistant that can use tools to accomplish tasks.

You have access to the following tools:
{tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}
"""
        
        return PromptTemplate(
            input_variables=["input", "tools", "tool_names", "agent_scratchpad"],
            template=template
        )

# Example usage (requires OpenAI API key)
# langchain_system = LangChainAgentSystem(api_key="your-api-key")
# agent = langchain_system.create_agent()
# result = agent.invoke("What is 25 * 4 + 10?")
# print(result)
```

## Practical Exercises

### Exercise 1: Build a Research Agent
Create an autonomous research agent that can:
- Search multiple sources
- Synthesize information
- Generate reports
- Cite sources properly

### Exercise 2: Multi-Agent Debate System
Implement a multi-agent system where agents:
- Take different perspectives on a topic
- Debate and argue
- Reach consensus or identify disagreements
- Learn from discussions

### Exercise 3: Self-Improving Game Agent
Build an agent that:
- Plays a game (e.g., chess, tic-tac-toe)
- Learns from wins and losses
- Adapts strategy over time
- Explains its reasoning

## Assessment Questions

1. **What makes an agent "agentic"?**
   - Autonomy in decision-making
   - Goal-directed behavior
   - Ability to use tools
   - Learning from experience

2. **How do ReAct agents differ from traditional agents?**
   - Interleave reasoning and acting
   - Explicit thought processes
   - More interpretable behavior
   - Better handling of complex tasks

3. **What are the challenges in multi-agent systems?**
   - Coordination and communication
   - Conflict resolution
   - Load balancing
   - Maintaining consistency

## Key Takeaways

- Agentic AI systems can autonomously accomplish complex tasks
- ReAct pattern combines reasoning and action for better performance
- Memory systems enable agents to learn from experience
- Planning capabilities allow agents to break down complex problems
- Multi-agent systems can solve problems beyond single agent capabilities
- Self-reflection enables continuous improvement
- Tool use expands agent capabilities dramatically
- Frameworks like LangChain simplify agent development

## Next Steps

In the next module, we'll explore Conformal Prediction for uncertainty quantification in NLP, enabling agents to know when they don't know and make more reliable predictions.

