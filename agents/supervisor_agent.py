"""
Supervisor Agent that coordinates and manages the workflow between specialized agents.
"""

from typing import Dict, Any, Optional, List, Tuple
from langchain.schema import HumanMessage, SystemMessage
from agents.base_agent import BaseAgent
from agents.research_agent import ResearchAgent
from agents.creative_agent import CreativeAgent
from agents.execution_agent import ExecutionAgent
import uuid
from datetime import datetime


class SupervisorAgent(BaseAgent):
    """Supervisor agent that coordinates and manages the workflow between specialized agents."""
    
    def __init__(self, agent_id: str = "supervisor_agent", agent_name: str = "Supervisor Agent",
                 model_name: str = "gpt-4o-mini", temperature: float = 0.5,
                 short_term_memory=None, long_term_memory=None):
        """
        Initialize the Supervisor Agent.
        
        Args:
            agent_id: Unique identifier for the agent
            agent_name: Human-readable name for the agent
            model_name: OpenAI model to use (GPT-4 for better reasoning)
            temperature: Temperature for model responses
            short_term_memory: Short-term memory instance
            long_term_memory: Long-term memory instance
        """
        super().__init__(
            agent_id=agent_id,
            agent_name=agent_name,
            model_name=model_name,
            temperature=temperature,
            short_term_memory=short_term_memory,
            long_term_memory=long_term_memory
        )
        
        # Initialize specialized agents
        self.research_agent = ResearchAgent(
            short_term_memory=short_term_memory,
            long_term_memory=long_term_memory
        )
        self.creative_agent = CreativeAgent(
            short_term_memory=short_term_memory,
            long_term_memory=long_term_memory
        )
        self.execution_agent = ExecutionAgent(
            short_term_memory=short_term_memory,
            long_term_memory=long_term_memory
        )
        
        # Agent registry
        self.agents = {
            "research": self.research_agent,
            "creative": self.creative_agent,
            "execution": self.execution_agent
        }
        
        # Workflow state
        self.current_workflow = None
        self.workflow_history = []
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the Supervisor Agent."""
        return """You are a Supervisor Agent that coordinates and manages a team of specialized agents. 
Your capabilities include:

1. **Task Analysis**: Analyze incoming tasks and determine the best approach
2. **Agent Selection**: Choose the most appropriate agent(s) for each task
3. **Workflow Orchestration**: Coordinate multiple agents in complex workflows
4. **Quality Control**: Review and validate agent outputs
5. **Conflict Resolution**: Handle disagreements or conflicts between agents
6. **Performance Monitoring**: Track agent performance and optimize workflows
7. **Resource Management**: Allocate tasks efficiently across available agents
8. **Communication Management**: Facilitate communication between agents and users

Available Agents:
- **Research Agent**: Specialized in gathering, analyzing, and synthesizing information
- **Creative Agent**: Specialized in generating creative content, ideas, and innovative solutions
- **Execution Agent**: Specialized in implementing and executing tasks with practical solutions

When processing tasks:
- Analyze the task requirements thoroughly
- Determine which agent(s) are best suited for the task
- Create a logical workflow if multiple agents are needed
- Provide clear instructions to each agent
- Synthesize and validate the results
- Ensure the final output meets the user's needs

Your responses should be strategic, well-organized, and demonstrate effective coordination."""
    
    def can_handle_task(self, task_description: str) -> bool:
        """
        Determine if this agent can handle a specific task.
        
        Args:
            task_description: Description of the task
            
        Returns:
            bool: True if the agent can handle the task (supervisor can handle any task)
        """
        # Supervisor can handle any task by delegating to appropriate agents
        return True
    
    def process_task(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a task by coordinating the appropriate specialized agents.
        
        Args:
            task_description: Description of the task to process
            context: Additional context for the task
            
        Returns:
            Dict containing the coordinated result and metadata
        """
        try:
            # Update agent state
            self.update_agent_state(current_task=task_description)
            
            # Generate session ID for this workflow
            session_id = context.get("session_id", str(uuid.uuid4())) if context else str(uuid.uuid4())
            
            # Analyze task and determine workflow
            workflow_plan = self._analyze_task_and_plan_workflow(task_description, context)
            
            # Execute the workflow
            workflow_results = self._execute_workflow(workflow_plan, session_id, context)
            
            # Synthesize final result
            final_result = self._synthesize_results(workflow_results, task_description, context)
            
            # Store workflow in memory
            self._store_workflow_in_memory(workflow_plan, workflow_results, final_result, session_id)
            
            # Create supervisor result
            result = {
                "agent_id": self.agent_id,
                "agent_name": self.agent_name,
                "task_description": task_description,
                "workflow_plan": workflow_plan,
                "agent_results": workflow_results,
                "final_response": final_result,
                "coordination_score": self._calculate_coordination_score(workflow_results),
                "efficiency_score": self._calculate_efficiency_score(workflow_plan, workflow_results),
                "success_rate": self._calculate_overall_success_rate(workflow_results),
                "agents_used": list(workflow_results.keys()),
                "timestamp": self._get_current_timestamp()
            }
            
            # Add to task history
            self.add_task_to_history(task_description, result)
            
            # Reset current task
            self.update_agent_state(current_task=None)
            
            return result
            
        except Exception as e:
            error_result = {
                "agent_id": self.agent_id,
                "agent_name": self.agent_name,
                "task_description": task_description,
                "error": str(e),
                "success_rate": 0.0,
                "timestamp": self._get_current_timestamp()
            }
            self.add_task_to_history(task_description, error_result)
            return error_result
    
    def _analyze_task_and_plan_workflow(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze the task and create a workflow plan."""
        # Create messages for task analysis
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""
Task Analysis Request: {task_description}

Please analyze this task and create a workflow plan that includes:
1. Which agent(s) should be involved
2. The order of execution
3. How agents should collaborate
4. Expected outputs from each agent
5. Quality criteria for the final result

Available Agents:
- Research Agent: For information gathering, analysis, and research
- Creative Agent: For creative content, ideas, and innovative solutions
- Execution Agent: For practical implementation and task execution

Provide a structured workflow plan with clear steps and agent assignments.
""")
        ]
        
        # Generate workflow plan
        response = self._generate_response(messages, context)
        
        # Parse the response into a structured workflow plan
        workflow_plan = self._parse_workflow_plan(response, task_description)
        
        return workflow_plan
    
    def _parse_workflow_plan(self, response: str, task_description: str) -> Dict[str, Any]:
        """Parse the LLM response into a structured workflow plan."""
        # Simple parsing logic - in a real implementation, you might use more sophisticated parsing
        workflow_plan = {
            "task_description": task_description,
            "steps": [],
            "agents_involved": [],
            "estimated_duration": "unknown",
            "complexity": "medium"
        }
        
        # Determine which agents to involve based on task keywords
        task_lower = task_description.lower()
        
        # Check for research needs
        if any(word in task_lower for word in ["research", "analyze", "investigate", "study", "find"]):
            workflow_plan["agents_involved"].append("research")
            workflow_plan["steps"].append({
                "step": 1,
                "agent": "research",
                "action": "Gather and analyze relevant information",
                "output": "Research findings and insights"
            })
        
        # Check for creative needs
        if any(word in task_lower for word in ["create", "design", "generate", "innovate", "creative"]):
            workflow_plan["agents_involved"].append("creative")
            step_num = len(workflow_plan["steps"]) + 1
            workflow_plan["steps"].append({
                "step": step_num,
                "agent": "creative",
                "action": "Generate creative ideas and solutions",
                "output": "Creative concepts and innovative approaches"
            })
        
        # Check for execution needs
        if any(word in task_lower for word in ["implement", "execute", "build", "deploy", "create"]):
            workflow_plan["agents_involved"].append("execution")
            step_num = len(workflow_plan["steps"]) + 1
            workflow_plan["steps"].append({
                "step": step_num,
                "agent": "execution",
                "action": "Implement practical solutions",
                "output": "Actionable implementation plan"
            })
        
        # If no specific agents identified, use all three in sequence
        if not workflow_plan["agents_involved"]:
            workflow_plan["agents_involved"] = ["research", "creative", "execution"]
            workflow_plan["steps"] = [
                {
                    "step": 1,
                    "agent": "research",
                    "action": "Research and analyze the topic",
                    "output": "Research findings"
                },
                {
                    "step": 2,
                    "agent": "creative",
                    "action": "Generate creative ideas",
                    "output": "Creative solutions"
                },
                {
                    "step": 3,
                    "agent": "execution",
                    "action": "Create implementation plan",
                    "output": "Actionable plan"
                }
            ]
        
        return workflow_plan
    
    def _execute_workflow(self, workflow_plan: Dict[str, Any], session_id: str, 
                         context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute the workflow plan by coordinating the agents."""
        workflow_results = {}
        previous_results = {}
        
        for step in workflow_plan["steps"]:
            agent_name = step["agent"]
            agent = self.agents[agent_name]
            
            # Prepare task for the agent
            agent_task = self._prepare_agent_task(step, previous_results, context)
            
            # Execute the agent task
            agent_result = agent.process_task(agent_task, {
                "session_id": session_id,
                "workflow_step": step["step"],
                "previous_results": previous_results,
                "context": context
            })
            
            # Store result
            workflow_results[agent_name] = agent_result
            previous_results[agent_name] = agent_result
            
            # Store coordination message in short-term memory
            self._store_in_short_term_memory(
                session_id=session_id,
                message=f"Step {step['step']}: {agent_name} agent completed task - {step['action']}",
                metadata={
                    "workflow_step": step["step"],
                    "agent": agent_name,
                    "task_type": "coordination"
                }
            )
        
        return workflow_results
    
    def _prepare_agent_task(self, step: Dict[str, Any], previous_results: Dict[str, Any], 
                           context: Optional[Dict[str, Any]] = None) -> str:
        """Prepare a specific task for an agent based on the workflow step."""
        base_task = step["action"]
        
        # Add context from previous agent results
        if previous_results:
            context_parts = []
            for agent_name, result in previous_results.items():
                if "response" in result:
                    context_parts.append(f"Previous {agent_name} agent output: {result['response'][:200]}...")
            
            if context_parts:
                base_task += f"\n\nContext from previous steps:\n" + "\n".join(context_parts)
        
        # Add original context if available
        if context and "original_task" in context:
            base_task += f"\n\nOriginal task: {context['original_task']}"
        
        return base_task
    
    def _synthesize_results(self, workflow_results: Dict[str, Any], task_description: str,
                           context: Optional[Dict[str, Any]] = None) -> str:
        """Synthesize the results from all agents into a final response."""
        # Create messages for synthesis
        synthesis_prompt = f"""
Task: {task_description}

Agent Results:
"""
        
        for agent_name, result in workflow_results.items():
            if "response" in result:
                synthesis_prompt += f"\n{agent_name.upper()} AGENT:\n{result['response']}\n"
        
        synthesis_prompt += """
Please synthesize these results into a comprehensive, well-organized final response that:
1. Addresses the original task completely
2. Integrates insights from all agents
3. Provides a clear, actionable conclusion
4. Maintains the quality and depth of each agent's contribution
5. Presents the information in a logical, easy-to-follow structure

Focus on creating a cohesive response that leverages the strengths of each agent.
"""
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=synthesis_prompt)
        ]
        
        # Generate synthesized response
        final_response = self._generate_response(messages, context)
        
        return final_response
    
    def _store_workflow_in_memory(self, workflow_plan: Dict[str, Any], workflow_results: Dict[str, Any],
                                 final_result: str, session_id: str):
        """Store the workflow information in memory systems."""
        # Store in short-term memory
        workflow_summary = f"Workflow completed: {len(workflow_plan['steps'])} steps, {len(workflow_results)} agents"
        self._store_in_short_term_memory(
            session_id=session_id,
            message=workflow_summary,
            metadata={
                "task_type": "workflow_completion",
                "agents_used": list(workflow_results.keys()),
                "steps_count": len(workflow_plan["steps"])
            }
        )
        
        # Store valuable workflow patterns in long-term memory
        if self._is_valuable_workflow(final_result):
            self._store_in_long_term_memory(
                content=f"Workflow: {workflow_plan['task_description']}\nResult: {final_result}",
                category="workflow_patterns",
                metadata={
                    "agents_used": list(workflow_results.keys()),
                    "steps_count": len(workflow_plan["steps"]),
                    "agent_id": self.agent_id,
                    "workflow_type": "multi_agent_coordination"
                }
            )
    
    def _calculate_coordination_score(self, workflow_results: Dict[str, Any]) -> float:
        """Calculate how well the agents were coordinated."""
        if not workflow_results:
            return 0.0
        
        # Check if all agents completed successfully
        successful_agents = sum(1 for result in workflow_results.values() 
                              if result.get("success_rate", 0) > 0.5)
        
        return successful_agents / len(workflow_results)
    
    def _calculate_efficiency_score(self, workflow_plan: Dict[str, Any], 
                                  workflow_results: Dict[str, Any]) -> float:
        """Calculate the efficiency of the workflow execution."""
        if not workflow_results:
            return 0.0
        
        # Simple efficiency metric based on number of agents used vs. results quality
        avg_success_rate = sum(result.get("success_rate", 0) for result in workflow_results.values()) / len(workflow_results)
        
        # Penalize for using too many agents unnecessarily
        agent_efficiency = min(len(workflow_results) / 3, 1.0)  # Assume 3 agents is optimal
        
        return (avg_success_rate * 0.7) + (agent_efficiency * 0.3)
    
    def _calculate_overall_success_rate(self, workflow_results: Dict[str, Any]) -> float:
        """Calculate the overall success rate of the workflow."""
        if not workflow_results:
            return 0.0
        
        success_rates = [result.get("success_rate", 0) for result in workflow_results.values()]
        return sum(success_rates) / len(success_rates)
    
    def _is_valuable_workflow(self, final_result: str) -> bool:
        """Determine if the workflow result contains valuable content worth storing."""
        # Check if result contains substantial content
        if len(final_result) < 100:
            return False
        
        # Check for valuable indicators
        valuable_indicators = [
            "workflow", "coordination", "synthesis", "integration", "comprehensive",
            "analysis", "solution", "implementation", "strategy", "plan"
        ]
        
        result_lower = final_result.lower()
        return any(indicator in result_lower for indicator in valuable_indicators)
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get the status of all agents in the system."""
        status = {
            "supervisor": self.get_agent_info(),
            "agents": {}
        }
        
        for agent_name, agent in self.agents.items():
            status["agents"][agent_name] = agent.get_agent_info()
        
        return status
    
    def get_workflow_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get the workflow execution history."""
        if limit:
            return self.workflow_history[-limit:]
        return self.workflow_history.copy()
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.now().isoformat() 