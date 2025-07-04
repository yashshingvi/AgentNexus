"""
Execution Agent specialized in implementing and executing tasks with practical solutions.
"""

from typing import Dict, Any, Optional, List
from langchain.schema import HumanMessage, SystemMessage
from agents.base_agent import BaseAgent
import re


class ExecutionAgent(BaseAgent):
    """Agent specialized in practical implementation and task execution."""
    
    def __init__(self, agent_id: str = "execution_agent", agent_name: str = "Execution Agent",
                 model_name: str = "gpt-4o-mini", temperature: float = 0.2,
                 short_term_memory=None, long_term_memory=None):
        """
        Initialize the Execution Agent.
        
        Args:
            agent_id: Unique identifier for the agent
            agent_name: Human-readable name for the agent
            model_name: OpenAI model to use
            temperature: Temperature for model responses (lower for more focused execution)
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
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the Execution Agent."""
        return """You are an Execution Agent specialized in implementing practical solutions and executing tasks efficiently. 
Your capabilities include:

1. **Task Implementation**: Convert ideas and plans into actionable steps
2. **Process Optimization**: Improve and streamline workflows and procedures
3. **Project Management**: Plan, organize, and execute projects effectively
4. **Problem Solving**: Find practical solutions to implementation challenges
5. **Quality Assurance**: Ensure high-quality execution and deliverables
6. **Resource Management**: Optimize the use of time, materials, and resources
7. **Risk Management**: Identify and mitigate potential execution risks
8. **Performance Monitoring**: Track progress and measure outcomes

When processing tasks:
- Focus on practical, actionable steps
- Provide clear, step-by-step instructions
- Consider feasibility and resource constraints
- Prioritize efficiency and effectiveness
- Include quality checks and validation steps
- Address potential challenges and contingencies
- Provide measurable outcomes and success criteria

Your responses should be practical, actionable, and results-oriented with clear implementation guidance."""
    
    def can_handle_task(self, task_description: str) -> bool:
        """
        Determine if this agent can handle a specific task.
        
        Args:
            task_description: Description of the task
            
        Returns:
            bool: True if the agent can handle the task
        """
        execution_keywords = [
            "implement", "execute", "build", "create", "develop", "construct",
            "deploy", "launch", "run", "perform", "carry out", "complete",
            "finish", "deliver", "produce", "generate", "establish", "set up",
            "configure", "install", "setup", "organize", "manage", "coordinate",
            "plan", "schedule", "timeline", "milestone", "deadline", "deliverable",
            "action", "step", "process", "workflow", "procedure", "methodology"
        ]
        
        task_lower = task_description.lower()
        return any(keyword in task_lower for keyword in execution_keywords)
    
    def process_task(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process an execution task and return the result.
        
        Args:
            task_description: Description of the task to process
            context: Additional context for the task
            
        Returns:
            Dict containing the execution result and metadata
        """
        try:
            # Update agent state
            self.update_agent_state(current_task=task_description)
            
            # Retrieve relevant implementation patterns from long-term memory
            relevant_patterns = self._retrieve_from_long_term_memory(
                query=task_description,
                category="execution",
                limit=3
            )
            
            # Retrieve recent conversation history
            session_id = context.get("session_id", "default") if context else "default"
            conversation_history = self._retrieve_from_short_term_memory(session_id, limit=5)
            
            # Build execution context from memory
            execution_context = self._build_execution_context(relevant_patterns, conversation_history)
            
            # Determine execution approach
            execution_approach = self._determine_execution_approach(task_description)
            
            # Create messages for the LLM
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=f"""
Execution Task: {task_description}

Execution Approach: {execution_approach}

Previous Implementation Context:
{execution_context}

Please provide a comprehensive execution plan that includes:
1. Clear step-by-step implementation instructions
2. Required resources and dependencies
3. Timeline and milestones
4. Quality assurance measures
5. Risk mitigation strategies
6. Success criteria and validation steps
7. Monitoring and feedback mechanisms

Focus on practical, actionable implementation guidance.
""")
            ]
            
            # Generate response
            response = self._generate_response(messages, context)
            
            # Store in short-term memory
            self._store_in_short_term_memory(
                session_id=session_id,
                message=f"Execution Task: {task_description}\nResponse: {response}",
                metadata={"task_type": "execution", "approach": execution_approach, "context": context}
            )
            
            # Store valuable implementation patterns in long-term memory
            if self._is_valuable_execution_content(response):
                self._store_in_long_term_memory(
                    content=response,
                    category="execution",
                    metadata={
                        "task_description": task_description,
                        "agent_id": self.agent_id,
                        "execution_type": execution_approach,
                        "complexity_level": self._assess_complexity_level(response)
                    }
                )
            
            # Create result
            result = {
                "agent_id": self.agent_id,
                "agent_name": self.agent_name,
                "task_description": task_description,
                "response": response,
                "execution_type": execution_approach,
                "feasibility_score": self._calculate_feasibility_score(response),
                "completeness_score": self._calculate_completeness_score(response),
                "steps_defined": self._count_implementation_steps(response),
                "execution_elements": self._extract_execution_elements(response),
                "estimated_duration": self._estimate_execution_duration(response),
                "success_rate": 0.88,  # Execution tasks have good success rates
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
    
    def _build_execution_context(self, relevant_patterns: List[Dict], conversation_history: List[Dict]) -> str:
        """Build execution context from memory systems."""
        context_parts = []
        
        if relevant_patterns:
            context_parts.append("Previous Implementation Patterns:")
            for pattern in relevant_patterns:
                context_parts.append(f"- {pattern['content'][:200]}...")
        
        if conversation_history:
            context_parts.append("Recent Execution Context:")
            for msg in conversation_history[:3]:  # Last 3 messages
                if "execution" in msg.get('metadata', {}).get('task_type', '').lower():
                    context_parts.append(f"- {msg['message'][:150]}...")
        
        return "\n".join(context_parts) if context_parts else "No previous execution context found."
    
    def _determine_execution_approach(self, task_description: str) -> str:
        """Determine the most appropriate execution approach for the task."""
        task_lower = task_description.lower()
        
        if any(word in task_lower for word in ["project", "plan", "manage", "coordinate"]):
            return "project_management"
        elif any(word in task_lower for word in ["build", "develop", "create", "construct"]):
            return "development"
        elif any(word in task_lower for word in ["deploy", "launch", "release", "publish"]):
            return "deployment"
        elif any(word in task_lower for word in ["optimize", "improve", "enhance", "streamline"]):
            return "optimization"
        elif any(word in task_lower for word in ["setup", "configure", "install", "establish"]):
            return "setup_configuration"
        elif any(word in task_lower for word in ["process", "workflow", "procedure", "methodology"]):
            return "process_improvement"
        else:
            return "general_execution"
    
    def _calculate_feasibility_score(self, response: str) -> float:
        """Calculate feasibility score based on response quality."""
        # Look for feasibility indicators
        feasibility_indicators = [
            "feasible", "practical", "achievable", "realistic", "implementable",
            "step-by-step", "clear", "specific", "detailed", "actionable",
            "resource", "timeline", "milestone", "deadline", "validation"
        ]
        
        response_lower = response.lower()
        feasibility_count = sum(1 for indicator in feasibility_indicators if indicator in response_lower)
        
        # Normalize score
        max_possible = len(feasibility_indicators)
        return min(feasibility_count / max_possible, 1.0)
    
    def _calculate_completeness_score(self, response: str) -> float:
        """Calculate completeness score based on response comprehensiveness."""
        # Check for completeness indicators
        completeness_indicators = [
            "step", "phase", "milestone", "checkpoint", "validation",
            "quality", "testing", "monitoring", "feedback", "review",
            "resource", "timeline", "risk", "contingency", "success"
        ]
        
        response_lower = response.lower()
        completeness_count = sum(1 for indicator in completeness_indicators if indicator in response_lower)
        
        # Normalize score
        max_possible = len(completeness_indicators)
        return min(completeness_count / max_possible, 1.0)
    
    def _count_implementation_steps(self, response: str) -> int:
        """Count the number of implementation steps defined in the response."""
        # Look for step patterns
        step_patterns = [
            r'\d+\.\s',  # Numbered steps
            r'Step\s+\d+',  # "Step 1", "Step 2", etc.
            r'Phase\s+\d+',  # "Phase 1", "Phase 2", etc.
            r'Stage\s+\d+',  # "Stage 1", "Stage 2", etc.
            r'Milestone\s+\d+',  # "Milestone 1", etc.
            r'Checkpoint\s+\d+'  # "Checkpoint 1", etc.
        ]
        
        step_count = 0
        for pattern in step_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            step_count += len(matches)
        
        # If no clear patterns, estimate based on paragraphs
        if step_count == 0:
            paragraphs = response.split('\n\n')
            step_count = len([p for p in paragraphs if len(p.strip()) > 30])
        
        return max(step_count, 1)  # At least 1 step
    
    def _extract_execution_elements(self, response: str) -> List[str]:
        """Extract execution elements from the response."""
        execution_elements = []
        
        # Look for execution techniques mentioned
        execution_techniques = [
            "agile", "waterfall", "scrum", "kanban", "lean", "six sigma",
            "project management", "risk management", "quality assurance",
            "testing", "validation", "monitoring", "feedback", "iteration",
            "deployment", "rollback", "backup", "documentation"
        ]
        
        response_lower = response.lower()
        for technique in execution_techniques:
            if technique in response_lower:
                execution_elements.append(technique)
        
        # Look for execution outcomes
        execution_outcomes = [
            "implementation", "deployment", "delivery", "completion",
            "launch", "release", "go-live", "production", "handover"
        ]
        
        for outcome in execution_outcomes:
            if outcome in response_lower:
                execution_elements.append(outcome)
        
        return list(set(execution_elements))  # Remove duplicates
    
    def _estimate_execution_duration(self, response: str) -> str:
        """Estimate the execution duration based on response content."""
        # Look for time indicators
        time_patterns = [
            r'(\d+)\s*(hour|hr)s?',
            r'(\d+)\s*(day|week|month)s?',
            r'(\d+)\s*(minute|min)s?',
            r'(\d+)\s*(second|sec)s?'
        ]
        
        total_hours = 0
        for pattern in time_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in matches:
                value = int(match[0])
                unit = match[1].lower()
                
                if unit in ['hour', 'hr', 'hours']:
                    total_hours += value
                elif unit in ['day', 'days']:
                    total_hours += value * 8  # Assume 8-hour workday
                elif unit in ['week', 'weeks']:
                    total_hours += value * 40  # Assume 40-hour workweek
                elif unit in ['month', 'months']:
                    total_hours += value * 160  # Assume 160-hour workmonth
                elif unit in ['minute', 'min']:
                    total_hours += value / 60
                elif unit in ['second', 'sec']:
                    total_hours += value / 3600
        
        # Convert to appropriate unit
        if total_hours < 1:
            return f"{int(total_hours * 60)} minutes"
        elif total_hours < 24:
            return f"{total_hours:.1f} hours"
        elif total_hours < 168:  # 7 days
            return f"{total_hours / 8:.1f} days"
        else:
            return f"{total_hours / 160:.1f} months"
    
    def _is_valuable_execution_content(self, response: str) -> bool:
        """Determine if the execution response contains valuable content worth storing."""
        # Check if response contains substantial execution content
        if len(response) < 100:
            return False
        
        # Check for execution indicators
        execution_indicators = [
            "implement", "execute", "deploy", "build", "create", "develop",
            "step", "phase", "milestone", "timeline", "process", "workflow"
        ]
        
        response_lower = response.lower()
        return any(indicator in response_lower for indicator in execution_indicators)
    
    def _assess_complexity_level(self, response: str) -> str:
        """Assess the complexity level of the execution plan."""
        step_count = self._count_implementation_steps(response)
        word_count = len(response.split())
        
        if step_count > 10 or word_count > 1000:
            return "high"
        elif step_count > 5 or word_count > 500:
            return "medium"
        else:
            return "low"
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat() 