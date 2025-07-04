"""
Base agent class that provides common functionality for all specialized agents.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from memory.short_term_memory import ShortTermMemory
from memory.long_term_memory import LongTermMemory
import uuid
import os
from datetime import datetime


class BaseAgent(ABC):
    """Base class for all agents in the multi-agent system."""
    
    def __init__(self, agent_id: str, agent_name: str, model_name: str = "gpt-4o-mini",
                 temperature: float = 0.7, short_term_memory: Optional[ShortTermMemory] = None,
                 long_term_memory: Optional[LongTermMemory] = None):
        """
        Initialize the base agent.
        
        Args:
            agent_id: Unique identifier for the agent
            agent_name: Human-readable name for the agent
            model_name: OpenAI model to use
            temperature: Temperature for model responses
            short_term_memory: Short-term memory instance
            long_term_memory: Long-term memory instance
        """
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature
        )
        
        # Memory systems
        self.short_term_memory = short_term_memory
        self.long_term_memory = long_term_memory
        
        # Agent state
        self.is_active = True
        self.task_history = []
        self.current_task = None
        
        # Initialize agent-specific system prompt
        self.system_prompt = self._get_system_prompt()
    
    @abstractmethod
    def _get_system_prompt(self) -> str:
        """
        Get the system prompt for this specific agent.
        
        Returns:
            str: System prompt defining the agent's role and capabilities
        """
        pass
    
    @abstractmethod
    def can_handle_task(self, task_description: str) -> bool:
        """
        Determine if this agent can handle a specific task.
        
        Args:
            task_description: Description of the task
            
        Returns:
            bool: True if the agent can handle the task
        """
        pass
    
    @abstractmethod
    def process_task(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a task and return the result.
        
        Args:
            task_description: Description of the task to process
            context: Additional context for the task
            
        Returns:
            Dict containing the task result and metadata
        """
        pass
    
    def _store_in_short_term_memory(self, session_id: str, message: str, metadata: Optional[Dict] = None):
        """Store message in short-term memory if available."""
        if self.short_term_memory:
            self.short_term_memory.store_conversation(
                session_id=session_id,
                message=message,
                agent_id=self.agent_id,
                metadata=metadata
            )
    
    def _retrieve_from_short_term_memory(self, session_id: str, limit: int = 5) -> List[Dict]:
        """Retrieve recent conversation history from short-term memory."""
        if self.short_term_memory:
            return self.short_term_memory.get_conversation_history(session_id, limit)
        return []
    
    def _store_in_long_term_memory(self, content: str, category: str, metadata: Optional[Dict] = None) -> str:
        """Store knowledge in long-term memory if available."""
        if self.long_term_memory:
            return self.long_term_memory.store_knowledge(content, category, metadata)
        return ""
    
    def _retrieve_from_long_term_memory(self, query: str, category: Optional[str] = None, limit: int = 5) -> List[Dict]:
        """Retrieve relevant knowledge from long-term memory."""
        if self.long_term_memory:
            return self.long_term_memory.retrieve_knowledge(query, category, limit)
        return []
    
    def _generate_response(self, messages: List, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a response using the LLM.
        
        Args:
            messages: List of messages to send to the LLM
            context: Additional context for the response
            
        Returns:
            str: Generated response
        """
        try:
            # Add context to the last message if provided
            if context and messages:
                last_message = messages[-1]
                if isinstance(last_message, HumanMessage):
                    context_str = f"\n\nContext: {context}"
                    messages[-1] = HumanMessage(content=last_message.content + context_str)
            
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get information about this agent.
        
        Returns:
            Dict containing agent information
        """
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "is_active": self.is_active,
            "task_count": len(self.task_history),
            "current_task": self.current_task
        }
    
    def update_agent_state(self, is_active: Optional[bool] = None, current_task: Optional[str] = None):
        """
        Update the agent's state.
        
        Args:
            is_active: Whether the agent is active
            current_task: Current task being processed
        """
        if is_active is not None:
            self.is_active = is_active
        if current_task is not None:
            self.current_task = current_task
    
    def add_task_to_history(self, task_description: str, result: Dict[str, Any]):
        """
        Add a completed task to the agent's history.
        
        Args:
            task_description: Description of the completed task
            result: Result of the task
        """
        task_record = {
            "task_id": str(uuid.uuid4()),
            "description": task_description,
            "result": result,
            "timestamp": datetime.now().isoformat(),
            "agent_id": self.agent_id
        }
        self.task_history.append(task_record)
        
        # Store in long-term memory for learning
        if self.long_term_memory:
            self.long_term_memory.store_task_pattern(
                task_type=self.__class__.__name__,
                solution=str(result),
                success_rate=result.get("success_rate", 0.5),
                metadata={"agent_id": self.agent_id, "task_description": task_description}
            )
    
    def get_task_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get the agent's task history.
        
        Args:
            limit: Maximum number of tasks to return
            
        Returns:
            List of task records
        """
        if limit:
            return self.task_history[-limit:]
        return self.task_history.copy() 