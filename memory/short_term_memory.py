"""
Short-term memory system using Redis for temporary storage of conversation context
and immediate task-related information.
"""

import json
import time
from typing import Dict, List, Any, Optional
import redis
from datetime import datetime, timedelta


class ShortTermMemory:
    """Redis-based short-term memory for storing temporary conversation context."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", ttl: int = 3600):
        """
        Initialize short-term memory with Redis.
        
        Args:
            redis_url: Redis connection URL
            ttl: Time-to-live for memory entries in seconds (default: 1 hour)
        """
        self.redis_client = redis.from_url(redis_url)
        self.ttl = ttl
    
    def store_conversation(self, session_id: str, message: str, agent_id: str, 
                          metadata: Optional[Dict] = None) -> bool:
        """
        Store a conversation message in short-term memory.
        
        Args:
            session_id: Unique session identifier
            message: The message content
            agent_id: ID of the agent that sent the message
            metadata: Additional metadata about the message
            
        Returns:
            bool: True if successfully stored
        """
        try:
            conversation_key = f"conversation:{session_id}"
            message_data = {
                "message": message,
                "agent_id": agent_id,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            
            # Add to conversation list
            self.redis_client.lpush(conversation_key, json.dumps(message_data))
            self.redis_client.expire(conversation_key, self.ttl)
            
            return True
        except Exception as e:
            print(f"Error storing conversation: {e}")
            return False
    
    def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """
        Retrieve conversation history for a session.
        
        Args:
            session_id: Unique session identifier
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of conversation messages
        """
        try:
            conversation_key = f"conversation:{session_id}"
            messages = self.redis_client.lrange(conversation_key, 0, limit - 1)
            return [json.loads(msg) for msg in messages]
        except Exception as e:
            print(f"Error retrieving conversation history: {e}")
            return []
    
    def store_task_state(self, task_id: str, state: Dict[str, Any]) -> bool:
        """
        Store task state information.
        
        Args:
            task_id: Unique task identifier
            state: Task state dictionary
            
        Returns:
            bool: True if successfully stored
        """
        try:
            task_key = f"task_state:{task_id}"
            self.redis_client.setex(
                task_key, 
                self.ttl, 
                json.dumps(state)
            )
            return True
        except Exception as e:
            print(f"Error storing task state: {e}")
            return False
    
    def get_task_state(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve task state information.
        
        Args:
            task_id: Unique task identifier
            
        Returns:
            Task state dictionary or None if not found
        """
        try:
            task_key = f"task_state:{task_id}"
            state_data = self.redis_client.get(task_key)
            return json.loads(state_data) if state_data else None
        except Exception as e:
            print(f"Error retrieving task state: {e}")
            return None
    
    def store_agent_context(self, agent_id: str, context: Dict[str, Any]) -> bool:
        """
        Store agent-specific context information.
        
        Args:
            agent_id: Agent identifier
            context: Context dictionary
            
        Returns:
            bool: True if successfully stored
        """
        try:
            context_key = f"agent_context:{agent_id}"
            self.redis_client.setex(
                context_key,
                self.ttl,
                json.dumps(context)
            )
            return True
        except Exception as e:
            print(f"Error storing agent context: {e}")
            return False
    
    def get_agent_context(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve agent-specific context information.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Context dictionary or None if not found
        """
        try:
            context_key = f"agent_context:{agent_id}"
            context_data = self.redis_client.get(context_key)
            return json.loads(context_data) if context_data else None
        except Exception as e:
            print(f"Error retrieving agent context: {e}")
            return None
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear all data for a specific session.
        
        Args:
            session_id: Session identifier to clear
            
        Returns:
            bool: True if successfully cleared
        """
        try:
            conversation_key = f"conversation:{session_id}"
            self.redis_client.delete(conversation_key)
            return True
        except Exception as e:
            print(f"Error clearing session: {e}")
            return False
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.
        
        Returns:
            Dictionary with memory statistics
        """
        try:
            info = self.redis_client.info()
            return {
                "used_memory": info.get("used_memory", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands_processed": info.get("total_commands_processed", 0)
            }
        except Exception as e:
            print(f"Error getting memory stats: {e}")
            return {} 