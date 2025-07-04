"""
Long-term memory system using ChromaDB for persistent storage of knowledge,
learned patterns, and historical data.
"""

import os
import json
from typing import Dict, List, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
from datetime import datetime
import uuid


class LongTermMemory:
    """ChromaDB-based long-term memory for storing persistent knowledge and patterns."""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize long-term memory with ChromaDB.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
        """
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create collections for different types of knowledge
        self._initialize_collections()
    
    def _initialize_collections(self):
        """Initialize ChromaDB collections for different knowledge types."""
        try:
            # Knowledge base collection
            self.knowledge_collection = self.client.get_or_create_collection(
                name="knowledge_base",
                metadata={"description": "General knowledge and facts"}
            )
            
            # Task patterns collection
            self.patterns_collection = self.client.get_or_create_collection(
                name="task_patterns",
                metadata={"description": "Learned task patterns and solutions"}
            )
            
            # Agent interactions collection
            self.interactions_collection = self.client.get_or_create_collection(
                name="agent_interactions",
                metadata={"description": "Historical agent interactions and outcomes"}
            )
            
            # User preferences collection
            self.preferences_collection = self.client.get_or_create_collection(
                name="user_preferences",
                metadata={"description": "User preferences and settings"}
            )
            
        except Exception as e:
            print(f"Error initializing collections: {e}")
    
    def store_knowledge(self, content: str, category: str, metadata: Optional[Dict] = None) -> str:
        """
        Store knowledge in the long-term memory.
        
        Args:
            content: Knowledge content to store
            category: Category of knowledge
            metadata: Additional metadata
            
        Returns:
            str: ID of the stored knowledge
        """
        try:
            knowledge_id = str(uuid.uuid4())
            metadata = metadata or {}
            metadata.update({
                "category": category,
                "timestamp": datetime.now().isoformat(),
                "type": "knowledge"
            })
            
            self.knowledge_collection.add(
                documents=[content],
                metadatas=[metadata],
                ids=[knowledge_id]
            )
            
            return knowledge_id
        except Exception as e:
            print(f"Error storing knowledge: {e}")
            return ""
    
    def retrieve_knowledge(self, query: str, category: Optional[str] = None, 
                          limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant knowledge from long-term memory.
        
        Args:
            query: Search query
            category: Filter by category
            limit: Maximum number of results
            
        Returns:
            List of knowledge items with relevance scores
        """
        try:
            where_filter = {"category": category} if category else None
            
            results = self.knowledge_collection.query(
                query_texts=[query],
                n_results=limit,
                where=where_filter
            )
            
            knowledge_items = []
            for i in range(len(results['ids'][0])):
                knowledge_items.append({
                    "id": results['ids'][0][i],
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i]
                })
            
            return knowledge_items
        except Exception as e:
            print(f"Error retrieving knowledge: {e}")
            return []
    
    def store_task_pattern(self, task_type: str, solution: str, 
                          success_rate: float, metadata: Optional[Dict] = None) -> str:
        """
        Store a learned task pattern and solution.
        
        Args:
            task_type: Type of task
            solution: Solution approach
            success_rate: Success rate of this pattern
            metadata: Additional metadata
            
        Returns:
            str: ID of the stored pattern
        """
        try:
            pattern_id = str(uuid.uuid4())
            metadata = metadata or {}
            metadata.update({
                "task_type": task_type,
                "success_rate": success_rate,
                "timestamp": datetime.now().isoformat(),
                "type": "pattern"
            })
            
            self.patterns_collection.add(
                documents=[solution],
                metadatas=[metadata],
                ids=[pattern_id]
            )
            
            return pattern_id
        except Exception as e:
            print(f"Error storing task pattern: {e}")
            return ""
    
    def retrieve_task_patterns(self, task_type: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve relevant task patterns for a given task type.
        
        Args:
            task_type: Type of task
            limit: Maximum number of patterns to retrieve
            
        Returns:
            List of task patterns sorted by success rate
        """
        try:
            results = self.patterns_collection.query(
                query_texts=[task_type],
                n_results=limit,
                where={"task_type": task_type}
            )
            
            patterns = []
            for i in range(len(results['ids'][0])):
                patterns.append({
                    "id": results['ids'][0][i],
                    "solution": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i]
                })
            
            # Sort by success rate (descending)
            patterns.sort(key=lambda x: x['metadata'].get('success_rate', 0), reverse=True)
            return patterns
        except Exception as e:
            print(f"Error retrieving task patterns: {e}")
            return []
    
    def store_interaction(self, agent_id: str, user_input: str, response: str,
                         outcome: str, metadata: Optional[Dict] = None) -> str:
        """
        Store an agent interaction for learning purposes.
        
        Args:
            agent_id: ID of the agent
            user_input: User's input
            response: Agent's response
            outcome: Outcome of the interaction (success/failure)
            metadata: Additional metadata
            
        Returns:
            str: ID of the stored interaction
        """
        try:
            interaction_id = str(uuid.uuid4())
            metadata = metadata or {}
            metadata.update({
                "agent_id": agent_id,
                "outcome": outcome,
                "timestamp": datetime.now().isoformat(),
                "type": "interaction"
            })
            
            interaction_text = f"User: {user_input}\nAgent: {response}"
            
            self.interactions_collection.add(
                documents=[interaction_text],
                metadatas=[metadata],
                ids=[interaction_id]
            )
            
            return interaction_id
        except Exception as e:
            print(f"Error storing interaction: {e}")
            return ""
    
    def retrieve_similar_interactions(self, query: str, agent_id: Optional[str] = None,
                                    limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve similar past interactions.
        
        Args:
            query: Search query
            agent_id: Filter by agent ID
            limit: Maximum number of interactions to retrieve
            
        Returns:
            List of similar interactions
        """
        try:
            where_filter = {"agent_id": agent_id} if agent_id else None
            
            results = self.interactions_collection.query(
                query_texts=[query],
                n_results=limit,
                where=where_filter
            )
            
            interactions = []
            for i in range(len(results['ids'][0])):
                interactions.append({
                    "id": results['ids'][0][i],
                    "interaction": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i]
                })
            
            return interactions
        except Exception as e:
            print(f"Error retrieving interactions: {e}")
            return []
    
    def store_user_preference(self, user_id: str, preference_type: str, 
                             preference_value: str, metadata: Optional[Dict] = None) -> str:
        """
        Store user preferences for personalization.
        
        Args:
            user_id: User identifier
            preference_type: Type of preference
            preference_value: Preference value
            metadata: Additional metadata
            
        Returns:
            str: ID of the stored preference
        """
        try:
            preference_id = str(uuid.uuid4())
            metadata = metadata or {}
            metadata.update({
                "user_id": user_id,
                "preference_type": preference_type,
                "timestamp": datetime.now().isoformat(),
                "type": "preference"
            })
            
            self.preferences_collection.add(
                documents=[preference_value],
                metadatas=[metadata],
                ids=[preference_id]
            )
            
            return preference_id
        except Exception as e:
            print(f"Error storing user preference: {e}")
            return ""
    
    def get_user_preferences(self, user_id: str, preference_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve user preferences.
        
        Args:
            user_id: User identifier
            preference_type: Filter by preference type
            
        Returns:
            List of user preferences
        """
        try:
            where_filter = {"user_id": user_id}
            if preference_type:
                where_filter["preference_type"] = preference_type
            
            results = self.preferences_collection.get(where=where_filter)
            
            preferences = []
            for i in range(len(results['ids'])):
                preferences.append({
                    "id": results['ids'][i],
                    "value": results['documents'][i],
                    "metadata": results['metadatas'][i]
                })
            
            return preferences
        except Exception as e:
            print(f"Error retrieving user preferences: {e}")
            return []
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.
        
        Returns:
            Dictionary with memory statistics
        """
        try:
            stats = {
                "knowledge_count": self.knowledge_collection.count(),
                "patterns_count": self.patterns_collection.count(),
                "interactions_count": self.interactions_collection.count(),
                "preferences_count": self.preferences_collection.count(),
                "persist_directory": self.persist_directory
            }
            return stats
        except Exception as e:
            print(f"Error getting memory stats: {e}")
            return {}
    
    def clear_collection(self, collection_name: str) -> bool:
        """
        Clear a specific collection.
        
        Args:
            collection_name: Name of the collection to clear
            
        Returns:
            bool: True if successfully cleared
        """
        try:
            if collection_name == "knowledge_base":
                self.client.delete_collection("knowledge_base")
                self.knowledge_collection = self.client.create_collection("knowledge_base")
            elif collection_name == "task_patterns":
                self.client.delete_collection("task_patterns")
                self.patterns_collection = self.client.create_collection("task_patterns")
            elif collection_name == "agent_interactions":
                self.client.delete_collection("agent_interactions")
                self.interactions_collection = self.client.create_collection("agent_interactions")
            elif collection_name == "user_preferences":
                self.client.delete_collection("user_preferences")
                self.preferences_collection = self.client.create_collection("user_preferences")
            else:
                return False
            
            return True
        except Exception as e:
            print(f"Error clearing collection: {e}")
            return False 