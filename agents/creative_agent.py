"""
Creative Agent specialized in generating creative content, ideas, and innovative solutions.
"""

from typing import Dict, Any, Optional, List
from langchain.schema import HumanMessage, SystemMessage
from agents.base_agent import BaseAgent
import re


class CreativeAgent(BaseAgent):
    """Agent specialized in creative content generation and innovative thinking."""
    
    def __init__(self, agent_id: str = "creative_agent", agent_name: str = "Creative Agent",
                 model_name: str = "gpt-4o-mini", temperature: float = 0.8,
                 short_term_memory=None, long_term_memory=None):
        """
        Initialize the Creative Agent.
        
        Args:
            agent_id: Unique identifier for the agent
            agent_name: Human-readable name for the agent
            model_name: OpenAI model to use
            temperature: Temperature for model responses (higher for more creativity)
            short_term_memory: Short-term memory instance
            long_term_memory: Long-term memory instance
        """
        super().__init__(
            agent_id=agent_id,
            agent_name=agent_name,
            model=model_name,
            temperature=temperature,
            short_term_memory=short_term_memory,
            long_term_memory=long_term_memory
        )
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the Creative Agent."""
        return """You are a Creative Agent specialized in generating innovative ideas, creative content, and imaginative solutions. 
Your capabilities include:

1. **Idea Generation**: Create novel and innovative ideas across various domains
2. **Content Creation**: Generate creative writing, stories, poems, and artistic content
3. **Problem Solving**: Find creative and unconventional solutions to problems
4. **Design Thinking**: Apply design thinking principles to create user-centered solutions
5. **Brainstorming**: Facilitate creative brainstorming sessions
6. **Storytelling**: Create compelling narratives and stories
7. **Concept Development**: Develop new concepts and approaches
8. **Innovation**: Generate breakthrough ideas and innovations

When processing tasks:
- Think outside the box and explore unconventional approaches
- Combine different ideas and concepts in novel ways
- Consider multiple perspectives and viewpoints
- Focus on originality and uniqueness
- Balance creativity with practicality
- Provide multiple options and alternatives
- Use vivid language and engaging descriptions

Your responses should be imaginative, inspiring, and thought-provoking while remaining relevant and useful."""
    
    def can_handle_task(self, task_description: str) -> bool:
        """
        Determine if this agent can handle a specific task.
        
        Args:
            task_description: Description of the task
            
        Returns:
            bool: True if the agent can handle the task
        """
        creative_keywords = [
            "create", "generate", "design", "invent", "imagine", "brainstorm",
            "creative", "innovative", "original", "unique", "novel", "artistic",
            "story", "poem", "content", "idea", "concept", "solution",
            "write", "compose", "develop", "craft", "build", "construct",
            "visualize", "envision", "dream", "fantasy", "fiction", "art",
            "inspire", "motivate", "entertain", "engage", "captivate"
        ]
        
        task_lower = task_description.lower()
        return any(keyword in task_lower for keyword in creative_keywords)
    
    def process_task(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a creative task and return the result.
        
        Args:
            task_description: Description of the task to process
            context: Additional context for the task
            
        Returns:
            Dict containing the creative result and metadata
        """
        try:
            # Update agent state
            self.update_agent_state(current_task=task_description)
            
            # Retrieve relevant creative inspiration from long-term memory
            relevant_inspiration = self._retrieve_from_long_term_memory(
                query=task_description,
                category="creative",
                limit=3
            )
            
            # Retrieve recent conversation history
            session_id = context.get("session_id", "default") if context else "default"
            conversation_history = self._retrieve_from_short_term_memory(session_id, limit=5)
            
            # Build creative context from memory
            creative_context = self._build_creative_context(relevant_inspiration, conversation_history)
            
            # Determine creative approach
            creative_approach = self._determine_creative_approach(task_description)
            
            # Create messages for the LLM
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=f"""
Creative Task: {task_description}

Creative Approach: {creative_approach}

Previous Creative Context:
{creative_context}

Please provide a creative response that includes:
1. Multiple creative ideas or options
2. Detailed development of the best ideas
3. Explanation of the creative process
4. Potential applications or variations
5. Inspirational elements and unique perspectives

Be imaginative, original, and inspiring!
""")
            ]
            
            # Generate response
            response = self._generate_response(messages, context)
            
            # Store in short-term memory
            self._store_in_short_term_memory(
                session_id=session_id,
                message=f"Creative Task: {task_description}\nResponse: {response}",
                metadata={"task_type": "creative", "approach": creative_approach, "context": context}
            )
            
            # Store valuable creative ideas in long-term memory
            if self._is_valuable_creative_content(response):
                self._store_in_long_term_memory(
                    content=response,
                    category="creative",
                    metadata={
                        "task_description": task_description,
                        "agent_id": self.agent_id,
                        "creative_type": creative_approach,
                        "inspiration_level": self._assess_inspiration_level(response)
                    }
                )
            
            # Create result
            result = {
                "agent_id": self.agent_id,
                "agent_name": self.agent_name,
                "task_description": task_description,
                "response": response,
                "creative_type": creative_approach,
                "inspiration_score": self._calculate_inspiration_score(response),
                "originality_score": self._calculate_originality_score(response),
                "ideas_generated": self._count_ideas_generated(response),
                "creative_elements": self._extract_creative_elements(response),
                "success_rate": 0.85,  # Creative tasks have good success rates
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
    
    def _build_creative_context(self, relevant_inspiration: List[Dict], conversation_history: List[Dict]) -> str:
        """Build creative context from memory systems."""
        context_parts = []
        
        if relevant_inspiration:
            context_parts.append("Creative Inspiration:")
            for inspiration in relevant_inspiration:
                context_parts.append(f"- {inspiration['content'][:200]}...")
        
        if conversation_history:
            context_parts.append("Recent Creative Context:")
            for msg in conversation_history[:3]:  # Last 3 messages
                if "creative" in msg.get('metadata', {}).get('task_type', '').lower():
                    context_parts.append(f"- {msg['message'][:150]}...")
        
        return "\n".join(context_parts) if context_parts else "No previous creative context found."
    
    def _determine_creative_approach(self, task_description: str) -> str:
        """Determine the most appropriate creative approach for the task."""
        task_lower = task_description.lower()
        
        if any(word in task_lower for word in ["story", "narrative", "tale", "fiction"]):
            return "storytelling"
        elif any(word in task_lower for word in ["poem", "poetry", "verse", "rhyme"]):
            return "poetry"
        elif any(word in task_lower for word in ["design", "visual", "art", "graphic"]):
            return "visual_design"
        elif any(word in task_lower for word in ["idea", "concept", "innovation", "invention"]):
            return "ideation"
        elif any(word in task_lower for word in ["solution", "problem", "solve", "fix"]):
            return "creative_problem_solving"
        elif any(word in task_lower for word in ["brainstorm", "generate", "create"]):
            return "brainstorming"
        else:
            return "general_creativity"
    
    def _calculate_inspiration_score(self, response: str) -> float:
        """Calculate inspiration score based on response quality."""
        # Look for inspirational elements
        inspirational_indicators = [
            "imagine", "envision", "dream", "aspire", "inspire", "motivate",
            "breakthrough", "revolutionary", "groundbreaking", "innovative",
            "unique", "original", "creative", "artistic", "beautiful", "amazing"
        ]
        
        response_lower = response.lower()
        inspiration_count = sum(1 for indicator in inspirational_indicators if indicator in response_lower)
        
        # Normalize score
        max_possible = len(inspirational_indicators)
        return min(inspiration_count / max_possible, 1.0)
    
    def _calculate_originality_score(self, response: str) -> float:
        """Calculate originality score based on response uniqueness."""
        # Simple heuristic: longer, more detailed responses with varied vocabulary
        words = response.split()
        unique_words = set(words)
        
        if len(words) == 0:
            return 0.0
        
        # Vocabulary diversity
        vocabulary_diversity = len(unique_words) / len(words)
        
        # Length factor
        length_factor = min(len(response) / 500, 1.0)
        
        # Combine factors
        return (vocabulary_diversity * 0.6) + (length_factor * 0.4)
    
    def _count_ideas_generated(self, response: str) -> int:
        """Count the number of distinct ideas generated in the response."""
        # Look for numbered lists, bullet points, or idea indicators
        idea_patterns = [
            r'\d+\.\s',  # Numbered lists
            r'[-*•]\s',  # Bullet points
            r'Idea\s+\d+',  # "Idea 1", "Idea 2", etc.
            r'Option\s+\d+',  # "Option 1", "Option 2", etc.
            r'Alternative\s+\d+',  # "Alternative 1", etc.
            r'Concept\s+\d+'  # "Concept 1", etc.
        ]
        
        idea_count = 0
        for pattern in idea_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            idea_count += len(matches)
        
        # If no clear patterns, estimate based on paragraphs
        if idea_count == 0:
            paragraphs = response.split('\n\n')
            idea_count = len([p for p in paragraphs if len(p.strip()) > 50])
        
        return max(idea_count, 1)  # At least 1 idea
    
    def _extract_creative_elements(self, response: str) -> List[str]:
        """Extract creative elements from the response."""
        creative_elements = []
        
        # Look for creative techniques mentioned
        creative_techniques = [
            "metaphor", "analogy", "visualization", "brainstorming", "mind mapping",
            "lateral thinking", "design thinking", "prototyping", "iteration",
            "collaboration", "experimentation", "playfulness", "curiosity"
        ]
        
        response_lower = response.lower()
        for technique in creative_techniques:
            if technique in response_lower:
                creative_elements.append(technique)
        
        # Look for creative outcomes
        creative_outcomes = [
            "story", "poem", "design", "concept", "prototype", "solution",
            "innovation", "creation", "artwork", "composition"
        ]
        
        for outcome in creative_outcomes:
            if outcome in response_lower:
                creative_elements.append(outcome)
        
        return list(set(creative_elements))  # Remove duplicates
    
    def _is_valuable_creative_content(self, response: str) -> bool:
        """Determine if the creative response contains valuable content worth storing."""
        # Check if response contains substantial creative content
        if len(response) < 100:
            return False
        
        # Check for creative indicators
        creative_indicators = [
            "creative", "innovative", "original", "unique", "imaginative",
            "story", "poem", "design", "concept", "idea", "solution"
        ]
        
        response_lower = response.lower()
        return any(indicator in response_lower for indicator in creative_indicators)
    
    def _assess_inspiration_level(self, response: str) -> str:
        """Assess the inspiration level of the creative content."""
        inspiration_score = self._calculate_inspiration_score(response)
        
        if inspiration_score > 0.7:
            return "high"
        elif inspiration_score > 0.4:
            return "medium"
        else:
            return "low"
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat() 