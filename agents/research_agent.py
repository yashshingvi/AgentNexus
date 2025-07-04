"""
Research Agent specialized in gathering, analyzing, and synthesizing information.
"""

from typing import Dict, Any, Optional, List
from langchain.schema import HumanMessage, SystemMessage
from agents.base_agent import BaseAgent
import re


class ResearchAgent(BaseAgent):
    """Agent specialized in research, information gathering, and analysis."""
    
    def __init__(self, agent_id: str = "research_agent", agent_name: str = "Research Agent",
                 model_name: str = "gpt-4o-mini", temperature: float = 0.3,
                 short_term_memory=None, long_term_memory=None):
        """
        Initialize the Research Agent.
        
        Args:
            agent_id: Unique identifier for the agent
            agent_name: Human-readable name for the agent
            model_name: OpenAI model to use
            temperature: Temperature for model responses (lower for more focused research)
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
        """Get the system prompt for the Research Agent."""
        return """You are a Research Agent specialized in gathering, analyzing, and synthesizing information. 
Your capabilities include:

1. **Information Gathering**: Collect relevant information from various sources
2. **Data Analysis**: Analyze data to identify patterns, trends, and insights
3. **Fact Verification**: Verify facts and cross-reference information
4. **Report Generation**: Create comprehensive research reports
5. **Trend Analysis**: Identify and analyze trends in data
6. **Comparative Analysis**: Compare different sources, methods, or approaches
7. **Literature Review**: Conduct thorough literature reviews
8. **Market Research**: Analyze market trends, competitors, and opportunities

When processing tasks:
- Always provide well-structured, evidence-based responses
- Include relevant sources and citations when possible
- Present information in a clear, organized manner
- Highlight key findings and insights
- Consider multiple perspectives and viewpoints
- Maintain objectivity and accuracy

Your responses should be comprehensive, well-researched, and actionable."""
    
    def can_handle_task(self, task_description: str) -> bool:
        """
        Determine if this agent can handle a specific task.
        
        Args:
            task_description: Description of the task
            
        Returns:
            bool: True if the agent can handle the task
        """
        research_keywords = [
            "research", "analyze", "investigate", "study", "examine", "explore",
            "gather", "collect", "find", "search", "look up", "verify",
            "report", "summary", "overview", "trend", "pattern", "insight",
            "compare", "contrast", "review", "survey", "market", "data",
            "information", "facts", "evidence", "sources", "literature"
        ]
        
        task_lower = task_description.lower()
        return any(keyword in task_lower for keyword in research_keywords)
    
    def process_task(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a research task and return the result.
        
        Args:
            task_description: Description of the task to process
            context: Additional context for the task
            
        Returns:
            Dict containing the research result and metadata
        """
        try:
            # Update agent state
            self.update_agent_state(current_task=task_description)
            
            # Retrieve relevant knowledge from long-term memory
            relevant_knowledge = self._retrieve_from_long_term_memory(
                query=task_description,
                category="research",
                limit=3
            )
            
            # Retrieve recent conversation history
            session_id = context.get("session_id", "default") if context else "default"
            conversation_history = self._retrieve_from_short_term_memory(session_id, limit=5)
            
            # Build context from memory
            memory_context = self._build_memory_context(relevant_knowledge, conversation_history)
            
            # Create messages for the LLM
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=f"""
Task: {task_description}

Previous Research Context:
{memory_context}

Please provide a comprehensive research response including:
1. Key findings and insights
2. Relevant data and evidence
3. Analysis and interpretation
4. Recommendations or conclusions
5. Sources or references (if applicable)
""")
            ]
            
            # Generate response
            response = self._generate_response(messages, context)
            
            # Store in short-term memory
            self._store_in_short_term_memory(
                session_id=session_id,
                message=f"Research Task: {task_description}\nResponse: {response}",
                metadata={"task_type": "research", "context": context}
            )
            
            # Store valuable insights in long-term memory
            if self._is_valuable_research(response):
                self._store_in_long_term_memory(
                    content=response,
                    category="research",
                    metadata={
                        "task_description": task_description,
                        "agent_id": self.agent_id,
                        "insight_type": "research_finding"
                    }
                )
            
            # Create result
            result = {
                "agent_id": self.agent_id,
                "agent_name": self.agent_name,
                "task_description": task_description,
                "response": response,
                "research_type": self._classify_research_type(task_description),
                "confidence_score": self._calculate_confidence(response),
                "sources_used": self._extract_sources(response),
                "key_findings": self._extract_key_findings(response),
                "success_rate": 0.9,  # Research tasks typically have high success rates
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
    
    def _build_memory_context(self, relevant_knowledge: List[Dict], conversation_history: List[Dict]) -> str:
        """Build context from memory systems."""
        context_parts = []
        
        if relevant_knowledge:
            context_parts.append("Relevant Previous Research:")
            for knowledge in relevant_knowledge:
                context_parts.append(f"- {knowledge['content'][:200]}...")
        
        if conversation_history:
            context_parts.append("Recent Conversation Context:")
            for msg in conversation_history[:3]:  # Last 3 messages
                context_parts.append(f"- {msg['message'][:150]}...")
        
        return "\n".join(context_parts) if context_parts else "No relevant previous context found."
    
    def _classify_research_type(self, task_description: str) -> str:
        """Classify the type of research being requested."""
        task_lower = task_description.lower()
        
        if any(word in task_lower for word in ["market", "competitor", "industry"]):
            return "market_research"
        elif any(word in task_lower for word in ["trend", "pattern", "analysis"]):
            return "trend_analysis"
        elif any(word in task_lower for word in ["compare", "contrast", "versus"]):
            return "comparative_analysis"
        elif any(word in task_lower for word in ["review", "literature", "study"]):
            return "literature_review"
        elif any(word in task_lower for word in ["verify", "fact", "check"]):
            return "fact_verification"
        else:
            return "general_research"
    
    def _calculate_confidence(self, response: str) -> float:
        """Calculate confidence score based on response quality."""
        # Simple heuristic: longer, more detailed responses get higher confidence
        word_count = len(response.split())
        if word_count > 500:
            return 0.9
        elif word_count > 300:
            return 0.8
        elif word_count > 150:
            return 0.7
        else:
            return 0.5
    
    def _extract_sources(self, response: str) -> List[str]:
        """Extract sources mentioned in the response."""
        # Look for common source patterns
        source_patterns = [
            r'https?://[^\s]+',
            r'\[([^\]]+)\]',
            r'\(([^)]+)\)',
            r'Source: ([^\n]+)',
            r'Reference: ([^\n]+)'
        ]
        
        sources = []
        for pattern in source_patterns:
            matches = re.findall(pattern, response)
            sources.extend(matches)
        
        return list(set(sources))  # Remove duplicates
    
    def _extract_key_findings(self, response: str) -> List[str]:
        """Extract key findings from the response."""
        # Look for sentences that might contain key findings
        sentences = response.split('.')
        key_findings = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and any(word in sentence.lower() for word in 
                                        ["finding", "discovered", "revealed", "shows", "indicates", "suggests"]):
                key_findings.append(sentence)
        
        return key_findings[:5]  # Return top 5 findings
    
    def _is_valuable_research(self, response: str) -> bool:
        """Determine if the research response contains valuable insights worth storing."""
        # Check if response contains substantial content
        if len(response) < 100:
            return False
        
        # Check for key indicators of valuable research
        valuable_indicators = [
            "research", "study", "analysis", "finding", "discovery",
            "trend", "pattern", "insight", "evidence", "data"
        ]
        
        response_lower = response.lower()
        return any(indicator in response_lower for indicator in valuable_indicators)
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat() 