# Multi-Agent LangGraph System

A sophisticated multi-agent system built with Pure Python & Langgraph that features a supervisor agent coordinating three specialized agents with both short-term and long-term memory capabilities.

## 🏗️ System Architecture

### Agents

1. **👑 Supervisor Agent** - Coordinates and manages workflows between specialized agents
2. **🔍 Research Agent** - Specialized in gathering, analyzing, and synthesizing information
3. **🎨 Creative Agent** - Specialized in generating creative content, ideas, and innovative solutions
4. **⚙️ Execution Agent** - Specialized in implementing and executing tasks with practical solutions

### Memory Systems

- **🧠 Short-term Memory (Redis)** - Stores conversation context and immediate task information
- **💾 Long-term Memory (ChromaDB)** - Stores persistent knowledge, patterns, and historical data

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment

Create a `.env` file based on `env_example.txt`:

```bash
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Redis Configuration (for short-term memory)
REDIS_URL=redis://localhost:6379

# ChromaDB Configuration (for long-term memory)
CHROMA_PERSIST_DIRECTORY=./chroma_db

# Agent Configuration
SUPERVISOR_MODEL=gpt-4
AGENT_MODEL=gpt-4o-mini
TEMPERATURE=0.7
MAX_TOKENS=2000
```

### 3. Start Redis (for short-term memory)

**On Windows:**
```bash
# Download and install Redis from https://github.com/microsoftarchive/redis/releases
redis-server
```

**On macOS:**
```bash
brew install redis
brew services start redis
```

**On Linux:**
```bash
sudo apt-get install redis-server
sudo systemctl start redis
```

### 4. Run the System

```bash
python main.py
```

## 🎯 Usage Examples

### Sample Tasks

The system comes with pre-built sample tasks that demonstrate different agent capabilities:

1. **Research Task**: "Research the latest trends in artificial intelligence and machine learning for 2024"
2. **Creative Task**: "Create a creative marketing campaign for a new eco-friendly smartphone"
3. **Execution Task**: "Create a step-by-step implementation plan for setting up a CI/CD pipeline"
4. **Complex Multi-Agent Task**: "Design and implement a comprehensive customer feedback system for an e-commerce platform"

### Interactive Mode

You can also run the system in interactive mode to enter your own tasks:

```bash
python main.py
# Choose option 2 for interactive mode
```

## 🔧 System Features

### Agent Capabilities

#### Supervisor Agent
- **Task Analysis**: Analyzes incoming tasks and determines the best approach
- **Agent Selection**: Chooses the most appropriate agent(s) for each task
- **Workflow Orchestration**: Coordinates multiple agents in complex workflows
- **Quality Control**: Reviews and validates agent outputs
- **Performance Monitoring**: Tracks agent performance and optimizes workflows

#### Research Agent
- **Information Gathering**: Collects relevant information from various sources
- **Data Analysis**: Analyzes data to identify patterns, trends, and insights
- **Fact Verification**: Verifies facts and cross-references information
- **Report Generation**: Creates comprehensive research reports
- **Trend Analysis**: Identifies and analyzes trends in data

#### Creative Agent
- **Idea Generation**: Creates novel and innovative ideas across various domains
- **Content Creation**: Generates creative writing, stories, poems, and artistic content
- **Problem Solving**: Finds creative and unconventional solutions to problems
- **Design Thinking**: Applies design thinking principles to create user-centered solutions
- **Brainstorming**: Facilitates creative brainstorming sessions

#### Execution Agent
- **Task Implementation**: Converts ideas and plans into actionable steps
- **Process Optimization**: Improves and streamlines workflows and procedures
- **Project Management**: Plans, organizes, and executes projects effectively
- **Quality Assurance**: Ensures high-quality execution and deliverables
- **Resource Management**: Optimizes the use of time, materials, and resources

### Memory Features

#### Short-term Memory (Redis)
- **Conversation History**: Stores recent conversation context
- **Task State**: Maintains current task state information
- **Agent Context**: Stores agent-specific context information
- **Session Management**: Manages session-specific data with TTL

#### Long-term Memory (ChromaDB)
- **Knowledge Base**: Stores general knowledge and facts
- **Task Patterns**: Stores learned task patterns and solutions
- **Agent Interactions**: Stores historical agent interactions and outcomes
- **User Preferences**: Stores user preferences and settings

## 📊 Performance Metrics

The system tracks various performance metrics:

- **Success Rate**: Overall task completion success
- **Coordination Score**: How well agents were coordinated
- **Efficiency Score**: Workflow execution efficiency
- **Confidence Score**: Agent confidence in their responses
- **Memory Usage**: Memory system statistics

## 🛠️ Customization

### Adding New Agents

1. Create a new agent class inheriting from `BaseAgent`
2. Implement the required abstract methods
3. Add the agent to the supervisor's agent registry

### Modifying Memory Systems

The memory systems are modular and can be extended:
- Add new collection types to `LongTermMemory`
- Extend `ShortTermMemory` with new data types
- Implement custom memory backends

### Configuration

Modify agent behavior by adjusting:
- Model parameters (temperature, max_tokens)
- Memory TTL settings
- Agent selection logic
- Workflow orchestration rules

## 🔍 Troubleshooting

### Common Issues

1. **Redis Connection Error**
   - Ensure Redis is running: `redis-cli ping`
   - Check Redis URL in `.env` file

2. **OpenAI API Error**
   - Verify your API key is correct
   - Check your OpenAI account balance

3. **ChromaDB Error**
   - Ensure write permissions to the chroma_db directory
   - Check available disk space

4. **Import Errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python path and module structure

### Debug Mode

Enable debug logging by setting environment variables:
```bash
export DEBUG=1
export LOG_LEVEL=DEBUG
```

## 📈 Advanced Usage

### Programmatic API

```python
from agents.supervisor_agent import SupervisorAgent
from memory.short_term_memory import ShortTermMemory
from memory.long_term_memory import LongTermMemory

# Initialize memory systems
short_term_memory = ShortTermMemory()
long_term_memory = LongTermMemory()

# Initialize supervisor
supervisor = SupervisorAgent(
    short_term_memory=short_term_memory,
    long_term_memory=long_term_memory
)

# Process a task
result = supervisor.process_task("Your task description here")
print(result['final_response'])
```

### Memory Management

```python
# Store knowledge in long-term memory
long_term_memory.store_knowledge(
    content="Important information",
    category="research",
    metadata={"source": "user_input"}
)

# Retrieve relevant knowledge
knowledge = long_term_memory.retrieve_knowledge("query", limit=5)

# Store conversation in short-term memory
short_term_memory.store_conversation(
    session_id="session_123",
    message="User message",
    agent_id="supervisor_agent"
)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- LLMs Powered by [OpenAI](https://openai.com/)
- Memory systems using [Redis](https://redis.io/) and [ChromaDB](https://www.trychroma.com/) 
