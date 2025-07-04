"""
Main script to demonstrate the multi-agent LangGraph system with supervisor coordination.
"""

import os
import sys
from dotenv import load_dotenv
from typing import Dict, Any
import uuid
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our custom modules
from memory.short_term_memory import ShortTermMemory
from memory.long_term_memory import LongTermMemory
from agents.supervisor_agent import SupervisorAgent


def load_environment():
    """Load environment variables from .env file."""
    load_dotenv()
    
    # Check for required environment variables
    required_vars = ['OPENAI_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        print("Please create a .env file with the required variables (see env_example.txt)")
        return False
    
    return True


def initialize_memory_systems():
    """Initialize short-term and long-term memory systems."""
    try:
        # Initialize short-term memory (Redis)
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        short_term_memory = ShortTermMemory(redis_url=redis_url)
        print("✅ Short-term memory (Redis) initialized")
        
        # Initialize long-term memory (ChromaDB)
        chroma_persist_dir = os.getenv('CHROMA_PERSIST_DIRECTORY', './chroma_db')
        long_term_memory = LongTermMemory(persist_directory=chroma_persist_dir)
        print("✅ Long-term memory (ChromaDB) initialized")
        
        return short_term_memory, long_term_memory
        
    except Exception as e:
        print(f"❌ Error initializing memory systems: {e}")
        print("Note: Make sure Redis is running for short-term memory")
        return None, None


def initialize_supervisor_agent(short_term_memory, long_term_memory):
    """Initialize the supervisor agent with memory systems."""
    try:
        supervisor = SupervisorAgent(
            short_term_memory=short_term_memory,
            long_term_memory=long_term_memory
        )
        print("✅ Supervisor agent initialized")
        return supervisor
        
    except Exception as e:
        print(f"❌ Error initializing supervisor agent: {e}")
        return None


def display_agent_status(supervisor):
    """Display the status of all agents in the system."""
    print("\n🤖 Agent System Status:")
    print("=" * 50)
    
    status = supervisor.get_agent_status()
    
    # Supervisor status
    supervisor_info = status["supervisor"]
    print(f"👑 Supervisor Agent:")
    print(f"   - ID: {supervisor_info['agent_id']}")
    print(f"   - Model: {supervisor_info['model_name']}")
    print(f"   - Status: {'🟢 Active' if supervisor_info['is_active'] else '🔴 Inactive'}")
    print(f"   - Tasks completed: {supervisor_info['task_count']}")
    
    # Specialized agents status
    print(f"\n🔧 Specialized Agents:")
    for agent_name, agent_info in status["agents"].items():
        status_icon = "🟢" if agent_info['is_active'] else "🔴"
        print(f"   {status_icon} {agent_info['agent_name']}:")
        print(f"      - Model: {agent_info['model_name']}")
        print(f"      - Temperature: {agent_info['temperature']}")
        print(f"      - Tasks completed: {agent_info['task_count']}")


def run_sample_tasks(supervisor):
    """Run sample tasks to demonstrate the multi-agent system."""
    print("\n🚀 Running Sample Tasks")
    print("=" * 50)
    
    # Sample tasks to demonstrate different agent capabilities
    sample_tasks = [
        {
            "name": "Research Task",
            "description": "Research the latest trends in artificial intelligence and machine learning for 2024",
            "expected_agents": ["research"]
        },
        {
            "name": "Creative Task", 
            "description": "Create a creative marketing campaign for a new eco-friendly smartphone",
            "expected_agents": ["creative"]
        },
        {
            "name": "Execution Task",
            "description": "Create a step-by-step implementation plan for setting up a CI/CD pipeline",
            "expected_agents": ["execution"]
        },
        {
            "name": "Complex Multi-Agent Task",
            "description": "Design and implement a comprehensive customer feedback system for an e-commerce platform",
            "expected_agents": ["research", "creative", "execution"]
        }
    ]
    
    for i, task in enumerate(sample_tasks, 1):
        print(f"\n📋 Task {i}: {task['name']}")
        print(f"Description: {task['description']}")
        print(f"Expected agents: {', '.join(task['expected_agents'])}")
        print("-" * 50)
        
        # Generate unique session ID for this task
        session_id = str(uuid.uuid4())
        
        # Create context for the task
        context = {
            "session_id": session_id,
            "original_task": task['description'],
            "task_type": task['name'].lower().replace(" ", "_"),
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Process the task with the supervisor
            print("🔄 Processing task...")
            result = supervisor.process_task(task['description'], context)
            
            # Display results
            print(f"✅ Task completed successfully!")
            print(f"📊 Success rate: {result['success_rate']:.2%}")
            print(f"🎯 Coordination score: {result['coordination_score']:.2%}")
            print(f"⚡ Efficiency score: {result['efficiency_score']:.2%}")
            print(f"🤖 Agents used: {', '.join(result['agents_used'])}")
            
            # Display final response (truncated for readability)
            final_response = result['final_response']
            if len(final_response) > 500:
                print(f"📝 Final response (truncated):\n{final_response[:500]}...")
            else:
                print(f"📝 Final response:\n{final_response}")
            
            print("\n" + "="*50)
            
        except Exception as e:
            print(f"❌ Error processing task: {e}")
            print("\n" + "="*50)


def display_memory_stats(supervisor):
    """Display memory system statistics."""
    print("\n🧠 Memory System Statistics")
    print("=" * 50)
    
    try:
        # Short-term memory stats
        if supervisor.short_term_memory:
            short_term_stats = supervisor.short_term_memory.get_memory_stats()
            print("📊 Short-term Memory (Redis):")
            for key, value in short_term_stats.items():
                print(f"   - {key}: {value}")
        
        # Long-term memory stats
        if supervisor.long_term_memory:
            long_term_stats = supervisor.long_term_memory.get_memory_stats()
            print("\n📊 Long-term Memory (ChromaDB):")
            for key, value in long_term_stats.items():
                print(f"   - {key}: {value}")
                
    except Exception as e:
        print(f"❌ Error retrieving memory stats: {e}")


def interactive_mode(supervisor):
    """Run the system in interactive mode for user input."""
    print("\n🎮 Interactive Mode")
    print("=" * 50)
    print("Enter your tasks below. Type 'quit' to exit, 'status' to see agent status, 'stats' for memory stats.")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\n💬 Enter your task: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'status':
                display_agent_status(supervisor)
                continue
            elif user_input.lower() == 'stats':
                display_memory_stats(supervisor)
                continue
            elif not user_input:
                continue
            
            # Generate unique session ID
            session_id = str(uuid.uuid4())
            
            # Create context
            context = {
                "session_id": session_id,
                "original_task": user_input,
                "timestamp": datetime.now().isoformat()
            }
            
            print("🔄 Processing your task...")
            
            # Process the task
            result = supervisor.process_task(user_input, context)
            
            # Display results
            print(f"\n✅ Task completed!")
            print(f"📊 Success rate: {result['success_rate']:.2%}")
            print(f"🤖 Agents used: {', '.join(result['agents_used'])}")
            print(f"\n📝 Response:\n{result['final_response']}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main function to run the multi-agent system."""
    print("🤖 Multi-Agent LangGraph System")
    print("=" * 50)
    
    # Load environment variables
    if not load_environment():
        return
    
    # Initialize memory systems
    short_term_memory, long_term_memory = initialize_memory_systems()
    if short_term_memory is None or long_term_memory is None:
        print("❌ Failed to initialize memory systems. Exiting.")
        return
    
    # Initialize supervisor agent
    supervisor = initialize_supervisor_agent(short_term_memory, long_term_memory)
    if supervisor is None:
        print("❌ Failed to initialize supervisor agent. Exiting.")
        return
    
    # Display initial status
    display_agent_status(supervisor)
    
    # Ask user for mode
    print("\n🎯 Choose your mode:")
    print("1. Run sample tasks (demonstration)")
    print("2. Interactive mode (enter your own tasks)")
    print("3. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == '1':
                run_sample_tasks(supervisor)
                display_memory_stats(supervisor)
                break
            elif choice == '2':
                interactive_mode(supervisor)
                break
            elif choice == '3':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main() 