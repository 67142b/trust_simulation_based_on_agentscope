"""
Main entry point for the Trust Behavior Simulation.
"""
import asyncio
import os
import sys
from simulator import TrustBehaviorSimulator


async def main():
    """Main function to run the trust behavior simulation."""
    
    # Check for required environment variables
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("Error: DASHSCOPE_API_KEY environment variable is not set.")
        print("Please set your DashScope API key:")
        print("export DASHSCOPE_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    print("Trust Behavior Simulation")
    print("=" * 50)
    print("This simulation uses BDI agents to model trust behavior in economic games.")
    print("Agents will participate in Trust Games and Dictator Games.")
    print("Each agent has unique personality traits based on the Big Five model.")
    print()
    
    try:
        # Create and run simulator
        simulator = TrustBehaviorSimulator("config.json")
        results = await simulator.run_full_simulation()
        
        # Print final summary
        print("\n" + "=" * 50)
        print("SIMULATION COMPLETED SUCCESSFULLY")
        print("=" * 50)
        
        if "simulation_summary" in results:
            summary = results["simulation_summary"]
            print(f"Total agents: {summary.get('total_agents', 0)}")
            print(f"Experiments run: {', '.join(summary.get('experiments_run', []))}")
        
        print(f"\nResults saved to: {simulator.data_saver.session_dir}")
        print("Check the generated files for detailed analysis and visualizations.")
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    except Exception as e:
        print(f"\nError during simulation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())