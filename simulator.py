"""
Main trust behavior simulator integrating BDI agents and economic games.
"""
import asyncio
import json
import os
import random
from typing import List, Dict, Any
import numpy as np
from datetime import datetime

from agentscope.message import Msg
from agentscope.pipeline import MsgHub, fanout_pipeline, sequential_pipeline
from agentscope.model import DashScopeChatModel

from agents.bdi_agent import BDITrustAgent
from experiments.trust_game import MultiTrustGame
from experiments.dictator_game import MultiDictatorGame
from utils.agent_factory import AgentFactory
from utils.data_saver import DataSaver


class TrustBehaviorSimulator:
    """
    Main simulator for trust behavior experiments using BDI agents.
    """
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the simulator.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.agents: List[BDITrustAgent] = []
        self.results: Dict[str, Any] = {}
        self.data_saver = DataSaver(self.config["output_config"])
        
        # Initialize model
        self.model_config = self.config["model_config"]
        self.model_config["api_key"] = os.getenv(
            self.model_config["api_key_env"], 
            "your-api-key-here"
        )
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    async def setup_agents(self) -> None:
        """Setup BDI agents with diverse personalities."""
        print("Setting up agents...")
        
        agent_factory = AgentFactory(self.config)
        num_agents = self.config["experiment_config"]["num_agents"]
        
        # Create agents with diverse personalities
        self.agents = await agent_factory.create_agent_pool(num_agents)
        
        print(f"Created {len(self.agents)} agents:")
        for agent in self.agents:
            print(f"  - {agent.name} (age: {agent.age}, education: {agent.education})")
            print(f"    Traits: {agent.personality_traits}")
    
    async def run_trust_game_experiment(self) -> Dict[str, Any]:
        """Run the trust game experiment."""
        print("\n" + "="*50)
        print("RUNNING TRUST GAME EXPERIMENT")
        print("="*50)
        
        trust_game = MultiTrustGame(self.agents, self.config["experiment_config"])
        num_rounds = self.config["experiment_config"]["num_rounds"]
        
        # Run tournament
        results = await trust_game.run_tournament(num_rounds)
        
        # Get statistics
        summary_stats = trust_game.trust_game.get_summary_stats()
        rankings = trust_game.get_agent_rankings()
        
        experiment_results = {
            "experiment_type": "trust_game",
            "timestamp": datetime.now().isoformat(),
            "config": self.config["experiment_config"],
            "results": results,
            "summary_stats": summary_stats,
            "rankings": rankings,
            "agent_states": [agent.get_agent_state() for agent in self.agents]
        }
        
        self.results["trust_game"] = experiment_results
        
        # Print summary
        print(f"\nTrust Game Summary:")
        print(f"  Total rounds: {summary_stats.get('total_rounds', 0)}")
        print(f"  Average investment: {summary_stats.get('avg_investment', 0):.2f}")
        print(f"  Average return: {summary_stats.get('avg_return', 0):.2f}")
        print(f"  Trust rate: {summary_stats.get('trust_rate', 0):.2%}")
        print(f"  Profitable investments: {summary_stats.get('profitable_investments', 0):.2%}")
        
        print(f"\nTop 3 Agents by Profit:")
        for i, ranking in enumerate(rankings[:3]):
            print(f"  {i+1}. {ranking['name']}: {ranking['total_profit']:.2f}")
        
        return experiment_results
    
    async def run_dictator_game_experiment(self) -> Dict[str, Any]:
        """Run the dictator game experiment."""
        print("\n" + "="*50)
        print("RUNNING DICTATOR GAME EXPERIMENT")
        print("="*50)
        
        dictator_game = MultiDictatorGame(self.agents, self.config["experiment_config"])
        num_rounds = max(1, self.config["experiment_config"]["num_rounds"] // 2)  # Fewer rounds for dictator game
        
        # Run tournament
        results = await dictator_game.run_tournament(num_rounds)
        
        # Get statistics
        summary_stats = dictator_game.dictator_game.get_summary_stats()
        rankings = dictator_game.get_agent_rankings()
        personality_correlations = dictator_game.analyze_personality_correlations()
        
        experiment_results = {
            "experiment_type": "dictator_game",
            "timestamp": datetime.now().isoformat(),
            "config": self.config["experiment_config"],
            "results": results,
            "summary_stats": summary_stats,
            "rankings": rankings,
            "personality_correlations": personality_correlations,
            "agent_states": [agent.get_agent_state() for agent in self.agents]
        }
        
        self.results["dictator_game"] = experiment_results
        
        # Print summary
        print(f"\nDictator Game Summary:")
        print(f"  Total rounds: {summary_stats.get('total_rounds', 0)}")
        print(f"  Average allocation: {summary_stats.get('avg_allocation', 0):.2f}")
        print(f"  Average generosity: {summary_stats.get('avg_generosity_ratio', 0):.2%}")
        print(f"  Fair splits: {summary_stats.get('fair_splits', 0):.2%}")
        print(f"  Fully generous: {summary_stats.get('fully_generous', 0):.2%}")
        print(f"  Fully selfish: {summary_stats.get('fully_selfish', 0):.2%}")
        
        print(f"\nTop 3 Most Generous Agents:")
        for i, ranking in enumerate(rankings[:3]):
            print(f"  {i+1}. {ranking['name']}: {ranking['avg_generosity']:.2%}")
        
        if personality_correlations:
            print(f"\nPersonality-Giving Correlations:")
            for trait, stats in personality_correlations.items():
                print(f"  {trait}: {stats['avg_generosity']:.2%} (n={stats['count']})")
        
        return experiment_results
    
    async def run_mixed_experiment(self) -> Dict[str, Any]:
        """Run a mixed experiment with both games using concurrent execution."""
        print("\n" + "="*50)
        print("RUNNING MIXED EXPERIMENT (CONCURRENT)")
        print("="*50)
        
        # Create two separate agent groups for concurrent experiments
        mid_point = len(self.agents) // 2
        group1 = self.agents[:mid_point]
        group2 = self.agents[mid_point:]
        
        # Setup experiments
        trust_game = MultiTrustGame(group1, self.config["experiment_config"])
        dictator_game = MultiDictatorGame(group2, self.config["experiment_config"])
        
        # Run experiments concurrently
        num_rounds = self.config["experiment_config"]["num_rounds"]
        
        print(f"Running trust game with {len(group1)} agents...")
        print(f"Running dictator game with {len(group2)} agents...")
        
        # Execute concurrently
        trust_task = asyncio.create_task(trust_game.run_tournament(num_rounds))
        dictator_task = asyncio.create_task(dictator_game.run_tournament(num_rounds // 2))
        
        # Wait for both to complete
        trust_results, dictator_results = await asyncio.gather(trust_task, dictator_task)
        
        mixed_results = {
            "experiment_type": "mixed_concurrent",
            "timestamp": datetime.now().isoformat(),
            "config": self.config["experiment_config"],
            "trust_game": {
                "results": trust_results,
                "summary_stats": trust_game.trust_game.get_summary_stats(),
                "rankings": trust_game.get_agent_rankings()
            },
            "dictator_game": {
                "results": dictator_results,
                "summary_stats": dictator_game.dictator_game.get_summary_stats(),
                "rankings": dictator_game.get_agent_rankings(),
                "personality_correlations": dictator_game.analyze_personality_correlations()
            }
        }
        
        self.results["mixed"] = mixed_results
        
        print(f"\nMixed Experiment Completed:")
        print(f"  Trust game rounds: {len(trust_results)}")
        print(f"  Dictator game rounds: {len(dictator_results)}")
        
        return mixed_results
    
    async def run_reflection_session(self) -> None:
        """Run a reflection session where agents discuss their experiences."""
        print("\n" + "="*50)
        print("RUNNING REFLECTION SESSION")
        print("="*50)
        
        async with MsgHub(
            participants=self.agents,
            announcement=Msg(
                "moderator",
                "Welcome to the reflection session! Please share your experiences "
                "from the economic games. What did you learn about trust, fairness, "
                "and decision-making? How did your personality influence your choices?",
                "system"
            )
        ) as hub:
            
            # Each agent reflects on their experience
            reflection_tasks = []
            for agent in self.agents:
                task = asyncio.create_task(
                    self._agent_reflection(agent, hub)
                )
                reflection_tasks.append(task)
            
            # Run reflections concurrently with some delay
            for i, task in enumerate(reflection_tasks):
                await task
                if i < len(reflection_tasks) - 1:
                    await asyncio.sleep(0.5)  # Small delay between reflections
    
    async def _agent_reflection(self, agent: BDITrustAgent, hub: MsgHub) -> None:
        """Individual agent reflection."""
        reflection_prompt = f"""
        Please reflect on your experience in the economic games:
        
        1. What did you learn about trust and fairness?
        2. How did your personality traits influence your decisions?
        3. What strategies did you use?
        4. How did other agents' behavior affect your choices?
        5. What would you do differently next time?
        
        Your personality: {agent.personality_traits}
        Your final endowment: {agent.endowment}
        Games played: {len(agent.game_history)}
        
        Share your thoughts in a thoughtful way.
        """
        
        reflection_msg = await agent.reply(
            context=reflection_prompt,
            game_type="reflection",
            other_players=[a.name for a in self.agents if a != agent]
        )
        
        # Broadcast reflection to all agents
        await hub.broadcast(reflection_msg)
    
    async def run_full_simulation(self) -> Dict[str, Any]:
        """Run the complete simulation with all experiments."""
        print("Starting Trust Behavior Simulation")
        print("="*60)
        
        # Setup agents
        await self.setup_agents()
        
        # Run experiments based on configuration
        experiments = self.config["experiment_config"]["experiments"]
        
        if "trust_game" in experiments:
            await self.run_trust_game_experiment()
        
        if "dictator_game" in experiments:
            await self.run_dictator_game_experiment()
        
        if "mixed" in experiments or len(experiments) > 1:
            await self.run_mixed_experiment()
        
        # Run reflection session
        await self.run_reflection_session()
        
        # Save results
        await self.data_saver.save_results(self.results)
        
        print("\n" + "="*60)
        print("SIMULATION COMPLETED")
        print("="*60)
        print(f"Results saved to: {self.data_saver.results_dir}")
        
        return self.results
    
    def get_comprehensive_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive analysis of all results."""
        if not self.results:
            return {}
        
        analysis = {
            "simulation_summary": {
                "total_agents": len(self.agents),
                "experiments_run": list(self.results.keys()),
                "timestamp": datetime.now().isoformat()
            },
            "cross_experiment_analysis": {},
            "personality_analysis": {},
            "trust_behavior_patterns": {}
        }
        
        # Cross-experiment correlations
        if "trust_game" in self.results and "dictator_game" in self.results:
            trust_rankings = {r["name"]: r["total_profit"] 
                            for r in self.results["trust_game"]["rankings"]}
            generosity_rankings = {r["name"]: r["avg_generosity"] 
                                 for r in self.results["dictator_game"]["rankings"]}
            
            # Find agents who participated in both
            common_agents = set(trust_rankings.keys()) & set(generosity_rankings.keys())
            
            if common_agents:
                correlations = []
                for agent_name in common_agents:
                    correlations.append({
                        "agent": agent_name,
                        "trust_profit": trust_rankings[agent_name],
                        "generosity": generosity_rankings[agent_name]
                    })
                
                analysis["cross_experiment_analysis"]["trust_generosity_correlation"] = correlations
        
        # Personality-based analysis
        personality_groups = {}
        for agent in self.agents:
            primary_trait = max(agent.personality_traits.items(), key=lambda x: x[1])[0]
            if primary_trait not in personality_groups:
                personality_groups[primary_trait] = []
            personality_groups[primary_trait].append(agent.name)
        
        analysis["personality_analysis"]["groups"] = personality_groups
        
        return analysis