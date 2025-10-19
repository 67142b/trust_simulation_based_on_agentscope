"""
Dictator Game implementation for multi-agent simulation.
"""
import asyncio
import json
from typing import List, Dict, Any, Tuple
import numpy as np
from dataclasses import dataclass

from agentscope.message import Msg
from agentscope.pipeline import MsgHub
from agents.bdi_agent import BDITrustAgent


@dataclass
class DictatorGameResult:
    """Result of a dictator game round."""
    dictator: str
    receiver: str
    total_amount: float
    allocated_amount: float
    kept_amount: float
    generosity_ratio: float
    round_number: int


class DictatorGame:
    """
    Dictator Game implementation where one agent (dictator) decides
    how to split a sum of money with another agent (receiver).
    """
    
    def __init__(
        self,
        total_amount: float = 100.0,
        min_allocation: float = 0.0,
        max_allocation: float = 100.0
    ):
        """
        Initialize dictator game.
        
        Args:
            total_amount: Total amount to be split
            min_allocation: Minimum amount to allocate to receiver
            max_allocation: Maximum amount to allocate to receiver
        """
        self.total_amount = total_amount
        self.min_allocation = min_allocation
        self.max_allocation = max_allocation
        self.results: List[DictatorGameResult] = []
    
    async def play_round(
        self,
        dictator: BDITrustAgent,
        receiver: BDITrustAgent,
        round_number: int = 1
    ) -> DictatorGameResult:
        """
        Play one round of dictator game.
        
        Args:
            dictator: The agent who decides the split
            receiver: The agent who receives the allocation
            round_number: Current round number
            
        Returns:
            DictatorGameResult with the outcome
        """
        # Dictator decides how much to allocate
        context = f"""
        You are the DICTATOR in a dictator game.
        You have {self.total_amount} to split between yourself and {receiver.name}.
        
        Rules:
        - You can allocate between {self.min_allocation} and {self.max_allocation} to {receiver.name}
        - You keep the remaining amount
        - The receiver has no say in this decision
        - Consider fairness, empathy, and your personal values
        
        Your personality traits influence your decision:
        - High agreeableness: More likely to be generous
        - High openness: More likely to consider different perspectives
        - High conscientiousness: More likely to follow social norms
        
        Your history with this player: {self._get_interaction_history(dictator, receiver)}
        """
        
        decision_msg = await dictator.reply(
            context=context,
            game_type="dictator_game",
            other_players=[receiver.name]
        )
        
        # Parse allocation amount
        allocated_amount = self._parse_amount(decision_msg.content)
        allocated_amount = max(self.min_allocation, min(self.max_allocation, allocated_amount))
        
        kept_amount = self.total_amount - allocated_amount
        generosity_ratio = allocated_amount / self.total_amount
        
        # Update endowments
        dictator.endowment += kept_amount
        receiver.endowment += allocated_amount
        
        # Create result
        result = DictatorGameResult(
            dictator=dictator.name,
            receiver=receiver.name,
            total_amount=self.total_amount,
            allocated_amount=allocated_amount,
            kept_amount=kept_amount,
            generosity_ratio=generosity_ratio,
            round_number=round_number
        )
        
        self.results.append(result)
        
        # Agents observe the outcome
        await dictator.observe(Msg(
            name="system",
            content=f"You allocated {allocated_amount} to {receiver.name} and kept {kept_amount}.",
            role="system"
        ))
        
        await receiver.observe(Msg(
            name=dictator.name,
            content=f"I allocated {allocated_amount} to you from the total of {self.total_amount}.",
            role="assistant"
        ))
        
        return result
    
    def _parse_amount(self, content: str) -> float:
        """Parse monetary amount from agent response."""
        #print(content)
        content = content[0]["text"]
        try:
            # Try to parse JSON response
            response_data = json.loads(content)
            amount = response_data.get("amount", self.total_amount / 2)
        except json.JSONDecodeError:
            # Fallback: extract number from text
            import re
            numbers = re.findall(r'\d+\.?\d*', content)
            amount = float(numbers[0]) if numbers else self.total_amount / 2
        
        return float(amount)
    
    def _get_interaction_history(self, agent1: BDITrustAgent, agent2: BDITrustAgent) -> str:
        """Get interaction history between two agents."""
        history = []
        for result in self.results:
            if result.dictator == agent1.name and result.receiver == agent2.name:
                history.append(
                    f"Round {result.round_number}: "
                    f"Allocated {result.allocated_amount}/{result.total_amount} "
                    f"(generosity: {result.generosity_ratio:.2f})"
                )
        
        return "; ".join(history) if history else "No previous interactions"
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of all played games."""
        if not self.results:
            return {}
        
        allocations = [r.allocated_amount for r in self.results]
        generosity_ratios = [r.generosity_ratio for r in self.results]
        
        return {
            "total_rounds": len(self.results),
            "avg_allocation": np.mean(allocations),
            "avg_generosity_ratio": np.mean(generosity_ratios),
            "max_allocation": np.max(allocations),
            "min_allocation": np.min(allocations),
            "fully_generous": np.mean([r.generosity_ratio == 1.0 for r in self.results]),
            "fully_selfish": np.mean([r.generosity_ratio == 0.0 for r in self.results]),
            "fair_splits": np.mean([0.4 <= r.generosity_ratio <= 0.6 for r in self.results])
        }


class MultiDictatorGame:
    """
    Multi-agent dictator game tournament using MsgHub for coordination.
    """
    
    def __init__(
        self,
        agents: List[BDITrustAgent],
        game_config: Dict[str, Any]
    ):
        """
        Initialize multi-agent dictator game.
        
        Args:
            agents: List of agents to participate
            game_config: Game configuration parameters
        """
        self.agents = agents
        self.game_config = game_config
        self.dictator_game = DictatorGame(
            total_amount=game_config.get("initial_endowment", 100.0)
        )
        self.tournament_results: List[DictatorGameResult] = []
    
    async def run_tournament(self, num_rounds: int = 3) -> List[DictatorGameResult]:
        """
        Run a tournament where each agent plays as dictator against others.
        
        Args:
            num_rounds: Number of rounds per pair
            
        Returns:
            List of all game results
        """
        print(f"Starting Dictator Game Tournament with {len(self.agents)} agents")
        
        # Use MsgHub for coordination
        async with MsgHub(
            participants=self.agents,
            announcement=Msg(
                "system",
                "Welcome to the Dictator Game Tournament! Each agent will take turns "
                "being the dictator and deciding how to split money with other agents. "
                "This game measures generosity and fairness.",
                "system"
            )
        ) as hub:
            
            # Create tasks for all pairs to run concurrently
            tasks = []
            semaphore = asyncio.Semaphore(10)  # Limit concurrent executions to prevent resource exhaustion
            
            # Create tasks for each pair
            for dictator in self.agents:
                for receiver in self.agents:
                    if dictator != receiver:  # Don't play against yourself
                        task = asyncio.create_task(
                            self._play_pair_series_with_semaphore(dictator, receiver, num_rounds, hub, semaphore)
                        )
                        tasks.append(task)
            
            # Run all tasks concurrently
            await asyncio.gather(*tasks)
        
        self.tournament_results = self.dictator_game.results
        return self.tournament_results
    
    async def _play_pair_series_with_semaphore(
        self,
        dictator: BDITrustAgent,
        receiver: BDITrustAgent,
        num_rounds: int,
        hub: MsgHub,
        semaphore: asyncio.Semaphore
    ) -> None:
        """Play a series of games between two agents with semaphore control."""
        async with semaphore:
            await self._play_pair_series(dictator, receiver, num_rounds, hub)
    
    async def _play_pair_series(
        self,
        dictator: BDITrustAgent,
        receiver: BDITrustAgent,
        num_rounds: int,
        hub: MsgHub
    ) -> None:
        """Play a series of games between two agents."""
        
        for round_num in range(1, num_rounds + 1):
            print(f"Round {round_num}: {dictator.name} (dictator) -> {receiver.name} (receiver)")
            
            result = await self.dictator_game.play_round(dictator, receiver, round_num)
            
            # Announce results to all agents
            await hub.broadcast(Msg(
                "game_master",
                f"Dictator Game Round {round_num}: {dictator.name} allocated "
                f"{result.allocated_amount}/{result.total_amount} to {receiver.name} "
                f"(generosity: {result.generosity_ratio:.2f})",
                "system"
            ))
            
            # Small delay between rounds
            await asyncio.sleep(0.1)
    
    def get_agent_rankings(self) -> List[Dict[str, Any]]:
        """Get rankings of agents by generosity and total received."""
        agent_stats = {}
        
        # Initialize stats
        for agent in self.agents:
            agent_stats[agent.name] = {
                "total_given": 0,
                "total_received": 0,
                "avg_generosity": 0,
                "times_as_dictator": 0,
                "times_as_receiver": 0
            }
        
        # Calculate statistics
        for result in self.tournament_results:
            # Dictator stats
            if result.dictator in agent_stats:
                agent_stats[result.dictator]["total_given"] += result.allocated_amount
                agent_stats[result.dictator]["times_as_dictator"] += 1
            
            # Receiver stats
            if result.receiver in agent_stats:
                agent_stats[result.receiver]["total_received"] += result.allocated_amount
                agent_stats[result.receiver]["times_as_receiver"] += 1
        
        # Calculate averages
        for agent_name, stats in agent_stats.items():
            if stats["times_as_dictator"] > 0:
                stats["avg_generosity"] = stats["total_given"] / (
                    stats["times_as_dictator"] * self.dictator_game.total_amount
                )
        
        # Sort by average generosity
        rankings = sorted(
            agent_stats.items(),
            key=lambda x: x[1]["avg_generosity"],
            reverse=True
        )
        
        return [{"name": name, **stats} for name, stats in rankings]
    
    def analyze_personality_correlations(self) -> Dict[str, Any]:
        """Analyze correlations between personality traits and generosity."""
        # This would require more sophisticated statistical analysis
        # For now, return basic aggregation by personality traits
        
        personality_generosity = {}
        
        for result in self.tournament_results:
            # Find the dictator agent
            dictator_agent = None
            for agent in self.agents:
                if agent.name == result.dictator:
                    dictator_agent = agent
                    break
            
            if dictator_agent:
                # Categorize by primary personality trait
                primary_trait = max(
                    dictator_agent.personality_traits.items(),
                    key=lambda x: x[1]
                )[0]
                
                if primary_trait not in personality_generosity:
                    personality_generosity[primary_trait] = []
                
                personality_generosity[primary_trait].append(result.generosity_ratio)
        
        # Calculate averages
        correlations = {}
        for trait, ratios in personality_generosity.items():
            correlations[trait] = {
                "avg_generosity": np.mean(ratios),
                "count": len(ratios),
                "std_dev": np.std(ratios)
            }
        
        return correlations