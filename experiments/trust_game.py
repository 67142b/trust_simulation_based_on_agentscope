"""
Trust Game implementation for multi-agent simulation.
"""
import asyncio
import json
from typing import List, Dict, Any, Tuple
import numpy as np
from dataclasses import dataclass

from agentscope.message import Msg
from agentscope.pipeline import MsgHub, sequential_pipeline
from agents.bdi_agent import BDITrustAgent


@dataclass
class TrustGameResult:
    """Result of a trust game round."""
    investor: str
    trustee: str
    investment_amount: float
    return_amount: float
    investor_profit: float
    trustee_profit: float
    round_number: int


class TrustGame:
    """
    Trust Game implementation where one agent (investor) can invest money
    in another agent (trustee), and the trustee can return a portion.
    """
    
    def __init__(
        self,
        multiplier: float = 3.0,
        min_amount: float = 0.0,
        max_amount: float = 100.0
    ):
        """
        Initialize trust game.
        
        Args:
            multiplier: Multiplier for invested amount
            min_amount: Minimum investment amount
            max_amount: Maximum investment amount
        """
        self.multiplier = multiplier
        self.min_amount = min_amount
        self.max_amount = max_amount
        self.results: List[TrustGameResult] = []
    
    async def play_round(
        self,
        investor: BDITrustAgent,
        trustee: BDITrustAgent,
        round_number: int = 1
    ) -> TrustGameResult:
        """
        Play one round of trust game.
        
        Args:
            investor: The agent who invests money
            trustee: The agent who receives and returns money
            round_number: Current round number
            
        Returns:
            TrustGameResult with the outcome
        """
        # Step 1: Investor decides how much to invest
        investment_context = f"""
        You are the INVESTOR in a trust game.
        Your current endowment: {investor.endowment}
        The trustee's endowment: {trustee.endowment}
        
        Rules:
        - You can invest between {self.min_amount} and {self.max_amount}
        - The invested amount will be multiplied by {self.multiplier}
        - The trustee will decide how much to return to you
        
        Your history with this player: {self._get_interaction_history(investor, trustee)}
        """
        
        investment_msg = await investor.reply(
            context=investment_context,
            game_type="trust_game_investment",
            other_players=[trustee.name]
        )
        
        # Parse investment amount
        investment_amount = self._parse_amount(investment_msg.content, investor.endowment)
        investment_amount = max(self.min_amount, min(self.max_amount, investment_amount))
        
        # Update investor's endowment
        investor.endowment -= investment_amount
        
        # Step 2: Trustee receives investment and decides return amount
        total_received = investment_amount * self.multiplier
        trustee.endowment += total_received
        
        return_context = f"""
        You are the TRUSTEE in a trust game.
        You received {total_received} from {investor.name} (original investment: {investment_amount})
        Your current endowment: {trustee.endowment}
        
        Rules:
        - You can return any amount between 0 and {total_received}
        - Consider fairness, trust, and future interactions
        
        Your history with this player: {self._get_interaction_history(trustee, investor)}
        """
        
        return_msg = await trustee.reply(
            context=return_context,
            game_type="trust_game_return",
            other_players=[investor.name]
        )
        
        # Parse return amount
        return_amount = self._parse_amount(return_msg.content, total_received)
        return_amount = max(0, min(total_received, return_amount))
        
        # Update endowments
        trustee.endowment -= return_amount
        investor.endowment += return_amount
        
        # Calculate profits
        investor_profit = return_amount - investment_amount
        trustee_profit = total_received - return_amount
        
        # Create result
        result = TrustGameResult(
            investor=investor.name,
            trustee=trustee.name,
            investment_amount=investment_amount,
            return_amount=return_amount,
            investor_profit=investor_profit,
            trustee_profit=trustee_profit,
            round_number=round_number
        )
        
        self.results.append(result)
        
        # Agents observe each other's actions
        await investor.observe(Msg(
            name=trustee.name,
            content=f"I returned {return_amount} to you.",
            role="assistant"
        ))
        
        await trustee.observe(Msg(
            name=investor.name,
            content=f"I invested {investment_amount} in you.",
            role="assistant"
        ))
        
        return result
    
    def _parse_amount(self, content: str, max_amount: float) -> float:
        """Parse monetary amount from agent response."""
        content = content[0]['text']
        try:
            # Try to parse JSON response
            #print("content:", content)
            response_data = json.loads(content)
            amount = response_data.get("amount", 0)
        except json.JSONDecodeError:
            # Fallback: extract number from text
            import re
            numbers = re.findall(r'\d+\.?\d*', content)
            amount = float(numbers[0]) if numbers else 0
        
        return float(amount)
    
    def _get_interaction_history(self, agent1: BDITrustAgent, agent2: BDITrustAgent) -> str:
        """Get interaction history between two agents."""
        history = []
        for result in self.results:
            if (result.investor == agent1.name and result.trustee == agent2.name) or \
               (result.investor == agent2.name and result.trustee == agent1.name):
                history.append(
                    f"Round {result.round_number}: "
                    f"Investment: {result.investment_amount}, "
                    f"Return: {result.return_amount}"
                )
        
        return "; ".join(history) if history else "No previous interactions"
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of all played games."""
        if not self.results:
            return {}
        
        investments = [r.investment_amount for r in self.results]
        returns = [r.return_amount for r in self.results]
        investor_profits = [r.investor_profit for r in self.results]
        trustee_profits = [r.trustee_profit for r in self.results]
        
        return {
            "total_rounds": len(self.results),
            "avg_investment": np.mean(investments),
            "avg_return": np.mean(returns),
            "avg_investor_profit": np.mean(investor_profits),
            "avg_trustee_profit": np.mean(trustee_profits),
            "trust_rate": np.mean([r.return_amount > 0 for r in self.results]),
            "profitable_investments": np.mean([r.investor_profit > 0 for r in self.results])
        }


class MultiTrustGame:
    """
    Multi-agent trust game tournament using MsgHub for coordination.
    """
    
    def __init__(
        self,
        agents: List[BDITrustAgent],
        game_config: Dict[str, Any]
    ):
        """
        Initialize multi-agent trust game.
        
        Args:
            agents: List of agents to participate
            game_config: Game configuration parameters
        """
        self.agents = agents
        self.game_config = game_config
        self.trust_game = TrustGame(
            multiplier=game_config.get("trust_game_multiplier", 3.0)
        )
        self.tournament_results: List[TrustGameResult] = []
    
    async def run_tournament(self, num_rounds: int = 5) -> List[TrustGameResult]:
        """
        Run a tournament where each agent plays against each other.
        
        Args:
            num_rounds: Number of rounds per pair
            
        Returns:
            List of all game results
        """
        print(f"Starting Trust Game Tournament with {len(self.agents)} agents")
        
        # Use MsgHub for coordination
        async with MsgHub(
            participants=self.agents,
            announcement=Msg(
                "system",
                "Welcome to the Trust Game Tournament! You will play multiple rounds "
                "of trust games against other agents. Make decisions based on your "
                "personality, beliefs, and desires.",
                "system"
            )
        ) as hub:
            
            # Create tasks for all pairs to run concurrently
            tasks = []
            semaphore = asyncio.Semaphore(10)  # Limit concurrent executions to prevent resource exhaustion
            
            # Create tasks for each pair
            for i, agent1 in enumerate(self.agents):
                for j, agent2 in enumerate(self.agents):
                    if i != j:  # Don't play against yourself
                        task = asyncio.create_task(
                            self._play_pair_series_with_semaphore(agent1, agent2, num_rounds, hub, semaphore)
                        )
                        tasks.append(task)
            
            # Run all tasks concurrently
            await asyncio.gather(*tasks)
        
        self.tournament_results = self.trust_game.results
        return self.tournament_results
    
    async def _play_pair_series_with_semaphore(
        self,
        agent1: BDITrustAgent,
        agent2: BDITrustAgent,
        num_rounds: int,
        hub: MsgHub,
        semaphore: asyncio.Semaphore
    ) -> None:
        """Play a series of games between two agents with semaphore control."""
        async with semaphore:
            await self._play_pair_series(agent1, agent2, num_rounds, hub)
    
    async def _play_pair_series(
        self,
        agent1: BDITrustAgent,
        agent2: BDITrustAgent,
        num_rounds: int,
        hub: MsgHub
    ) -> None:
        """Play a series of games between two agents."""
        
        for round_num in range(1, num_rounds + 1):
            print(f"Round {round_num}: {agent1.name} vs {agent2.name}")
            
            # Alternate roles: agent1 as investor, agent2 as trustee
            result1 = await self.trust_game.play_round(agent1, agent2, round_num)
            
            # Announce results to all agents
            await hub.broadcast(Msg(
                "game_master",
                f"Round {round_num} result: {agent1.name} invested {result1.investment_amount}, "
                f"{agent2.name} returned {result1.return_amount}",
                "system"
            ))
            
            # Small delay between rounds
            await asyncio.sleep(0.1)
    
    def get_agent_rankings(self) -> List[Dict[str, Any]]:
        """Get rankings of agents by total profit."""
        agent_stats = {}
        
        for agent in self.agents:
            total_profit = 0
            games_played = 0
            
            for result in self.tournament_results:
                if result.investor == agent.name:
                    total_profit += result.investor_profit
                    games_played += 1
                elif result.trustee == agent.name:
                    total_profit += result.trustee_profit
                    games_played += 1
            
            agent_stats[agent.name] = {
                "total_profit": total_profit,
                "avg_profit": total_profit / games_played if games_played > 0 else 0,
                "games_played": games_played
            }
        
        # Sort by total profit
        rankings = sorted(
            agent_stats.items(),
            key=lambda x: x[1]["total_profit"],
            reverse=True
        )
        
        return [{"name": name, **stats} for name, stats in rankings]