"""
Factory class for creating BDI agents with diverse personalities.
"""
import random
import asyncio
from typing import List, Dict, Any

from agents.bdi_agent import BDITrustAgent


class AgentFactory:
    """
    Factory for creating BDI agents with diverse personalities and backgrounds.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize agent factory.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_config = config["model_config"]
        self.agent_config = config["agent_config"]
        
        # Sample data for agent generation
        self.names = [
            "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry",
            "Iris", "Jack", "Kate", "Leo", "Maya", "Noah", "Olivia", "Peter",
            "Quinn", "Rachel", "Sam", "Tina", "Uma", "Victor", "Wendy", "Xavier"
        ]
        
        self.education_levels = [
            "High School", "Bachelor's Degree", "Master's Degree", "PhD", "MBA"
        ]
        
        self.professions = [
            "Teacher", "Engineer", "Doctor", "Artist", "Entrepreneur", 
            "Scientist", "Manager", "Consultant", "Designer", "Writer"
        ]
    
    async def create_agent_pool(self, num_agents: int) -> List[BDITrustAgent]:
        """
        Create a pool of diverse BDI agents.
        
        Args:
            num_agents: Number of agents to create
            
        Returns:
            List of BDI agents
        """
        agents = []
        
        # Create diverse personality profiles
        personality_profiles = self._generate_personality_profiles(num_agents)
        
        for i in range(num_agents):
            agent = await self._create_single_agent(i, personality_profiles[i])
            agents.append(agent)
        
        return agents
    
    def _generate_personality_profiles(self, num_agents: int) -> List[Dict[str, float]]:
        """
        Generate diverse personality profiles using Big Five traits.
        
        Args:
            num_agents: Number of profiles to generate
            
        Returns:
            List of personality trait dictionaries
        """
        profiles = []
        
        # Define some archetypal personality profiles
        archetypes = [
            # High agreeableness, high openness - The Altruist
            {"openness": 0.8, "conscientiousness": 0.6, "extraversion": 0.7, 
             "agreeableness": 0.9, "neuroticism": 0.3},
            
            # Low agreeableness, high conscientiousness - The Pragmatist
            {"openness": 0.4, "conscientiousness": 0.9, "extraversion": 0.5, 
             "agreeableness": 0.2, "neuroticism": 0.4},
            
            # High extraversion, low neuroticism - The Optimist
            {"openness": 0.7, "conscientiousness": 0.6, "extraversion": 0.9, 
             "agreeableness": 0.7, "neuroticism": 0.1},
            
            # High neuroticism, low extraversion - The Anxious
            {"openness": 0.5, "conscientiousness": 0.4, "extraversion": 0.2, 
             "agreeableness": 0.6, "neuroticism": 0.9},
            
            # Balanced profile - The Moderate
            {"openness": 0.5, "conscientiousness": 0.5, "extraversion": 0.5, 
             "agreeableness": 0.5, "neuroticism": 0.5},
            
            # High openness, high neuroticism - The Creative Worrier
            {"openness": 0.9, "conscientiousness": 0.4, "extraversion": 0.6, 
             "agreeableness": 0.5, "neuroticism": 0.8},
            
            # Low openness, high agreeableness - The Traditionalist
            {"openness": 0.2, "conscientiousness": 0.8, "extraversion": 0.4, 
             "agreeableness": 0.8, "neuroticism": 0.3},
            
            # High extraversion, low agreeableness - The Dominant
            {"openness": 0.6, "conscientiousness": 0.7, "extraversion": 0.9, 
             "agreeableness": 0.2, "neuroticism": 0.4}
        ]
        
        for i in range(num_agents):
            if i < len(archetypes):
                # Use predefined archetype
                base_profile = archetypes[i % len(archetypes)].copy()
            else:
                # Generate random profile
                base_profile = {
                    "openness": random.uniform(0.1, 0.9),
                    "conscientiousness": random.uniform(0.1, 0.9),
                    "extraversion": random.uniform(0.1, 0.9),
                    "agreeableness": random.uniform(0.1, 0.9),
                    "neuroticism": random.uniform(0.1, 0.9)
                }
            
            # Add some random variation
            for trait in base_profile:
                variation = random.uniform(-0.1, 0.1)
                base_profile[trait] = max(0.0, min(1.0, base_profile[trait] + variation))
            
            profiles.append(base_profile)
        
        return profiles
    
    async def _create_single_agent(
        self, 
        index: int, 
        personality: Dict[str, float]
    ) -> BDITrustAgent:
        """
        Create a single BDI agent.
        
        Args:
            index: Agent index
            personality: Personality traits
            
        Returns:
            BDI agent instance
        """
        # Generate agent attributes
        name = self.names[index % len(self.names)]
        if index >= len(self.names):
            name += f"_{index // len(self.names) + 1}"
        
        age = random.randint(18, 65)
        education = random.choice(self.education_levels)
        initial_endowment = self.config["experiment_config"]["initial_endowment"]
        
        # Create agent
        agent = BDITrustAgent(
            name=name,
            age=age,
            personality_traits=personality,
            education=education,
            initial_endowment=initial_endowment,
            model_config=self.model_config,
            agent_config=self.agent_config
        )
        
        # Initialize desires based on personality
        await self._initialize_agent_desires(agent, personality)
        
        return agent
    
    async def _initialize_agent_desires(
        self, 
        agent: BDITrustAgent, 
        personality: Dict[str, float]
    ) -> None:
        """
        Initialize agent desires based on personality traits.
        
        Args:
            agent: The agent to initialize
            personality: Personality traits
        """
        # Base desires for all agents
        agent.add_desire("Maximize personal earnings", priority=0.8, urgency=0.0)
        
        # Personality-based desires
        if personality["agreeableness"] > 0.7:
            agent.add_desire("Be fair to others", priority=0.9, urgency=0.0)
            agent.add_desire("Maintain good relationships", priority=0.8, urgency=0.0)
        
        if personality["openness"] > 0.7:
            agent.add_desire("Explore new strategies", priority=0.6, urgency=0.0)
            agent.add_desire("Learn from others", priority=0.7, urgency=0.0)
        
        if personality["conscientiousness"] > 0.7:
            agent.add_desire("Follow social norms", priority=0.8, urgency=0.0)
            agent.add_desire("Be consistent in decisions", priority=0.7, urgency=0.0)
        
        if personality["extraversion"] > 0.7:
            agent.add_desire("Social interaction", priority=0.6, urgency=0.0)
            agent.add_desire("Be influential", priority=0.5, urgency=0.0)
        
        if personality["neuroticism"] > 0.7:
            agent.add_desire("Avoid risks", priority=0.8, urgency=0.0)
            agent.add_desire("Seek security", priority=0.9, urgency=0.0)
        else:
            agent.add_desire("Take calculated risks", priority=0.6, urgency=0.0)
        
        # Add some initial beliefs
        agent.beliefs = [
            # These will be populated during the game
        ]
    
    def create_specialized_agent(
        self, 
        agent_type: str, 
        **kwargs
    ) -> BDITrustAgent:
        """
        Create a specialized agent for specific roles.
        
        Args:
            agent_type: Type of agent to create
            **kwargs: Additional parameters
            
        Returns:
            Specialized BDI agent
        """
        if agent_type == "trusting":
            personality = {
                "openness": 0.7, "conscientiousness": 0.6, "extraversion": 0.8,
                "agreeableness": 0.9, "neuroticism": 0.3
            }
        elif agent_type == "skeptical":
            personality = {
                "openness": 0.4, "conscientiousness": 0.8, "extraversion": 0.3,
                "agreeableness": 0.2, "neuroticism": 0.6
            }
        elif agent_type == "generous":
            personality = {
                "openness": 0.8, "conscientiousness": 0.5, "extraversion": 0.7,
                "agreeableness": 0.9, "neuroticism": 0.4
            }
        elif agent_type == "selfish":
            personality = {
                "openness": 0.3, "conscientiousness": 0.7, "extraversion": 0.5,
                "agreeableness": 0.1, "neuroticism": 0.5
            }
        else:
            # Random personality
            personality = {
                "openness": random.uniform(0.1, 0.9),
                "conscientiousness": random.uniform(0.1, 0.9),
                "extraversion": random.uniform(0.1, 0.9),
                "agreeableness": random.uniform(0.1, 0.9),
                "neuroticism": random.uniform(0.1, 0.9)
            }
        
        # Override with provided parameters
        personality.update(kwargs.get("personality", {}))
        
        return BDITrustAgent(
            name=kwargs.get("name", f"Agent_{agent_type}"),
            age=kwargs.get("age", random.randint(18, 65)),
            personality_traits=personality,
            education=kwargs.get("education", random.choice(self.education_levels)),
            initial_endowment=kwargs.get("endowment", self.config["experiment_config"]["initial_endowment"]),
            model_config=self.model_config,
            agent_config=self.agent_config
        )