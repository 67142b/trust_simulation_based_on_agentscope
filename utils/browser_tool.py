"""
Browser tool for agents to access external knowledge during decision making.
"""
import asyncio
import json
import re
from typing import Dict, Any, List, Optional
import aiohttp
from bs4 import BeautifulSoup


class BrowserTool:
    """
    Browser tool for agents to search and retrieve web information.
    """
    
    def __init__(self, max_results: int = 5, timeout: int = 10):
        """
        Initialize browser tool.
        
        Args:
            max_results: Maximum number of search results
            timeout: Request timeout in seconds
        """
        self.max_results = max_results
        self.timeout = timeout
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def search_trust_game_info(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for information about trust games and economic experiments.
        
        Args:
            query: Search query
            
        Returns:
            List of search results with title, url, and snippet
        """
        # This is a mock implementation since we don't have real search API
        # In a real implementation, you would use Google Search API, Bing API, etc.
        
        mock_results = [
            {
                "title": "Trust Game - Wikipedia",
                "url": "https://en.wikipedia.org/wiki/Trust_game",
                "snippet": "The trust game is an economic experiment that measures trust in economic transactions.",
                "relevance": 0.9
            },
            {
                "title": "Dictator Game - Experimental Economics",
                "url": "https://en.wikipedia.org/wiki/Dictator_game",
                "snippet": "The dictator game is a simple economic experiment used to study fairness and generosity.",
                "relevance": 0.8
            },
            {
                "title": "Big Five Personality Traits and Economic Behavior",
                "url": "https://example.com/personality-economics",
                "snippet": "Research on how personality traits influence economic decision-making and trust behavior.",
                "relevance": 0.7
            }
        ]
        
        return mock_results[:self.max_results]
    
    async def fetch_page_content(self, url: str) -> Optional[str]:
        """
        Fetch content from a URL.
        
        Args:
            url: URL to fetch
            
        Returns:
            Page content as string or None if failed
        """
        if not self.session:
            return None
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    return self._extract_main_content(content)
        except Exception as e:
            print(f"Error fetching {url}: {e}")
        
        return None
    
    def _extract_main_content(self, html: str) -> str:
        """
        Extract main content from HTML.
        
        Args:
            html: HTML content
            
        Returns:
            Extracted text content
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text from main content areas
            content_areas = [
                soup.find("main"),
                soup.find("article"),
                soup.find("div", class_="content"),
                soup.find("div", class_="main"),
                soup.body
            ]
            
            for area in content_areas:
                if area:
                    text = area.get_text()
                    # Clean up text
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text = ' '.join(chunk for chunk in chunks if chunk)
                    
                    # Limit length
                    if len(text) > 2000:
                        text = text[:2000] + "..."
                    
                    return text
            
            return ""
        except Exception:
            return ""
    
    async def get_trust_game_strategies(self) -> str:
        """
        Get information about trust game strategies.
        
        Returns:
            Strategy information as string
        """
        strategies = """
        Trust Game Strategies:
        
        1. Reciprocal Strategy: Return a proportion of what you receive (typically 50-100%)
        2. Fairness Strategy: Aim for equal outcomes between both players
        3. Profit Maximization: Keep most of the returns for yourself
        4. Trust Building: Return more than expected to build future trust
        5. Cautious Approach: Return minimal amounts to minimize risk
        
        Factors influencing decisions:
        - Personality traits (agreeableness, conscientiousness)
        - Previous interactions and reputation
        - Cultural norms and fairness expectations
        - Risk tolerance and future interaction expectations
        """
        
        return strategies.strip()
    
    async def get_dictator_game_insights(self) -> str:
        """
        Get insights about dictator game behavior.
        
        Returns:
            Dictator game insights as string
        """
        insights = """
        Dictator Game Research Insights:
        
        Typical Behavior Patterns:
        1. Fair Split (40-60%): Most common, reflects social norms
        2. Selfish (0-20%): Pure self-interest
        3. Generous (80-100%): Strong altruism or social signaling
        4. Minimal Giving (1-10%): Token gestures
        
        Influencing Factors:
        - Anonymity effects (less anonymous = more giving)
        - Social distance (closer = more giving)
        - Framing effects (how the game is presented)
        - Personality traits (agreeableness, empathy)
        - Cultural background
        - Presence of observers
        
        Average giving across studies: 20-30% of total amount
        """
        
        return insights.strip()
    
    async def get_personality_economics_info(self, personality: Dict[str, float]) -> str:
        """
        Get information about how specific personality traits affect economic behavior.
        
        Args:
            personality: Personality trait scores
            
        Returns:
            Personality-economics information
        """
        info = []
        
        if personality.get("agreeableness", 0) > 0.7:
            info.append("High agreeableness suggests more cooperative and fair behavior in economic games.")
        
        if personality.get("conscientiousness", 0) > 0.7:
            info.append("High conscientiousness indicates adherence to social norms and consistent decision-making.")
        
        if personality.get("openness", 0) > 0.7:
            info.append("High openness suggests willingness to try novel strategies and consider different perspectives.")
        
        if personality.get("extraversion", 0) > 0.7:
            info.append("High extraversion may lead to more social considerations in decision-making.")
        
        if personality.get("neuroticism", 0) > 0.7:
            info.append("High neuroticism suggests risk-averse behavior and concern about negative outcomes.")
        
        return " ".join(info) if info else "Personality traits have balanced influence on economic behavior."


class EnhancedBrowserTool(BrowserTool):
    """
    Enhanced browser tool with caching and advanced search capabilities.
    """
    
    def __init__(self, max_results: int = 5, timeout: int = 10, cache_size: int = 100):
        """
        Initialize enhanced browser tool.
        
        Args:
            max_results: Maximum number of search results
            timeout: Request timeout in seconds
            cache_size: Maximum number of cached results
        """
        super().__init__(max_results, timeout)
        self.cache = {}
        self.cache_size = cache_size
    
    async def search_with_cache(self, query: str) -> List[Dict[str, Any]]:
        """
        Search with caching to avoid duplicate requests.
        
        Args:
            query: Search query
            
        Returns:
            Cached or fresh search results
        """
        if query in self.cache:
            return self.cache[query]
        
        results = await self.search_trust_game_info(query)
        
        # Add to cache (simple LRU would be better for production)
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[query] = results
        return results
    
    async def get_contextual_advice(
        self, 
        game_type: str, 
        personality: Dict[str, float],
        situation: str
    ) -> str:
        """
        Get contextual advice based on game type, personality, and situation.
        
        Args:
            game_type: Type of game (trust_game, dictator_game)
            personality: Personality traits
            situation: Current situation description
            
        Returns:
            Contextual advice
        """
        advice_parts = []
        
        # Get general game information
        if game_type == "trust_game":
            advice_parts.append(await self.get_trust_game_strategies())
        elif game_type == "dictator_game":
            advice_parts.append(await self.get_dictator_game_insights())
        
        # Get personality-specific advice
        personality_advice = await self.get_personality_economics_info(personality)
        advice_parts.append(f"Personality Considerations: {personality_advice}")
        
        # Add situational advice based on keywords
        if "first round" in situation.lower():
            advice_parts.append("First rounds often involve signaling and reputation building.")
        
        if "low trust" in situation.lower() or "suspicious" in situation.lower():
            advice_parts.append("In low-trust situations, consider building trust through generous initial moves.")
        
        if "repeated interaction" in situation.lower():
            advice_parts.append("Repeated interactions allow for reputation effects and reciprocal strategies.")
        
        return "\n\n".join(advice_parts)