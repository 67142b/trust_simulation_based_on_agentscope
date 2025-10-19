# -*- coding: utf-8 -*-
"""
BDI (Belief-Desire-Intention) Agent implementation for trust simulation.
"""
import json
import asyncio
import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import os
from agentscope.agent import AgentBase
from agentscope.message import Msg
from agentscope.model import DashScopeChatModel
from agentscope.memory import InMemoryMemory
from agentscope.memory import Mem0LongTermMemory  # 添加长期记忆导入
from agentscope.embedding import DashScopeTextEmbedding  # 添加嵌入模型导入
from agentscope.formatter import DashScopeMultiAgentFormatter,DashScopeChatFormatter
from utils.browser_tool import EnhancedBrowserTool


@dataclass
class Belief:
    """Agent's beliefs about the world and other agents."""
    content: str
    confidence: float
    source: str
    timestamp: float = field(default_factory=lambda: asyncio.get_event_loop().time())


@dataclass
class Desire:
    """Agent's desires or goals."""
    content: str
    priority: float
    urgency: float = 0.0


@dataclass
class Intention:
    """Agent's intentions or planned actions."""
    content: str
    plan: List[str]
    status: str = "pending"  # pending, executing, completed, failed


class BDITrustAgent(AgentBase):
    """
    BDI-based Trust Agent that simulates human trust behavior in economic games.
    """
    
    def __init__(
        self,
        name: str,
        age: int,
        personality_traits: Dict[str, float],
        education: str,
        initial_endowment: float,
        model_config: Dict[str, Any],
        agent_config: Dict[str, Any],
    ):
        """Initialize BDI Trust Agent."""
        super().__init__()
        
        # Basic attributes
        self.name = name
        self.age = age
        self.personality_traits = personality_traits
        self.education = education
        self.endowment = initial_endowment
        
        # BDI components
        self.beliefs: List[Belief] = []
        self.desires: List[Desire] = []
        self.intentions: List[Intention] = []
        
        # Memory system
        self.memory = InMemoryMemory()
        self.memory_limit = agent_config.get("memory_limit", 50)
        self.enable_cot = agent_config.get("enable_cot", True)
        self.enable_browser_tool = agent_config.get("enable_browser_tool", False)
        
        # Parameter update mechanism
        self.use_llm_for_parameter_updates = agent_config.get("use_llm_for_parameter_updates", False)
        
        # 从agent_config中读取短期和长期记忆的最大长度配置
        self.max_short_term_memory_length = agent_config.get("max_short_term_memory_length", 20)
        self.max_long_term_memory_length = agent_config.get("max_long_term_memory_length", 5)

        processed_model_config = model_config.copy()
        if "api_key_env" in processed_model_config:
            processed_model_config["api_key"] = os.getenv(
                processed_model_config["api_key_env"], 
                "your-api-key-here"
            )
            # Remove api_key_env as DashScopeChatModel doesn't expect it
            del processed_model_config["api_key_env"]
        
        # Long-term memory system
        self.long_term_memory = None
        if agent_config.get("enable_long_term_memory", False):
            try:
                # 初始化长期记忆
                self.long_term_memory = Mem0LongTermMemory(
                    agent_name=self.name,
                    user_name=f"user_{self.name}",
                    model=DashScopeChatModel(**processed_model_config),
                    embedding_model=DashScopeTextEmbedding(
                        model_name="text-embedding-v2",
                        api_key=os.getenv("DASHSCOPE_API_KEY"),
                    ),
                    on_disk=True,
                )
            except Exception as e:
                print(f"Failed to initialize long-term memory for {self.name}: {e}")
                self.long_term_memory = None
        
        # Browser tool for external knowledge
        self.browser_tool = None
        if self.enable_browser_tool:
            self.browser_tool = EnhancedBrowserTool()
        
        self.model = DashScopeChatModel(**processed_model_config)
        #self.model = DashScopeChatModel(**model_config)
        self.formatter = DashScopeMultiAgentFormatter()
        self.formatter1 = DashScopeChatFormatter()
        
        # Game history
        self.game_history: List[Dict[str, Any]] = []
        self.trust_scores: Dict[str, float] = {}
        
        # Initialize system prompt
        self.sys_prompt = self._generate_system_prompt()
        
    def _generate_system_prompt(self) -> str:
        """Generate system prompt based on agent's persona."""
        traits_desc = ", ".join([f"{k}: {v}" for k, v in self.personality_traits.items()])
        
        # 添加关于长期记忆的说明
        long_term_memory_note = ""
        if self.long_term_memory:
            long_term_memory_note = "\nYou also have access to long-term memories that help you remember important information across different conversations and sessions."
        
        # Add note about parameter updates if LLM-based updates are enabled
        parameter_update_note = ""
        if self.use_llm_for_parameter_updates:
            parameter_update_note = """
When updating your mental state, you can also provide numerical values for parameter changes:
- Include "trust_change": a numerical value (e.g., -0.1 to 0.1) to indicate how much your trust in other players has changed
- Include "confidence": a numerical value between 0.0 and 1.0 to indicate your confidence in a belief
"""

        return f"""You are {self.name}, a {self.age}-year-old {self.education} participating in economic games.
        
Your personality traits (Big Five model) are: {traits_desc}

You have an initial endowment of {self.endowment} units.

Your cognitive architecture follows the BDI model:
- Beliefs: What you believe about the world and other players
- Desires: What you want to achieve (maximize earnings, be fair, build trust, etc.)
- Intentions: Your planned actions based on your beliefs and desires

{long_term_memory_note}
{parameter_update_note}

In each game, you should:
1. Update your beliefs about other players based on their actions
2. Consider your desires (earnings, fairness, trust, etc.)
3. Form intentions and decide on actions
4. Reflect on the outcomes and update your mental state

{self._get_cot_prompt() if self.enable_cot else ""}

Always respond with your decision and reasoning in JSON format:
{{
    "action": "your_action",
    "amount": numeric_amount,
    "reasoning": "detailed_reasoning",
    "belief_update": "what_you_learned",
    "trust_change": "how_your_trust_changed"
}}"""

    def _get_cot_prompt(self) -> str:
        """Get enhanced chain-of-thought prompt with browser tool integration."""
        base_prompt = """
Think step-by-step (Chain of Thought):
1. Analyze the current situation and other players' previous actions
2. Consider your personality traits and how they influence your decision
3. Evaluate potential outcomes and risks
4. Balance self-interest with fairness and trust considerations
5. Make a decision that aligns with your beliefs and desires
"""
        
        if self.enable_browser_tool:
            base_prompt += """
6. Search for relevant information about game strategies and economic behavior
7. Incorporate external knowledge into your decision-making process
"""
        
        return base_prompt

    async def observe(self, msg: Msg | list[Msg] | None) -> None:
        """Observe messages and update beliefs."""
        if msg is None:
            return
            
        if isinstance(msg, list):
            for m in msg:
                await self._process_observation(m)
        else:
            await self._process_observation(msg)
    
    async def _process_observation(self, msg: Msg) -> None:
        """Process a single observation."""
        # Store in memory
        self.memory.add(msg)
        
        # Store in long-term memory if enabled
        if self.long_term_memory:
            try:
                await self.long_term_memory.record(msgs=[msg])
            except Exception as e:
                print(f"Failed to record to long-term memory: {e}")
        
        # Update beliefs based on observation
        if msg.name != self.name:
            # Extract content from structured messages if possible
            content = self._extract_message_content(msg)
            belief_content = f"{msg.name} said: {content}"
            confidence = self._calculate_belief_confidence(msg)
            
            belief = Belief(
                content=belief_content,
                confidence=confidence,
                source=msg.name,
            )
            self.beliefs.append(belief)
            
            # Update trust scores based on the action/content
            self._update_trust_score(msg.name, msg.content)
            
            # Log the belief update
            print(f"[{self.name}] New belief formed: '{belief_content}' (confidence: {confidence:.3f})")
        
        # Limit memory size
        if len(self.beliefs) > self.memory_limit:
            self.beliefs = self.beliefs[-self.memory_limit:]
    
    def _extract_message_content(self, msg: Msg) -> str:
        """Extract content from potentially structured messages."""
        content = msg.content
        
        # Handle different content types
        if isinstance(content, list):
            # Handle list content, e.g., [{"text": "some text"}, ...]
            if len(content) > 0 and isinstance(content[0], dict) and "text" in content[0]:
                return content[0]["text"]
            else:
                # Join list elements into a string
                return " ".join(str(item) for item in content)
        elif isinstance(content, dict):
            # Handle dict content, e.g., {"text": "some text"}
            if "text" in content:
                return content["text"]
            elif "content" in content:
                return content["content"]
            elif "message" in content:
                return content["message"]
            else:
                return str(content)
        else:
            return str(content)
    
    def _calculate_belief_confidence(self, msg: Msg) -> float:
        """Calculate confidence in a belief based on various factors."""
        # Check if we should use LLM for parameter updates
        if self.use_llm_for_parameter_updates:
            # Try to extract belief confidence directly from structured message content
            confidence = self._extract_belief_confidence_from_message(msg)
            if confidence is not None:
                return confidence
                
        # Fall back to rule-based calculation
        base_confidence = 0.5
        
        # Adjust based on personality traits
        # Neuroticism: Higher neuroticism leads to lower confidence
        neuroticism = self.personality_traits.get("neuroticism", 0.5)
        base_confidence *= (1.1 - 0.2 * neuroticism)  # Scale between 0.9x to 1.1x
        
        # Conscientiousness: Higher conscientiousness leads to higher confidence
        conscientiousness = self.personality_traits.get("conscientiousness", 0.5)
        base_confidence *= (0.9 + 0.2 * conscientiousness)  # Scale between 0.9x to 1.1x
        
        # Openness: Higher openness might lead to more questioning of beliefs
        openness = self.personality_traits.get("openness", 0.5)
        base_confidence *= (1.05 - 0.1 * openness)  # Scale between 0.95x to 1.05x
        
        # Extraversion: More extraverted agents might be more confident in social situations
        extraversion = self.personality_traits.get("extraversion", 0.5)
        base_confidence *= (0.95 + 0.1 * extraversion)  # Scale between 0.95x to 1.05x
        
        # Agreeableness: More agreeable agents might be more trusting of others' statements
        agreeableness = self.personality_traits.get("agreeableness", 0.5)
        base_confidence *= (0.98 + 0.04 * agreeableness)  # Scale between 0.98x to 1.02x
        
        # Adjust based on source trustworthiness
        source_trust = self.trust_scores.get(msg.name, 0.5)
        # Trust score affects confidence - higher trust leads to higher confidence
        base_confidence = base_confidence * 0.7 + source_trust * 0.3  # Weighted combination
        
        # Adjust based on message content complexity
        content_length = len(str(msg.content))
        if content_length > 500:  # Very long message
            base_confidence *= 0.9  # Slightly lower confidence for complex messages
        elif content_length < 50:  # Very short message
            base_confidence *= 1.05  # Slightly higher confidence for simple messages
            
        # Adjust based on consistency with existing beliefs
        consistency_score = self._assess_belief_consistency(msg.content)
        base_confidence = base_confidence * 0.8 + consistency_score * 0.2  # Weighted combination
        
        return min(1.0, max(0.0, base_confidence))
    
    def _extract_belief_confidence_from_message(self, msg: Msg) -> Optional[float]:
        """
        Extract belief confidence from structured message content provided by the LLM.
        
        Args:
            msg: The message containing potential structured data
            
        Returns:
            The belief confidence value if found, None otherwise
        """
        # Try to parse message content as JSON if it looks like structured data
        content = msg.content
        content_data = {}
        
        # Handle different content types
        if isinstance(content, list):
            # Handle list content, e.g., [{"text": "some text"}, ...]
            if len(content) > 0 and isinstance(content[0], dict) and "text" in content[0]:
                content_str = content[0]["text"]
            else:
                # Join list elements into a string
                content_str = " ".join(str(item) for item in content)
        elif isinstance(content, dict):
            # Handle dict content, e.g., {"text": "some text"}
            if "text" in content:
                content_str = content["text"]
            else:
                content_str = str(content)
        else:
            content_str = str(content)
            
        # Try to parse content as JSON if it looks like structured response
        try:
            # Clean up the content string to make it valid JSON if possible
            cleaned_content = content_str.strip()
            if (cleaned_content.startswith('{') and cleaned_content.endswith('}')) or \
               (cleaned_content.startswith('[') and cleaned_content.endswith(']')):
                content_data = json.loads(cleaned_content)
        except json.JSONDecodeError:
            # If JSON parsing fails, treat as plain text
            pass
            
        # Look for confidence in the structured data
        if content_data and isinstance(content_data, dict):
            # Look for confidence in various possible keys
            confidence_keys = ["confidence", "belief_confidence", "certainty"]
            for key in confidence_keys:
                confidence = content_data.get(key)
                if confidence is not None:
                    try:
                        # Convert to float if it's a string representation of a number
                        if isinstance(confidence, str):
                            confidence_value = float(confidence)
                        elif isinstance(confidence, (int, float)):
                            confidence_value = float(confidence)
                        else:
                            continue
                            
                        # Ensure confidence is within valid range
                        return max(0.0, min(1.0, confidence_value))
                    except ValueError:
                        # If conversion fails, continue to next key
                        continue
        return None
    
    def _assess_belief_consistency(self, new_content) -> float:
        """Assess how consistent a new belief is with existing beliefs."""
        if not self.beliefs:
            return 0.5  # Neutral if no existing beliefs
        
        # Handle different content types
        if isinstance(new_content, list):
            # Handle list content, e.g., [{"text": "some text"}, ...]
            if len(new_content) > 0 and isinstance(new_content[0], dict) and "text" in new_content[0]:
                content_str = new_content[0]["text"]
            else:
                # Join list elements into a string
                content_str = " ".join(str(item) for item in new_content)
        elif isinstance(new_content, dict):
            # Handle dict content, e.g., {"text": "some text"}
            if "text" in new_content:
                content_str = new_content["text"]
            else:
                content_str = str(new_content)
        else:
            content_str = str(new_content)
        
        # Simple keyword-based consistency check
        new_content_lower = content_str.lower()
        consistent_count = 0
        total_count = 0
        
        # Check against recent beliefs
        recent_beliefs = self.beliefs[-5:]  # Check last 5 beliefs
        for belief in recent_beliefs:
            belief_content_lower = belief.content.lower()
            total_count += 1
            
            # Simple overlap check
            if any(word in belief_content_lower for word in new_content_lower.split()):
                consistent_count += 1
                
        if total_count == 0:
            return 0.5
            
        consistency_ratio = consistent_count / total_count
        # Map ratio to confidence score (0.0 to 1.0)
        return consistency_ratio
    
    def _update_trust_score(self, other_agent: str, action) -> None:
        """Update trust score for another agent based on their actions."""
        if other_agent not in self.trust_scores:
            self.trust_scores[other_agent] = 0.5  # Neutral trust
        
        # Convert action to string if it's not already
        action_str = ""
        if isinstance(action, list):
            # Handle list content, e.g., [{"text": "some text"}, ...]
            if len(action) > 0 and isinstance(action[0], dict) and "text" in action[0]:
                action_str = action[0]["text"]
            else:
                # Join list elements into a string
                action_str = " ".join(str(item) for item in action)
        elif isinstance(action, dict):
            # Handle dict content, e.g., {"text": "some text"}
            if "text" in action:
                action_str = action["text"]
            else:
                action_str = str(action)
        else:
            action_str = str(action)
        
        # Try to parse action as JSON if it looks like a structured response
        action_data = {}
        try:
            # Clean up the action string to make it valid JSON if possible
            cleaned_action = action_str.strip()
            if (cleaned_action.startswith('{') and cleaned_action.endswith('}')) or \
               (cleaned_action.startswith('[') and cleaned_action.endswith(']')):
                action_data = json.loads(cleaned_action)
        except json.JSONDecodeError:
            # If JSON parsing fails, treat as plain text
            pass
        
        # Check if we should use LLM for parameter updates
        if self.use_llm_for_parameter_updates:
            # Try to extract parameter updates from the action data
            llm_trust_change = self._extract_parameter_updates_from_action(action_data, "trust")
            if llm_trust_change is not None:
                trust_change = llm_trust_change
            else:
                # Fall back to rule-based calculation if LLM didn't provide updates
                trust_change = self._calculate_trust_change_from_action(action_str, action_data)
        else:
            # Use rule-based calculation
            trust_change = self._calculate_trust_change_from_action(action_str, action_data)
        
        # Update trust score
        self.trust_scores[other_agent] += trust_change
        
        # Keep trust scores bounded
        self.trust_scores[other_agent] = max(0.0, min(1.0, self.trust_scores[other_agent]))
        
        # Log the trust update
        print(f"[{self.name}] Trust update for {other_agent}: {trust_change:+.3f} -> {self.trust_scores[other_agent]:.3f}")

    def _extract_parameter_updates_from_action(self, action_data: Dict[str, Any], parameter_type: str) -> Optional[float]:
        """
        Extract parameter updates from structured action data provided by the LLM.
        
        Args:
            action_data: The structured data from the LLM response
            parameter_type: The type of parameter to extract ("trust", "personality", etc.)
            
        Returns:
            The parameter update value if found, None otherwise
        """
        if not action_data or not isinstance(action_data, dict):
            return None
            
        # Try to extract trust change directly from the response
        if parameter_type == "trust":
            # Look for trust_change in the action data
            trust_change = action_data.get("trust_change")
            if trust_change is not None:
                try:
                    # Convert to float if it's a string representation of a number
                    if isinstance(trust_change, str):
                        return float(trust_change)
                    elif isinstance(trust_change, (int, float)):
                        return float(trust_change)
                except ValueError:
                    # If conversion fails, return None
                    pass
            return None
        # Could add other parameter types here in the future
        return None
    
    def _calculate_trust_change_from_action(self, action_str: str, action_data: Dict[str, Any]) -> float:
        """Calculate trust change based on action content and structure."""
        trust_change = 0.0
        
        # If we have structured action data, use it for more accurate assessment
        if action_data and isinstance(action_data, dict):
            # Look for specific keys in structured responses
            action_type = action_data.get("action", "").lower()
            amount = action_data.get("amount", 0)
            reasoning = action_data.get("reasoning", "").lower()
            belief_update = action_data.get("belief_update", "").lower()
            
            # Assess trust based on action type and amount in economic games
            if action_type in ["invest", "send", "cooperate", "share"]:
                # Positive actions that show trust
                trust_change += 0.05
                # Amount affects trust - higher amounts show more trust
                if isinstance(amount, (int, float)) and amount > 0:
                    trust_change += min(0.1, amount / 100.0)  # Scale by amount but cap it
                    
            elif action_type in ["defect", "steal", "cheat"]:
                # Negative actions that break trust
                trust_change -= 0.15
                
            # Analyze reasoning for cooperative language
            cooperative_words = ["fair", "generous", "help", "cooperate", "together", "mutual benefit"]
            selfish_words = ["myself", "selfish", "only", "just me", "personal gain"]
            
            for word in cooperative_words:
                if word in reasoning or word in belief_update:
                    trust_change += 0.03
                    
            for word in selfish_words:
                if word in reasoning or word in belief_update:
                    trust_change -= 0.03
                    
        else:
            # For unstructured text, use keyword-based assessment with improved logic
            action_lower = action_str.lower()
            
            # Cooperative/positive keywords
            if any(word in action_lower for word in ["fair", "generous", "help", "cooperate", "trust"]):
                trust_change += 0.05
                
            # Selfish/negative keywords
            if any(word in action_lower for word in ["selfish", "unfair", "cheat", "steal", "lie"]):
                trust_change -= 0.1
                
            # Context-aware modifiers
            if "a lot" in action_lower or "much" in action_lower:
                # Indicates significant action, amplify effect
                trust_change *= 1.5
                
            if "little" in action_lower or "small" in action_lower:
                # Indicates minimal action, reduce effect
                trust_change *= 0.5
                
        # Adjust trust change based on personality traits
        # More agreeable agents might be more trusting
        agreeableness = self.personality_traits.get("agreeableness", 0.5)
        trust_change *= (0.8 + 0.4 * agreeableness)  # Scale between 0.8x to 1.2x
        
        # More neurotic agents might be less trusting
        neuroticism = self.personality_traits.get("neuroticism", 0.5)
        trust_change *= (1.1 - 0.2 * neuroticism)  # Scale between 0.9x to 1.1x
        
        # Ensure trust change stays within reasonable bounds
        return max(-0.3, min(0.3, trust_change))
    
    def add_desire(self, content: str, priority: float, urgency: float = 0.0) -> None:
        """Add a new desire."""
        desire = Desire(content=content, priority=priority, urgency=urgency)
        self.desires.append(desire)
        
        # Sort desires by priority and urgency
        self.desires.sort(key=lambda d: -(d.priority + d.urgency))
    
    def add_intention(self, content: str, plan: List[str]) -> None:
        """Add a new intention."""
        intention = Intention(content=content, plan=plan)
        self.intentions.append(intention)
    
    def get_current_beliefs(self) -> List[Belief]:
        """Get current beliefs about the situation."""
        return self.beliefs[-10:]  # Return recent beliefs
    
    def get_top_desires(self, n: int = 3) -> List[Desire]:
        """Get top n desires."""
        return self.desires[:n]
    
    def get_active_intentions(self) -> List[Intention]:
        """Get currently active intentions."""
        return [i for i in self.intentions if i.status in ["pending", "executing"]]
    
    async def reply(self, *args: Any, **kwargs: Any) -> Msg:
        """Generate a reply based on current BDI state with enhanced reasoning."""
        # Extract context from arguments
        context = kwargs.get("context", "")
        game_type = kwargs.get("game_type", "unknown")
        other_players = kwargs.get("other_players", [])
        
        # Get external knowledge if browser tool is enabled
        external_knowledge = ""
        if self.enable_browser_tool and self.browser_tool:
            try:
                async with self.browser_tool:
                    external_knowledge = await self.browser_tool.get_contextual_advice(
                        game_type, self.personality_traits, context
                    )
            except Exception as e:
                print(f"Browser tool error for {self.name}: {e}")
                external_knowledge = "Unable to access external knowledge."
        
        # Retrieve relevant long-term memories if enabled
        long_term_memories = []
        if self.long_term_memory:
            try:
                # 构造查询消息以检索相关记忆
                query_msg = Msg(
                    name="system",
                    content=f"Game context: {context}. Game type: {game_type}. Other players: {', '.join(other_players)}",
                    role="system"
                )
                long_term_memories = await self.long_term_memory.retrieve(msg=[query_msg])
            except Exception as e:
                print(f"Failed to retrieve from long-term memory: {e}")
        
        # Construct enhanced prompt with BDI state and external knowledge
        prompt = self._construct_enhanced_bdi_prompt(
            context, game_type, other_players, external_knowledge, long_term_memories
        )

        user_msg = Msg("user", prompt, "user")
        await self.memory.add(user_msg)
        
        # 优化记忆处理机制，防止提示词过长
        memory_messages = await self.memory.get_memory()
        
        # 使用从agent_config中读取的配置限制短期记忆数量
        if len(memory_messages) > self.max_short_term_memory_length:
            # 只保留最近的几条消息
            memory_messages = memory_messages[-self.max_short_term_memory_length:]
        
        # 构造prompt，包含系统提示和记忆
        formatted_messages = [Msg("system", self.sys_prompt, "system")]
        formatted_messages.extend(memory_messages)
        
        # 如果有长期记忆，也添加到prompt中
        current_long_term_memories = long_term_memories[:self.max_long_term_memory_length] if long_term_memories else []
        if current_long_term_memories:
            formatted_messages.append(Msg("system", "Relevant past experiences:", "system"))
            # 使用从agent_config中读取的配置限制长期记忆数量
            for memory in current_long_term_memories:
                formatted_messages.append(Msg("system", f"- {memory}", "system"))
        
        prompt = await self.formatter.format(formatted_messages)
        
        # 检查prompt长度，如果过长则进一步优化
        while len(prompt) > 10000:  # 假设10000字符接近模型上下文限制
            # 分别减半短期和长期记忆，直至长度合法
            if len(memory_messages) > 1:
                # 减半短期记忆
                memory_messages = memory_messages[len(memory_messages)//2:]
            elif len(current_long_term_memories) > 1:
                # 减半长期记忆
                current_long_term_memories = current_long_term_memories[len(current_long_term_memories)//2:]
            else:
                # 如果都已经减到最少但仍过长，则截断
                break
                
            # 重新构造prompt
            formatted_messages = [Msg("system", self.sys_prompt, "system")]
            formatted_messages.extend(memory_messages)
            
            if current_long_term_memories:
                formatted_messages.append(Msg("system", "Relevant past experiences:", "system"))
                for memory in current_long_term_memories:
                    formatted_messages.append(Msg("system", f"- {memory}", "system"))
            
            prompt = await self.formatter.format(formatted_messages)
                
        # Generate response
        response = await self.model(prompt)
        
        # 记录响应到长期记忆
        if self.long_term_memory and response.content:
            try:
                response_msg = Msg(name=self.name, content=str(response.content), role="assistant")
                await self.long_term_memory.record(msgs=[response_msg])
            except Exception as e:
                print(f"Failed to record response to long-term memory: {e}")
        
        try:
            # Parse JSON response - 修改为正确处理response.content[0]['text']
            response_text = response.content[0]['text']
            response_data = json.loads(response_text)
            
            # Update intentions based on response
            self._update_intentions(response_data)
            
            # Store in game history
            self.game_history.append({
                "round": len(self.game_history) + 1,
                "game_type": game_type,
                "action": response_data.get("action", ""),
                "amount": response_data.get("amount", 0),
                "reasoning": response_data.get("reasoning", ""),
                "external_knowledge_used": bool(external_knowledge),
                "timestamp": asyncio.get_event_loop().time()
            })
            
            return Msg(
                name=self.name,
                content=response.content,
                role="assistant",
            )
            
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return Msg(
                name=self.name,
                content=response.content,
                role="assistant",
            )
    
    def _construct_enhanced_bdi_prompt(
        self, 
        context: str, 
        game_type: str, 
        other_players: List[str],
        external_knowledge: str = "",
        long_term_memories: List[str] = None
    ) -> str:
        """Construct enhanced prompt including BDI state and external knowledge."""
        prompt = f"{self.sys_prompt}\n\n"
        
        prompt += f"Current Game: {game_type}\n"
        prompt += f"Context: {context}\n"
        prompt += f"Other Players: {', '.join(other_players)}\n"
        prompt += f"Current Endowment: {self.endowment}\n\n"
        
        # Add current beliefs
        prompt += "Current Beliefs:\n"
        for belief in self.get_current_beliefs():
            prompt += f"- {belief.content} (confidence: {belief.confidence:.2f})\n"
        
        # Add top desires
        prompt += "\nTop Desires:\n"
        for desire in self.get_top_desires():
            prompt += f"- {desire.content} (priority: {desire.priority:.2f})\n"
        
        # Add trust scores
        prompt += "\nTrust Scores:\n"
        for player, score in self.trust_scores.items():
            prompt += f"- {player}: {score:.2f}\n"
        
        # Add external knowledge if available
        if external_knowledge:
            prompt += f"\nExternal Knowledge:\n{external_knowledge}\n"
        
        # Add relevant long-term memories if available
        if long_term_memories:
            prompt += "\nRelevant Past Experiences:\n"
            for memory in long_term_memories[:3]:  # 只显示最相关的3条记忆
                prompt += f"- {memory}\n"
        
        # Add enhanced reasoning guidance
        prompt += f"\nEnhanced Reasoning Process:\n"
        prompt += f"1. Situation Analysis: What is the current state and what are the key factors?\n"
        prompt += f"2. Personality Assessment: How do your traits {self.personality_traits} influence your approach?\n"
        prompt += f"3. Strategic Thinking: What are the potential outcomes and their probabilities?\n"
        prompt += f"4. Ethical Considerations: What are the fairness and trust implications?\n"
        prompt += f"5. Decision Integration: Combine all factors to make an optimal choice.\n"
        
        prompt += "\nBased on your enhanced BDI analysis, make your decision:"
        
        return prompt
    
    def _construct_bdi_prompt(
        self, 
        context: str, 
        game_type: str, 
        other_players: List[str]
    ) -> str:
        """Construct prompt including current BDI state (legacy method)."""
        return self._construct_enhanced_bdi_prompt(context, game_type, other_players, "", [])
    
    def _update_intentions(self, response_data: Dict[str, Any]) -> None:
        """Update intentions based on response."""
        action = response_data.get("action", "")
        
        # Complete current intentions if applicable
        for intention in self.get_active_intentions():
            if action in intention.content:
                intention.status = "completed"
        
        # Create new intention for next action
        if action:
            self.add_intention(
                content=f"Execute {action}",
                plan=[f"Prepare for {action}", f"Execute {action}", f"Reflect on {action}"]
            )
    
    async def handle_interrupt(self, *args: Any, **kwargs: Any) -> Msg:
        """Handle interruption during reply."""
        return Msg(
            name=self.name,
            content="Action was interrupted. I need to reconsider.",
            role="assistant",
        )
    
    def get_agent_state(self) -> Dict[str, Any]:
        """Get current agent state for saving."""
        return {
            "name": self.name,
            "age": self.age,
            "personality_traits": self.personality_traits,
            "education": self.education,
            "endowment": self.endowment,
            "beliefs": [
                {
                    "content": b.content,
                    "confidence": b.confidence,
                    "source": b.source,
                    "timestamp": b.timestamp
                } for b in self.beliefs
            ],
            "desires": [
                {
                    "content": d.content,
                    "priority": d.priority,
                    "urgency": d.urgency
                } for d in self.desires
            ],
            "intentions": [
                {
                    "content": i.content,
                    "plan": i.plan,
                    "status": i.status
                } for i in self.intentions
            ],
            "trust_scores": self.trust_scores,
            "game_history": self.game_history
        }