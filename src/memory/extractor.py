import logging
from typing import List, Union, Dict, Any
from src.llm.client import llm_client
from src.memory.models import Fact, FactExtractionResponse

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = """You are an expert AI Memory Extraction engine.
Your mission is to extract durable, persistent, and actionable facts about the user from the provided conversation.

Guidelines:
1. ATOMIC & SELF-CONTAINED: Each fact must be a standalone statement in third person (e.g., "User prefers Python over JavaScript", "User is building an AI memory engine", "User lives in Berlin").
2. DURABLE & VALUABLE: Focus on user preferences, skills, technical stacks, background, goals, personal traits, and strict constraints.
3. CATEGORIZATION: Classify each fact into one of these exact categories:
   - 'profile': Identity, name, age, location, residence, education, job role.
   - 'preference': Work habits, coding style, tool/framework preferences (e.g. "prefers dark mode").
   - 'skill': Languages, technologies, tools, and technical proficiencies.
   - 'project': Active projects, architectures, applications, and goals.
   - 'constraint': Strict limitations, rules, budget, or architectural constraints.
   - 'other': Any other durable fact.
4. IGNORE NOISE: Do NOT extract temporary conversational noise, greetings, transient task instructions (e.g. "Fix line 20"), or pleasantries ("Thanks", "Hello").
5. NO DUPLICATE FACTS: Combine related statements into crisp, atomic points.
6. If no durable user facts are present in the conversation, return an empty list of facts.
"""


class FactExtractor:
    """Extracts atomic, structured facts from conversations or text inputs."""

    def __init__(self):
        self.llm = llm_client

    def extract_facts(self, conversation: Union[str, List[Dict[str, Any]]]) -> List[Fact]:
        """
        Extracts atomic facts from a raw text string or conversation history.
        """
        formatted_input = self._format_conversation(conversation)
        if not formatted_input.strip():
            return []

        prompt = f"""Extract all durable facts about the user from this conversation transcript:

--- TRANSCRIPT START ---
{formatted_input}
--- TRANSCRIPT END ---
"""

        try:
            result = self.llm.generate_structured(
                prompt=prompt,
                response_model=FactExtractionResponse,
                system_instruction=EXTRACTION_SYSTEM_PROMPT,
            )
            logger.info(f"Extracted {len(result.facts)} facts from conversation.")
            return result.facts
        except Exception as e:
            logger.error(f"Error during fact extraction: {e}")
            return []

    def _format_conversation(self, conversation: Union[str, List[Dict[str, Any]]]) -> str:
        """Formats string or structured message turns into readable transcript."""
        if isinstance(conversation, str):
            return conversation.strip()

        lines = []
        for msg in conversation:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)


fact_extractor = FactExtractor()
