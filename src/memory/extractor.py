import logging
from typing import List, Union, Dict, Any
from src.llm.client import llm_client
from src.memory.models import Fact, FactExtractionResponse

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = """You are an expert AI Memory Extraction & Knowledge Graph engine.
Your mission is to extract durable, persistent, and actionable facts about the user from the provided conversation, along with Knowledge Graph Entity Triples.

Guidelines:
1. ATOMIC & SELF-CONTAINED: Each fact must be a standalone statement in third person (e.g., "User prefers Python over JavaScript", "User is building an AI memory engine at Owting", "User lives in Providence and plans to move to Boston").
2. ACTIONABLE & INFORMATIVE: Focus on user preferences, skills, technical stacks, active session tasks/goals, background, personal traits, and strict constraints.
3. CATEGORIZATION: Classify each fact into: 'profile', 'preference', 'skill', 'project', 'constraint', 'other'.
4. KNOWLEDGE GRAPH TRIPLES: For EACH fact, break down the core entities and relationships into 1 to 3 distinct 'triples' (Subject, Relation, Object).
   Examples:
   - Fact: "User works as a software developer for a company called Owting."
     -> Triple 1: Subject: "User", Relation: "works_at", Object: "Owting"
     -> Triple 2: Subject: "User", Relation: "role", Object: "Software Developer"
   - Fact: "User lives in Providence, Rhode Island and plans to move to Boston."
     -> Triple 1: Subject: "User", Relation: "lives_in", Object: "Providence"
     -> Triple 2: Subject: "User", Relation: "plans_move_to", Object: "Boston"
     -> Triple 3: Subject: "Providence", Relation: "relocating_to", Object: "Boston"
   - Fact: "User is planning to expand the Owting flagship project."
     -> Triple 1: Subject: "Owting", Relation: "has_project", Object: "Flagship Project"
     -> Triple 2: Subject: "User", Relation: "expanding", Object: "Flagship Project"
5. IGNORE TRIVIAL NOISE: Do NOT extract pure greetings ("hi", "hello"), one-word pleasantries ("thanks", "bye"), or meaningless filler.
6. NO DUPLICATE FACTS: Combine related statements into crisp, atomic points.
7. If no informative facts are present in the conversation, return an empty list of facts.
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
