import json
import logging
import re
from typing import Optional, Type, TypeVar, List
from pydantic import BaseModel

from src.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """Unified LLM and Embedding Client supporting Gemini and OpenAI."""

    def __init__(self):
        self._gemini_client = None
        self._openai_client = None

    @property
    def gemini(self):
        """Lazy initialization of Google GenAI client."""
        if self._gemini_client is None:
            from google import genai

            api_key = settings.active_api_key
            if not api_key:
                raise ValueError("No API key configured for Gemini. Please set OPENAI_API_KEY or GEMINI_API_KEY in .env")
            self._gemini_client = genai.Client(api_key=api_key)
        return self._gemini_client

    @property
    def openai(self):
        """Lazy initialization of OpenAI client."""
        if self._openai_client is None:
            from openai import OpenAI

            api_key = settings.active_api_key
            if not api_key:
                raise ValueError("No API key configured for OpenAI. Please set OPENAI_API_KEY in .env")
            self._openai_client = OpenAI(
                api_key=api_key,
                base_url=settings.openai_base_url,
            )
        return self._openai_client

    def _extract_retry_delay(self, error: Exception, default_delay: float = 15.0) -> float:
        """Extract recommended retry delay from rate limit error message or return default."""
        err_str = str(error)
        match = re.search(r"retry in (\d+(\.\d+)?)s", err_str, re.IGNORECASE)
        if match:
            return min(float(match.group(1)) + 1.0, 35.0)
        match_delay = re.search(r"retryDelay':\s*'(\d+)s'", err_str)
        if match_delay:
            return min(float(match_delay.group(1)) + 1.0, 35.0)
        return default_delay

    def embed_text(self, text: str) -> List[float]:
        """Generate vector embedding for a given text with retry backoff."""
        max_retries = 5
        for attempt in range(max_retries):
            try:
                if settings.provider == "gemini":
                    response = self.gemini.models.embed_content(
                        model=settings.embedding_model,
                        contents=text,
                    )
                    return response.embeddings[0].values
                else:
                    response = self.openai.embeddings.create(
                        model=settings.embedding_model,
                        input=text,
                    )
                    return response.data[0].embedding
            except Exception as e:
                if ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) and attempt < max_retries - 1:
                    wait_time = self._extract_retry_delay(e, default_delay=(attempt + 1) * 6.0)
                    logger.warning(f"Rate limit hit in embed_text. Waiting {wait_time:.1f}s before retry... ({e})")
                    import time
                    time.sleep(wait_time)
                else:
                    raise

    def generate_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system_instruction: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> T:
        """
        Generate structured output from LLM matching the provided Pydantic model with retry backoff.
        """
        target_model = model_name or settings.extraction_model
        max_retries = 5

        for attempt in range(max_retries):
            try:
                if settings.provider == "gemini":
                    full_prompt = prompt
                    if system_instruction:
                        full_prompt = f"System Instruction: {system_instruction}\n\n{prompt}"

                    response = self.gemini.models.generate_content(
                        model=target_model,
                        contents=full_prompt,
                        config={
                            "response_mime_type": "application/json",
                            "response_schema": response_model,
                        },
                    )
                    raw_text = response.text or "{}"
                    return response_model.model_validate_json(raw_text)
                else:
                    messages = []
                    if system_instruction:
                        messages.append({"role": "system", "content": system_instruction})
                    messages.append({"role": "user", "content": prompt})

                    try:
                        response = self.openai.beta.chat.completions.parse(
                            model=target_model,
                            messages=messages,
                            response_format=response_model,
                        )
                        return response.choices[0].message.parsed
                    except Exception:
                        response = self.openai.chat.completions.create(
                            model=target_model,
                            messages=messages,
                            response_format={"type": "json_object"},
                        )
                        content = response.choices[0].message.content or "{}"
                        return response_model.model_validate_json(content)
            except Exception as e:
                err_text = str(e)
                is_transient = any(code in err_text for code in ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE"])
                if is_transient and attempt < max_retries - 1:
                    wait_time = self._extract_retry_delay(e, default_delay=(attempt + 1) * 8.0)
                    logger.warning(f"Transient error ({'429' if '429' in err_text else '503'}) hit in generate_structured. Waiting {wait_time:.1f}s before retry...")
                    import time
                    time.sleep(wait_time)
                else:
                    raise


llm_client = LLMClient()
