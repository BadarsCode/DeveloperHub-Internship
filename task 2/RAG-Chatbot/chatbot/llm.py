import os
from typing import Any

from dotenv import load_dotenv


class _ResponseWrapper:
    def __init__(self, content: str):
        self.content = content


class ChatGemini:
    """Small wrapper to mimic `llm.invoke(...).content` style."""

    def __init__(
        self,
        gemini_api_key: str,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ):
        import google.generativeai as genai

        genai.configure(api_key=gemini_api_key)
        self.genai = genai
        self.model = genai.GenerativeModel(model_name)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def invoke(self, prompt: str) -> _ResponseWrapper:
        response = self.model.generate_content(
            prompt,
            generation_config=self.genai.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            ),
        )
        text = getattr(response, "text", "") or ""
        return _ResponseWrapper(text)


def get_llm(
    provider: str = "gemini",
    model_name: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    api_key: str | None = None,
) -> Any:
    """Create an LLM client compatible with `.invoke(...).content`."""
    load_dotenv()
    provider_lower = provider.lower().strip()

    if provider_lower == "gemini":
        gemini_api_key = (api_key or "").strip() or os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("Missing GEMINI_API_KEY in environment")
        selected_model = model_name or "gemini-2.5-flash"
        return ChatGemini(
            gemini_api_key=gemini_api_key,
            model_name=selected_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    raise ValueError("Unsupported provider. Use 'gemini'")
