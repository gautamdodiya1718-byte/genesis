from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LLMOutput:
    text: str
    model_used: str
    backend: str


class AirLLMProvider:
    """Optional AirLLM backend (can be very slow on CPU-only systems)."""

    def __init__(self, model_id: str):
        try:
            from airllm import AutoModel
        except ImportError as e:
            raise RuntimeError("airllm is required for airllm backend") from e

        self.model_id = model_id
        self._model = AutoModel.from_pretrained(model_id)

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> LLMOutput:
        text = self._model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        if isinstance(text, list):
            text = "\n".join(str(t) for t in text)
        return LLMOutput(text=str(text).strip(), model_used=self.model_id, backend="airllm")
