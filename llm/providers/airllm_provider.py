from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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

    def _extract_generated_ids(self, generation: Any) -> Any:
        """Normalize different AirLLM generation return types to token ids."""
        if hasattr(generation, "sequences"):
            return generation.sequences
        if isinstance(generation, dict) and "sequences" in generation:
            return generation["sequences"]
        return generation

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> LLMOutput:
        input_ids = self._model.tokenizer(prompt, return_tensors="pt").input_ids
        generation = self._model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        generated_ids = self._extract_generated_ids(generation)
        text = self._model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        if isinstance(text, list):
            text = "\n".join(t.strip() for t in text if isinstance(t, str) and t.strip())
        return LLMOutput(text=str(text).strip(), model_used=self.model_id, backend="airllm")
