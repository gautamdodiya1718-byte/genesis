from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMOutput:
    text: str
    model_used: str
    backend: str


class LlamaCppProvider:
    """CPU-first provider using llama-cpp-python GGUF models."""

    def __init__(self, model_path: str, n_ctx: int = 2048, n_threads: Optional[int] = None):
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise RuntimeError("llama-cpp-python is required for llama_cpp backend") from e

        self.model_path = model_path
        self._llama = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False,
        )

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> LLMOutput:
        out = self._llama(
            prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            echo=False,
        )
        text = out["choices"][0]["text"].strip()
        return LLMOutput(text=text, model_used=self.model_path, backend="llama_cpp")
