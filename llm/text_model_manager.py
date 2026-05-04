from __future__ import annotations

import time
from typing import Any, Dict

from llm.providers.airllm_provider import AirLLMProvider
from llm.providers.llama_cpp_provider import LlamaCppProvider


class TextModelManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.provider = None
        self.backend = cfg.get_nested("llm.backend", "llama_cpp")
        self._last_load_s = None

    def _load_if_needed(self) -> None:
        if self.provider is not None:
            return

        if self.backend == "airllm":
            model_id = self.cfg.get_nested("llm.model_id", "Qwen/Qwen2.5-3B-Instruct")
            self.provider = AirLLMProvider(model_id=model_id)
        else:
            model_path = self.cfg.get_nested("llm.model_path", "")
            if not model_path:
                raise RuntimeError("llm.model_path is required for llama_cpp backend")
            self.provider = LlamaCppProvider(
                model_path=model_path,
                n_ctx=int(self.cfg.get_nested("llm.max_seq_len", 1024)),
                n_threads=self.cfg.get_nested("llm.n_threads", None),
            )

        self._last_load_s = time.time()

    def generate(self, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> Dict[str, Any]:
        self._load_if_needed()
        started = time.time()
        out = self.provider.generate_text(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        duration = time.time() - started
        return {
            "output_text": out.text,
            "model_used": out.model_used,
            "backend": out.backend,
            "duration_s": duration,
            "status": "done",
        }

    def status(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "loaded": self.provider is not None,
            "last_load_s": self._last_load_s,
        }
