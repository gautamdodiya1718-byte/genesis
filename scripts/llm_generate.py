from __future__ import annotations

import argparse

from core.config import load_config, parse_cli_overrides
from llm import TextModelManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Local LLM generation (CPU-first)")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("overrides", nargs="*", help="dot-notation overrides e.g. llm.backend=airllm")
    args = parser.parse_args()

    overrides = parse_cli_overrides(args.overrides)
    cfg = load_config(args.config, overrides=overrides)

    manager = TextModelManager(cfg)
    result = manager.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(result["output_text"])


if __name__ == "__main__":
    main()
