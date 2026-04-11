"""
models/generation/prompt_engine.py
------------------------------------
Template-based prompt generation system. Merged from AutoDiff.
"""

from __future__ import annotations
import random, yaml, logging
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Prompt:
    text: str
    negative: str
    category: str
    style: str
    subject: str
    seed: Optional[int] = None
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "text": self.text, "negative": self.negative,
            "category": self.category, "style": self.style,
            "subject": self.subject, "seed": self.seed,
            **self.metadata,
        }


DEFAULT_NEGATIVE = (
    "blurry, low quality, low resolution, distorted, deformed, "
    "watermark, text, signature, ugly, duplicate"
)

DEFAULT_TEMPLATES = {
    "landscape": {"subjects": ["mountain at sunset", "tropical beach", "autumn forest", "desert canyon"]},
    "portrait":  {"subjects": ["elderly craftsman", "musician on stage", "scientist in lab"]},
    "architecture": {"subjects": ["gothic cathedral", "modern skyscraper", "ancient ruins"]},
    "nature":    {"subjects": ["butterfly macro", "underwater coral", "cherry blossom"]},
    "abstract":  {"subjects": ["fluid color explosion", "geometric fractals", "light refraction"]},
    "animals":   {"subjects": ["wolf at moonlight", "tiger in jungle", "whale breaching"]},
}

DEFAULT_STYLES = [
    "photorealistic", "oil painting", "watercolor", "digital art",
    "pencil sketch", "cinematic", "studio photography", "illustration",
]

DEFAULT_QUALITY = [
    "highly detailed", "sharp focus", "professional photography",
    "4k resolution", "masterpiece", "award winning",
]


class PromptEngine:
    def __init__(self, templates_path: str = "configs/prompt_templates.yaml"):
        self.templates = DEFAULT_TEMPLATES
        self.styles = DEFAULT_STYLES
        self.quality_suffixes = DEFAULT_QUALITY
        self.negative_prompts = {"default": DEFAULT_NEGATIVE}
        self._load(templates_path)

    def _load(self, path: str) -> None:
        try:
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            self.styles    = data.get("styles", self.styles)
            self.quality_suffixes = data.get("quality_suffixes", self.quality_suffixes)
            self.negative_prompts = data.get("negative_prompts", self.negative_prompts)
            self.templates = data.get("templates", self.templates)
            logger.info(f"Prompt templates loaded: {len(self.templates)} categories")
        except FileNotFoundError:
            logger.warning(f"Template file not found: {path}, using defaults")

    def generate(
        self,
        category: Optional[str] = None,
        style: Optional[str] = None,
        count: int = 1,
        seed: Optional[int] = None,
    ) -> List[Prompt]:
        if seed is not None:
            random.seed(seed)
        return [self._one(category, style, seed=(seed+i if seed else None)) for i in range(count)]

    def _one(self, category=None, style=None, seed=None) -> Prompt:
        if category not in self.templates:
            category = random.choice(list(self.templates.keys()))
        subjects = self.templates[category].get("subjects", [f"{category} scene"])
        subject  = random.choice(subjects)
        sel_style = style or random.choice(self.styles)
        quality   = random.choice(self.quality_suffixes)
        text = f"{subject}, {sel_style}, {quality}"
        negative = self.negative_prompts.get("default", DEFAULT_NEGATIVE)
        return Prompt(text=text, negative=negative, category=category,
                      style=sel_style, subject=subject, seed=seed)

    def generate_batch(self, categories=None, count_per_category: int = 2) -> List[Prompt]:
        cats = categories or list(self.templates.keys())
        out = []
        for cat in cats:
            out.extend(self.generate(category=cat, count=count_per_category))
        return out

    def from_raw(self, text: str) -> Prompt:
        return Prompt(text=text, negative=self.negative_prompts.get("default", DEFAULT_NEGATIVE),
                      category="custom", style="custom", subject=text)

    def list_categories(self) -> List[str]: return list(self.templates.keys())
    def list_styles(self) -> List[str]: return list(self.styles)
