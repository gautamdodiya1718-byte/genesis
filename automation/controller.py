"""
automation/controller.py
-------------------------
Genesis Unified Automation Controller.

Merges AutoDiff AutomationController with LocalDiffusion training pipeline.
This is the "brain" that wires all subsystems into a coherent autonomous loop.

Full autonomous cycle:
  1. Generate synthetic images (LCM / SD — auto-selected)
  2. Crawl web images (Openverse + Wikimedia)
  3. Caption all crawled images (BLIP)
  4. Add everything to the dataset (with pHash dedup)
  5. [Optional] Trigger VAE training after N cycles
  6. [Optional] Trigger diffusion fine-tuning after N cycles
  7. Export dataset in training-ready format

The controller is designed so any individual step can be run standalone
OR the full pipeline runs autonomously with configurable cycle intervals.
"""

from __future__ import annotations
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from core.config import ConfigNode
from core.logger import setup_logging
from core.model_manager import ModelManager
from core.image_utils import load_image

from models.generation.lcm_generator import AdaptiveGenerator
from models.generation.prompt_engine import PromptEngine, Prompt
from models.captioning.captioner import ImageCaptioner
from crawler.web_crawler import ImageCrawler, DownloadResult
from dataset.builder import DatasetBuilder

logger = logging.getLogger(__name__)


class GenesisController:
    """
    Unified autonomous controller for the Genesis system.

    Handles both CPU-only (pretrained) mode and GPU training mode.
    Automatically selects the right generator (LCM vs SD) based on config.
    """

    def __init__(self, cfg: ConfigNode):
        self.cfg = cfg
        setup_logging(
            log_level=cfg.system.get("log_level", "INFO"),
            log_dir=os.path.join(cfg.system.output_dir, "logs"),
        )

        logger.info(f"Genesis {cfg.system.version} starting | device={cfg.system.device}")

        # Shared infrastructure
        self.model_manager = ModelManager(cfg.model_cache_dir)

        # Core subsystems (all lazy-loaded on first use)
        self.generator    = AdaptiveGenerator(cfg, self.model_manager)
        self.captioner    = ImageCaptioner(cfg, self.model_manager)
        self.crawler      = ImageCrawler(cfg)
        self.dataset      = DatasetBuilder(cfg)
        self.prompt_engine = PromptEngine(
            templates_path=cfg.prompts.get("templates_file", "configs/prompt_templates.yaml")
        )

        # Output dirs
        self.gen_dir = Path(cfg.system.output_dir) / "generated_images"
        self.gen_dir.mkdir(parents=True, exist_ok=True)

        self._cycle_count = 0
        self._total_generated = 0
        self._total_crawled = 0

    # ══════════════════════════════════════════════════════════
    # Main entry points
    # ══════════════════════════════════════════════════════════

    def run(self) -> None:
        """Run the full automation pipeline (main entry point)."""
        ac = self.cfg.automation
        max_cycles    = ac.max_cycles
        interval      = ac.cycle_interval_seconds
        train_every_n = ac.get("train_every_n_cycles", 0)

        logger.info(
            f"Starting Genesis automation | "
            f"max_cycles={max_cycles} | interval={interval}s"
        )

        for cycle in range(max_cycles):
            self._cycle_count += 1
            start = time.time()

            logger.info(f"{'='*55}")
            logger.info(f"  Cycle {cycle+1}/{max_cycles}")
            logger.info(f"{'='*55}")

            self.run_cycle()

            # Optional: trigger training after N cycles
            if train_every_n > 0 and self._cycle_count % train_every_n == 0:
                n = self.dataset.stats()["total_images"]
                logger.info(f"Training trigger: {n} images accumulated")
                self._run_training_if_enabled()

            elapsed = time.time() - start
            logger.info(f"Cycle {cycle+1} done in {elapsed:.0f}s")

            if interval > 0 and cycle < max_cycles - 1:
                logger.info(f"Waiting {interval}s until next cycle...")
                time.sleep(interval)

        self.dataset.print_stats()
        logger.info("Genesis automation complete")

    def run_cycle(self) -> None:
        """Execute a single automation cycle."""
        ac = self.cfg.automation
        tasks = ac.tasks

        if tasks.get("generate_images", True):
            self.step_generate()

        if tasks.get("crawl_images", True):
            results = self.step_crawl()
            if tasks.get("caption_images", True):
                self.step_caption_and_add(results)

        if tasks.get("build_dataset", True):
            self.step_finalize_dataset()

    # ══════════════════════════════════════════════════════════
    # Individual pipeline steps
    # ══════════════════════════════════════════════════════════

    def step_generate(
        self,
        count: Optional[int] = None,
        categories: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Step 1: Generate synthetic images using LCM or SD.
        Adds generated images directly to the dataset (prompt = caption).

        Returns:
            List of saved image paths
        """
        n = count or self.cfg.automation.generate_count
        cats = categories or list(self.cfg.prompts.get("categories", []))

        prompts: List[Prompt] = self.prompt_engine.generate_batch(
            categories=cats or None,
            count_per_category=max(1, n // max(len(cats), 1)),
        )
        prompts = prompts[:n]

        logger.info(f"Generating {len(prompts)} images ({self.generator.mode} mode)")

        saved_paths = []
        for i, prompt in enumerate(prompts):
            try:
                images = self.generator.generate(
                    prompt=prompt.text,
                    negative_prompt=prompt.negative,
                    num_images=1,
                    seed=prompt.seed,
                )
                paths = self.generator.save_images(
                    images, str(self.gen_dir),
                    prefix=f"gen_{prompt.category}",
                    start_idx=self._total_generated + i,
                )
                # Add to dataset — use prompt as caption
                for path, img in zip(paths, images):
                    self.dataset.add_image(
                        image=img,
                        caption=prompt.text,
                        source="generated",
                        prompt=prompt.text,
                        extra_meta={"category": prompt.category, "style": prompt.style},
                    )
                saved_paths.extend(paths)
            except Exception as e:
                logger.error(f"Generation failed for prompt {i}: {e}")

        self._total_generated += len(saved_paths)
        logger.info(f"Generated {len(saved_paths)} images (total: {self._total_generated})")
        return saved_paths

    def step_crawl(
        self,
        queries: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
    ) -> List[DownloadResult]:
        """
        Step 2: Crawl web images from configured sources.

        Returns:
            List of DownloadResult objects (local paths + metadata)
        """
        n_queries = self.cfg.automation.crawl_queries
        sources = sources or ["openverse", "wikimedia"]

        # Auto-generate crawl queries from prompt templates if not provided
        if not queries:
            prompts = self.prompt_engine.generate_batch(count_per_category=1)
            queries = [p.subject for p in prompts[:n_queries]]

        logger.info(f"Crawling {len(queries)} queries: {queries}")
        results = self.crawler.crawl_multiple(
            queries=queries,
            sources=sources,
            max_per_query=self.cfg.crawler.max_images_per_query,
        )
        self._total_crawled += len(results)
        logger.info(f"Crawled {len(results)} images (total: {self._total_crawled})")
        return results

    def step_caption_and_add(
        self,
        results: List[DownloadResult],
    ) -> Tuple[int, int]:
        """
        Step 3: Caption crawled images and add to dataset.

        Returns:
            (added, rejected) counts
        """
        if not results:
            return 0, 0

        logger.info(f"Captioning {len(results)} crawled images")

        # Caption all images
        paths = [r.local_path for r in results]
        caption_results = self.captioner.caption_batch(paths)

        # Build caption map
        caption_map = {
            cr["path"]: cr["caption"]
            for cr in caption_results
            if cr["caption"]
        }

        # Add to dataset
        items = []
        for r in results:
            cap = caption_map.get(r.local_path, r.query)
            items.append({
                "image":   r.local_path,
                "caption": cap,
                "source":  r.source,
                "url":     r.url,
                "prompt":  r.query,
            })

        added, rejected = self.dataset.add_batch(items)
        logger.info(f"Dataset update | added={added} | rejected={rejected}")
        return added, rejected

    def step_generate_variations(
        self,
        source_image_path: str,
        prompts: Optional[List[str]] = None,
        num_variations: int = 3,
        strength: float = 0.7,
    ) -> List[str]:
        """
        Step 4 (optional): Generate img2img variations from a source image.
        Useful for domain-specific augmentation.
        """
        if not prompts:
            raw_prompts = self.prompt_engine.generate(count=num_variations)
            prompts = [p.text for p in raw_prompts]

        logger.info(f"Generating {num_variations} variations of {source_image_path}")
        saved = []
        for i, prompt in enumerate(prompts[:num_variations]):
            try:
                images = self.generator.generate_img2img(
                    prompt=prompt,
                    init_image=source_image_path,
                    strength=strength,
                )
                paths = self.generator.save_images(
                    images, str(self.gen_dir),
                    prefix="variation",
                    start_idx=self._total_generated + i,
                )
                for path, img in zip(paths, images):
                    self.dataset.add_image(
                        image=img, caption=prompt,
                        source="variation",
                        extra_meta={"source_image": source_image_path, "strength": strength},
                    )
                saved.extend(paths)
            except Exception as e:
                logger.error(f"Variation generation failed: {e}")

        return saved

    def step_finalize_dataset(self) -> None:
        """Final step: print stats and export JSONL."""
        self.dataset.print_stats()
        try:
            export_path = self.dataset.export(fmt="jsonl")
            logger.info(f"Dataset exported → {export_path}")
        except Exception as e:
            logger.warning(f"Export failed: {e}")

    # ══════════════════════════════════════════════════════════
    # Training integration
    # ══════════════════════════════════════════════════════════

    def _run_training_if_enabled(self) -> None:
        """Trigger GPU training phases if configured."""
        ac = self.cfg.automation
        tasks = ac.get("tasks", {})

        if tasks.get("train_vae", False):
            logger.info("Triggering VAE training phase")
            self._train_vae()

        if tasks.get("train_diffusion", False):
            logger.info("Triggering diffusion training phase")
            self._train_diffusion()

    def _train_vae(self) -> None:
        """Launch VAE training on the accumulated dataset."""
        try:
            from models.vae.vae import VAE
            from models.vae.encoder import Encoder
            from models.vae.decoder import Decoder
            from training.vae_trainer import VAETrainer
            from torch.utils.data import DataLoader

            logger.info("Building VAE training dataset...")
            hf_ds = self.dataset.to_hf_dataset()

            import torchvision.transforms as T
            transform = T.Compose([
                T.Resize((512, 512)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.5]*3, [0.5]*3),
            ])

            class SimpleDS:
                def __init__(self, hf_dataset, transform):
                    self.ds = hf_dataset; self.t = transform
                def __len__(self): return len(self.ds)
                def __getitem__(self, i):
                    return self.t(self.ds[i]["image"].convert("RGB")), ""

            ds = SimpleDS(hf_ds, transform)
            loader = DataLoader(ds, batch_size=self.cfg.vae_training.batch_size, shuffle=True)

            vae = VAE(self.cfg)
            trainer = VAETrainer(self.cfg, vae, loader)
            trainer.train()
            logger.info("VAE training complete")

        except Exception as e:
            logger.error(f"VAE training failed: {e}", exc_info=True)

    def _train_diffusion(self) -> None:
        """Launch diffusion training on the accumulated dataset."""
        try:
            from models.vae.vae import VAE
            from models.diffusion.unet import UNet
            from models.diffusion.scheduler import NoiseScheduler
            from models.text_encoder.encoder import TextEncoder
            from models.diffusion.diffusion import LatentDiffusion
            from training.diffusion_trainer import DiffusionTrainer
            from torch.utils.data import DataLoader

            logger.info("Building diffusion training dataset...")
            hf_ds = self.dataset.to_hf_dataset()

            import torchvision.transforms as T
            transform = T.Compose([
                T.Resize((512, 512)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.5]*3, [0.5]*3),
            ])

            class CaptionDS:
                def __init__(self, hf_dataset, transform):
                    self.ds = hf_dataset; self.t = transform
                def __len__(self): return len(self.ds)
                def __getitem__(self, i):
                    item = self.ds[i]
                    return self.t(item["image"].convert("RGB")), item.get("caption","")

            ds = CaptionDS(hf_ds, transform)
            loader = DataLoader(
                ds, batch_size=self.cfg.training.batch_size,
                shuffle=True, num_workers=self.cfg.dataset.num_workers,
            )

            # Build models
            cfg = self.cfg
            vae = VAE(cfg)
            unet = UNet(
                in_channels=cfg.diffusion.in_channels,
                out_channels=cfg.diffusion.out_channels,
                base_channels=cfg.diffusion.base_channels,
                channel_multipliers=list(cfg.diffusion.channel_multipliers),
                context_dim=cfg.diffusion.context_dim,
            )
            text_encoder = TextEncoder(cfg)
            text_encoder.load()
            scheduler = NoiseScheduler(
                timesteps=cfg.diffusion.timesteps,
                beta_schedule=cfg.diffusion.beta_schedule,
            )

            model = LatentDiffusion(cfg, vae, unet, text_encoder, scheduler)
            trainer = DiffusionTrainer(cfg, model, loader)
            trainer.train()
            logger.info("Diffusion training complete")

        except Exception as e:
            logger.error(f"Diffusion training failed: {e}", exc_info=True)

    # ── Resource management ────────────────────────────────────

    def unload_models(self) -> None:
        """Free GPU/CPU memory by unloading all loaded models."""
        self.generator.unload()
        self.captioner.unload()
        logger.info("All models unloaded")
