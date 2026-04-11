# Genesis — Unified Generative AI System

**v0.2.0** · CPU + GPU · Fully Local · No API Keys

Genesis merges two previously separate systems — **LocalDiffusion** (GPU training) and **AutoDiff** (CPU autonomous pipeline) — into a single, coherent codebase. It can run entirely on CPU with pretrained models, or be trained from scratch on a GPU.

---

## What's New in v0.2

| Feature | Description | Speedup |
|---|---|---|
| **Flash Attention** | `F.scaled_dot_product_attention` — auto-activates on PyTorch 2.0+ | 2-4× VRAM reduction |
| **LCM Generator** | 4-step generation via LCM scheduler | 5-10× faster on CPU |
| **ONNX Export** | U-Net → ONNXRuntime for CPU inference | 2-3× faster |
| **VAE Trainer** | Full VAE training with perceptual + adversarial loss | Enables domain customization |
| **LatentDiffusion** | Unified wrapper: VAE + UNet + TextEncoder + Scheduler | Clean training API |
| **Unified Config** | Single YAML + CLI overrides for both systems | One config to rule them all |
| **GenesisController** | Merged automation controller — triggers training after N cycles | Fully autonomous |

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
playwright install chromium   # For Unsplash crawling
```

### CPU-Only (No GPU, No Training)

```bash
# Full autonomous cycle: generate → crawl → caption → dataset
python scripts/auto_run.py

# Generate 5 landscape images using LCM (fast, 4 steps)
python scripts/generate.py --category landscape --count 5 --lcm

# Crawl 50 images about "tropical beach"
python scripts/crawl.py --query "tropical beach" --max 50

# Caption all crawled images
python scripts/caption.py --dir outputs/crawled_images --output captions.json

# Build + export training-ready dataset
python scripts/build_dataset.py --image_dir outputs/crawled_images --export --format jsonl
```

### GPU Training (Full Pipeline)

```bash
# Phase 1: Train the VAE on your images (50 epochs ~4h on RTX 3090)
python scripts/train_vae.py --image_dir /path/to/images

# Phase 2: Train the diffusion U-Net (100 epochs ~24h on RTX 3090)
python scripts/train.py --vae_checkpoint outputs/vae_training/vae_final.pt

# Generate with your trained model
python scripts/generate.py --checkpoint outputs/checkpoints/step_00010000 --prompt "..."
```

### ONNX Export (2-3× CPU Speedup)

```bash
# Export pretrained SD U-Net to ONNX
python scripts/export_onnx.py --pretrained --benchmark

# Generate using ONNX
python scripts/generate.py --prompt "mountain" generation.use_onnx=true
```

---

## Architecture

```
Genesis/
├── core/                    # Shared infrastructure
│   ├── config.py            # Unified YAML + CLI config system
│   ├── image_utils.py       # PIL helpers, pHash deduplication
│   ├── logger.py            # Console + rotating file logging
│   └── model_manager.py     # HuggingFace auto-download + cache
│
├── models/
│   ├── vae/                 # Variational Autoencoder
│   │   ├── encoder.py       # Image → (mean, log_var)
│   │   ├── decoder.py       # Latent → image
│   │   └── vae.py           # Full VAE with KL loss
│   ├── diffusion/
│   │   ├── attention.py     # Flash Attention + SpatialTransformer
│   │   ├── unet.py          # Latent diffusion U-Net
│   │   ├── scheduler.py     # DDPM + DDIM noise schedulers
│   │   └── diffusion.py     # ★ NEW: LatentDiffusion unified wrapper
│   ├── text_encoder/
│   │   └── encoder.py       # CLIP + T5 text encoding
│   ├── generation/
│   │   ├── sd_generator.py  # Standard SD 1.5 (CPU)
│   │   ├── lcm_generator.py # ★ NEW: LCM 4-step generator (CPU)
│   │   └── prompt_engine.py # YAML template prompt system
│   └── captioning/
│       └── captioner.py     # BLIP / BLIP-2 / ViT-GPT2
│
├── training/
│   ├── losses.py            # ★ NEW: Perceptual + SSIM + PatchGAN losses
│   ├── vae_trainer.py       # ★ NEW: VAE training pipeline
│   └── diffusion_trainer.py # Diffusion U-Net training (EMA, AMP, compile)
│
├── inference/
│   ├── pipeline.py          # CustomPipeline for trained Genesis models
│   └── onnx_exporter.py     # ★ NEW: ONNX export + ONNXRuntime inference
│
├── crawler/
│   └── web_crawler.py       # Openverse + Wikimedia + Playwright
│
├── dataset/
│   └── builder.py           # pHash dedup + JSONL/CSV/DreamBooth export
│
├── automation/
│   └── controller.py        # ★ NEW: Unified GenesisController
│
├── scripts/
│   ├── auto_run.py          # Full autonomous pipeline
│   ├── generate.py          # Image generation
│   ├── train_vae.py         # ★ NEW: VAE training
│   ├── train.py             # Diffusion training
│   ├── export_onnx.py       # ★ NEW: ONNX export + benchmark
│   ├── crawl.py             # Web crawling
│   ├── caption.py           # Image captioning
│   └── build_dataset.py     # Dataset management
│
└── configs/
    ├── base.yaml            # Master configuration
    └── prompt_templates.yaml # Generation prompt templates
```

---

## Performance Reference

| Task | Model | Resolution | Time |
|---|---|---|---|
| Generate (LCM) | LCM_Dreamshaper | 512×512, 4 steps | ~2-5 min (CPU) |
| Generate (SD) | SD 1.5 | 512×512, 20 steps | ~10-25 min (CPU) |
| Generate (ONNX) | SD 1.5 + ONNXRuntime | 512×512, 20 steps | ~4-10 min (CPU) |
| Generate (GPU) | SD 1.5 | 512×512, 50 steps | ~5-15 sec (GPU) |
| Caption | BLIP-base | any | ~2-5 sec |
| VAE train epoch | Custom VAE | 512×512 | ~20-40 min/epoch (GPU) |
| Diffusion train step | Custom UNet | 512×512 | ~0.5-2 sec/step (GPU) |

---

## Config Overrides

Any config value can be overridden at the CLI:

```bash
python scripts/generate.py \
    generation.num_inference_steps=4 \
    generation.width=256 \
    lcm.enabled=true \
    system.seed=42
```

---

## Roadmap (Next Steps)

- [ ] ControlNet — structural conditioning (edge/depth/pose maps)
- [ ] LoRA fine-tuning — efficient domain adaptation in <100MB
- [ ] Prompt Evolution — CLIP-scored self-improving prompt system
- [ ] FAISS deduplication index — O(log n) near-duplicate search
- [ ] FastAPI web UI — browser-based generation + dataset browser
- [ ] Aspect ratio bucketing — better composition at non-square ratios
- [ ] Video generation — pseudo-3D conv + temporal attention
