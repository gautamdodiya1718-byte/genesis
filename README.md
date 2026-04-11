# Genesis AI Studio

**v0.5** · Local AI Image Generation · CPU + GPU · No API Keys Required

Genesis is a fully local, autonomous AI image generation platform that combines web crawling, dataset building, model training, and inference into a single unified pipeline. It runs entirely on CPU with pretrained models, or can be trained from scratch on a GPU.

---

## Features

| Feature | Description |
|---|---|
| **LCM Generator** | 4-step generation via LCM scheduler — 5-10x faster on CPU |
| **Flash Attention** | Auto-activates on PyTorch 2.0+ with 2-4x VRAM reduction |
| **ONNX Export** | U-Net to ONNXRuntime for 2-3x faster CPU inference |
| **VAE Trainer** | Full VAE training with perceptual + adversarial loss |
| **Web Crawler** | Openverse + Wikimedia + Playwright image crawler |
| **Active Learning** | Auto-detects model weaknesses and expands dataset |
| **Unified Config** | Single YAML + CLI overrides for all modules |
| **GenesisController** | Autonomous controller — triggers training after N cycles |

---

## Project Structure

```
genesis/
├── core/                    # Config, logging, image utils, model manager
├── models/
│   ├── vae/                 # Variational Autoencoder (encoder, decoder)
│   ├── diffusion/           # U-Net, attention, schedulers
│   ├── text_encoder/        # CLIP + T5 text encoding
│   ├── generation/          # SD 1.5, LCM, prompt engine
│   └── captioning/          # BLIP / BLIP-2 / ViT-GPT2
├── training/                # Diffusion + VAE trainers, checkpointing
├── inference/               # Custom pipeline, ONNX export, optimization
├── crawler/                 # Web image crawler
├── dataset/                 # Builder, dedup, active learning, lifecycle
├── evaluation/              # FID, CLIP score, aesthetic scoring
├── automation/              # GenesisController autonomous pipeline
├── api/                     # FastAPI server, prompt logger, feedback store
├── scripts/                 # CLI entry points for all operations
└── configs/                 # base.yaml + prompt_templates.yaml
```

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
playwright install chromium
```

### CPU-Only (No GPU Required)

```bash
# Full autonomous cycle: generate → crawl → caption → dataset
python scripts/auto_run.py

# Generate images using LCM (fast, 4 steps)
python scripts/generate.py --category landscape --count 5 --lcm

# Crawl images from the web
python scripts/crawl.py --query "tropical beach" --max 50

# Caption all crawled images
python scripts/caption.py --dir outputs/crawled_images --output captions.json

# Build training-ready dataset
python scripts/build_dataset.py --image_dir outputs/crawled_images --export --format jsonl
```

### GPU Training

```bash
# Phase 1: Train VAE on your images
python scripts/train_vae.py --image_dir /path/to/images

# Phase 2: Train diffusion U-Net
python scripts/train.py --vae_checkpoint outputs/vae_training/vae_final.pt

# Generate with your trained model
python scripts/generate.py --checkpoint outputs/checkpoints/step_00010000 --prompt "your prompt"
```

### ONNX Export (2-3x CPU Speedup)

```bash
python scripts/export_onnx.py --pretrained --benchmark
python scripts/generate.py --prompt "mountain" generation.use_onnx=true
```

---

## Performance Reference

| Task | Model | Resolution | Time |
|---|---|---|---|
| Generate (LCM) | LCM Dreamshaper | 512x512, 4 steps | ~2-5 min (CPU) |
| Generate (SD) | SD 1.5 | 512x512, 20 steps | ~10-25 min (CPU) |
| Generate (ONNX) | SD 1.5 + ONNXRuntime | 512x512, 20 steps | ~4-10 min (CPU) |
| Generate (GPU) | SD 1.5 | 512x512, 50 steps | ~5-15 sec (GPU) |
| Caption | BLIP-base | any | ~2-5 sec |
| VAE train/epoch | Custom VAE | 512x512 | ~20-40 min (GPU) |

---

## CLI Config Overrides

Any config value can be overridden at the command line:

```bash
python scripts/generate.py \
    generation.num_inference_steps=4 \
    generation.width=256 \
    lcm.enabled=true \
    system.seed=42
```

---

## Roadmap

- [ ] ControlNet — structural conditioning (edge/depth/pose maps)
- [ ] LoRA fine-tuning — efficient domain adaptation in under 100MB
- [ ] Prompt Evolution — CLIP-scored self-improving prompt system
- [ ] FAISS deduplication index — fast near-duplicate search
- [ ] FastAPI Web UI — browser-based generation + dataset browser
- [ ] Aspect ratio bucketing — better composition at non-square ratios
- [ ] Video generation — temporal attention + pseudo-3D conv

---

## License

MIT
