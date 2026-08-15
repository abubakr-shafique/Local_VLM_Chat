# 🖼️ Local Multimodal Chatbot

A fully local chatbot GUI built with **PyTorch**, **Hugging Face Transformers**, and **Gradio**. Chat with open-source **vision-language**, **text**, and **coding** models — including image conversations — using weights stored on your own machine. After the one-time weight download, no data ever leaves your computer.

## Features

- 💬 Multi-turn chat with token-by-token streaming responses
- 🖼️ Image understanding with VLMs (upload via 📎 or paste directly with Ctrl+V)
- 🔀 Model picker — switch between multiple locally stored open-source models at runtime
- 🎯 **Automatic image input enable/disable** — image upload is only enabled for vision-language models
- 💻 Dedicated coding models (Qwen Coder, Devstral) alongside vision and text models
- 🗜️ 4-bit / 8-bit quantization (bitsandbytes) so 7B–14B models fit a 16 GB GPU
- ⚙️ Adjustable system prompt, temperature, top-p/top-k, repetition penalty, max tokens
- 💾 Save conversations to JSON, 🧹 clear chat, ⏹️ stop generation
- 🔒 100% offline inference

## Hardware target

Designed for **32 GB RAM + 16 GB VRAM** (e.g., RTX 4060 Ti 16 GB / 4080 / 5080 class).

### Vision-language models (image chat)

| Model | ~VRAM @ 4-bit | Notes |
|---|---|---|
| Qwen3.5-9B | ~6 GB | **Default.** Newest gen, natively multimodal, 262k context, Apache 2.0 |
| Qwen3.5-4B | ~3.5 GB | Fastest new-gen VLM |
| DeepSeek-VL2-Tiny | ~3 GB | MoE VLM (1B active) — strong OCR/chart understanding, runs fully on 16 GB |
| DeepSeek-VL2-Small | ~10 GB | MoE VLM (2.8B active) — higher quality, still fits at 4-bit |
| Qwen2.5-VL-7B-Instruct | ~6.5 GB | Proven; excellent OCR/document understanding |
| Qwen2.5-VL-3B-Instruct | ~3.5 GB | Lightweight image Q&A |
| Gemma-3-4B-it | ~4.5 GB | 128k context (gated) |
| Gemma-3-12B-it | ~8 GB | Best Gemma 3 VLM that fits (gated) |
| Llama-3.2-11B-Vision-Instruct | ~8 GB | Meta vision model (gated) |

### Text models

| Model | ~VRAM @ 4-bit | Notes |
|---|---|---|
| Qwen3-14B | ~9.5 GB | Hybrid reasoning — streams its `<think>` block first |
| Qwen3-8B | ~5.5 GB | Compact hybrid-reasoning option |
| Qwen2.5-14B-Instruct | ~9.5 GB | Strong previous-gen instruct |
| Llama-3.1-8B-Instruct | ~5.5 GB | Reliable Meta model (gated) |
| Mistral-7B-Instruct-v0.3 | ~5 GB | Fast, Apache 2.0 |

### Coding models

| Model | ~VRAM @ 4-bit | Notes |
|---|---|---|
| Qwen2.5-Coder-7B-Instruct | ~5 GB | Fast coding specialist |
| Qwen2.5-Coder-14B-Instruct | ~9.5 GB | Best coding quality that fits fully on 16 GB |
| Devstral-Small-2507 | ~13 GB | Mistral's 24B agentic-coding model, Apache 2.0 |
| Qwen3-Coder-30B-A3B-Instruct | ~17 GB | ⚠️ Top local coding MoE (3B active) but exceeds 16 GB — runs via CPU offload (`device_map="auto"` uses your 32 GB RAM); expect slower tokens |

VRAM figures are estimates for 4-bit weights; add ~1–3 GB headroom for KV cache and image tokens depending on context length and image resolution.

## Installation

Python 3.10–3.12 recommended. Linux (or WSL2 on Windows) gives the smoothest experience.

```bash
# 1. Create an environment
conda create --name local_VLM python=3.12
conda activate local_VLM       # Windows

# 2. Install PyTorch with CUDA (pick the wheel matching your driver)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 3. Install the rest
pip install -r requirements.txt

# 4. Qwen3.5 models need a recent transformers — make sure you're up to date
pip install -U transformers
```

## Downloading model weights

Weights are stored locally in `./models/` (override with the `CHATBOT_MODELS_DIR` env var).

```bash
python download_models.py --list                       # show registered models
python download_models.py --models "Qwen3.5-9B"        # download one
python download_models.py                              # download all
```

**Gated models** (Llama, Gemma) require a free Hugging Face account: accept the license on the model's HF page, then run `huggingface-cli login` with an access token before downloading.

## Running

```bash
python app.py
```

Open http://127.0.0.1:7860, pick a model, press **Load model**, and chat.

- **Vision-language models**: image upload (📎) is enabled.
- **Text/coding models**: image upload is automatically disabled.

Attach images with the 📎 button or paste straight from the clipboard (Ctrl+V). Images are only processed when a VLM is loaded; with a text/coding model they are replaced by a placeholder note.

## Project structure

```
├── app.py               # Gradio GUI (chat, image upload, model picker, settings)
├── model_manager.py     # PyTorch/Transformers backend: load/unload, quantized streaming generation
├── model_registry.py    # Registry of supported models + local storage path + capability checks
├── chat_utils.py        # Message conversion and image extraction utilities
├── download_models.py   # One-shot weight downloader (Hugging Face → ./models)
├── requirements.txt
├── models/              # Local weights live here (created by download_models.py)
└── chats/               # Saved conversations (JSON)
```

## Design note: Transformers vs. the vLLM engine

This app uses the **PyTorch + Transformers** backend rather than the vLLM serving engine, deliberately:

- vLLM pre-allocates a fixed fraction of VRAM per engine instance and is designed to serve **one** model per process — it cannot hot-swap between several locally stored models inside a running GUI.
- Transformers + bitsandbytes allows true load/unload on demand and 4-bit NF4 quantization, which is what makes 7B–14B models usable on a 16 GB card.

If you later want maximum throughput for a single fixed model, serve it with `vllm serve ./models/Qwen3.5-9B` and point an OpenAI-compatible client at it.

## Adding a new model

1. Add an entry to `MODEL_REGISTRY` in `model_registry.py` (repo id, `kind`, `category`, VRAM estimate).
2. Run `python download_models.py --models "<YourKey>"`.
3. Select it in the UI. Any model whose architecture is supported by `AutoModelForImageTextToText` (VLMs) or `AutoModelForCausalLM` (text/coding) works out of the box.

The `kind` field determines whether image input is enabled:
- `"kind": "vlm"` → image upload enabled
- `"kind": "llm"` → image upload disabled

## Troubleshooting

- **Qwen3.5 fails to load with an unrecognized-architecture error** → your transformers is too old: `pip install -U transformers`.
- **CUDA out of memory** → use 4-bit, pick a smaller model, lower *Max new tokens*, or close other GPU apps. Full precision is only practical for ≤4B models on 16 GB.
- **Qwen3-Coder-30B is very slow** → expected: it doesn't fit in 16 GB, so some layers run from system RAM. Use Qwen2.5-Coder-14B for full-GPU speed.
- **Qwen3 / Qwen3.5 output starts with a `<think>` block** → these are hybrid-reasoning models; the reasoning is streamed before the final answer.
- **401 / access error when downloading** → the model is gated; accept its license on HF and run `huggingface-cli login`.
- **bitsandbytes fails to install on native Windows** → use WSL2, or upgrade (`pip install -U bitsandbytes`; recent releases ship Windows wheels).
- **Slow first response after loading** → normal; the first forward pass warms up kernels.
- **bf16 unsupported on very old GPUs** → change `torch.bfloat16` to `torch.float16` in `model_manager.py`.
- **Optional speedup** → install FlashAttention (`pip install flash-attn --no-build-isolation`, Linux only) and set `attn_implementation="flash_attention_2"` in `model_manager.py`.
- **Image upload is disabled but I want to use images** → make sure you've selected a vision-language model (e.g., Qwen3.5-9B, Qwen2.5-VL-7B-Instruct). Text-only and coding models do not support image input.

## Licenses

The app code is yours to use freely. Model weights are governed by their own licenses (Qwen/Qwen3.5/Qwen-Coder: Apache 2.0; Mistral/Devstral: Apache 2.0; Gemma: Gemma Terms of Use; Llama: Llama Community License) — review them before redistribution or commercial use.
