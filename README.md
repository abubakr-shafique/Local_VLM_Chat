# 🖼️ Local Multimodal Chatbot

A fully local chatbot GUI built with **PyTorch**, **Hugging Face Transformers**, and **Gradio**.
Chat with open-source **text LLMs** and **vision-language models (VLMs)** — including image
conversations — using weights stored on your own machine. After the one-time weight download,
no data ever leaves your computer.

## Features

- 💬 Multi-turn chat with token-by-token streaming responses
- 🖼️ Image understanding with VLMs (upload or paste images from the clipboard)
- 🔀 Model picker — switch between multiple locally stored open-source models at runtime
- 🗜️ 4-bit / 8-bit quantization (bitsandbytes) so 7B–14B models fit a 16 GB GPU
- ⚙️ Adjustable system prompt, temperature, top-p/top-k, repetition penalty, max tokens
- 💾 Save conversations to JSON, 🧹 clear chat, ⏹️ stop generation
- 🔒 100% offline inference

## Hardware target

Designed for **32 GB RAM + 16 GB VRAM** (e.g., RTX 4060 Ti 16 GB / 4080 / 5080 class).

| Model | Type | ~VRAM @ 4-bit | Notes |
|---|---|---|---|
| Qwen2.5-VL-7B-Instruct | VLM | ~6.5 GB | **Recommended default** — best balance of quality/speed |
| Qwen2.5-VL-3B-Instruct | VLM | ~3.5 GB | Fastest vision option |
| Gemma-3-4B-it | VLM | ~4.5 GB | Strong small VLM, 128k context (gated) |
| Gemma-3-12B-it | VLM | ~8 GB | Highest-quality VLM that fits (gated) |
| Llama-3.2-11B-Vision-Instruct | VLM | ~8 GB | Meta vision model (gated) |
| Qwen2.5-14B-Instruct | Text | ~9.5 GB | Best text-only quality that fits |
| Llama-3.1-8B-Instruct | Text | ~5.5 GB | Reliable Meta text model (gated) |
| Mistral-7B-Instruct-v0.3 | Text | ~5 GB | Fast, Apache 2.0 |

VRAM figures are estimates for 4-bit weights; add ~1–3 GB headroom for KV cache and image
tokens depending on context length and image resolution.

## Installation

Python 3.10–3.12 recommended. Linux (or WSL2 on Windows) gives the smoothest experience.

```bash
# 1. Create an environment
conda create --name local_VLM python=3.10
conda activate local_VLM       # Windows

# 2. Install PyTorch with CUDA (pick the wheel matching your driver)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 3. Install the rest
pip install -r requirements.txt
```

## Downloading model weights

Weights are stored locally in `./models/` (override with the `CHATBOT_MODELS_DIR` env var).

```bash
python download_models.py --list                              # show registered models
python download_models.py --models "Qwen2.5-VL-7B-Instruct"   # download one
python download_models.py                                     # download all
```

**Gated models** (Llama, Gemma) require a free Hugging Face account: accept the license on the
model's HF page, then run `huggingface-cli login` with an access token before downloading.

## Running

```bash
python app.py
```

Open http://127.0.0.1:7860, pick a model, press **Load model**, and chat. Attach images with
the 📎 button or paste from the clipboard. Images are only processed when a VLM is loaded;
with a text-only model they are replaced by a placeholder note.

## Project structure

```
├── app.py               # Gradio GUI (chat, image upload, model picker, settings)
├── model_manager.py     # PyTorch/Transformers backend: load/unload, quantized streaming generation
├── model_registry.py    # Registry of supported models + local storage path
├── download_models.py   # One-shot weight downloader (Hugging Face → ./models)
├── requirements.txt
├── models/              # Local weights live here (created by download_models.py)
└── chats/               # Saved conversations (JSON)
```

## Design note: Transformers vs. the vLLM engine

This app uses the **PyTorch + Transformers** backend rather than the vLLM serving engine,
deliberately:

- vLLM pre-allocates a fixed fraction of VRAM per engine instance and is designed to serve
  **one** model per process — it cannot hot-swap between several locally stored models inside
  a running GUI.
- Transformers + bitsandbytes allows true load/unload on demand and 4-bit NF4 quantization,
  which is what makes 7B–14B models usable on a 16 GB card.

If you later want maximum throughput for a single fixed model, serve it with
`vllm serve ./models/Qwen2.5-VL-7B-Instruct` and point an OpenAI-compatible client at it.

## Adding a new model

1. Add an entry to `MODEL_REGISTRY` in `model_registry.py` (repo id, `kind`, VRAM estimate).
2. Run `python download_models.py --models "<YourKey>"`.
3. Select it in the UI. Any model whose architecture is supported by
   `AutoModelForImageTextToText` (VLMs) or `AutoModelForCausalLM` (text) works out of the box.

## Troubleshooting

- **CUDA out of memory** → use 4-bit, pick a smaller model, lower *Max new tokens*, or close
  other GPU apps. Full precision is only practical for ≤4B models on 16 GB.
- **401 / access error when downloading** → the model is gated; accept its license on HF and
  run `huggingface-cli login`.
- **bitsandbytes fails to install on native Windows** → use WSL2, or upgrade
  (`pip install -U bitsandbytes`; recent releases ship Windows wheels).
- **Slow first response after loading** → normal; the first forward pass warms up kernels.
- **bf16 unsupported on very old GPUs** → change `torch.bfloat16` to `torch.float16` in
  `model_manager.py`.
- **Optional speedup** → install FlashAttention (`pip install flash-attn --no-build-isolation`,
  Linux only) and set `attn_implementation="flash_attention_2"` in `model_manager.py`.

## Licenses

The app code is yours to use freely. Model weights are governed by their own licenses
(Qwen: Apache 2.0; Mistral: Apache 2.0; Gemma: Gemma Terms of Use; Llama: Llama Community
License) — review them before redistribution or commercial use.
