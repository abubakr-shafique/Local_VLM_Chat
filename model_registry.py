"""Registry of supported open-source models and local storage layout.

Add new models by adding an entry to MODEL_REGISTRY:
  - repo_id:        Hugging Face repo id (weights are downloaded from here)
  - kind:           "vlm" (vision-language, accepts images) or "llm" (text only)
  - vram_4bit_gb:   rough VRAM estimate at 4-bit quantization (for the UI)
  - gated:          True if the HF repo requires accepting a license + token
  - notes:          short description shown in docs
"""
import os
from pathlib import Path

MODELS_DIR = Path(os.environ.get("CHATBOT_MODELS_DIR", Path(__file__).parent / "models"))

MODEL_REGISTRY = {
    # ---------------- Vision-language models (image chat) ----------------
    "Qwen2.5-VL-7B-Instruct": {
        "repo_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "kind": "vlm",
        "vram_4bit_gb": 6.5,
        "gated": False,
        "notes": "Best all-round VLM for a 16 GB card; excellent OCR/document understanding.",
    },
    "Qwen2.5-VL-3B-Instruct": {
        "repo_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "kind": "vlm",
        "vram_4bit_gb": 3.5,
        "gated": False,
        "notes": "Fastest vision option; great for quick image Q&A.",
    },
    "Gemma-3-4B-it": {
        "repo_id": "google/gemma-3-4b-it",
        "kind": "vlm",
        "vram_4bit_gb": 4.5,
        "gated": True,
        "notes": "Strong small Google VLM with 128k context.",
    },
    "Gemma-3-12B-it": {
        "repo_id": "google/gemma-3-12b-it",
        "kind": "vlm",
        "vram_4bit_gb": 8.0,
        "gated": True,
        "notes": "Highest-quality VLM that still fits in 16 GB at 4-bit.",
    },
    "Llama-3.2-11B-Vision-Instruct": {
        "repo_id": "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "kind": "vlm",
        "vram_4bit_gb": 8.0,
        "gated": True,
        "notes": "Meta's vision model; solid general image reasoning.",
    },
    # ----------------------------- Text-only -----------------------------
    "Qwen2.5-14B-Instruct": {
        "repo_id": "Qwen/Qwen2.5-14B-Instruct",
        "kind": "llm",
        "vram_4bit_gb": 9.5,
        "gated": False,
        "notes": "Best text-only quality that fits your VRAM at 4-bit.",
    },
    "Llama-3.1-8B-Instruct": {
        "repo_id": "meta-llama/Llama-3.1-8B-Instruct",
        "kind": "llm",
        "vram_4bit_gb": 5.5,
        "gated": True,
        "notes": "Reliable Meta text model.",
    },
    "Mistral-7B-Instruct-v0.3": {
        "repo_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "kind": "llm",
        "vram_4bit_gb": 5.0,
        "gated": False,
        "notes": "Fast, permissively licensed text model.",
    },
}

DEFAULT_MODEL = "Qwen2.5-VL-7B-Instruct"
