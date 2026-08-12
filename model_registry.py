"""Registry of supported open-source models and local storage layout.

Add new models by adding an entry to MODEL_REGISTRY:
- repo_id: Hugging Face repo id (weights are downloaded from here)
- kind: "vlm" (vision-language, accepts images) or "llm" (text only)
- category: "vision" | "text" | "coding" (display grouping in the UI)
- vram_4bit_gb: rough VRAM estimate at 4-bit quantization (for the UI)
- gated: True if the HF repo requires accepting a license + token
- notes: short description shown in docs
"""
import os
from pathlib import Path

MODELS_DIR = Path(os.environ.get("CHATBOT_MODELS_DIR", Path(__file__).parent / "models"))

MODEL_REGISTRY = {
    # ---------------- Vision-language models (image chat) ----------------
    "Qwen3.5-9B": {
        "repo_id": "Qwen/Qwen3.5-9B",
        "kind": "vlm",
        "category": "vision",
        "vram_4bit_gb": 6.0,
        "gated": False,
        "notes": "Newest Qwen generation — natively multimodal, 262k context, Apache 2.0. "
                 "Requires a recent transformers release.",
    },
    "Qwen3.5-4B": {
        "repo_id": "Qwen/Qwen3.5-4B",
        "kind": "vlm",
        "category": "vision",
        "vram_4bit_gb": 3.5,
        "gated": False,
        "notes": "Smallest practical Qwen3.5 — natively multimodal and very fast.",
    },
    "DeepSeek-VL2-Tiny": {
        "repo_id": "deepseek-ai/deepseek-vl2-tiny",
        "kind": "vlm",
        "category": "vision",
        "vram_4bit_gb": 3.0,
        "gated": False,
        "notes": "MoE VLM (1B active params) — runs fully on 16GB, strong OCR/chart understanding.",
    },
    "DeepSeek-VL2-Small": {
        "repo_id": "deepseek-ai/deepseek-vl2-small",
        "kind": "vlm",
        "category": "vision",
        "vram_4bit_gb": 10.0,
        "gated": False,
        "notes": "MoE VLM (2.8B active) — better quality than Tiny; fits at 4-bit with headroom.",
    },
    "Qwen2.5-VL-7B-Instruct": {
        "repo_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "kind": "vlm",
        "category": "vision",
        "vram_4bit_gb": 6.5,
        "gated": False,
        "notes": "Proven all-round VLM; excellent OCR/document understanding.",
    },
    "Qwen2.5-VL-3B-Instruct": {
        "repo_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "kind": "vlm",
        "category": "vision",
        "vram_4bit_gb": 3.5,
        "gated": False,
        "notes": "Lightweight VLM; great for quick image Q&A.",
    },
    "Gemma-3-4B-it": {
        "repo_id": "google/gemma-3-4b-it",
        "kind": "vlm",
        "category": "vision",
        "vram_4bit_gb": 4.5,
        "gated": True,
        "notes": "Strong small Google VLM with 128k context.",
    },
    "Gemma-3-12B-it": {
        "repo_id": "google/gemma-3-12b-it",
        "kind": "vlm",
        "category": "vision",
        "vram_4bit_gb": 8.0,
        "gated": True,
        "notes": "Highest-quality Gemma 3 VLM that still fits in 16 GB at 4-bit.",
    },
    "Llama-3.2-11B-Vision-Instruct": {
        "repo_id": "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "kind": "vlm",
        "category": "vision",
        "vram_4bit_gb": 8.0,
        "gated": True,
        "notes": "Meta's vision model; solid general image reasoning.",
    },
    # ----------------------------- Text-only -----------------------------
    "Qwen3-14B": {
        "repo_id": "Qwen/Qwen3-14B",
        "kind": "llm",
        "category": "text",
        "vram_4bit_gb": 9.5,
        "gated": False,
        "notes": "Hybrid reasoning model — streams its block before the answer.",
    },
    "Qwen3-8B": {
        "repo_id": "Qwen/Qwen3-8B",
        "kind": "llm",
        "category": "text",
        "vram_4bit_gb": 5.5,
        "gated": False,
        "notes": "Compact hybrid-reasoning Qwen3; fast and capable.",
    },
    "Qwen2.5-14B-Instruct": {
        "repo_id": "Qwen/Qwen2.5-14B-Instruct",
        "kind": "llm",
        "category": "text",
        "vram_4bit_gb": 9.5,
        "gated": False,
        "notes": "Strong previous-gen instruct model.",
    },
    "Llama-3.1-8B-Instruct": {
        "repo_id": "meta-llama/Llama-3.1-8B-Instruct",
        "kind": "llm",
        "category": "text",
        "vram_4bit_gb": 5.5,
        "gated": True,
        "notes": "Reliable Meta text model.",
    },
    "Mistral-7B-Instruct-v0.3": {
        "repo_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "kind": "llm",
        "category": "text",
        "vram_4bit_gb": 5.0,
        "gated": False,
        "notes": "Fast, permissively licensed text model.",
    },
    # ------------------------------ Coding -------------------------------
    "Qwen2.5-Coder-7B-Instruct": {
        "repo_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "kind": "llm",
        "category": "coding",
        "vram_4bit_gb": 5.0,
        "gated": False,
        "notes": "Fast coding specialist; best code model per size in its class.",
    },
    "Qwen2.5-Coder-14B-Instruct": {
        "repo_id": "Qwen/Qwen2.5-Coder-14B-Instruct",
        "kind": "llm",
        "category": "coding",
        "vram_4bit_gb": 9.5,
        "gated": False,
        "notes": "Best coding quality that fits fully on a 16 GB card.",
    },
    "Devstral-Small-2507": {
        "repo_id": "mistralai/Devstral-Small-2507",
        "kind": "llm",
        "category": "coding",
        "vram_4bit_gb": 13.0,
        "gated": False,
        "notes": "Mistral's 24B agentic-coding model, Apache 2.0.",
    },
    "Qwen3-Coder-30B-A3B-Instruct": {
        "repo_id": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "kind": "llm",
        "category": "coding",
        "vram_4bit_gb": 17.0,
        "gated": False,
        "notes": "Top local coding MoE (3B active). ⚠️ Exceeds 16 GB at 4-bit — runs with "
                 "CPU offload via device_map='auto' (slower generation).",
    },
}

DEFAULT_MODEL = "Qwen3.5-9B"


def is_vision_model(model_key: str) -> bool:
    """Return True if the model supports vision (image) input."""
    cfg = MODEL_REGISTRY.get(model_key)
    if cfg is None:
        return False
    return cfg.get("kind") == "vlm"


def get_model_kind(model_key: str) -> str:
    """Return 'vlm' or 'llm' for the given model key."""
    cfg = MODEL_REGISTRY.get(model_key)
    if cfg is None:
        return "unknown"
    return cfg.get("kind", "unknown")