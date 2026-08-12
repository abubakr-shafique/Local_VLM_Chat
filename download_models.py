"""Download model weights from Hugging Face into ./models for fully local use.

Examples:
    python download_models.py --list
    python download_models.py --models "Qwen3.5-9B"
    python download_models.py  # downloads every registered model
"""
import argparse
from typing import List, Optional

from huggingface_hub import snapshot_download

from model_registry import MODEL_REGISTRY, MODELS_DIR

# Skip duplicate/alternate framework weights to save disk space.
IGNORE_PATTERNS = [
    "original/*",
    "*.pth",
    "*.gguf",
    "*.onnx",
    "*.msgpack",
    "flax_model*",
    "rust_model*",
    "tf_model*",
]


def download(key: str) -> None:
    """Download a single model by registry key."""
    cfg = MODEL_REGISTRY[key]
    dest = MODELS_DIR / key
    print(f"⬇️ {cfg['repo_id']} -> {dest}")
    snapshot_download(
        repo_id=cfg["repo_id"],
        local_dir=str(dest),
        ignore_patterns=IGNORE_PATTERNS,
        max_workers=8,
    )
    print(f"✅ Done: {key}\n")


def main(
    models: Optional[List[str]] = None,
    list_only: bool = False,
) -> None:
    """Main entry point for the download script.

    Args:
        models: List of model keys to download, or None for all.
        list_only: If True, only list registered models.
    """
    if list_only:
        for key, cfg in MODEL_REGISTRY.items():
            gate = " (gated — HF token required)" if cfg["gated"] else ""
            print(
                f" {key:32s} {cfg['kind']:4s} ~{cfg['vram_4bit_gb']:>4} GB @4-bit{gate}"
            )
        return

    keys = models or list(MODEL_REGISTRY.keys())
    for key in keys:
        if key not in MODEL_REGISTRY:
            print(
                f"⚠️ Unknown model '{key}' — skipping. Use --list to see options."
            )
            continue
        download(key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Registry keys to download (default: all).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List registered models and exit.",
    )
    args = parser.parse_args()

    main(models=args.models, list_only=args.list)