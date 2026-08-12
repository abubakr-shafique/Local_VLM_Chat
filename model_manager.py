"""Model loading, unloading and streaming generation (PyTorch + Transformers).

This module keeps exactly one model resident on the GPU at a time and provides
a streaming chat interface for both text-only and vision-language models.
"""
import gc
import threading
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

from model_registry import MODEL_REGISTRY, MODELS_DIR, is_vision_model

QUANT_MODES = ["4-bit (recommended)", "8-bit", "Full precision (bf16)"]


class ModelManager:
    """Manages a single loaded model and provides streaming chat generation."""

    def __init__(self):
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.current_key: Optional[str] = None
        self.kind: Optional[str] = None  # "vlm" | "llm"
        self.is_qwen_vl = False

    # ------------------------------------------------------------------ utils
    @staticmethod
    def is_downloaded(key: str) -> bool:
        """Check if model weights exist locally."""
        return (MODELS_DIR / key / "config.json").exists()

    @staticmethod
    def _vram_gb() -> float:
        """Return current VRAM usage in GB (0.0 if CUDA unavailable)."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 ** 3
        return 0.0

    # ---------------------------------------------------------------- loading
    def load(self, key: str, quant_mode: str) -> str:
        """Load a model by registry key with specified quantization.

        Returns a status message string.
        """
        if key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model '{key}'.")
        if self.current_key == key:
            return f"✅ **{key}** is already loaded."
        if not self.is_downloaded(key):
            raise FileNotFoundError(
                f"Weights for '{key}' not found at `{MODELS_DIR / key}`.\n\n"
                f"Download them first:\n```\npython download_models.py --models \"{key}\"\n```"
            )

        self.unload()
        cfg = MODEL_REGISTRY[key]
        path = str(MODELS_DIR / key)

        kwargs: Dict[str, Any] = {"device_map": "auto", "torch_dtype": torch.bfloat16}
        if quant_mode.startswith("4-bit"):
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif quant_mode.startswith("8-bit"):
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        # Detect Qwen2.5-VL for special handling
        self.is_qwen_vl = "Qwen2.5-VL" in key or "Qwen2_5_VL" in key

        if cfg["kind"] == "vlm":
            if self.is_qwen_vl:
                from transformers import Qwen2_5_VLForConditionalGeneration
                model_cls = Qwen2_5_VLForConditionalGeneration
            else:
                model_cls = AutoModelForImageTextToText
            self.processor = AutoProcessor.from_pretrained(path)
            self.tokenizer = self.processor.tokenizer
        else:
            model_cls = AutoModelForCausalLM
            self.processor = None
            self.tokenizer = AutoTokenizer.from_pretrained(path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        try:
            self.model = model_cls.from_pretrained(path, attn_implementation="sdpa", **kwargs)
        except Exception:
            self.model = model_cls.from_pretrained(path, attn_implementation="eager", **kwargs)
        self.model.eval()

        self.current_key = key
        self.kind = cfg["kind"]
        return f"✅ Loaded **{key}** — {quant_mode}, {self._vram_gb():.1f} GiB VRAM in use."

    def unload(self) -> str:
        """Unload the current model and free GPU memory."""
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.current_key = None
        self.kind = None
        self.is_qwen_vl = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return "🗑️ Model unloaded."

    # ------------------------------------------------------------- generation
    def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        images: Optional[List[Image.Image]] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.05,
    ):
        """Yield decoded text chunks for the given OpenAI-style chat messages.

        Args:
            messages: List of message dicts with {"role": ..., "content": ...}.
                      content can be a string or a list of {"type": "text"|"image", ...} blocks.
            images: For VLMs: list of PIL images. For text-only models: ignored.
        """
        if self.model is None:
            raise RuntimeError("No model loaded.")

        if self.kind == "vlm":
            if self.is_qwen_vl:
                # Qwen2.5-VL requires special processing with qwen_vl_utils
                try:
                    from qwen_vl_utils import process_vision_info

                    # For Qwen2.5-VL, we need to include image data in the messages
                    # process_vision_info expects images to be referenced with image_url or image
                    if images:
                        # Replace image placeholders with actual image data
                        image_index = 0
                        for msg in messages:
                            if msg["role"] == "user" and isinstance(msg.get("content"), list):
                                new_content = []
                                for block in msg["content"]:
                                    if block.get("type") == "image":
                                        # Add image data using the format Qwen expects
                                        if image_index < len(images):
                                            # Use the image directly
                                            new_content.append({
                                                "type": "image",
                                                "image": images[image_index]
                                            })
                                            image_index += 1
                                        else:
                                            new_content.append(block)
                                    else:
                                        new_content.append(block)
                                msg["content"] = new_content

                    conversation = [
                        {"role": msg["role"], "content": msg["content"]}
                        for msg in messages
                    ]

                    prompt = self.processor.apply_chat_template(
                        conversation,
                        tokenize=False,
                        add_generation_prompt=True,
                    )

                    image_inputs, video_inputs = process_vision_info(conversation)

                    proc_kwargs = {
                        "text": [prompt],
                        "images": image_inputs if image_inputs else None,
                        "videos": video_inputs if video_inputs else None,
                        "return_tensors": "pt",
                        "padding": True,
                    }
                    inputs = self.processor(**proc_kwargs)

                except ImportError:
                    raise ImportError(
                        "Qwen2.5-VL requires qwen_vl-utils. Install it:\n"
                        "pip install qwen-vl-utils"
                    )
            else:
                # Generic VLM (Llama, Gemma, etc.)
                template_engine = self.processor
                prompt = template_engine.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                proc_kwargs = {"text": [prompt], "return_tensors": "pt"}
                if images:
                    proc_kwargs["images"] = images
                inputs = self.processor(**proc_kwargs)
        else:
            # Text-only model
            template_engine = self.tokenizer
            prompt = template_engine.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(prompt, return_tensors="pt")

        device = self.model.device
        inputs = {
            k: v.to(device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        gen_kwargs = dict(**inputs, streamer=streamer, max_new_tokens=int(max_new_tokens))
        if temperature and float(temperature) > 0:
            gen_kwargs.update(
                do_sample=True,
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
            )
        else:
            gen_kwargs.update(do_sample=False)
        if repetition_penalty and float(repetition_penalty) != 1.0:
            gen_kwargs["repetition_penalty"] = float(repetition_penalty)

        def _generate():
            with torch.inference_mode():
                self.model.generate(**gen_kwargs)

        worker = threading.Thread(target=_generate, daemon=True)
        worker.start()
        for chunk in streamer:
            yield chunk
        worker.join()