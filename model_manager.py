"""Model loading, unloading and streaming generation (PyTorch + Transformers)."""
import gc
import threading

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

from model_registry import MODEL_REGISTRY, MODELS_DIR

QUANT_MODES = ["4-bit (recommended)", "8-bit", "Full precision (bf16)"]


class ModelManager:
    """Keeps exactly one model resident on the GPU at a time."""

    def __init__(self):
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.current_key = None
        self.kind = None  # "vlm" | "llm"

    # ------------------------------------------------------------------ utils
    @staticmethod
    def is_downloaded(key):
        return (MODELS_DIR / key / "config.json").exists()

    @staticmethod
    def _vram_gb():
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 ** 3
        return 0.0

    # ---------------------------------------------------------------- loading
    def load(self, key, quant_mode):
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

        kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16}
        if quant_mode.startswith("4-bit"):
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif quant_mode.startswith("8-bit"):
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        if cfg["kind"] == "vlm":
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

    def unload(self):
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.current_key = None
        self.kind = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return "🗑️ Model unloaded."

    # ------------------------------------------------------------- generation
    def stream_chat(self, messages, images, max_new_tokens=1024, temperature=0.7,
                    top_p=0.9, top_k=50, repetition_penalty=1.05):
        """Yield decoded text chunks for the given OpenAI-style chat messages."""
        if self.model is None:
            raise RuntimeError("No model loaded.")

        template_engine = self.processor if self.kind == "vlm" else self.tokenizer
        prompt = template_engine.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if self.kind == "vlm":
            proc_kwargs = {"text": [prompt], "return_tensors": "pt"}
            if images:
                proc_kwargs["images"] = images
            inputs = self.processor(**proc_kwargs)
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt")

        device = self.model.device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

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
