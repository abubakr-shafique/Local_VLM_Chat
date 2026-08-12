"""Local multimodal chatbot — Gradio GUI.

Run:  python app.py   →  http://127.0.0.1:7860
"""
import datetime
import json
from pathlib import Path

import gradio as gr
from PIL import Image

from model_manager import QUANT_MODES, ModelManager
from model_registry import DEFAULT_MODEL, MODEL_REGISTRY

manager = ModelManager()
CHATS_DIR = Path(__file__).parent / "chats"
CHATS_DIR.mkdir(exist_ok=True)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


# --------------------------------------------------------------------- helpers
def model_status_markdown():
    lines = []
    for key, cfg in MODEL_REGISTRY.items():
        state = "✅ downloaded" if manager.is_downloaded(key) else "⬇️ not downloaded"
        kind = "VLM 🖼️" if cfg["kind"] == "vlm" else "LLM 📝"
        lines.append(f"- **{key}** ({kind}, ~{cfg['vram_4bit_gb']} GB VRAM @ 4-bit) — {state}")
    return "\n".join(lines)


def history_to_messages(history, allow_images):
    """Convert Gradio messages-format history to OpenAI-style chat messages."""
    messages = []
    for msg in history:
        role, content = msg["role"], msg["content"]

        if role == "assistant":
            text = content if isinstance(content, str) else ""
            if messages and messages[-1]["role"] == "assistant":
                messages[-1]["content"] += text
            else:
                messages.append({"role": "assistant", "content": text})
            continue

        if isinstance(content, str):
            block = {"type": "text", "text": content}
        elif isinstance(content, dict) and "path" in content:
            path = content["path"]
            if Path(path).suffix.lower() in IMAGE_EXTS:
                if allow_images:
                    block = {"type": "image", "image": Image.open(path).convert("RGB")}
                else:
                    block = {"type": "text", "text": "[image omitted — text-only model loaded]"}
            else:
                block = {"type": "text", "text": f"[attached file: {Path(path).name}]"}
        else:
            continue

        if messages and messages[-1]["role"] == "user" and isinstance(messages[-1]["content"], list):
            messages[-1]["content"].append(block)
        else:
            messages.append({"role": "user", "content": [block]})
    return messages


def extract_images(messages):
    images = []
    for m in messages:
        if isinstance(m["content"], list):
            images.extend(b["image"] for b in m["content"] if b.get("type") == "image")
    return images


# ------------------------------------------------------------------- callbacks
def ui_load_model(key, quant_mode):
    try:
        msg = manager.load(key, quant_mode)
    except Exception as exc:
        msg = f"❌ {exc}"
    return msg, model_status_markdown()


def ui_unload_model():
    return manager.unload(), model_status_markdown()


def add_message(history, message):
    for path in message.get("files", []):
        history.append({"role": "user", "content": {"path": path}})
    text = (message.get("text") or "").strip()
    if text:
        history.append({"role": "user", "content": text})
    return history, gr.MultimodalTextbox(value=None, interactive=True)


def bot_response(history, system_prompt, max_new_tokens, temperature, top_p, top_k,
                 repetition_penalty):
    if not history or history[-1]["role"] != "user":
        yield history
        return
    if manager.model is None:
        history.append({
            "role": "assistant",
            "content": "⚠️ No model is loaded. Select a model above and press **Load model**.",
        })
        yield history
        return

    allow_images = manager.kind == "vlm"
    messages = history_to_messages(history, allow_images)
    if system_prompt and system_prompt.strip():
        messages.insert(0, {"role": "system", "content": system_prompt.strip()})
    images = extract_images(messages)

    history.append({"role": "assistant", "content": ""})
    try:
        stream = manager.stream_chat(
            messages, images,
            max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty,
        )
        for chunk in stream:
            history[-1]["content"] += chunk
            yield history
    except Exception as exc:
        history[-1]["content"] = f"❌ Generation error: {exc}"
        yield history


def save_chat(history):
    if not history:
        gr.Info("Nothing to save yet.")
        return
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = CHATS_DIR / f"chat_{stamp}.json"
    out.write_text(json.dumps(history, indent=2, ensure_ascii=False, default=str))
    gr.Info(f"Conversation saved to {out}")


def clear_chat():
    return [], gr.MultimodalTextbox(value=None, interactive=True)


# ------------------------------------------------------------------------- UI
def build_app():
    with gr.Blocks(title="Local Multimodal Chatbot", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🖼️ Local Multimodal Chatbot\n"
            "Fully local inference with PyTorch + Hugging Face Transformers — chat with "
            "text-only LLMs or vision-language models (VLMs) from locally stored weights."
        )

        with gr.Row():
            model_dd = gr.Dropdown(
                choices=list(MODEL_REGISTRY.keys()), value=DEFAULT_MODEL,
                label="Model", scale=3,
            )
            quant_rd = gr.Radio(QUANT_MODES, value=QUANT_MODES[0], label="Precision", scale=3)
            load_btn = gr.Button("🔄 Load model", variant="primary", scale=1)
            unload_btn = gr.Button("🗑️ Unload", scale=1)

        status_md = gr.Markdown("No model loaded.")

        with gr.Accordion("Local weights status", open=False):
            weights_md = gr.Markdown(model_status_markdown())
            refresh_btn = gr.Button("Refresh")

        with gr.Accordion("⚙️ Generation settings", open=False):
            system_tb = gr.Textbox(label="System prompt", value="You are a helpful assistant.")
            with gr.Row():
                max_tok_sl = gr.Slider(64, 8192, value=1024, step=64, label="Max new tokens")
                temp_sl = gr.Slider(0.0, 2.0, value=0.7, step=0.05,
                                    label="Temperature (0 = greedy)")
                top_p_sl = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="Top-p")
                top_k_sl = gr.Slider(0, 200, value=50, step=1, label="Top-k")
                rep_pen_sl = gr.Slider(1.0, 2.0, value=1.05, step=0.05,
                                       label="Repetition penalty")

        chatbot = gr.Chatbot(type="messages", multimodal=True, height=480, label="Conversation")
        chat_input = gr.MultimodalTextbox(
            placeholder="Type a message and/or attach an image…",
            file_types=["image"], sources=["upload", "clipboard"],
            label="Your message",
        )
        with gr.Row():
            clear_btn = gr.Button("🧹 Clear chat")
            save_btn = gr.Button("💾 Save chat")
            stop_btn = gr.Button("⏹️ Stop generation")

        load_btn.click(ui_load_model, [model_dd, quant_rd], [status_md, weights_md])
        unload_btn.click(ui_unload_model, None, [status_md, weights_md])
        refresh_btn.click(lambda: model_status_markdown(), None, weights_md)

        submit_event = (
            chat_input.submit(add_message, [chatbot, chat_input], [chatbot, chat_input])
            .then(
                bot_response,
                [chatbot, system_tb, max_tok_sl, temp_sl, top_p_sl, top_k_sl, rep_pen_sl],
                chatbot,
            )
        )
        stop_btn.click(lambda: None, None, None, cancels=submit_event)
        clear_btn.click(clear_chat, None, [chatbot, chat_input])
        save_btn.click(save_chat, chatbot, None)

    return demo


if __name__ == "__main__":
    build_app().queue().launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
