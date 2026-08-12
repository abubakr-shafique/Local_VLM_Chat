"""Local multimodal chatbot — Gradio GUI.

Run: python app.py → http://127.0.0.1:7860

Features:
- Model selection with automatic image input enable/disable based on model capability.
- Text-only models: image upload is disabled.
- Vision-language models: image upload is enabled.
- Streaming generation with adjustable parameters.
- Save/clear chat functionality.
"""
import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from chat_utils import extract_images_from_history, format_user_message, history_to_messages
from model_manager import QUANT_MODES, ModelManager
from model_registry import DEFAULT_MODEL, MODEL_REGISTRY, is_vision_model

manager = ModelManager()
CHATS_DIR = Path(__file__).parent / "chats"
CHATS_DIR.mkdir(exist_ok=True)

CATEGORY_LABELS = {"vision": "VLM 🖼️", "text": "LLM 📝", "coding": "Coding 💻"}


# --------------------------------------------------------------------- helpers
def model_status_markdown() -> str:
    """Generate a markdown summary of model download status."""
    lines = []
    for key, cfg in MODEL_REGISTRY.items():
        state = "✅ downloaded" if manager.is_downloaded(key) else "⬇️ not downloaded"
        tag = CATEGORY_LABELS.get(cfg.get("category", cfg["kind"]), cfg["kind"])
        lines.append(
            f"- **{key}** ({tag}, ~{cfg['vram_4bit_gb']} GB VRAM @ 4-bit) — {state}"
        )
    return "\n".join(lines)


# ------------------------------------------------------------------- callbacks
def ui_load_model(
    key: str,
    quant_mode: str,
) -> Tuple[str, str]:
    """Load a model and return status messages."""
    try:
        msg = manager.load(key, quant_mode)
    except Exception as exc:
        msg = f"❌ {exc}"
    return msg, model_status_markdown()


def ui_unload_model() -> Tuple[str, str]:
    """Unload the current model and return status messages."""
    return manager.unload(), model_status_markdown()


def on_model_change(
    model_key: str,
) -> Dict[str, Any]:
    """Update image input availability based on model capability.

    Returns a dict to enable/disable the MultimodalTextbox file upload.
    """
    is_vlm = is_vision_model(model_key)
    # Enable file upload only for vision-language models
    return gr.update(interactive=is_vlm)


def add_message(
    history: List[Dict[str, Any]],
    message: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Add a user message to chat history.

    Args:
        history: Current chat history.
        message: Message dict from Gradio MultimodalTextbox.

    Returns:
        Updated history and a reset MultimodalTextbox.
    """
    # Debug: print the raw message
    print("\n=== add_message DEBUG ===")
    print(f"Raw message: {message}")
    print(f"Message type: {type(message)}")
    
    # Extract text and files from the message
    text = ""
    file_paths = []

    if isinstance(message, dict):
        text = (message.get("text") or "").strip()
        files = message.get("files") or []
        
        print(f"Extracted text: '{text}'")
        print(f"Extracted files: {files}")

        # Handle alternate key names
        if not files:
            for key in ("file_paths", "images", "image_paths"):
                if key in message and message[key]:
                    files = message[key]
                    print(f"Found files via alternate key '{key}': {files}")
                    break

        if isinstance(files, str):
            file_paths = [files]
        elif isinstance(files, (list, tuple)):
            file_paths = [str(f) for f in files if f]

    elif isinstance(message, str):
        text = message.strip()
    else:
        text = str(message)

    # Format the message
    formatted = format_user_message(text, file_paths if file_paths else None)
    print(f"Formatted message: {formatted}")
    
    history.append(formatted)

    # Keep interactive state based on current model
    is_vlm = manager.kind == "vlm" if manager.kind else True
    print(f"Interactive state: {is_vlm}")
    print("=========================\n")
    
    return history, gr.update(value=None, interactive=is_vlm)


def bot_response(
    history: List[Dict[str, Any]],
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
):
    """Generate a streaming response from the loaded model.

    Yields updated chat history after each token chunk.
    """
    # Ensure there is a user message to respond to
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

    # Debug output
    print("\n=== bot_response DEBUG ===")
    print(f"Model kind: {manager.kind}")
    print(f"Allow images: {allow_images}")
    print(f"History length: {len(history)}")
    print(f"Last message: {history[-1]}")

    # Convert history to model messages
    messages = history_to_messages(history, allow_images)
    print(f"Messages: {messages[-1] if messages else 'None'}")

    # Extract images for VLMs
    if allow_images:
        images = extract_images_from_history(history, allow_images)
        print(f"Extracted {len(images)} images")
        for i, img in enumerate(images):
            print(f"  Image {i}: {img.size} - {img.mode}")
    else:
        images = None
        print("Images not allowed (text-only model)")

    # Add system prompt if provided
    if system_prompt and system_prompt.strip():
        messages.insert(0, {"role": "system", "content": system_prompt.strip()})

    # Initialize assistant response
    history.append({"role": "assistant", "content": ""})

    try:
        stream = manager.stream_chat(
            messages,
            images,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )

        for chunk in stream:
            history[-1]["content"] += chunk
            yield history

    except Exception as exc:
        import traceback
        history[-1]["content"] = f"❌ Generation error: {exc}\n\n{traceback.format_exc()}"
        yield history


def save_chat(history: List[Dict[str, Any]]) -> None:
    """Save conversation history to a JSON file."""
    if not history:
        gr.Info("Nothing to save yet.")
        return

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = CHATS_DIR / f"chat_{stamp}.json"
    out.write_text(json.dumps(history, indent=2, ensure_ascii=False, default=str))
    gr.Info(f"Conversation saved to {out}")


def clear_chat() -> Tuple[List, Dict[str, Any]]:
    """Clear chat history and reset input."""
    is_vlm = manager.kind == "vlm" if manager.kind else True
    return [], gr.update(value=None, interactive=is_vlm)


# ------------------------------------------------------------------------- UI
def build_app() -> gr.Blocks:
    """Build and return the Gradio application."""
    with gr.Blocks(title="Local Multimodal Chatbot") as demo:
        gr.Markdown(
            "# 🖼️ Local Multimodal Chatbot\n"
            "Fully local inference with PyTorch + Hugging Face Transformers — chat with "
            "vision-language, text, and coding models from locally stored weights."
        )

        # Model selection row
        with gr.Row():
            model_dd = gr.Dropdown(
                choices=list(MODEL_REGISTRY.keys()),
                value=DEFAULT_MODEL,
                label="Model",
                scale=3,
            )
            quant_rd = gr.Radio(
                QUANT_MODES,
                value=QUANT_MODES[0],
                label="Precision",
                scale=3,
            )
            load_btn = gr.Button("🔄 Load model", variant="primary", scale=1)
            unload_btn = gr.Button("🗑️ Unload", scale=1)

        status_md = gr.Markdown("No model loaded.")

        # Model status accordion
        with gr.Accordion("Local weights status", open=False):
            weights_md = gr.Markdown(model_status_markdown())
            refresh_btn = gr.Button("Refresh")

        # Generation settings accordion
        with gr.Accordion("⚙️ Generation settings", open=False):
            system_tb = gr.Textbox(
                label="System prompt",
                value="You are a helpful assistant.",
            )
            with gr.Row():
                max_tok_sl = gr.Slider(
                    64, 8192, value=1024, step=64, label="Max new tokens"
                )
                temp_sl = gr.Slider(
                    0.0, 2.0, value=0.7, step=0.05, label="Temperature (0 = greedy)"
                )
                top_p_sl = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="Top-p")
                top_k_sl = gr.Slider(0, 200, value=50, step=1, label="Top-k")
                rep_pen_sl = gr.Slider(
                    1.0, 2.0, value=1.05, step=0.05, label="Repetition penalty"
                )

        # Chatbot and input
        chatbot = gr.Chatbot(height=480, label="Conversation")

        # Initialize image input based on default model
        default_is_vlm = is_vision_model(DEFAULT_MODEL)
        chat_input = gr.MultimodalTextbox(
            placeholder="Type a message, attach an image (📎), or paste one from your clipboard…",
            file_types=["image"],
            sources=["upload"],
            file_count="multiple",
            label="Your message",
            interactive=default_is_vlm,  # Disable for text-only default
        )

        # Action buttons
        with gr.Row():
            clear_btn = gr.Button("🧹 Clear chat")
            save_btn = gr.Button("💾 Save chat")
            stop_btn = gr.Button("⏹️ Stop generation")

        # Wire up events
        load_btn.click(
            ui_load_model,
            [model_dd, quant_rd],
            [status_md, weights_md],
        )
        unload_btn.click(
            ui_unload_model,
            None,
            [status_md, weights_md],
        )
        refresh_btn.click(
            lambda: model_status_markdown(),
            None,
            weights_md,
        )

        # Update image input availability when model changes
        model_dd.change(
            on_model_change,
            [model_dd],
            [chat_input],
        )

        # Chat submission
        submit_event = (
            chat_input.submit(
                add_message,
                [chatbot, chat_input],
                [chatbot, chat_input],
            )
            .then(
                bot_response,
                [
                    chatbot,
                    system_tb,
                    max_tok_sl,
                    temp_sl,
                    top_p_sl,
                    top_k_sl,
                    rep_pen_sl,
                ],
                chatbot,
            )
        )

        stop_btn.click(lambda: None, None, None, cancels=submit_event)
        clear_btn.click(clear_chat, None, [chatbot, chat_input])
        save_btn.click(save_chat, chatbot, None)

    return demo


if __name__ == "__main__":
    build_app().queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        theme=gr.themes.Soft(),
    )