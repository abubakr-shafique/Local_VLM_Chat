"""Utilities for chat message conversion and image handling.

This module provides helpers to:
- Convert Gradio message history to OpenAI-style messages.
- Extract PIL images from message history for VLMs.
- Check model capabilities and format messages appropriately.
"""
import io
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


def history_to_messages(
    history: List[Dict[str, Any]],
    allow_images: bool = True,
) -> List[Dict[str, Any]]:
    """Convert Gradio messages-format history to OpenAI-style chat messages.

    For VLMs: images are represented as {"type": "image"} placeholders.
    The actual image data is passed separately via extract_images_from_history().

    Args:
        history: List of message dicts from Gradio Chatbot.
        allow_images: If False, replace image blocks with text placeholders.

    Returns:
        List of message dicts in OpenAI chat format.
    """
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

        # User message: should be a list of blocks (text/image/file)
        if isinstance(content, list):
            block_list = []
            for b in content:
                if isinstance(b, dict):
                    # Handle text blocks
                    if b.get("type") == "text" and "text" in b:
                        block_list.append({"type": "text", "text": b["text"]})
                    # Handle image blocks
                    elif b.get("type") == "image":
                        if allow_images:
                            block_list.append({"type": "image"})
                        else:
                            block_list.append({"type": "text", "text": "[image omitted]"})
                    # Handle file blocks (Gradio 6.x format)
                    elif b.get("type") == "file":
                        if allow_images:
                            block_list.append({"type": "image"})
                        else:
                            block_list.append({"type": "text", "text": "[file]"})
                    # Handle blocks with path or image data
                    elif "path" in b or "image" in b:
                        if allow_images:
                            block_list.append({"type": "image"})
                        else:
                            block_list.append({"type": "text", "text": "[image]"})
            
            if block_list:
                messages.append({"role": "user", "content": block_list})
            else:
                messages.append({"role": "user", "content": [{"type": "text", "text": ""}]})
        else:
            # Fallback: treat as plain text
            messages.append({"role": "user", "content": [{"type": "text", "text": str(content)}]})

    return messages


def extract_images_from_history(
    history: List[Dict[str, Any]],
    allow_images: bool = True,
) -> List[Image.Image]:
    """Extract actual PIL images from history for VLM generation.

    Returns a list of PIL images in the order they appear in the conversation.

    Args:
        history: List of message dicts from Gradio Chatbot.
        allow_images: If False, return an empty list.

    Returns:
        List of PIL.Image objects.
    """
    if not allow_images:
        return []

    images = []
    for msg in history:
        content = msg.get("content")
        if msg["role"] != "user":
            continue

        if not isinstance(content, list):
            content = [content]

        for b in content:
            if not isinstance(b, dict):
                continue
            
            # Handle file blocks (Gradio 6.x format) - this is the key fix
            if b.get("type") == "file" and "file" in b:
                file_data = b["file"]
                if isinstance(file_data, dict) and "path" in file_data:
                    path = file_data["path"]
                    if isinstance(path, str) and Path(path).suffix.lower() in IMAGE_EXTS:
                        try:
                            img = Image.open(path).convert("RGB")
                            images.append(img)
                            print(f"  [extract_images] Loaded image from: {path}")
                        except Exception as e:
                            print(f"  [extract_images] Warning: Could not load image from {path}: {e}")
            
            # Handle image with path
            elif b.get("type") == "image" and "path" in b:
                path = b["path"]
                if isinstance(path, str) and Path(path).suffix.lower() in IMAGE_EXTS:
                    try:
                        img = Image.open(path).convert("RGB")
                        images.append(img)
                    except Exception as e:
                        print(f"Warning: Could not load image from {path}: {e}")
            
            # Handle image with image data
            elif b.get("type") == "image" and "image" in b:
                img = b["image"]
                if isinstance(img, Image.Image):
                    images.append(img.convert("RGB"))
                elif hasattr(img, "convert"):
                    images.append(img.convert("RGB"))
                else:
                    # Try to convert from other formats
                    try:
                        img_pil = Image.fromarray(img)
                        images.append(img_pil.convert("RGB"))
                    except Exception:
                        pass
            
            # Handle base64 encoded images
            elif b.get("type") == "image" and "base64" in b:
                import base64
                try:
                    img_data = base64.b64decode(b["base64"])
                    img = Image.open(io.BytesIO(img_data)).convert("RGB")
                    images.append(img)
                except Exception:
                    pass

    return images


def format_user_message(
    text: str,
    file_paths: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Format a user message with text and optional image attachments.

    Args:
        text: The text content of the message.
        file_paths: Optional list of image file paths.

    Returns:
        A message dict suitable for Gradio Chatbot history.
    """
    if file_paths is None:
        file_paths = []

    content = []
    if text:
        content.append({"type": "text", "text": text})

    for path in file_paths:
        if path:
            content.append({"type": "image", "path": str(path)})

    if not content:
        content = [{"type": "text", "text": ""}]

    return {"role": "user", "content": content}