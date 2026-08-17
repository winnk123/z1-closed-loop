"""
Logging, JSON, and image-encoding helpers used across the codebase.
"""
import os
import json
import base64
import re
from io import BytesIO
from typing import Any, Dict, List, Union

import cv2
import numpy as np
from PIL import Image as PILImage

try:
    from colorama import Fore, init as _colorama_init
    _colorama_init(autoreset=True)
except ImportError:  # pragma: no cover - optional dependency
    class _NoColour:
        RESET = CYAN = GREEN = BLUE = YELLOW = RED = MAGENTA = ""

        def __getattr__(self, _name):
            return ""

    Fore = _NoColour()


# ---------------------------------------------------------------------------
# Coloured stdout logging
# ---------------------------------------------------------------------------

def print_step(step_num: int, description: str):
    """Print step information."""
    print(Fore.CYAN + f"\n[STEP {step_num}] {description}")


def print_action(action: str, details: str = ""):
    """Print action information."""
    print(Fore.GREEN + f"[ACTION] {action}" + (f" - {details}" if details else ""))


def print_info(info: str):
    """Print general information."""
    print(Fore.BLUE + f"[INFO] {info}")


def print_warning(warning: str):
    """Print warning information."""
    print(Fore.YELLOW + f"[WARNING] {warning}")


def print_error(error: str):
    """Print error information."""
    print(Fore.RED + f"[ERROR] {error}")


def print_success(success: str):
    """Print success information."""
    print(Fore.GREEN + f"[SUCCESS] {success}")


def print_model_interaction(
    model_name: str,
    prompt: str,
    response: str,
    speed: float = None,
    duration: float = None,
    prompt_speed: float = None,
):
    """Pretty-print a model call. Truncates base64 image payloads in the prompt."""
    print(Fore.YELLOW + "=" * 60)
    print(Fore.YELLOW + f"MODEL: {model_name}")
    print(Fore.CYAN + "-" * 20 + " PROMPT " + "-" * 20)

    clean_prompt = str(prompt)
    if "data:image" in clean_prompt:
        clean_prompt = re.sub(
            r"data:image/[^;]+;base64,[a-zA-Z0-9+/=]+",
            "[IMAGE_BASE64_DATA]",
            clean_prompt,
        )
    print(clean_prompt)

    print(Fore.GREEN + "-" * 20 + " RESPONSE " + "-" * 20)
    print(response)

    if any(v is not None for v in (speed, duration, prompt_speed)):
        print(Fore.MAGENTA + "-" * 20 + " STATS " + "-" * 20)
        if prompt_speed is not None:
            print(Fore.MAGENTA + f"Input speed:  {prompt_speed:.2f} tokens/s")
        if speed is not None:
            print(Fore.MAGENTA + f"Output speed: {speed:.2f} tokens/s")
        if duration is not None:
            print(Fore.MAGENTA + f"Duration:     {duration:.2f}s")
    print(Fore.YELLOW + "=" * 60)


# ---------------------------------------------------------------------------
# Output / image helpers
# ---------------------------------------------------------------------------

def save_output(output_dir: str, filename: str, content: Union[Dict, List, str]):
    """Save output to local file."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    if isinstance(content, (dict, list)):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(content, f, indent=2, ensure_ascii=False)
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(content))
    print_action(f"Saved output file: {filename}")


def numpy_to_base64(img_np: np.ndarray) -> str:
    """Convert a NumPy array (OpenCV BGR image) to a base64 JPEG string.

    Images larger than 1024 px on either side are downscaled proportionally
    before encoding at JPEG quality 85.
    """
    if img_np is None:
        return ""

    h, w = img_np.shape[:2]
    max_size = 1024
    if h > max_size or w > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_np = cv2.resize(img_np, (new_w, new_h))
        print_warning(f"Numpy image resized to {new_w}x{new_h}")

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
    success, buffer = cv2.imencode(".jpg", img_np, encode_param)
    if not success:
        print_error("Failed to encode image to JPEG")
        return ""

    return base64.b64encode(buffer).decode("utf-8")


def img_to_base64(img_path: str) -> str:
    """Read an image file from disk and convert it to a base64 JPEG string.

    Images larger than 256 px on either side are thumbnailed (aspect-ratio
    preserved) before encoding at JPEG quality 85.  The 256 px cap is
    intentionally smaller than the 1024 px cap used by ``numpy_to_base64`` to
    keep file-based payloads compact.
    """
    print_action(f"Converting image to Base64: {img_path}")
    if not os.path.exists(img_path):
        print_error(f"Image file not found: {img_path}")
        return ""
    try:
        img = PILImage.open(img_path).convert("RGB")

        max_size = 256
        if img.width > max_size or img.height > max_size:
            img.thumbnail((max_size, max_size))
            print_warning(f"Image resized to {img.size} to avoid server overload")

        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print_error(f"Failed to convert image to base64: {e}")
        return ""


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def safe_json_loads(text: str) -> Dict[str, Any]:
    """Robust JSON extraction from an LLM response.

    Falls back through three layers:
    1. Direct ``json.loads`` parse.
    2. Extract the first ``{...}`` block and fix common syntax errors.
    3. Manual regex extraction of known key fields.
    """
    print_action("Parsing JSON response")

    # Layer 1: direct parse
    try:
        result = json.loads(text)
        print_success("JSON parsed successfully")
        return result
    except json.JSONDecodeError:
        print_warning("Direct parsing failed, attempting to extract JSON content")

    # Layer 2: extract first JSON object
    json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not json_match:
        print_error("JSON parsing failed - No JSON structure found")
        print_warning(f"[Original Content]:\n{text}")
        return {}

    json_str = json_match.group()

    # Layer 3: fix common JSON errors and retry
    try:
        # Replace single quotes with double quotes
        json_str = json_str.replace("'", '"')
        # Fix unquoted property names
        json_str = re.sub(r"(\{|\,\s*)(\w+)\s*:", r'\1"\2":', json_str)
        # Fix unquoted string values
        json_str = re.sub(r":\s*([a-zA-Z_][a-zA-Z0-9_]*)(\s*[,}])", r':"\1"\2', json_str)
        # Fix boolean and null values
        json_str = re.sub(r":\s*(true|false|null)\s*([,}])", r":\1\2", json_str)
        # Remove trailing commas
        json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
        # Fix numbers
        json_str = re.sub(r":\s*(\d+\.?\d*)\s*([,}])", r":\1\2", json_str)
        # Fix strings in arrays
        json_str = re.sub(r"\[\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\]", r'["\1"]', json_str)

        print_info(f"Fixed JSON: {json_str}")

        result = json.loads(json_str)
        print_success("JSON parsed successfully (after fix)")
        return result

    except json.JSONDecodeError as e:
        print_error(f"JSON parsing failed: {e}")
        print_warning(f"[Fixed JSON Content]:\n{json_str}")
        print_warning(f"[Original Content]:\n{text}")

        # Fallback: manual extraction of known key fields
        try:
            description_match = (
                re.search(r'"description"\s*:\s*"([^"]*)"', text, re.IGNORECASE)
                or re.search(r"description[^:]*:\s*([^\n,}]*)", text, re.IGNORECASE)
            )
            reasoning_match = (
                re.search(r'"reasoning"\s*:\s*"([^"]*)"', text, re.IGNORECASE)
                or re.search(r"reasoning[^:]*:\s*([^\n,}]*)", text, re.IGNORECASE)
            )
            turn_match = (
                re.search(r'"turn_direction"\s*:\s*"([^"]*)"', text, re.IGNORECASE)
                or re.search(r"turn_direction[^:]*:\s*([^\n,}]*)", text, re.IGNORECASE)
            )

            result = {}
            if description_match:
                result["description"] = description_match.group(1).strip("\"' ")
            if reasoning_match:
                result["reasoning"] = reasoning_match.group(1).strip("\"' ")
            if turn_match:
                result["turn_direction"] = turn_match.group(1).strip("\"' ").lower()

            if result:
                print_success("Manual extraction of key info successful")
                return result
            else:
                return {}

        except Exception as e2:
            print_error(f"Manual extraction also failed: {e2}")
            return {}
