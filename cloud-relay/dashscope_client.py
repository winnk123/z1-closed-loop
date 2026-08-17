"""
DashScope Qwen3.5 Vision API client for LA (Language-Action) and VA (Vision-Action).

Based on uni-lavira/ai_client/vision_client.py, adapted for DashScope API.
Provides OpenAI-compatible interface to DashScope's Qwen multimodal models.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Tuple

from openai import OpenAI

logger = logging.getLogger(__name__)

_MAX_RETRIES: int = 3
_RETRY_DELAY: float = 10.0

# DashScope OpenAI-compatible endpoint
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class DashScopeVisionClient:
    """DashScope / 阿里百炼 OpenAI-compatible client for LA + VA inference."""

    def __init__(
        self,
        api_key: str,
        model: str = "qwen3.5-27b",
        base_url: str = DASHSCOPE_BASE_URL,
    ) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model

        self.stats: Dict[str, Dict[str, int]] = {
            "Language Action Model": {
                "calls": 0, "input_tokens": 0,
                "output_tokens": 0, "total_tokens": 0,
            },
            "Vision Action Model": {
                "calls": 0, "input_tokens": 0,
                "output_tokens": 0, "total_tokens": 0,
            },
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _prepend_no_think(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepend /no_think to suppress Qwen3 chain-of-thought tokens."""
        messages = list(messages)
        if messages and messages[0].get("role") == "system":
            first = dict(messages[0])
            first["content"] = "/no_think " + first["content"]
            messages[0] = first
        else:
            messages.insert(0, {"role": "system", "content": "/no_think"})
        return messages

    def _call_once(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int,
        temperature: float,
        stats_key: str,
    ) -> Tuple[str, Dict[str, Any]]:
        t0 = time.time()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            # Keep the model in direct-answer mode at the API level.  The
            # /no_think system prefix remains as a prompt-level fallback.
            extra_body={"enable_thinking": False},
        )
        duration = time.time() - t0
        usage = getattr(response, "usage", None)

        self.stats[stats_key]["calls"] += 1
        if usage:
            self.stats[stats_key]["input_tokens"] += usage.prompt_tokens or 0
            self.stats[stats_key]["output_tokens"] += usage.completion_tokens or 0
            self.stats[stats_key]["total_tokens"] += usage.total_tokens or 0

        content = response.choices[0].message.content
        info = {
            "duration": duration,
            "usage": {
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            } if usage else {},
        }
        return content, info

    # ------------------------------------------------------------------ #
    # Core generate
    # ------------------------------------------------------------------ #
    def generate(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        stats_key: str = "Vision Action Model",
        _attempt: int = 0,
    ) -> Tuple[str, Dict[str, Any]]:
        """Send a multimodal chat completion and return (content, info_dict)."""
        messages = self._prepend_no_think(messages)

        try:
            return self._call_once(messages, max_new_tokens, temperature, stats_key)
        except Exception as exc:
            logger.error(
                "[DashScope] API error (%s / %s): %s",
                stats_key, self.model_name, exc,
            )

            if _attempt >= _MAX_RETRIES - 1:
                raise

            time.sleep(_RETRY_DELAY)
            return self.generate(
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stats_key=stats_key,
                _attempt=_attempt + 1,
            )

    def la_generate(
        self, messages: List[Dict[str, Any]], **kw: Any
    ) -> Tuple[str, Dict[str, Any]]:
        """LA: strategic reasoning with panorama images."""
        return self.generate(messages, stats_key="Language Action Model", **kw)

    def va_generate(
        self, messages: List[Dict[str, Any]], **kw: Any
    ) -> Tuple[str, Dict[str, Any]]:
        """VA: tactical bbox detection with single view."""
        return self.generate(messages, stats_key="Vision Action Model", **kw)

    # ------------------------------------------------------------------ #
    # Usage statistics
    # ------------------------------------------------------------------ #
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "base_url": str(self.client.base_url),
        }

    def get_stats(self) -> Dict[str, Any]:
        la = self.stats["Language Action Model"]
        va = self.stats["Vision Action Model"]
        return {
            "la_calls": la["calls"],
            "la_tokens": la["total_tokens"],
            "va_calls": va["calls"],
            "va_tokens": va["total_tokens"],
            "total_calls": la["calls"] + va["calls"],
            "total_tokens": la["total_tokens"] + va["total_tokens"],
        }
