"""Helpers for writing strict JSON execution artifacts."""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def json_safe(value: Any) -> Any:
    """Replace non-finite numeric audit values with JSON ``null`` values."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value
