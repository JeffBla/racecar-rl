"""Shared helpers for RaceCar observation preprocessing."""

from __future__ import annotations

import cv2
import numpy as np


def process_obs(obs: np.ndarray) -> np.ndarray:
    """Convert RGB obs (C,H,W) to grayscale 1x96x96 uint8."""
    if obs.ndim != 3:
        raise ValueError(f"Expected observation with 3 dims (C,H,W), received shape {obs.shape}")
    frame = cv2.cvtColor(obs.transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (96, 96))
    return np.expand_dims(frame.astype(np.uint8), axis=0)
