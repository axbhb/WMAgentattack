"""Leakage-resistant trajectory-level splitting."""

from __future__ import annotations

import random

from wmagentattack.schema import TrajectoryRecord


def split_trajectories(
    trajectories: list[TrajectoryRecord],
    seed: int = 7,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
):
    shuffled = list(trajectories)
    random.Random(seed).shuffle(shuffled)
    train_end = int(len(shuffled) * train_ratio)
    val_end = train_end + int(len(shuffled) * val_ratio)
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }

