"""
Modulo Core di Valutazione e Streaming Video/GIF (Play2Learn).
Scrittura incrementale direttamente su disco con footprint RAM costante (~1 MB).
"""

from __future__ import annotations

import collections
from pathlib import Path
from typing import Any, Callable

import imageio
import numpy as np


def record_episode_streaming(
    env: Any,
    model: Any,
    video_path: Path,
    fps: int = 50,
    max_steps: int = 1500,
    deterministic: bool = True,
    outcome_fn: Callable[[Any, float, bool, bool, Any], tuple[bool, bool, bool, Any]] | None = None,
) -> tuple[float, int, bool, dict[str, Any]]:
    """
    Registra un episodio in streaming direttamente su file mp4 (H.264).
    Non memorizza l'intera sequenza di frame in memoria RAM (consumo costante < 1 MB).
    """
    video_path.parent.mkdir(parents=True, exist_ok=True)
    obs, info = env.reset()

    writer = imageio.get_writer(
        str(video_path),
        fps=fps,
        codec="libx264",
        output_params=["-pix_fmt", "yuv420p"],
    )

    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    is_success = False

    try:
        frame = env.render()
        if frame is not None:
            writer.append_data(frame)

        while not (terminated or truncated) and steps < max_steps:
            action, _ = model.predict(obs, deterministic=deterministic)
            action_item = action.item() if hasattr(action, "item") and not hasattr(action, "__len__") else action
            obs, reward, terminated, truncated, info = env.step(action_item)
            total_reward += float(reward)
            steps += 1

            frame = env.render()
            if frame is not None:
                writer.append_data(frame)

        if outcome_fn is not None:
            unwrapped = getattr(env, "unwrapped", env)
            succ, crash, tout, _ = outcome_fn(obs, reward, terminated, truncated, unwrapped)
            is_success = succ
        else:
            is_success = bool(info.get("is_success", False))

    finally:
        writer.close()

    return total_reward, steps, is_success, info


def create_milestone_gif(
    frames_or_images: list[np.ndarray],
    gif_path: Path,
    fps: int = 10,
) -> Path:
    """Crea una GIF animata dai frame estratti dai checkpoint di apprendimento."""
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(gif_path), frames_or_images, fps=fps, loop=0)
    return gif_path
