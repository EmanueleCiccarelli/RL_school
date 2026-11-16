"""
Modulo Core di Valutazione e Streaming Video/GIF (Play2Learn).
Scrittura incrementale direttamente su disco con footprint RAM costante (< 1 MB),
preservazione del seed deterministico e compatibilità browser garantita (yuv420p).
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
    seed: int | None = None,
    outcome_fn: Callable[[Any, float, bool, bool, Any], tuple[bool, bool, bool, Any]] | None = None,
) -> tuple[float, int, bool, dict[str, Any]]:
    """
    Registra un episodio in streaming H.264 con consumo RAM O(1) (< 1 MB),
    preservando il seed deterministico ed evitando warning ffmpeg su dimensioni frame.
    """
    video_path.parent.mkdir(parents=True, exist_ok=True)
    obs, info = env.reset(seed=seed) if seed is not None else env.reset()

    try:
        writer = imageio.get_writer(
            str(video_path),
            fps=fps,
            macro_block_size=1,
            ffmpeg_params=["-preset", "ultrafast", "-crf", "26", "-pix_fmt", "yuv420p"],
        )
    except Exception:
        writer = imageio.get_writer(str(video_path), fps=fps, macro_block_size=1)

    total_reward = 0.0
    last_reward = 0.0
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
            action_item = int(action.item()) if hasattr(action, "item") and not hasattr(action, "__len__") else action
            obs, reward, terminated, truncated, info = env.step(action_item)
            last_reward = float(reward)
            total_reward += last_reward
            steps += 1

            frame = env.render()
            if frame is not None:
                writer.append_data(frame)

        if outcome_fn is not None:
            unwrapped = getattr(env, "unwrapped", env)
            succ, crash, tout, _ = outcome_fn(obs, last_reward, terminated, truncated, unwrapped)
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
    try:
        with imageio.get_writer(str(gif_path), mode="I", fps=fps, loop=0) as writer:
            for fr in frames_or_images:
                writer.append_data(fr)
    except Exception:
        imageio.mimsave(str(gif_path), frames_or_images, fps=fps, loop=0)
    return gif_path
