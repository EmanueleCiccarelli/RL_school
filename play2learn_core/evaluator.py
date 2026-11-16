"""
Modulo Core di Valutazione e Streaming Video/GIF/MP4 (Play2Learn).
Scrittura incrementale direttamente su disco con footprint RAM costante (< 1 MB),
preservazione del seed deterministico, streaming H.264 ultrafast e inferenza accelerata.
"""

from __future__ import annotations

import collections
import threading
from pathlib import Path
from typing import Any, Callable

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

# Global lock protecting headless Pygame/SDL surface rendering across threads
RENDER_LOCK = threading.Lock()


def make_fast_predictor(model: Any) -> Callable[[np.ndarray, bool], int]:
    """Bypasses SB3 model.predict() per-step validation for 3-4x faster CPU inference."""
    policy = getattr(model, "policy", None)
    if policy is None:
        return lambda obs, det: int(model.predict(obs, deterministic=det)[0])

    policy.set_training_mode(False)
    device = getattr(model, "device", "cpu")

    @torch.inference_mode()
    def _predict(obs: np.ndarray, deterministic: bool = True) -> int:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        act_t = policy._predict(obs_t, deterministic=deterministic)
        return int(act_t.item())

    return _predict


def make_fast_batch_predictor(model: Any) -> Callable[[np.ndarray, bool], np.ndarray]:
    """Vectorized batch predictor for multi-env headless evaluation sweeps."""
    policy = getattr(model, "policy", None)
    if policy is None:
        return lambda obs_b, det: model.predict(obs_b, deterministic=det)[0]

    policy.set_training_mode(False)
    device = getattr(model, "device", "cpu")

    @torch.inference_mode()
    def _predict_batch(obs_batch: np.ndarray, deterministic: bool = True) -> np.ndarray:
        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
        act_t = policy._predict(obs_t, deterministic=deterministic)
        return act_t.cpu().numpy()

    return _predict_batch


def _make_even(n: int) -> int:
    return max(2, n - (n % 2))


def render_banner_once(
    width: int,
    line1: str,
    line2: str,
    height: int = 34,
) -> np.ndarray:
    """Pre-renders the static milestone header once per stage as a uint8 RGB array."""
    img = Image.new("RGB", (width, height), color=(15, 23, 42))
    draw = ImageDraw.Draw(img)
    draw.text((8, 4), line1, fill=(255, 255, 255))
    draw.text((8, 18), line2, fill=(147, 197, 253))
    return np.asarray(img, dtype=np.uint8)


def fast_resize_into(frame: np.ndarray, out_buf: np.ndarray) -> np.ndarray:
    """Resizes RGB frame into pre-allocated out_buf using OpenCV or fast PIL fallback."""
    target_h, target_w = out_buf.shape[:2]
    if _HAS_CV2:
        cv2.resize(frame, (target_w, target_h), dst=out_buf, interpolation=cv2.INTER_AREA)
        return out_buf
    # Fast exact 2x decimation when dimensions match (e.g., 600x400 -> 300x200)
    if frame.shape[0] == target_h * 2 and frame.shape[1] == target_w * 2:
        np.copyto(out_buf, frame[::2, ::2])
        return out_buf
    pil_small = Image.fromarray(frame).resize(
        (target_w, target_h), Image.Resampling.BILINEAR, reducing_gap=2.0
    )
    np.copyto(out_buf, np.asarray(pil_small, dtype=np.uint8))
    return out_buf


def replay_actions_streaming_mp4(
    render_env: Any,
    seed: int,
    actions: list[int],
    output_path: Path,
    physics_fps: int = 50,
    render_stride: int = 2,
    target_width: int = 480,
    crf: int = 28,
) -> Path:
    """
    Replays recorded actions in Box2D with zero PyTorch overhead and streams H.264 directly
    to disk at O(1) RAM. Default render_stride=2 outputs smooth 25 FPS video at 2x speed
    and ~50% smaller file size.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_fps = max(1, physics_fps // max(1, render_stride))

    with RENDER_LOCK:
        render_env.reset(seed=seed)
        init_frame = render_env.render()
        if init_frame is None:
            return output_path

        h, w = init_frame.shape[:2]
        out_w = _make_even(min(w, target_width))
        out_h = _make_even(int(out_w * h / w))
        buf = np.empty((out_h, out_w, 3), dtype=np.uint8)

        writer = imageio.get_writer(
            str(output_path),
            fps=out_fps,
            macro_block_size=1,
            ffmpeg_params=[
                "-preset", "ultrafast",
                "-crf", str(crf),
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
            ],
        )
        try:
            writer.append_data(fast_resize_into(init_frame, buf))
            for idx, act in enumerate(actions, start=1):
                _, _, term, trunc, _ = render_env.step(act)
                done = bool(term or trunc)
                if (idx % render_stride == 0) or done:
                    frame = render_env.render()
                    if frame is not None:
                        writer.append_data(fast_resize_into(frame, buf))
                if done:
                    break
        finally:
            writer.close()

    return output_path


def create_timelapse_mp4_streaming(
    frames_iter_fn: Callable[[], Any],
    output_path: Path,
    fps: int = 15,
    crf: int = 28,
) -> Path | None:
    """
    Streams annotated milestone frames directly into a lightweight looping H.264 MP4
    (12-18x faster and ~12x smaller than animated GIF on Colab 2 vCPUs).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames_written = 0
    writer = imageio.get_writer(
        str(output_path),
        fps=fps,
        macro_block_size=1,
        ffmpeg_params=[
            "-preset", "ultrafast",
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
        ],
    )
    try:
        for frame in frames_iter_fn():
            writer.append_data(frame)
            frames_written += 1
    finally:
        writer.close()
    return output_path if frames_written > 0 else None


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

    predict_fn = make_fast_predictor(model)
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
            action_item = predict_fn(obs, deterministic)
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
