"""
Modulo di Valutazione Deterministica e Generazione Video per Hill Climb Racing.
Scrittura streaming diretta con impatto memoria trascurabile (~1 MB).
"""

from __future__ import annotations

import collections
import time
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from play2learn_core.evaluator import create_milestone_gif, record_episode_streaming
from play2learn_hillclimb.config import ENV_ID, EVAL_SEEDS, MIN_EVAL_EPISODES
from play2learn_hillclimb.environment import HillClimbRewardWrapper, classify_hill_climb_outcome


def evaluate_and_record(
    model: Any,
    output_dir: Path | str,
    n_episodes: int = MIN_EVAL_EPISODES,
    seeds: list[int] | None = None,
    env_kwargs: dict[str, Any] | None = None,
    reward_fn: Callable | None = None,
    sweep_timeout_seconds: float = 60.0,
    max_steps_per_episode: int = 1200,
) -> dict[str, Any]:
    """Valuta l'agente su 20 scenari collinari deterministici e salva video ad alta fedeltà."""
    out_dir = Path(output_dir)
    videos_dir = out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    actual_seeds = list(seeds or EVAL_SEEDS)[:n_episodes]
    final_kwargs = dict(env_kwargs or {})

    history: list[dict[str, Any]] = []
    success_videos: list[Path] = []
    failure_videos: list[Path] = []
    all_videos: list[Path] = []

    start_time = time.time()
    hard_timeout = max(sweep_timeout_seconds * 2.5, 20.0)

    for idx, s in enumerate(actual_seeds):
        if time.time() - start_time > hard_timeout:
            print(f"[play2learn] Timeout di sicurezza raggiunto dopo {idx} episodi di verifica.")
            break

        vid_path = videos_dir / f"episodio_{idx+1:02d}_seed_{s}.mp4"

        env = gym.make(ENV_ID, render_mode="rgb_array", **final_kwargs)
        if reward_fn is not None:
            env = HillClimbRewardWrapper(env, reward_fn=reward_fn)

        env.reset(seed=s)

        try:
            tot_rew, steps, is_succ, info = record_episode_streaming(
                env=env,
                model=model,
                video_path=vid_path,
                fps=50,
                max_steps=max_steps_per_episode,
                outcome_fn=classify_hill_climb_outcome,
            )
        finally:
            env.close()

        all_videos.append(vid_path)
        if is_succ:
            success_videos.append(vid_path)
        else:
            failure_videos.append(vid_path)

        telemetry = info.get("telemetry", {})
        dist = float(telemetry.get("max_distance", info.get("score", 0.0)))
        airtime = int(telemetry.get("airtime_steps", 0))
        air_pct = (airtime / max(1, steps)) * 100.0

        history.append({
            "episode": idx + 1,
            "seed": s,
            "distance": dist,
            "reward": tot_rew,
            "steps": steps,
            "max_speed": float(telemetry.get("max_speed", 0.0)),
            "airtime_pct": air_pct,
            "is_success": is_succ,
            "is_crashed": bool(info.get("is_crashed", False)),
            "is_timeout": bool(info.get("is_timeout", False)),
            "video_path": str(vid_path),
        })

    ep_count = max(1, len(history))
    succ_count = sum(1 for h in history if h["is_success"])
    succ_rate = (succ_count / ep_count) * 100.0
    mean_dist = float(np.mean([h["distance"] for h in history])) if history else 0.0
    max_dist = float(np.max([h["distance"] for h in history])) if history else 0.0
    mean_rew = float(np.mean([h["reward"] for h in history])) if history else 0.0
    mean_speed = float(np.mean([h["max_speed"] for h in history])) if history else 0.0
    mean_air = float(np.mean([h["airtime_pct"] for h in history])) if history else 0.0

    return {
        "eval_success_rate": succ_rate,
        "mean_distance": mean_dist,
        "max_distance_achieved": max_dist,
        "mean_reward": mean_rew,
        "mean_speed": mean_speed,
        "mean_airtime_pct": mean_air,
        "eval_successes": succ_count,
        "eval_crashes": sum(1 for h in history if h["is_crashed"]),
        "eval_timeouts": sum(1 for h in history if h["is_timeout"]),
        "episodes": ep_count,
        "history": history,
        "videos": all_videos,
        "success_videos": success_videos,
        "failure_videos": failure_videos,
        "best_video_path": (success_videos[0] if success_videos else (all_videos[0] if all_videos else None)),
    }


def crea_gif_evoluzione_apprendimento(
    checkpoints: list[tuple[int, Any]],
    output_gif_path: Path | str,
    env_kwargs: dict[str, Any] | None = None,
    reward_fn: Callable | None = None,
    seed: int = 142,
    max_frames_per_ckpt: int = 40,
) -> Path | None:
    """Genera la GIF timelapse che illustra visivamente l'avanzamento dell'auto nelle colline."""
    if not checkpoints:
        return None

    gif_path = Path(output_gif_path)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    frames_all: list[np.ndarray] = []

    final_kwargs = dict(env_kwargs or {})

    for timesteps, model in checkpoints:
        env = gym.make(ENV_ID, render_mode="rgb_array", **final_kwargs)
        if reward_fn is not None:
            env = HillClimbRewardWrapper(env, reward_fn=reward_fn)

        obs, _ = env.reset(seed=seed)
        step_count = 0
        terminated = False
        truncated = False

        while not (terminated or truncated) and step_count < max_frames_per_ckpt:
            action, _ = model.predict(obs, deterministic=True)
            act_item = action.item() if hasattr(action, "item") and not hasattr(action, "__len__") else action
            obs, reward, terminated, truncated, info = env.step(act_item)
            step_count += 1

            if step_count % 3 == 0:
                frame = env.render()
                if frame is not None:
                    # Sovraimpressione didattica del checkpoint
                    img = Image.fromarray(frame)
                    draw = ImageDraw.Draw(img)
                    dist = float(obs[0]) if hasattr(obs, "__getitem__") else 0.0
                    txt = f"Fase: {timesteps//1000}k step | Distanza: {dist:.1f} m"
                    draw.rectangle([10, 10, 360, 40], fill=(15, 23, 42, 220))
                    draw.text((18, 16), txt, fill=(56, 189, 248))
                    frames_all.append(np.array(img))

        env.close()

    if frames_all:
        return create_milestone_gif(frames_all, gif_path, fps=10)
    return None
