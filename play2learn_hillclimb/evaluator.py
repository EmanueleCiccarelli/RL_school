"""
Modulo di Valutazione Deterministica e Generazione Video per Hill Climb Racing.
Architettura a 2 fasi (sweep rapido headless + replay streaming dei video migliori e peggiori)
e generazione GIF timelapse con caricamento fedele degli state_dict dei checkpoint.
"""

from __future__ import annotations

import collections
import time
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch
from PIL import Image, ImageDraw

from play2learn_core.evaluator import (
    create_milestone_gif,
    record_episode_streaming,
    make_fast_predictor,
    RENDER_LOCK,
)
from play2learn_hillclimb.config import ENV_ID, EVAL_SEEDS, MIN_EVAL_EPISODES
from play2learn_hillclimb.environment import HillClimbRewardWrapper, classify_hill_climb_outcome


def _evaluate_single_episode_fast(
    env: gym.Env,
    model: Any,
    seed: int,
    max_steps: int = 1200,
    reward_fn: Callable | None = None,
    predict_fn: Callable[[np.ndarray, bool], int] | None = None,
) -> dict[str, Any]:
    """Valuta un episodio in modalita puramente headless senza rendering Pygame."""
    obs, info = env.reset(seed=seed)
    tot_rew = 0.0
    steps = 0
    terminated = False
    truncated = False
    is_success = False
    pred = predict_fn or make_fast_predictor(model)

    while not (terminated or truncated) and steps < max_steps:
        act_item = pred(obs, True)
        obs, reward, terminated, truncated, info = env.step(act_item)
        tot_rew += float(reward)
        steps += 1

    unwrapped = getattr(env, "unwrapped", env)
    succ, crash, tout, flip = classify_hill_climb_outcome(
        obs=obs,
        original_reward=tot_rew,
        terminated=terminated,
        truncated=truncated,
        unwrapped=unwrapped,
    )

    telemetry = info.get("telemetry", {})
    dist = float(telemetry.get("max_distance", info.get("score", 0.0)))
    airtime = int(telemetry.get("airtime_steps", 0))
    air_pct = (airtime / max(1, steps)) * 100.0

    return {
        "seed": seed,
        "distance": dist,
        "reward": tot_rew,
        "steps": steps,
        "max_speed": float(telemetry.get("max_speed", 0.0)),
        "airtime_pct": air_pct,
        "is_success": succ,
        "is_crashed": crash,
        "is_timeout": tout,
        "is_flipped": flip,
    }


def evaluate_and_record(
    model: Any,
    output_dir: Path | str,
    n_episodes: int = MIN_EVAL_EPISODES,
    seeds: list[int] | None = None,
    env_kwargs: dict[str, Any] | None = None,
    reward_fn: Callable | None = None,
    sweep_timeout_seconds: float = 60.0,
    max_steps_per_episode: int = 1200,
    max_videos_to_record: int = 6,
) -> dict[str, Any]:
    """
    Valuta l'agente con architettura a 2 fasi:
    1. Sweep rapido headless (render_mode=None) su tutti gli scenari deterministici (1-2s).
    2. Registrazione video in streaming (H.264 O(1) RAM) solo per i migliori successi e fallimenti.
    """
    out_dir = Path(output_dir)
    videos_dir = out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    actual_seeds = list(seeds or EVAL_SEEDS)[:n_episodes]
    final_kwargs = dict(env_kwargs or {})

    # FASE 1: Sweep rapido deterministico headless
    history: list[dict[str, Any]] = []
    fast_env = gym.make(ENV_ID, render_mode=None, **final_kwargs)
    if reward_fn is not None:
        fast_env = HillClimbRewardWrapper(fast_env, reward_fn=reward_fn)

    pred_fn = make_fast_predictor(model)
    try:
        for idx, s in enumerate(actual_seeds):
            ep_res = _evaluate_single_episode_fast(
                env=fast_env,
                model=model,
                seed=s,
                max_steps=max_steps_per_episode,
                reward_fn=reward_fn,
                predict_fn=pred_fn,
            )
            ep_res["episode"] = idx + 1
            ep_res["video_path"] = None
            history.append(ep_res)
    finally:
        fast_env.close()

    # Selezione episodi da registrare (es. fino a 3 successi e 3 fallimenti)
    succ_eps = [h for h in history if h["is_success"]]
    fail_eps = [h for h in history if not h["is_success"]]

    # Ordina per distanza decrescente
    succ_eps.sort(key=lambda x: x["distance"], reverse=True)
    fail_eps.sort(key=lambda x: x["distance"], reverse=True)

    vids_to_record: list[dict[str, Any]] = []
    vids_to_record.extend(succ_eps[:3])
    vids_to_record.extend(fail_eps[:3])
    if len(vids_to_record) < max_videos_to_record and history:
        for h in history:
            if h not in vids_to_record and len(vids_to_record) < max_videos_to_record:
                vids_to_record.append(h)

    # FASE 2: Registrazione video in streaming H.264
    success_videos: list[Path] = []
    failure_videos: list[Path] = []
    all_videos: list[Path] = []

    render_env = gym.make(ENV_ID, render_mode="rgb_array", **final_kwargs)
    if reward_fn is not None:
        render_env = HillClimbRewardWrapper(render_env, reward_fn=reward_fn)

    try:
        for ep_info in vids_to_record:
            s = ep_info["seed"]
            idx = ep_info["episode"]
            vid_path = videos_dir / f"episodio_{idx:02d}_seed_{s}.mp4"

            tot_rew, steps, is_succ, info = record_episode_streaming(
                env=render_env,
                model=model,
                video_path=vid_path,
                fps=50,
                max_steps=max_steps_per_episode,
                seed=s,
                outcome_fn=classify_hill_climb_outcome,
            )
            ep_info["video_path"] = str(vid_path)
            all_videos.append(vid_path)
            if is_succ:
                success_videos.append(vid_path)
            else:
                failure_videos.append(vid_path)
    finally:
        render_env.close()

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
    base_model: Any = None,
    seed: int = 142,
    max_steps_per_ckpt: int = 1000,
) -> Path | None:
    """
    Genera la GIF timelapse che illustra visivamente l'avanzamento dell'auto nelle colline,
    ricostruendo correttamente i pesi dai checkpoint state_dict (con ripristino dei pesi originali).
    """
    if not checkpoints:
        return None

    import imageio

    gif_path = Path(output_gif_path)
    gif_path.parent.mkdir(parents=True, exist_ok=True)

    orig_state = None
    if base_model is not None and hasattr(base_model, "policy"):
        try:
            orig_state = {k: v.detach().cpu().clone() for k, v in base_model.policy.state_dict().items()}
        except Exception:
            orig_state = None

    final_kwargs = dict(env_kwargs or {})
    env = gym.make(ENV_ID, render_mode="rgb_array", **final_kwargs)
    if reward_fn is not None:
        env = HillClimbRewardWrapper(env, reward_fn=reward_fn)

    frames_written = 0
    try:
        with imageio.get_writer(str(gif_path), mode="I", fps=12, loop=0) as writer:
            for timesteps, ckpt_or_model in checkpoints:
                # Ripristina i pesi se e uno state_dict salvato in RAM
                if isinstance(ckpt_or_model, dict) and base_model is not None:
                    try:
                        base_model.policy.load_state_dict(ckpt_or_model)
                        active_model = base_model
                    except Exception:
                        continue
                else:
                    active_model = ckpt_or_model

                pred_fn = make_fast_predictor(active_model)
                obs, _ = env.reset(seed=seed)
                step_count = 0
                terminated = False
                truncated = False

                while not (terminated or truncated) and step_count < max_steps_per_ckpt:
                    act_item = pred_fn(obs, True)
                    obs, reward, terminated, truncated, info = env.step(act_item)
                    step_count += 1

                    if step_count % 3 == 0 or terminated or truncated:
                        frame = env.render()
                        if frame is not None:
                            img = Image.fromarray(frame)
                            w, h = img.size
                            new_w = 480
                            new_h = int(h * (new_w / w))
                            img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

                            draw = ImageDraw.Draw(img)
                            dist = float(obs[0]) if hasattr(obs, "__getitem__") else 0.0
                            txt = f"Fase: {timesteps//1000}k step | Distanza: {dist:.1f} m"
                            draw.rectangle([10, 10, 260, 32], fill=(15, 23, 42, 220))
                            draw.text((16, 14), txt, fill=(56, 189, 248))
                            frame_arr = np.asarray(img)
                            writer.append_data(frame_arr)
                            frames_written += 1

                # Congela l'ultimo frame per 8 frame (~0.6s) per apprezzare il risultato raggiunto
                if 'frame_arr' in locals() and frame_arr is not None:
                    for _ in range(8):
                        writer.append_data(frame_arr)
                        frames_written += 1
    finally:
        env.close()
        if orig_state is not None and base_model is not None and hasattr(base_model, "policy"):
            base_model.policy.load_state_dict(orig_state)

    return gif_path if frames_written > 0 else None
