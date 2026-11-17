"""
Modulo di Valutazione e Registrazione Video Streaming per Play2Learn Hill Climb Racing.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Callable, Optional
import numpy as np

import gymnasium as gym

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

from play2learn_core.evaluator import (
    make_fast_predictor,
    replay_actions_streaming_mp4,
    record_episode_streaming,
    fast_resize_into,
    render_banner_once,
    create_timelapse_mp4_streaming,
)
from play2learn_hillclimb.config import (
    ENV_ID,
    MISSIONS,
    TARGET_DISTANCE_SUCCESS,
)
from play2learn_hillclimb.environment import HillClimbRewardWrapper


EVAL_SEEDS = [
    42, 137, 211, 389, 523, 641, 757, 863, 977, 1087,
    1193, 1301, 1427, 1543, 1657, 1789, 1901, 2017, 2137, 2251,
]


def evaluate_and_record(
    model: Any,
    mission: str = "campagna_standard",
    reward_fn: Optional[Callable] = None,
    output_dir: Optional[Path] = None,
    eval_seeds: list[int] = EVAL_SEEDS,
    max_steps: int = 1600,
    frame_skip: int = 3,
    max_video_successes: int = 3,
    max_video_failures: int = 3,
) -> dict[str, Any]:
    """
    Esegue la valutazione su seed deterministici registrando video H.264 streaming ad alta efficienza.
    """
    if not eval_seeds:
        return {
            "success_rate": 0.0,
            "crash_rate": 0.0,
            "timeout_rate": 0.0,
            "eval_success_rate": 0.0,
            "mean_distance": 0.0,
            "max_distance": 0.0,
            "mean_speed": 0.0,
            "mean_airtime_pct": 0.0,
            "mean_reward": 0.0,
            "total_episodes": 0,
            "history": [],
            "video_paths": [],
            "video_filenames": [],
        }

    mission_cfg = MISSIONS.get(mission, MISSIONS["campagna_standard"])
    difficulty = float(mission_cfg.get("difficulty", -150.0))
    original_noise = bool(mission_cfg.get("original_noise", False))

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    def _env_factory(render_mode: Optional[str] = "rgb_array", fskip: int = frame_skip) -> gym.Env:
        raw = gym.make(
            ENV_ID,
            render_mode=render_mode,
            action_space="discrete_3",
            reward_type="aggressive",
            difficulty=difficulty,
            original_noise=original_noise,
        )
        return HillClimbRewardWrapper(
            raw,
            reward_fn=reward_fn,
            target_distance=TARGET_DISTANCE_SUCCESS,
            frame_skip=fskip,
        )

    if model is not None and hasattr(model, "policy"):
        predict_fn = make_fast_predictor(model)
    elif model is not None and hasattr(model, "predict"):
        predict_fn = lambda obs, det=True: int(model.predict(obs, deterministic=det)[0])
    else:
        predict_fn = lambda obs, det=True: 1

    success_count = 0
    crash_count = 0
    timeout_count = 0

    distances: list[float] = []
    speeds: list[float] = []
    airtimes: list[float] = []
    flips_list: list[int] = []
    video_paths: list[Path] = []
    history: list[dict[str, Any]] = []

    saved_succ = 0
    saved_fail = 0

    eval_env = _env_factory(render_mode=None, fskip=frame_skip)
    try:
        for idx, seed in enumerate(eval_seeds):
            obs, info = eval_env.reset(seed=seed)
            step_count = 0
            ep_reward = 0.0
            actions: list[int] = []

            while step_count < max_steps:
                action = int(predict_fn(obs, True))
                actions.append(action)
                obs, reward, term, trunc, info = eval_env.step(action)
                ep_reward += float(reward)
                step_count += 1
                if term or trunc:
                    break

            telem = info.get("telemetry", {})
            dist = float(telem.get("distance", telem.get("max_distance", obs[0])))
            spd = float(telem.get("max_speed", 0.0))
            air = float(telem.get("airtime_pct", 0.0))
            flp = int(telem.get("flips", 0))

            is_succ = bool(telem.get("is_success", dist >= TARGET_DISTANCE_SUCCESS))
            is_crash = bool(telem.get("is_crashed", False))
            is_tout = bool(telem.get("is_timeout", not is_succ and not is_crash))

            if is_succ:
                success_count += 1
            elif is_crash:
                crash_count += 1
            else:
                timeout_count += 1

            distances.append(dist)
            speeds.append(spd)
            airtimes.append(air)
            flips_list.append(flp)

            record_entry = {
                "seed": seed,
                "distance": dist,
                "max_distance": dist,
                "max_speed": spd,
                "airtime_pct": air,
                "flips": flp,
                "is_success": is_succ,
                "is_crashed": is_crash,
                "is_timeout": is_tout,
                "reward": ep_reward,
            }
            history.append(record_entry)

            # 2. Registrazione streaming video su replay veloce delle azioni
            should_record = False
            outcome_tag = ""
            if is_succ and saved_succ < max_video_successes:
                should_record = True
                saved_succ += 1
                outcome_tag = "successo"
            elif not is_succ and saved_fail < max_video_failures:
                should_record = True
                saved_fail += 1
                outcome_tag = "fallimento"

            if should_record and output_dir is not None:
                vid_filename = f"episodio_{idx+1:02d}_{outcome_tag}_seed_{seed}.mp4"
                vid_path = output_dir / vid_filename
                rec_env = None
                try:
                    rec_env = _env_factory(render_mode="rgb_array", fskip=1)
                    expanded_actions = [a for a in actions for _ in range(frame_skip)]
                    replay_actions_streaming_mp4(
                        render_env=rec_env,
                        seed=seed,
                        actions=expanded_actions,
                        output_path=vid_path,
                        physics_fps=60,
                        render_stride=2,
                        target_width=640,
                        crf=28,
                    )
                    if vid_path.exists():
                        video_paths.append(vid_path)
                except Exception as e:
                    print(f"[play2learn] Errore video stream {vid_filename}: {e}")
                finally:
                    if rec_env is not None:
                        rec_env.close()
    finally:
        eval_env.close()

    total_runs = len(eval_seeds)
    succ_rate = (success_count / total_runs) * 100.0
    crash_rate = (crash_count / total_runs) * 100.0
    timeout_rate = (timeout_count / total_runs) * 100.0

    mean_dist = float(np.mean(distances)) if distances else 0.0
    max_dist = float(np.max(distances)) if distances else 0.0
    mean_speed = float(np.mean(speeds)) if speeds else 0.0
    mean_air = float(np.mean(airtimes)) if airtimes else 0.0

    results = {
        "success_rate": succ_rate,
        "crash_rate": crash_rate,
        "timeout_rate": timeout_rate,
        "eval_success_rate": succ_rate,
        "mean_distance": mean_dist,
        "max_distance": max_dist,
        "mean_speed": mean_speed,
        "mean_airtime_pct": mean_air,
        "total_episodes": total_runs,
        "history": history,
        "video_paths": video_paths,
        "video_filenames": [p.name for p in video_paths],
    }
    return results


def crea_gif_evoluzione_apprendimento(
    milestones: list[tuple[int, Any]],
    output_path: Path,
    model: Any = None,
    mission: str = "campagna_standard",
    seed: int = 42,
    max_steps_per_milestone: int = 150,
) -> Optional[Path]:
    """Genera una GIF animata mostrando come l'agente migliora attraverso i checkpoint."""
    if not milestones or not _PIL_AVAILABLE:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mission_cfg = MISSIONS.get(mission, MISSIONS["campagna_standard"])
    difficulty = float(mission_cfg.get("difficulty", -150.0))
    original_noise = bool(mission_cfg.get("original_noise", False))

    frames: list[Image.Image] = []

    final_weights = None
    if model is not None and hasattr(model, "policy"):
        try:
            final_weights = {k: v.detach().cpu().clone() for k, v in model.policy.state_dict().items()}
        except Exception:
            final_weights = None

    try:
        for step_num, snap in milestones:
            if isinstance(snap, dict) and model is not None and hasattr(model, "policy"):
                try:
                    model.policy.load_state_dict(snap)
                    pred = make_fast_predictor(model)
                except Exception:
                    pred = lambda obs, det=True: 1
            elif hasattr(snap, "predict"):
                pred = lambda obs, det=True: int(snap.predict(obs, deterministic=det)[0])
            else:
                pred = lambda obs, det=True: 1

            raw = gym.make(
                ENV_ID,
                render_mode="rgb_array",
                action_space="discrete_3",
                reward_type="aggressive",
                difficulty=difficulty,
                original_noise=original_noise,
            )
            env = HillClimbRewardWrapper(raw)
            obs, _ = env.reset(seed=seed)

            try:
                for s in range(max_steps_per_milestone):
                    act = pred(obs, True)
                    obs, r, term, trunc, _ = env.step(act)
                    if s % 3 == 0:
                        frame_arr = env.render()
                        if frame_arr is not None:
                            im = Image.fromarray(frame_arr)
                            w, h = im.size
                            frames.append(im.resize((w // 2, h // 2)))
                    if term or trunc:
                        break
            finally:
                env.close()
    finally:
        if final_weights is not None and model is not None and hasattr(model, "policy"):
            try:
                model.policy.load_state_dict(final_weights)
            except Exception:
                pass

    if len(frames) > 1:
        frames[0].save(
            str(output_path),
            save_all=True,
            append_images=frames[1:],
            duration=60,
            loop=0,
            optimize=True,
        )
        return output_path
    return None
