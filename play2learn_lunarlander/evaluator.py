"""
Modulo di valutazione e registrazione video/GIF ad alte prestazioni per Lunar Lander.
Procedura in 2 fasi:
- Fase 1: Fast Sweep deterministico senza rendering (inferenza diretta PyTorch @torch.inference_mode()).
- Fase 2: Replay fedele su Box2D dei soli target selezionati (fino a 2-3 successi e fallimenti),
  encoding MP4 ultrafast e generazione GIF diretta dai frame in RAM.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Any

import numpy as np
import torch

try:
    import gymnasium as gym
    from stable_baselines3 import PPO
    _GYM_AVAILABLE = True
except ImportError:
    gym = Any  # type: ignore
    PPO = Any  # type: ignore
    _GYM_AVAILABLE = False

from play2learn_lunarlander.environment import (
    CustomRewardWrapper,
    classify_lunar_lander_outcome,
)


def _make_fast_predictor(model: PPO) -> Callable[[np.ndarray, bool], int]:
    """
    Crea un predictor leggero che bypassa le validazioni per-step di SB3 `model.predict()`,
    accelerando lo sweep CPU di 3-4x.
    """
    policy = getattr(model, "policy", None)
    if policy is None:
        return lambda obs, det: int(model.predict(obs, deterministic=det)[0])

    policy.set_training_mode(False)
    device = model.device

    @torch.inference_mode()
    def _predict(obs: np.ndarray, deterministic: bool) -> int:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        act_t = policy._predict(obs_t, deterministic=deterministic)
        return int(act_t.item())

    return _predict


def _run_single_sweep_episode(
    env: gym.Env,
    predict_fn: Callable[[np.ndarray, bool], int],
    ep_seed: int,
    deterministic: bool,
    t_start: float,
    timeout_sec: float,
) -> tuple[dict[str, Any], bool]:
    """
    Esegue un singolo episodio completo di sweep.
    Se il timer scatta durante il volo, completa comunque l'episodio corrente.
    """
    obs, _ = env.reset(seed=ep_seed)
    done = False
    ep_rew = 0.0
    steps = 0
    ep_actions: list[int] = []
    last_info: dict[str, Any] = {}
    timeout_hit = False

    while not done:
        act_int = predict_fn(obs, deterministic)
        ep_actions.append(act_int)
        obs, rew, terminated, truncated, last_info = env.step(act_int)
        ep_rew += float(rew)
        steps += 1
        done = terminated or truncated

        if not timeout_hit and (time.perf_counter() - t_start >= timeout_sec):
            timeout_hit = True

    tel = last_info.get("telemetry", {})
    fuel = float(tel.get("fuel_used", 0.0))

    candidate = {
        "seed": ep_seed,
        "actions": ep_actions,
        "is_success": bool(last_info.get("is_success", False)),
        "is_crashed": bool(last_info.get("is_crashed", False)),
        "is_timeout": bool(last_info.get("is_timeout", False)),
        "is_off_target": bool(last_info.get("is_off_target", False)),
        "steps": steps,
        "fuel": fuel,
        "reward": ep_rew,
        "deterministic": deterministic,
    }
    return candidate, timeout_hit


def evaluate_and_record(
    model: PPO,
    env_id: str,
    reward_fn: Callable,
    video_dir: Path | str,
    target_successes: int = 2,
    target_failures: int = 2,
    sweep_timeout_seconds: float = 6.0,
    min_eval_episodes: int = 20,
    max_eval_episodes: int = 80,
    seed: int = 100,
    env_kwargs: dict[str, Any] | None = None,
    create_gif: bool = True,
    filename_prefix: str = "",
) -> dict[str, Any]:
    """
    Valuta l'agente in modo statisticamente imparziale (policy deterministica)
    e registra fino a `target_successes` successi e `target_failures` fallimenti.
    """
    env_kwargs = env_kwargs or {}
    video_dir = Path(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    predict_fn = _make_fast_predictor(model)

    try:
        sweep_env = gym.make(env_id, disable_env_checker=True, **env_kwargs)
    except TypeError:
        sweep_env = gym.make(env_id, **env_kwargs)
    sweep_env = CustomRewardWrapper(sweep_env, reward_fn)

    t_eval_start = time.perf_counter()
    eval_rewards: list[float] = []
    eval_lengths: list[int] = []
    eval_fuels: list[float] = []
    success_fuels: list[float] = []
    failure_fuels: list[float] = []
    eval_successes = 0
    eval_crashes = 0
    eval_timeouts = 0

    success_candidates: list[dict[str, Any]] = []
    failure_candidates: list[dict[str, Any]] = []

    # =========================================================================
    # FASE 1: FAST SWEEP DETERMINISTICO SENZA RENDERING
    # =========================================================================
    ep = 0
    timeout_hit = False

    while ep < max_eval_episodes and not timeout_hit:
        cand, timeout_hit = _run_single_sweep_episode(
            env=sweep_env,
            predict_fn=predict_fn,
            ep_seed=seed + ep,
            deterministic=True,
            t_start=t_eval_start,
            timeout_sec=sweep_timeout_seconds,
        )

        eval_rewards.append(cand["reward"])
        eval_lengths.append(cand["steps"])
        eval_fuels.append(cand["fuel"])

        if cand["is_success"]:
            eval_successes += 1
            success_fuels.append(cand["fuel"])
            success_candidates.append(cand)
        else:
            if cand["is_timeout"]:
                eval_timeouts += 1
            else:
                eval_crashes += 1
            failure_fuels.append(cand["fuel"])
            failure_candidates.append(cand)

        ep += 1
        have_enough_vids = (
            len(success_candidates) >= target_successes
            and len(failure_candidates) >= target_failures
        )
        if ep >= min_eval_episodes and have_enough_vids:
            break

    sweep_env.close()

    total_evaluated = max(1, len(eval_rewards))
    success_rate = (eval_successes / total_evaluated) * 100.0

    # =========================================================================
    # FASE 2: REPLAY MIRATO E RENDERING VIDEO (MIX DIDATTICO)
    # =========================================================================
    selected_successes = sorted(
        success_candidates,
        key=lambda c: c["fuel"],
    )[:target_successes]

    crashes_pool = [c for c in failure_candidates if c["is_crashed"]]
    timeouts_pool = [c for c in failure_candidates if c["is_timeout"]]
    selected_failures: list[dict[str, Any]] = []

    if crashes_pool and timeouts_pool and target_failures >= 2:
        selected_failures.append(crashes_pool[0])
        selected_failures.append(timeouts_pool[0])
    for c in failure_candidates:
        if len(selected_failures) >= target_failures:
            break
        if c not in selected_failures:
            selected_failures.append(c)

    all_selected = [(c, True) for c in selected_successes] + [(c, False) for c in selected_failures]

    success_videos: list[Path] = []
    failure_videos: list[Path] = []
    best_success_video: Path | None = None
    best_success_frames: list[np.ndarray] = []
    best_success_fuel = float("inf")
    pfx = f"{filename_prefix}_" if filename_prefix else ""

    if all_selected:
        try:
            render_env = gym.make(
                env_id,
                render_mode="rgb_array",
                disable_env_checker=True,
                **env_kwargs,
            )
        except TypeError:
            render_env = gym.make(env_id, render_mode="rgb_array", **env_kwargs)

        for cand, is_succ_cand in all_selected:
            render_env.reset(seed=cand["seed"])
            frames: list[np.ndarray] = []

            init_frame = render_env.render()
            if init_frame is not None:
                frames.append(init_frame)

            for act in cand["actions"]:
                _, _, term, trunc, _ = render_env.step(int(act))
                frame = render_env.render()
                if frame is not None:
                    frames.append(frame)
                if term or trunc:
                    break

            if not frames:
                continue

            if is_succ_cand:
                idx = len(success_videos) + 1
                v_path = video_dir / f"{pfx}successo_{idx}_step_{cand['steps']}.mp4"
                _save_frames_as_mp4(frames, v_path)
                success_videos.append(v_path)
                if cand["fuel"] < best_success_fuel:
                    best_success_fuel = cand["fuel"]
                    best_success_video = v_path
                    best_success_frames = frames
            else:
                idx = len(failure_videos) + 1
                causa = (
                    "fuori_bersaglio" if cand.get("is_off_target")
                    else ("crash" if cand["is_crashed"] else ("timeout" if cand["is_timeout"] else "fallimento"))
                )
                v_path = video_dir / f"{pfx}fallimento_{idx}_{causa}_step_{cand['steps']}.mp4"
                _save_frames_as_mp4(frames, v_path)
                failure_videos.append(v_path)

        render_env.close()

    # Genera GIF direttamente dai frame già in RAM (evita decode da disco)
    gif_path: Path | None = None
    if create_gif and best_success_video and best_success_frames:
        gif_path = _save_frames_as_gif(
            best_success_frames,
            best_success_video.with_suffix(".gif"),
        )

    return {
        "mean_reward": float(np.mean(eval_rewards)) if eval_rewards else 0.0,
        "std_reward": float(np.std(eval_rewards)) if eval_rewards else 0.0,
        "mean_length": float(np.mean(eval_lengths)) if eval_lengths else 0.0,
        "mean_fuel": float(np.mean(eval_fuels)) if eval_fuels else 0.0,
        "eval_success_rate": success_rate,
        "total_evaluated_episodes": total_evaluated,
        "eval_successes": eval_successes,
        "eval_crashes": eval_crashes,
        "eval_timeouts": eval_timeouts,
        "success_videos": success_videos,
        "failure_videos": failure_videos,
        "videos": success_videos + failure_videos,
        "gif_path": gif_path,
        "mean_fuel_success": float(np.mean(success_fuels)) if success_fuels else 0.0,
        "mean_fuel_failure": float(np.mean(failure_fuels)) if failure_fuels else 0.0,
    }


def _save_frames_as_mp4(frames: list[np.ndarray], output_path: Path, fps: int = 50) -> None:
    """Salva i frame in MP4 usando preset ultrafast (rapido e compatto)."""
    try:
        import imageio
        output_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(
            str(output_path),
            frames,
            fps=fps,
            macro_block_size=1,
            ffmpeg_params=["-preset", "ultrafast", "-crf", "26", "-pix_fmt", "yuv420p"],
        )
    except Exception:
        try:
            import imageio
            imageio.mimsave(str(output_path), frames, fps=fps, macro_block_size=1)
        except Exception as e:
            print(f"[play2learn] Errore nel salvataggio video {output_path.name}: {e}")


def _save_frames_as_gif(
    frames: list[np.ndarray],
    gif_output: Path,
    source_fps: int = 50,
    stride: int = 2,
) -> Path | None:
    """Crea la GIF animata direttamente dai frame in RAM campionati 1 ogni `stride`."""
    try:
        import imageio
        sampled = frames[::stride]
        if sampled:
            imageio.mimsave(
                str(gif_output),
                sampled,
                fps=max(10, source_fps // stride),
                loop=0,
            )
            return gif_output
    except Exception:
        pass
    return None


def crea_gif_evoluzione_apprendimento(
    model: Any,
    milestones: list[Any],
    env_id: str,
    output_path: Path | str,
    env_kwargs: dict[str, Any] | None = None,
    seed: int = 42,
    max_steps_per_episode: int = 1000,
    stride: int = 3,
    target_width: int = 400,
    fps: int = 20,
) -> Path | None:
    """
    Genera un timelapse video fluido e annotato che mostra visivamente come l'agente
    è passato dall'esplorazione iniziale (schianti) all'atterraggio morbido,
    confrontando le milestone di training nel medesimo scenario lunare.
    """
    if not milestones:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    env_kwargs = env_kwargs or {}

    try:
        import imageio
        import gymnasium as gym
    except ImportError as e:
        print(f"[play2learn] Modulo mancante per il video di evoluzione: {e}")
        return None

    policy = getattr(model, "policy", None)
    if policy is None:
        return None

    orig_state = {
        k: v.detach().cpu().clone()
        for k, v in policy.state_dict().items()
    }

    try:
        render_env = gym.make(
            env_id,
            render_mode="rgb_array",
            disable_env_checker=True,
            **env_kwargs,
        )
    except TypeError:
        render_env = gym.make(env_id, render_mode="rgb_array", **env_kwargs)

    if output_path.suffix.lower() == ".gif":
        output_path = output_path.with_suffix(".mp4")

    total_stages = len(milestones)

    try:
        def _open_timelapse():
            try:
                return imageio.get_writer(
                    str(output_path),
                    fps=fps,
                    macro_block_size=1,
                    ffmpeg_params=[
                        "-preset", "ultrafast",
                        "-crf", "28",
                        "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart",
                    ],
                )
            except Exception:
                return imageio.get_writer(str(output_path), fps=fps, macro_block_size=1)

        with _open_timelapse() as writer:
            for idx, item in enumerate(milestones):
                if isinstance(item, dict):
                    st = item.get("state_dict", item)
                    step_num = item.get("step", idx + 1)
                elif isinstance(item, (list, tuple)):
                    step_num, st = item[0], item[1]
                else:
                    continue

                policy.load_state_dict(st)
                predict_fn = _make_fast_predictor(model)

                obs, _ = render_env.reset(seed=seed)
                sampled_raw: list[np.ndarray] = []
                f_init = render_env.render()
                if f_init is not None:
                    sampled_raw.append(f_init)

                done = False
                step_count = 0

                while not done and step_count < max_steps_per_episode:
                    action = predict_fn(obs, deterministic=True)
                    obs, rew, term, trunc, _ = render_env.step(action)
                    step_count += 1
                    done = bool(term or trunc)

                    if step_count % stride == 0 or done:
                        f = render_env.render()
                        if f is not None:
                            sampled_raw.append(f)

                is_succ, is_crash, is_tout, _ = classify_lunar_lander_outcome(
                    obs=obs,
                    original_reward=0.0,
                    terminated=done,
                    truncated=False,
                    unwrapped=render_env.unwrapped,
                )
                esito_str = "Allunaggio Riuscito" if is_succ else ("Schianto" if is_crash else "Timeout")

                for frame_arr in sampled_raw:
                    annotated = _annotate_milestone_frame(
                        frame_arr,
                        stage_idx=idx + 1,
                        total_stages=total_stages,
                        step_num=step_num,
                        esito_str=esito_str,
                        target_width=target_width,
                    )
                    writer.append_data(annotated)

        return output_path
    except Exception as e:
        print(f"[play2learn] Errore nella generazione timelapse evoluzione: {e}")
        return None
    finally:
        render_env.close()
        policy.load_state_dict(orig_state)


def _annotate_milestone_frame(
    frame: np.ndarray,
    stage_idx: int,
    total_stages: int,
    step_num: int,
    esito_str: str,
    target_width: int = 400,
) -> np.ndarray:
    """Appone un banner informativo con stage, step di addestramento ed esito del volo."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.fromarray(frame)
        if target_width and img.width != target_width:
            aspect = img.height / img.width
            img = img.resize((target_width, int(target_width * aspect)), Image.Resampling.BILINEAR)

        draw = ImageDraw.Draw(img)
        banner_h = 28
        draw.rectangle([(0, 0), (img.width, banner_h)], fill=(20, 20, 30, 220))

        font = ImageFont.load_default()
        color_esito = (46, 125, 50) if "Riuscito" in esito_str else (198, 40, 40)
        draw.text((8, 8), f"Fase {stage_idx}/{total_stages} ({step_num:,} step)", fill=(240, 240, 255), font=font)
        draw.text((img.width - 120, 8), esito_str[:16], fill=color_esito, font=font)
        return np.array(img)
    except Exception:
        return frame
