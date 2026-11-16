"""
play2learn_lunarlander/evaluator.py (Optimized Architecture)
- Valutazione istantanea da RAM Episode Cache (< 1 ms)
- Fallback micro-sweep con predictor NumPy puro + Heuristic Early Termination
- Replay video parallelo su 2 vCPU (fork) con decimation zero-math frame[::2, ::2] (300x200)
- crea_gif_evoluzione_apprendimento: MP4/GIF timelapse on-demand
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch

from play2learn_core.agent import NumpyRolloutForward
from play2learn_core.evaluator import RENDER_LOCK, fast_resize_into, render_banner_once
from play2learn_lunarlander.config import HELIPAD_X_MAX, SAFE_LANDING_ANGLE
from play2learn_lunarlander.environment import (
    CustomRewardWrapper,
    classify_lunar_lander_outcome,
    get_cached_render_env,
    patch_lunarlander_physics,
)


def _make_fast_predictor(model: Any) -> Callable[[np.ndarray, bool], int]:
    """Usa NumpyRolloutForward (solo policy head, ~8 µs) con fallback PyTorch."""
    np_fwd = getattr(model, "_numpy_rollout_fwd", None)
    if np_fwd is not None and np_fwd._cache is not None:
        np_fwd._sync_weights()
        return np_fwd.predict_numpy

    policy = getattr(model, "policy", None)
    if policy is None:
        return lambda obs, det: int(model.predict(obs, deterministic=det)[0])

    try:
        fwd = NumpyRolloutForward(policy, seed=42)
        if fwd._cache is not None:
            return fwd.predict_numpy
    except Exception:
        pass

    policy.set_training_mode(False)
    device = model.device

    @torch.inference_mode()
    def _predict(obs: np.ndarray, deterministic: bool) -> int:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        return int(policy._predict(obs_t, deterministic=deterministic).item())

    return _predict


def _run_single_sweep_episode(
    env: gym.Env,
    predict_fn: Callable[[np.ndarray, bool], int],
    ep_seed: int,
    deterministic: bool,
    t_start: float,
    timeout_sec: float,
    fast_early_exit: bool = True,
) -> tuple[dict[str, Any], bool]:
    """
    Esegue un singolo episodio di sweep con Heuristic Early Termination opzionale:
    - Taglia schianti matematicamente irrecuperabili (|angle| > 45° e vy < -1.5)
    - Taglia atterraggi già stabilizzati tra le bandierine senza attendere Box2D sleep
    - Taglia hovering evidente dopo 500 step
    """
    obs, _ = env.reset(seed=ep_seed)
    done = False
    ep_rew = 0.0
    steps = 0
    ep_actions = bytearray()
    last_info: dict[str, Any] = {}
    timeout_hit = False
    settled_streak = 0
    early_outcome: str | None = None

    while not done:
        act_int = int(predict_fn(obs, deterministic))
        ep_actions.append(act_int)
        obs, rew, terminated, truncated, last_info = env.step(act_int)
        ep_rew += float(rew)
        steps += 1
        done = bool(terminated or truncated)

        if fast_early_exit and not done:
            x = obs.item(0)
            y = obs.item(1)
            vx = obs.item(2)
            vy = obs.item(3)
            ang = obs.item(4)
            vang = obs.item(5)
            legs = (obs.item(6) > 0.5) and (obs.item(7) > 0.5)

            # 1. Schianto irrecuperabile: risparmia fino a 300 frame Box2D
            if (abs(ang) > 0.785 and vy < -1.5) or (abs(x) > 0.94 and x * vx > 0.05):
                early_outcome = "crash"
                done = True
            # 2. Atterraggio stabile: evita attesa sleep Box2D (60-100 frame)
            elif (
                legs
                and act_int == 0
                and abs(x) <= HELIPAD_X_MAX
                and abs(y) <= 0.15
                and abs(ang) <= SAFE_LANDING_ANGLE
                and abs(vx) <= 0.08
                and abs(vy) <= 0.08
                and abs(vang) <= 0.08
            ):
                settled_streak += 1
                if settled_streak >= 10:
                    early_outcome = "success"
                    done = True
            else:
                settled_streak = 0

            # 3. Hovering palese: evita step 500..1000
            if not done and steps >= 500 and y > 0.45 and abs(vy) < 0.12:
                early_outcome = "timeout"
                done = True

        if not timeout_hit and (steps & 31) == 0 and (time.perf_counter() - t_start >= timeout_sec):
            timeout_hit = True

    fuel = float(getattr(env, "ep_fuel_used", last_info.get("telemetry", {}).get("fuel_used", 0.0)))

    if early_outcome == "success":
        is_succ, is_crash, is_tout, is_off = True, False, False, False
    elif early_outcome == "crash":
        is_succ, is_crash, is_tout = False, True, False
        is_off = abs(float(obs.item(0))) > HELIPAD_X_MAX
    elif early_outcome == "timeout":
        is_succ, is_crash, is_tout, is_off = False, False, True, False
    else:
        is_succ = bool(last_info.get("is_success", False))
        is_crash = bool(last_info.get("is_crashed", False))
        is_tout = bool(last_info.get("is_timeout", False))
        is_off = bool(last_info.get("is_off_target", False))

    candidate = {
        "seed": ep_seed,
        "actions": bytes(ep_actions),
        "is_success": is_succ,
        "is_crashed": is_crash,
        "is_timeout": is_tout,
        "is_off_target": is_off,
        "steps": steps,
        "fuel": fuel,
        "reward": ep_rew,
        "deterministic": deterministic,
    }
    return candidate, timeout_hit


def _fast_downscale_2x(frame: np.ndarray) -> np.ndarray:
    """Decimation 2x esatta (600x400 -> 300x200) in ~50 µs: taglia 4x il costo ffmpeg."""
    h, w = frame.shape[:2]
    if h >= 400 and w >= 600:
        return np.ascontiguousarray(frame[::2, ::2])
    return frame


def _render_single_mp4(
    render_env: gym.Env,
    seed: int,
    actions: bytes | list[int],
    output_path: Path,
    render_stride: int = 2,
    downscale_2x: bool = True,
) -> Path:
    """Rigioca le azioni in Box2D e streamma H.264 ultrafast a FPS fisicamente corretti."""
    import imageio

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_fps = max(1, 50 // max(1, render_stride))

    render_env.reset(seed=seed)
    writer = imageio.get_writer(
        str(output_path),
        fps=out_fps,
        macro_block_size=1,
        ffmpeg_params=[
            "-preset", "ultrafast",
            "-tune", "fastdecode",
            "-crf", "28",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
        ],
    )
    try:
        init_frame = render_env.render()
        if init_frame is not None:
            writer.append_data(_fast_downscale_2x(init_frame) if downscale_2x else init_frame)

        for act_idx, act in enumerate(actions, start=1):
            _, _, term, trunc, _ = render_env.step(int(act))
            done = bool(term or trunc)
            if (act_idx % render_stride == 0) or done:
                frame = render_env.render()
                if frame is not None:
                    writer.append_data(_fast_downscale_2x(frame) if downscale_2x else frame)
            if done:
                break
    finally:
        writer.close()

    return output_path


def _parallel_render_worker(
    tasks: list[tuple[int, bytes, str]],
    env_id: str,
    env_kwargs: dict[str, Any],
    render_stride: int,
    fast_solver: bool,
) -> None:
    """Worker isolato (propria istanza SDL/Pygame) per saturare entrambi i vCPU Colab."""
    try:
        env = gym.make(env_id, render_mode="rgb_array", disable_env_checker=True, **env_kwargs)
    except TypeError:
        env = gym.make(env_id, render_mode="rgb_array", **env_kwargs)
    patch_lunarlander_physics(env, disable_particles=False, fast_solver=fast_solver)
    try:
        for seed, actions, out_str in tasks:
            _render_single_mp4(env, seed, actions, Path(out_str), render_stride=render_stride, downscale_2x=True)
    finally:
        env.close()


def evaluate_and_record(
    model: Any,
    env_id: str,
    reward_fn: Callable,
    video_dir: Path | str,
    target_successes: int = 2,
    target_failures: int = 2,
    sweep_timeout_seconds: float = 4.0,
    min_eval_episodes: int = 20,
    max_eval_episodes: int = 60,
    seed: int = 100,
    env_kwargs: dict[str, Any] | None = None,
    create_gif: bool = False,
    filename_prefix: str = "",
    episode_cache: Any = None,
    fast_early_exit: bool = True,
    fast_solver: bool = True,
    render_stride: int = 2,
) -> dict[str, Any]:
    """
    Valuta l'agente sfruttando prima il RolloutEpisodeCache in RAM (0.001 s) e
    ricorrendo al micro-sweep NumPy con Early Termination solo se mancano episodi.
    Registra i video in parallelo sui 2 vCPU (300x200 H.264 ultrafast).
    """
    env_kwargs = env_kwargs or {}
    video_dir = Path(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    # Allinea i target video al limite visualizzato inline per non sprecare CPU
    target_successes = min(target_successes, 2)
    target_failures = min(target_failures, 2)

    t_eval_start = time.perf_counter()
    success_candidates: list[dict[str, Any]] = []
    failure_candidates: list[dict[str, Any]] = []

    eval_rewards: list[float] = []
    eval_lengths: list[int] = []
    eval_fuels: list[float] = []
    success_fuels: list[float] = []
    failure_fuels: list[float] = []
    eval_successes = 0
    eval_crashes = 0
    eval_timeouts = 0
    used_ram_cache = False

    # =========================================================================
    # FASE 1A: ESTRAZIONE ISTANTANEA DAL RAM EPISODE CACHE (ZERO SIMULAZIONE)
    # =========================================================================
    if episode_cache is not None and len(episode_cache) >= min_eval_episodes:
        recent_eps = episode_cache.get_recent(n=max(min_eval_episodes, 40))
        used_ram_cache = True
        for cand in recent_eps:
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

    # =========================================================================
    # FASE 1B: FALLBACK MICRO-SWEEP NUMPY + EARLY EXIT (SOLO SE NECESSARIO)
    # =========================================================================
    need_metrics = len(eval_rewards) < min_eval_episodes
    need_videos = (
        len(success_candidates) < target_successes
        or len(failure_candidates) < target_failures
    )

    if need_metrics or (need_videos and not used_ram_cache):
        predict_fn = _make_fast_predictor(model)
        try:
            sweep_env = gym.make(env_id, render_mode=None, disable_env_checker=True, **env_kwargs)
        except TypeError:
            sweep_env = gym.make(env_id, render_mode=None, **env_kwargs)
        patch_lunarlander_physics(sweep_env, disable_particles=True, fast_solver=fast_solver)
        sweep_env = CustomRewardWrapper(sweep_env, reward_fn, seed_base=seed, cache_start_step=-1)

        ep = 0
        timeout_hit = False
        try:
            while ep < max_eval_episodes and not timeout_hit:
                have_enough_vids = (
                    len(success_candidates) >= target_successes
                    and len(failure_candidates) >= target_failures
                )
                elapsed = time.perf_counter() - t_eval_start
                if len(eval_rewards) >= min_eval_episodes and (have_enough_vids or elapsed >= sweep_timeout_seconds):
                    break

                cand, timeout_hit = _run_single_sweep_episode(
                    env=sweep_env,
                    predict_fn=predict_fn,
                    ep_seed=seed + ep,
                    deterministic=True,
                    t_start=t_eval_start,
                    timeout_sec=sweep_timeout_seconds * 1.8,
                    fast_early_exit=fast_early_exit,
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
        finally:
            sweep_env.close()

    sweep_dt = time.perf_counter() - t_eval_start
    total_evaluated = max(1, len(eval_rewards))
    success_rate = (eval_successes / total_evaluated) * 100.0
    src_lbl = "RAM Episode Cache" if used_ram_cache else "Fast Early-Exit Sweep"
    print(
        f"[play2learn] Valutazione ({src_lbl}) completata in {sweep_dt:.3f}s "
        f"({total_evaluated} episodi): {len(success_candidates)} successi, "
        f"{len(failure_candidates)} fallimenti pronti."
    )

    # =========================================================================
    # FASE 2: REPLAY VIDEO MIRATO IN PARALLELO SU 2 vCPU (300x200 H.264)
    # =========================================================================
    selected_successes = sorted(
        success_candidates,
        key=lambda c: (not c["deterministic"], c["fuel"]),
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

    success_videos: list[Path] = []
    failure_videos: list[Path] = []
    render_jobs: list[tuple[int, bytes, str]] = []
    pfx = f"{filename_prefix}_" if filename_prefix else ""

    for idx, cand in enumerate(selected_successes, start=1):
        v_path = video_dir / f"{pfx}successo_{idx}_step_{cand['steps']}.mp4"
        success_videos.append(v_path)
        render_jobs.append((cand["seed"], bytes(cand["actions"]), str(v_path)))

    for idx, cand in enumerate(selected_failures, start=1):
        causa = (
            "fuori_bersaglio" if cand.get("is_off_target")
            else ("crash" if cand["is_crashed"] else ("timeout" if cand["is_timeout"] else "fallimento"))
        )
        v_path = video_dir / f"{pfx}fallimento_{idx}_{causa}_step_{cand['steps']}.mp4"
        failure_videos.append(v_path)
        render_jobs.append((cand["seed"], bytes(cand["actions"]), str(v_path)))

    if render_jobs:
        t_ren = time.perf_counter()
        use_parallel = (
            sys.platform.startswith("linux")
            and len(render_jobs) >= 2
            and "fork" in mp.get_all_start_methods()
        )
        if use_parallel:
            ctx = mp.get_context("fork")
            mid = len(render_jobs) // 2
            p1 = ctx.Process(target=_parallel_render_worker, args=(render_jobs[:mid], env_id, env_kwargs, render_stride, fast_solver))
            p2 = ctx.Process(target=_parallel_render_worker, args=(render_jobs[mid:], env_id, env_kwargs, render_stride, fast_solver))
            p1.start()
            p2.start()
            p1.join()
            p2.join()
        else:
            with RENDER_LOCK:
                render_env = get_cached_render_env(env_id, env_kwargs, fast_solver=fast_solver)
                for j_seed, j_acts, j_out in render_jobs:
                    _render_single_mp4(render_env, j_seed, j_acts, Path(j_out), render_stride=render_stride, downscale_2x=True)
        print(f"[play2learn] Rendering {len(render_jobs)} video completato in {time.perf_counter() - t_ren:.2f}s.")

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
        "gif_path": None,
        "mean_fuel_success": float(np.mean(success_fuels)) if success_fuels else 0.0,
        "mean_fuel_failure": float(np.mean(failure_fuels)) if failure_fuels else 0.0,
        "used_ram_cache": used_ram_cache,
    }


def crea_gif_evoluzione_apprendimento(
    model: Any,
    milestones: list[dict[str, Any]],
    env_id: str,
    output_path: Path | str,
    env_kwargs: dict[str, Any] | None = None,
    seed: int = 42,
    max_steps_per_episode: int = 1000,
    stride: int = 4,
    target_width: int = 400,
    fps: int = 15,
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
        print(f"[play2learn] Modulo mancante per la GIF di evoluzione: {e}")
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

    patch_lunarlander_physics(render_env, disable_particles=False, fast_solver=True)

    fasi_titoli = [
        "Fase 1: Primi Passi (Esplorazione)",
        "Fase 2: Accensione Motori",
        "Fase 3: Discesa Orientata",
        "Fase Finale: Atterraggio Controllato",
    ]
    total_stages = len(milestones)
    frames_written = 0

    output_path = Path(output_path)
    if output_path.suffix.lower() == ".gif":
        output_path = output_path.with_suffix(".mp4")

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

        with _open_timelapse() as gif_writer:
            for idx, m in enumerate(milestones):
                policy.load_state_dict(m["state_dict"])
                predict_fn = _make_fast_predictor(model)

                obs, _ = render_env.reset(seed=seed)
                sampled_raw: list[np.ndarray] = []
                f_init = render_env.render()
                if f_init is not None:
                    sampled_raw.append(f_init)

                done = False
                step_count = 0
                final_esito = "In volo..."

                while not done and step_count < max_steps_per_episode:
                    action = predict_fn(obs, deterministic=True)
                    obs, rew, term, trunc, _ = render_env.step(action)
                    step_count += 1
                    done = bool(term or trunc)

                    if step_count % stride == 0 or done:
                        f = render_env.render()
                        if f is not None:
                            sampled_raw.append(f)

                    if done:
                        is_succ, is_crash, is_tout, is_off = classify_lunar_lander_outcome(
                            obs=obs,
                            original_reward=float(rew),
                            terminated=bool(term),
                            truncated=bool(trunc),
                            unwrapped=render_env.unwrapped,
                        )
                        if is_succ:
                            final_esito = "[OK] Allunaggio Riuscito"
                        elif is_off:
                            final_esito = "[FUORI] Fuori Piazzola"
                        elif is_crash:
                            final_esito = "[CRASH] Schianto al Suolo"
                        elif is_tout:
                            final_esito = "[TIMEOUT] Tempo Scaduto"
                        else:
                            final_esito = "Fine Volo"

                if not done and step_count >= max_steps_per_episode:
                    final_esito = "[TIMEOUT] Durata Massima Raggiunta"

                if not sampled_raw:
                    continue

                step_val = m.get("step", 0)
                pct_val = m.get("percent", int((idx) / max(1, total_stages - 1) * 100))

                banner = render_banner_once(
                    width=target_width,
                    line1=f"Fase {idx + 1}/{total_stages} • Step {step_val:,} ({pct_val}%)",
                    line2=f"Esito: {final_esito} • Durata: {step_count} step",
                    height=34,
                )

                last_annotated = None
                for f in sampled_raw:
                    h, w = f.shape[:2]
                    target_h = max(2, int(target_width * h / w))
                    if target_h % 2 != 0:
                        target_h -= 1
                    buf = np.empty((target_h, target_width, 3), dtype=np.uint8)
                    fast_resize_into(f, buf)
                    buf[:34, :, :] = banner
                    last_annotated = buf
                    gif_writer.append_data(buf)
                    frames_written += 1

                if last_annotated is not None:
                    for _ in range(8):
                        gif_writer.append_data(last_annotated)
                        frames_written += 1
    except Exception as e:
        print(f"[play2learn] Errore nel salvataggio del timelapse evoluzione: {e}")
        return None
    finally:
        render_env.close()
        policy.load_state_dict(orig_state)

    return output_path if frames_written > 0 else None
