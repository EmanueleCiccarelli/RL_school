"""
Modulo di valutazione e registrazione video/GIF ad alte prestazioni.
Procedura in 2 fasi:
- Fase 1A: Fast Sweep deterministico (inferenza diretta PyTorch, metriche imparziali).
- Fase 1B: Caccia stocastica opzionale ai soli candidati video mancanti (senza alterare i KPI).
- Fase 2: Replay mirato su Box2D con streaming video diretto su disco (consumo RAM ~1 MB),
  pixel format compatibile yuv420p per browser e generazione GIF ottimizzata.
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
except ImportError:
    gym = Any  # type: ignore
    PPO = Any  # type: ignore

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


def _record_episode_streaming(
    render_env: gym.Env,
    seed: int,
    actions: list[int],
    output_path: Path,
    fps: int = 50,
    capture_gif_frames: bool = False,
    gif_stride: int = 2,
    gif_width: int = 400,
) -> list[np.ndarray]:
    """
    Rigioca un episodio e scrive direttamente i frame su disco in streaming (RAM costante ~1 MB).
    Supporta pixel format yuv420p per massima compatibilità con browser HTML5.
    """
    import imageio
    from PIL import Image

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gif_frames: list[np.ndarray] = []

    def _open_writer():
        try:
            return imageio.get_writer(
                str(output_path),
                fps=fps,
                macro_block_size=1,
                ffmpeg_params=["-preset", "ultrafast", "-crf", "26", "-pix_fmt", "yuv420p"],
            )
        except Exception:
            try:
                return imageio.get_writer(
                    str(output_path),
                    fps=fps,
                    macro_block_size=1,
                    ffmpeg_params=["-preset", "ultrafast", "-crf", "26"],
                )
            except Exception:
                return imageio.get_writer(str(output_path), fps=fps, macro_block_size=1)

    def _maybe_collect_gif(frame: np.ndarray, idx: int, force: bool = False) -> None:
        if not capture_gif_frames or (not force and idx % gif_stride != 0):
            return
        h, w = frame.shape[:2]
        target_h = max(1, int(gif_width * h / w))
        small = Image.fromarray(frame).resize((gif_width, target_h), Image.Resampling.BILINEAR)
        gif_frames.append(np.asarray(small))

    render_env.reset(seed=seed)
    with _open_writer() as writer:
        frame_idx = 0
        init_frame = render_env.render()
        if init_frame is not None:
            writer.append_data(init_frame)
            _maybe_collect_gif(init_frame, frame_idx)
            frame_idx += 1

        last_frame = init_frame
        for act in actions:
            _, _, term, trunc, _ = render_env.step(act)
            frame = render_env.render()
            if frame is not None:
                writer.append_data(frame)
                _maybe_collect_gif(frame, frame_idx)
                last_frame = frame
                frame_idx += 1
            if term or trunc:
                break

        if last_frame is not None and capture_gif_frames:
            _maybe_collect_gif(last_frame, frame_idx, force=True)

    return gif_frames


def evaluate_and_record(
    model: PPO,
    env_id: str,
    reward_fn: Callable,
    video_dir: Path | str,
    target_successes: int = 3,
    target_failures: int = 3,
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

    print(
        f"\n[play2learn] Valutazione dell'agente: ricerca di {target_successes} successi "
        f"e {target_failures} fallimenti..."
    )

    predict_fn = _make_fast_predictor(model)
    try:
        sweep_env = gym.make(env_id, render_mode=None, disable_env_checker=True, **env_kwargs)
    except TypeError:
        sweep_env = gym.make(env_id, render_mode=None, **env_kwargs)
    sweep_env = CustomRewardWrapper(sweep_env, reward_fn)

    success_candidates: list[dict[str, Any]] = []
    failure_candidates: list[dict[str, Any]] = []

    # Metriche ufficiali: calcolate SOLO sugli episodi deterministici (zero bias!)
    eval_rewards: list[float] = []
    eval_lengths: list[int] = []
    eval_fuels: list[float] = []
    success_fuels: list[float] = []
    failure_fuels: list[float] = []
    eval_successes: int = 0
    eval_crashes: int = 0
    eval_timeouts: int = 0

    t_sweep_start = time.perf_counter()
    timeout_triggered = False
    ep = 0

    # Budget di sicurezza esteso per garantire il completamento di min_eval_episodes
    hard_timeout_sec = max(sweep_timeout_seconds * 2.5, 15.0)

    try:
        # =========================================================================
        # FASE 1A: SWEEP DETERMINISTICO IMPARZIALE
        # =========================================================================
        det_budget_sec = sweep_timeout_seconds * 0.75
        while ep < max_eval_episodes and not timeout_triggered:
            have_enough_videos = (
                len(success_candidates) >= target_successes
                and len(failure_candidates) >= target_failures
            )
            elapsed = time.perf_counter() - t_sweep_start

            # Interrompe solo se abbiamo raggiunto min_eval_episodes E (abbiamo i video O è scaduto il budget)
            if ep >= min_eval_episodes and (have_enough_videos or elapsed >= det_budget_sec):
                break
            if elapsed >= hard_timeout_sec:
                break

            cand, timeout_triggered = _run_single_sweep_episode(
                env=sweep_env,
                predict_fn=predict_fn,
                ep_seed=seed + ep,
                deterministic=True,
                t_start=t_sweep_start,
                timeout_sec=hard_timeout_sec,
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

        # =========================================================================
        # FASE 1B: CACCIA RAPIDA STOCASTICA AI SOLI VIDEO MANCANTI (NON ALTERA I KPI)
        # =========================================================================
        stoch_ep = 0
        max_stoch_hunt = 25
        while (
            not timeout_triggered
            and stoch_ep < max_stoch_hunt
            and (time.perf_counter() - t_sweep_start < hard_timeout_sec)
            and (
                len(success_candidates) < target_successes
                or len(failure_candidates) < target_failures
            )
        ):
            cand, timeout_triggered = _run_single_sweep_episode(
                env=sweep_env,
                predict_fn=predict_fn,
                ep_seed=seed + 5000 + stoch_ep,
                deterministic=False,
                t_start=t_sweep_start,
                timeout_sec=hard_timeout_sec,
            )
            if cand["is_success"] and len(success_candidates) < target_successes:
                success_candidates.append(cand)
            elif (not cand["is_success"]) and len(failure_candidates) < target_failures:
                failure_candidates.append(cand)
            stoch_ep += 1
    finally:
        sweep_env.close()

    sweep_duration = time.perf_counter() - t_sweep_start
    total_evaluated = max(1, len(eval_rewards))
    success_rate = (eval_successes / total_evaluated) * 100.0

    print(
        f"[play2learn] Sweep completato in {sweep_duration:.2f}s "
        f"({total_evaluated} episodi deterministici + {stoch_ep} ricerca video): "
        f"{len(success_candidates)} successi, {len(failure_candidates)} fallimenti pronti."
    )

    # =========================================================================
    # FASE 2: REPLAY VIDEO MIRATO (STREAMING MP4 A BASSO CONSUMO RAM)
    # =========================================================================
    selected_successes = sorted(
        success_candidates,
        key=lambda c: (not c["deterministic"], c["fuel"]),
    )[:target_successes]

    crashes_pool = [c for c in failure_candidates if c["is_crashed"]]
    timeouts_pool = [c for c in failure_candidates if c["is_timeout"]]
    selected_failures: list[dict[str, Any]] = []
    if crashes_pool and timeouts_pool and target_failures >= 2:
        selected_failures.extend(crashes_pool[: target_failures - 1])
        selected_failures.append(timeouts_pool[0])
    for c in failure_candidates:
        if len(selected_failures) >= target_failures:
            break
        if c not in selected_failures:
            selected_failures.append(c)

    success_videos: list[Path] = []
    failure_videos: list[Path] = []
    best_success_fuel: float = float("inf")
    best_success_video: Path | None = None
    best_gif_frames: list[np.ndarray] = []

    to_render = (
        [("successo", c) for c in selected_successes]
        + [("fallimento", c) for c in selected_failures]
    )

    if to_render:
        print(
            f"[play2learn] Fase 2: Rendering video streaming ({len(selected_successes)} successi, "
            f"{len(selected_failures)} fallimenti)..."
        )
        try:
            render_env = gym.make(
                env_id,
                render_mode="rgb_array",
                disable_env_checker=True,
                **env_kwargs,
            )
        except TypeError:
            render_env = gym.make(env_id, render_mode="rgb_array", **env_kwargs)

        try:
            for kind, cand in to_render:
                pfx = f"{filename_prefix}_" if filename_prefix else ""
                if kind == "successo":
                    idx = len(success_videos) + 1
                    v_path = video_dir / f"{pfx}successo_{idx}_step_{cand['steps']}.mp4"
                    is_best = cand["fuel"] < best_success_fuel
                    gif_frames = _record_episode_streaming(
                        render_env=render_env,
                        seed=cand["seed"],
                        actions=cand["actions"],
                        output_path=v_path,
                        fps=50,
                        capture_gif_frames=is_best and create_gif,
                    )
                    success_videos.append(v_path)
                    if is_best:
                        best_success_fuel = cand["fuel"]
                        best_success_video = v_path
                        if gif_frames:
                            best_gif_frames = gif_frames
                else:
                    idx = len(failure_videos) + 1
                    if cand.get("is_off_target"):
                        causa = "fuori_bersaglio"
                    elif cand["is_crashed"]:
                        causa = "crash"
                    elif cand["is_timeout"]:
                        causa = "timeout"
                    else:
                        causa = "fallimento"
                    v_path = video_dir / f"{pfx}fallimento_{idx}_{causa}_step_{cand['steps']}.mp4"
                    _record_episode_streaming(
                        render_env=render_env,
                        seed=cand["seed"],
                        actions=cand["actions"],
                        output_path=v_path,
                        fps=50,
                        capture_gif_frames=False,
                    )
                    failure_videos.append(v_path)
        finally:
            render_env.close()

    # Genera GIF direttamente dai frame scalati catturati al volo in streaming
    gif_path: Path | None = None
    if create_gif and best_success_video and best_gif_frames:
        try:
            import imageio
            g_path = best_success_video.with_suffix(".gif")
            imageio.mimsave(str(g_path), best_gif_frames, fps=25, loop=0)
            gif_path = g_path
        except Exception as e:
            print(f"[play2learn] Avviso creazione GIF: {e}")

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


def crea_gif_evoluzione_apprendimento(
    model: PPO,
    milestones: list[dict[str, Any]],
    env_id: str,
    output_path: Path | str,
    env_kwargs: dict[str, Any] | None = None,
    seed: int = 42,
    max_steps_per_episode: int = 240,
    stride: int = 2,
    target_width: int = 400,
    fps: int = 15,
) -> Path | None:
    """
    Genera un timelapse GIF fluido e annotato che mostra visivamente come l'agente
    è passato dall'esplorazione iniziale (schianti) all'atterraggio morbido,
    confrontando le milestone di training nel medesimo scenario lunare.
    """
    if not milestones:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    env_kwargs = env_kwargs or {}

    try:
        from PIL import Image, ImageDraw
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

    fasi_titoli = [
        "Fase 1: Primi Passi (Esplorazione)",
        "Fase 2: Accensione Motori",
        "Fase 3: Discesa Orientata",
        "Fase Finale: Atterraggio Controllato",
    ]
    total_stages = len(milestones)
    frames_written = 0

    try:
        with imageio.get_writer(str(output_path), mode="I", fps=fps, loop=0) as gif_writer:
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
                last_frame = f_init

                while not done and step_count < max_steps_per_episode:
                    action = predict_fn(obs, deterministic=True)
                    obs, rew, term, trunc, _ = render_env.step(action)
                    step_count += 1
                    done = bool(term or trunc)

                    if step_count % stride == 0 or done:
                        f = render_env.render()
                        if f is not None:
                            sampled_raw.append(f)
                            last_frame = f

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
                    final_esito = "[VOLO] Hovering Prolungato"

                if not sampled_raw:
                    continue

                stage_title = fasi_titoli[idx] if idx < len(fasi_titoli) else f"Fase {idx + 1}/{total_stages}"
                step_val = m.get("step", step_count)
                pct_val = m.get("percent", int((idx + 1) / total_stages * 100))
                succ_val = m.get("success_rate", 0.0)

                last_annotated = None
                for f in sampled_raw:
                    h, w = f.shape[:2]
                    target_h = max(1, int(target_width * h / w))
                    img = Image.fromarray(f).resize((target_width, target_h), Image.Resampling.BILINEAR)
                    draw = ImageDraw.Draw(img)
                    draw.rectangle([(0, 0), (target_width, 34)], fill=(15, 23, 42))
                    draw.text((8, 4), f"{stage_title} - Step ~{step_val:,} ({pct_val}%)", fill=(255, 255, 255))
                    draw.text((8, 18), f"Esito: {final_esito} | Succ. mobile: {succ_val:.0f}%", fill=(147, 197, 253))
                    last_annotated = np.asarray(img)
                    gif_writer.append_data(last_annotated)
                    frames_written += 1

                # Congela l'ultimo frame annotato per 8 frame per una lettura chiara
                if last_annotated is not None:
                    for _ in range(8):
                        gif_writer.append_data(last_annotated)
                        frames_written += 1
    except Exception as e:
        print(f"[play2learn] Errore nel salvataggio della GIF evoluzione: {e}")
        return None
    finally:
        render_env.close()
        policy.load_state_dict(orig_state)

    return output_path if frames_written > 0 else None
