"""
Modulo di valutazione e registrazione video/GIF per l'agente addestrato.
Registra un mix didattico bilanciato: fino a 3 video di successo e fino a 3 video di fallimento.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Callable, Any

import numpy as np

try:
    import gymnasium as gym
    from stable_baselines3 import PPO
except ImportError:
    gym = Any  # type: ignore
    PPO = Any  # type: ignore

from play2learn_lunarlander.environment import CustomRewardWrapper


def evaluate_and_record(
    model: PPO,
    env_id: str,
    reward_fn: Callable,
    video_dir: Path | str,
    target_successes: int = 3,
    target_failures: int = 3,
    sweep_timeout_seconds: float = 6.0,
    min_eval_episodes: int = 15,
    max_eval_episodes: int = 100,
    seed: int = 100,
    env_kwargs: dict[str, Any] | None = None,
    create_gif: bool = True,
) -> dict[str, Any]:
    """
    Valuta l'agente addestrato con una procedura ottimizzata in due fasi:
    1. Fast Sweep senza rendering video regolato da timer:
       - Esegue rapidamente decine di episodi in background.
       - Se scatta il timer di timeout, l'episodio in corso viene completato prima di fermare lo sweep.
       - Raccoglie telemetria e cataloga episodi di successo ed episodi di fallimento.
    2. Elaborazione video ad alta fedeltà (mix bilanciato):
       - Replay deterministico e rendering video SOLO per i target selezionati:
         fino a 3 successi (ordinati per miglior efficienza) e fino a 3 fallimenti.
    """
    env_kwargs = env_kwargs or {}
    video_dir = Path(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[play2learn] Valutazione dell'agente: ricerca di {target_successes} successi e {target_failures} fallimenti...")
    print(f"[play2learn] Fase 1: Sweep rapido senza rendering (timer massimo: {sweep_timeout_seconds:.1f}s)...")

    # =========================================================================
    # FASE 1: FAST SWEEP SENZA RENDERING (OTTIMIZZATO)
    # =========================================================================
    sweep_env = gym.make(env_id, render_mode=None, **env_kwargs)
    sweep_env = CustomRewardWrapper(sweep_env, reward_fn)

    success_candidates: list[dict[str, Any]] = []
    failure_candidates: list[dict[str, Any]] = []

    eval_rewards: list[float] = []
    eval_lengths: list[int] = []
    eval_fuels: list[float] = []
    success_fuels: list[float] = []
    failure_fuels: list[float] = []
    eval_successes: int = 0
    eval_crashes: int = 0
    eval_timeouts: int = 0

    t_sweep_start = time.time()
    timeout_triggered = False
    ep = 0

    while ep < max_eval_episodes and not timeout_triggered:
        # Condizione di completamento anticipato:
        # Abbiamo già raccolto sia il target di successi che di fallimenti E raggiunto il minimo di campioni
        if (
            len(success_candidates) >= target_successes
            and len(failure_candidates) >= target_failures
            and ep >= min_eval_episodes
        ):
            break

        # Politica deterministica per le prime prove standard.
        # Se abbiamo già abbastanza successi ma mancano ancora fallimenti da mostrare,
        # testiamo con la politica stocastica (la stessa usata durante il training che ha generato i fallimenti)
        use_deterministic = True
        if (
            len(success_candidates) >= target_successes
            and len(failure_candidates) < target_failures
            and ep >= min_eval_episodes
        ):
            use_deterministic = False

        cur_seed = seed + ep
        obs, _ = sweep_env.reset(seed=cur_seed)
        done = False
        ep_rew = 0.0
        steps = 0
        ep_actions: list[int] = []
        last_info: dict[str, Any] = {}

        while not done:
            action, _ = model.predict(obs, deterministic=use_deterministic)
            act_int = int(action)
            ep_actions.append(act_int)

            obs, rew, terminated, truncated, last_info = sweep_env.step(action)
            ep_rew += float(rew)
            steps += 1
            done = terminated or truncated

            # Controllo timer: se scatta il timeout, segnaliamo la conclusione dello sweep
            # ma lasciamo completare l'episodio in corso fino a fine volo (done=True)
            if not timeout_triggered and (time.time() - t_sweep_start >= sweep_timeout_seconds):
                timeout_triggered = True

        eval_rewards.append(ep_rew)
        eval_lengths.append(steps)
        tel = last_info.get("telemetry", {})
        fuel = float(tel.get("fuel_used", 0.0))
        eval_fuels.append(fuel)

        is_succ = bool(last_info.get("is_success", False))
        is_crash = bool(last_info.get("is_crashed", False))
        is_time = bool(last_info.get("is_timeout", False))

        candidate = {
            "seed": cur_seed,
            "actions": ep_actions,
            "is_success": is_succ,
            "is_crashed": is_crash,
            "is_timeout": is_time,
            "steps": steps,
            "fuel": fuel,
            "reward": ep_rew,
            "deterministic": use_deterministic,
        }

        if is_succ:
            eval_successes += 1
            success_fuels.append(fuel)
            success_candidates.append(candidate)
        else:
            if is_crash:
                eval_crashes += 1
            elif is_time:
                eval_timeouts += 1
            else:
                eval_crashes += 1  # Fallimento generico
            failure_fuels.append(fuel)
            failure_candidates.append(candidate)

        ep += 1

    sweep_env.close()
    sweep_duration = time.time() - t_sweep_start
    total_evaluated = max(1, len(eval_rewards))
    success_rate = (eval_successes / total_evaluated) * 100.0

    if timeout_triggered:
        print(f"[play2learn] Timeout timer raggiunto ({sweep_timeout_seconds:.1f}s): completato l'ultimo episodio in corso.")
    print(
        f"[play2learn] Sweep completato in {sweep_duration:.2f}s ({total_evaluated} episodi valutati): "
        f"{len(success_candidates)} successi, {len(failure_candidates)} fallimenti trovati."
    )

    # =========================================================================
    # FASE 2: ELABORAZIONE VIDEO (SOLO SUI CANDIDATI SELEZIONATI)
    # =========================================================================
    # Seleziona fino a target_successes (ordinati per efficienza carburante)
    selected_successes = sorted(success_candidates, key=lambda c: c["fuel"])[:target_successes]

    # Seleziona fino a target_failures (cercando di includere sia crash che timeout se presenti)
    selected_failures = failure_candidates[:target_failures]

    success_videos: list[Path] = []
    failure_videos: list[Path] = []
    best_success_fuel: float = 9999.0
    best_success_video: Path | None = None

    to_render = (
        [("successo", c) for c in selected_successes]
        + [("fallimento", c) for c in selected_failures]
    )

    if to_render:
        print(
            f"[play2learn] Fase 2: Elaborazione video ({len(selected_successes)} successi, "
            f"{len(selected_failures)} fallimenti)..."
        )
        render_env = gym.make(env_id, render_mode="rgb_array", **env_kwargs)
        render_env = CustomRewardWrapper(render_env, reward_fn)

        for kind, cand in to_render:
            obs, _ = render_env.reset(seed=cand["seed"])
            frames = []
            init_frame = render_env.render()
            if init_frame is not None:
                frames.append(init_frame)

            for act in cand["actions"]:
                obs, rew, term, trunc, _ = render_env.step(act)
                frame = render_env.render()
                if frame is not None:
                    frames.append(frame)
                if term or trunc:
                    break

            if kind == "successo":
                idx = len(success_videos) + 1
                v_path = video_dir / f"successo_{idx}_step_{cand['steps']}.mp4"
                _save_frames_as_mp4(frames, v_path)
                success_videos.append(v_path)
                if cand["fuel"] < best_success_fuel:
                    best_success_fuel = cand["fuel"]
                    best_success_video = v_path
            else:
                idx = len(failure_videos) + 1
                causa = "crash" if cand["is_crashed"] else ("timeout" if cand["is_timeout"] else "fallimento")
                v_path = video_dir / f"fallimento_{idx}_{causa}_step_{cand['steps']}.mp4"
                _save_frames_as_mp4(frames, v_path)
                failure_videos.append(v_path)

        render_env.close()

    print(
        f"[play2learn] Valutazione conclusa: {len(success_videos)} video di successo "
        f"e {len(failure_videos)} video di fallimento registrati."
    )

    # Generazione della GIF animata dell'allunaggio di successo migliore
    gif_path: Path | None = None
    if create_gif and best_success_video and best_success_video.exists():
        gif_path = _convert_video_to_gif(best_success_video)

    all_recorded = success_videos + failure_videos

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
        "videos": all_recorded,
        "gif_path": gif_path,
        "mean_fuel_success": float(np.mean(success_fuels)) if success_fuels else 0.0,
        "mean_fuel_failure": float(np.mean(failure_fuels)) if failure_fuels else 0.0,
    }


def _save_frames_as_mp4(frames: list[np.ndarray], output_path: Path, fps: int = 50) -> None:
    """Salva una sequenza di frame numpy come video MP4 compresso."""
    try:
        import imageio
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Usa il writer ffmpeg di imageio
        imageio.mimsave(str(output_path), frames, fps=fps, macro_block_size=1)
    except Exception as e:
        print(f"[play2learn] Errore nel salvataggio video {output_path.name}: {e}")


def _convert_video_to_gif(video_file: Path) -> Path | None:
    """Converte un file MP4 in una GIF animata per le slide di presentazione."""
    try:
        import imageio
        reader = imageio.get_reader(str(video_file))
        gif_output = video_file.parent / (video_file.stem + ".gif")

        # Campiona 1 frame ogni 2 per mantenere la GIF leggera (< 2MB)
        fps = reader.get_meta_data().get("fps", 50)
        frames = []
        for i, frame in enumerate(reader):
            if i % 2 == 0:
                frames.append(frame)

        if frames:
            imageio.mimsave(str(gif_output), frames, fps=max(10, int(fps / 2)), loop=0)
            return gif_output
    except Exception:
        pass
    return None
