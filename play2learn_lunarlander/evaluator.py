"""
Modulo di valutazione e registrazione video/GIF per l'agente addestrato.
Registra un mix didattico bilanciato: fino a 3 video di successo e fino a 3 video di fallimento.
"""

from __future__ import annotations

import re
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
    max_eval_episodes: int = 30,
    seed: int = 100,
    env_kwargs: dict[str, Any] | None = None,
    create_gif: bool = True,
) -> dict[str, Any]:
    """
    Valuta l'agente addestrato registrando un mix bilanciato di prove:
    - fino a 3 video di allunaggi RIUSCITI (is_success == True)
    - fino a 3 video di FALLIMENTI (is_crashed o is_timeout)
    In questo modo gli studenti possono analizzare sia i comportamenti corretti
    che i punti critici su cui intervenire nella reward function.
    """
    env_kwargs = env_kwargs or {}
    video_dir = Path(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[play2learn] Valutazione dell'agente: ricerca di {target_successes} successi e {target_failures} fallimenti...")

    eval_env = gym.make(env_id, render_mode="rgb_array", **env_kwargs)
    eval_env = CustomRewardWrapper(eval_env, reward_fn)

    success_videos: list[Path] = []
    failure_videos: list[Path] = []

    eval_rewards: list[float] = []
    eval_lengths: list[int] = []
    eval_fuels: list[float] = []
    success_fuels: list[float] = []
    failure_fuels: list[float] = []
    eval_successes: int = 0
    eval_crashes: int = 0
    eval_timeouts: int = 0

    best_success_fuel: float = 9999.0
    best_success_video: Path | None = None

    import imageio

    for ep in range(max_eval_episodes):
        # Condizione di stop anticipato: se abbiamo già raccolto tutti i target desiderati
        if len(success_videos) >= target_successes and len(failure_videos) >= target_failures:
            break

        # Se abbiamo già raccolto tutti i successi ma mancano ancora video di fallimento
        # e dopo 12 tentativi deterministici non ne abbiamo trovati, testiamo in modalità stocastica
        # (la stessa usata in fase di addestramento che ha generato il tasso di fallimento)
        use_deterministic = True
        if len(success_videos) >= target_successes and len(failure_videos) < target_failures and ep >= 12:
            use_deterministic = False

        obs, _ = eval_env.reset(seed=seed + ep)
        done = False
        ep_rew = 0.0
        steps = 0
        last_info: dict[str, Any] = {}
        frames = []

        # Raccoglie i frame renderizzati dell'episodio
        initial_frame = eval_env.render()
        if initial_frame is not None:
            frames.append(initial_frame)

        while not done:
            action, _ = model.predict(obs, deterministic=use_deterministic)
            obs, rew, terminated, truncated, last_info = eval_env.step(action)
            frame = eval_env.render()
            if frame is not None:
                frames.append(frame)

            ep_rew += float(rew)
            steps += 1
            done = terminated or truncated

        eval_rewards.append(ep_rew)
        eval_lengths.append(steps)
        tel = last_info.get("telemetry", {})
        fuel = float(tel.get("fuel_used", 0.0))
        eval_fuels.append(fuel)

        is_succ = bool(last_info.get("is_success", False))
        is_crash = bool(last_info.get("is_crashed", False))
        is_time = bool(last_info.get("is_timeout", False))

        if is_succ:
            eval_successes += 1
            success_fuels.append(fuel)
            if len(success_videos) < target_successes and frames:
                v_path = video_dir / f"successo_{len(success_videos) + 1}_step_{steps}.mp4"
                _save_frames_as_mp4(frames, v_path)
                success_videos.append(v_path)
                if fuel < best_success_fuel:
                    best_success_fuel = fuel
                    best_success_video = v_path
        else:
            if is_crash:
                eval_crashes += 1
            elif is_time:
                eval_timeouts += 1
            else:
                eval_crashes += 1  # Fallimento generico classificato come crash
            failure_fuels.append(fuel)
            if len(failure_videos) < target_failures and frames:
                causa = "crash" if is_crash else ("timeout" if is_time else "fallimento")
                v_path = video_dir / f"fallimento_{len(failure_videos) + 1}_{causa}_step_{steps}.mp4"
                _save_frames_as_mp4(frames, v_path)
                failure_videos.append(v_path)

    eval_env.close()

    total_evaluated = max(1, len(eval_rewards))
    success_rate = (eval_successes / total_evaluated) * 100.0

    print(f"[play2learn] Valutazione completata ({total_evaluated} prove): "
          f"{len(success_videos)} video di successo, {len(failure_videos)} video di fallimento registrati.")

    # Generazione della GIF animata dell'allunaggio di successo migliore
    gif_path: Path | None = None
    if create_gif and best_success_video and best_success_video.exists():
        gif_path = _convert_video_to_gif(best_success_video)

    all_recorded = success_videos + failure_videos

    return {
        "mean_reward": float(np.mean(eval_rewards)),
        "std_reward": float(np.std(eval_rewards)),
        "mean_length": float(np.mean(eval_lengths)),
        "mean_fuel": float(np.mean(eval_fuels)),
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
