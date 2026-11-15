"""
Interfaccia Principale del Laboratorio Didattico per Hill Climb Racing (Play2Learn).
Fornisce la funzione didattica `run_lab` e la classe `LabResult`.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym

from play2learn_core.agent import make_parallel_envs
from play2learn_hillclimb.agent import HillClimbRewardTrackerCallback, build_hill_agent
from play2learn_hillclimb.config import (
    DEFAULT_N_ENVS,
    DEFAULT_TIMESTEPS,
    ENV_ID,
    MISSIONS,
    VECNORM_PARAMS,
)
from play2learn_hillclimb.diagnostics import diagnose_hill_climb_behavior
from play2learn_hillclimb.environment import HillClimbRewardWrapper, classify_hill_climb_outcome
from play2learn_hillclimb.evaluator import crea_gif_evoluzione_apprendimento, evaluate_and_record
from play2learn_hillclimb.interactive import gioca_vs_agente
from play2learn_hillclimb.reporting import genera_report_esperimento
from play2learn_hillclimb.storage import create_run_directory, save_run_metadata


class LabResult(dict):
    """Oggetto risultato arricchito con metodi didattici rapidi."""

    def __init__(self, data: dict[str, Any]):
        super().__init__(data)
        self.model = data.get("model")
        self.run_dir = Path(data.get("checkpoint_dir", "."))

    def report(self, autore: str = "Studente", titolo: str = "Report Hill Climb") -> dict[str, Path]:
        """Genera istantaneamente la scheda tecnica in PDF, HTML e Markdown."""
        return genera_report_esperimento(run_or_fav=self, autore=autore, titolo=titolo)

    def sfida(self, seed: int = 42) -> Any:
        """Lancia la sfida interattiva Split-Screen contro questo modello."""
        return gioca_vs_agente(
            model_or_result=self.model,
            seed=seed,
            mission=self.get("mission", "campagna_standard"),
            env_kwargs=self.get("env_kwargs"),
        )

    def preferito(self, nome: str, note: str = "") -> dict[str, Any]:
        """Salva questo esperimento nei preferiti personali."""
        from play2learn_hillclimb.favorites import salva_nei_preferiti
        return salva_nei_preferiti(self, nome=nome, note=note)


def run_lab(
    reward_fn: Callable | None = None,
    mission: str = "campagna_standard",
    total_timesteps: int = DEFAULT_TIMESTEPS,
    n_envs: int = DEFAULT_N_ENVS,
    learning_rate: float = 3e-4,
    crea_gif_evoluzione: bool = True,
    seed: int = 42,
    show_progress: bool = True,
) -> LabResult:
    """
    Funzione didattica principale per avviare l'addestramento dell'auto in Hill Climb Racing.
    """
    mission_cfg = MISSIONS.get(mission, MISSIONS["campagna_standard"])
    env_kwargs = dict(mission_cfg["env_kwargs"])

    run_dir = create_run_directory(mission=mission)
    checkpoints_dir = run_dir / "checkpoints"

    print("=" * 70)
    print(" 🏎️  PLAY2LEARN — LABORATORIO DI REINFORCEMENT LEARNING")
    print("     Ambiente: Hill Climb Racing (Farama Box2D + Perlin Noise)")
    print(f"     Tracciato: {mission_cfg['title']}")
    print(f"     Timesteps previsti: {total_timesteps:,} | Ambienti paralleli: {n_envs}")
    print("=" * 70)

    # 1. Creazione ambienti paralleli
    def _make_single_env(rank: int) -> Callable[[], gym.Env]:
        def _thunk() -> gym.Env:
            env = gym.make(ENV_ID, render_mode="rgb_array", **env_kwargs)
            env = HillClimbRewardWrapper(env, reward_fn=reward_fn)
            env.reset(seed=seed + rank * 100)
            return env
        return _thunk

    env_fns = [_make_single_env(i) for i in range(n_envs)]
    vec_env = make_parallel_envs(env_fns, use_subproc=(n_envs > 1))

    # Raccolta milestone per timelapse GIF evolutivo
    milestones: list[tuple[int, Any]] = []

    def _milestone_cb(step: int, mdl: Any) -> None:
        if crea_gif_evoluzione and len(milestones) < 6:
            milestones.append((step, mdl))

    callback = HillClimbRewardTrackerCallback(
        total_timesteps=total_timesteps,
        show_progress=show_progress,
        save_checkpoints=True,
        checkpoint_dir=checkpoints_dir,
        milestone_callback=_milestone_cb,
    )

    # 2. Creazione modello PPO
    model = build_hill_agent(
        env=vec_env,
        learning_rate=learning_rate,
        seed=seed,
    )

    t0 = time.time()
    try:
        model.learn(total_timesteps=total_timesteps, callback=callback)
    finally:
        vec_env.close()

    elapsed = time.time() - t0
    print(f"\n✅ Addestramento completato in {elapsed:.1f} s ({total_timesteps/max(0.1, elapsed):.0f} step/s)!")

    # 3. Valutazione e registrazione video streaming
    print("\n🎥 Avvio sessione di verifica su 20 scenari collinari deterministici...")
    eval_stats = evaluate_and_record(
        model=model,
        output_dir=run_dir,
        env_kwargs=env_kwargs,
        reward_fn=reward_fn,
    )

    # 4. GIF evoluzione apprendimento
    gif_path = None
    if crea_gif_evoluzione and milestones:
        print("🎬 Generazione timelapse GIF dell'evoluzione dell'apprendimento...")
        gif_path = crea_gif_evoluzione_apprendimento(
            checkpoints=milestones,
            output_gif_path=run_dir / "evoluzione_apprendimento.gif",
            env_kwargs=env_kwargs,
            reward_fn=reward_fn,
        )

    # 5. Diagnostica comportamentale
    diagnostics = diagnose_hill_climb_behavior(eval_stats)
    print("\n" + diagnostics["formatted_text"])

    # 6. Salvataggio metadati
    save_run_metadata(
        run_dir=run_dir,
        mission=mission,
        mission_cfg=mission_cfg,
        env_kwargs=env_kwargs,
        total_timesteps=total_timesteps,
        elapsed_seconds=elapsed,
        reward_fn=reward_fn,
        eval_stats=eval_stats,
        diagnostics=diagnostics,
    )

    # Salvataggio modello finale
    model.save(str(run_dir / "final_model.zip"))

    result_data = {
        "model": model,
        "eval_success_rate": eval_stats["eval_success_rate"],
        "mean_distance": eval_stats["mean_distance"],
        "max_distance": eval_stats["max_distance_achieved"],
        "mean_speed": eval_stats["mean_speed"],
        "checkpoint_dir": str(run_dir),
        "best_video_path": str(eval_stats["best_video_path"]) if eval_stats.get("best_video_path") else None,
        "gif_evoluzione_path": str(gif_path) if gif_path else None,
        "mission": mission,
        "env_kwargs": env_kwargs,
        "eval_metrics": eval_stats,
        "diagnostics": diagnostics,
    }

    print("\n🏆 Riepilogo Finale:")
    print(f"   • Distanza Massima: {eval_stats['max_distance_achieved']:.1f} m")
    print(f"   • Distanza Media:   {eval_stats['mean_distance']:.1f} m")
    print(f"   • Tasso Successo:   {eval_stats['eval_success_rate']:.1f}%")
    print(f"   • Cartella Run:     {run_dir.name}\n")

    return LabResult(result_data)


def dimostrazione_live(
    model_or_result: Any,
    seed: int = 42,
    mission: str = "campagna_standard",
) -> None:
    """Mostra una dimostrazione visiva della guida dell'agente."""
    model = getattr(model_or_result, "model", model_or_result)
    mission_cfg = MISSIONS.get(mission, MISSIONS["campagna_standard"])

    env = gym.make(ENV_ID, render_mode="human", **mission_cfg["env_kwargs"])
    obs, _ = env.reset(seed=seed)

    print("🚗 Dimostrazione live avviata... Premi ESC nella finestra Pygame per uscire.")
    for _ in range(1200):
        action, _ = model.predict(obs, deterministic=True)
        act_item = action.item() if hasattr(action, "item") and not hasattr(action, "__len__") else action
        obs, reward, terminated, truncated, info = env.step(act_item)
        env.render()
        if terminated or truncated:
            break

    env.close()
