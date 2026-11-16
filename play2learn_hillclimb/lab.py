"""
Interfaccia Principale del Laboratorio Didattico per Hill Climb Racing (Play2Learn).
Fornisce la funzione didattica `run_lab` e la classe `LabResult` con piena parita di API.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

from play2learn_core.agent import (
    configure_cpu_threads,
    get_optimal_vec_config,
    make_parallel_envs,
    AdaptiveBatchSizeCallback,
    attach_jit_rollout_forward,
)
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
from play2learn_hillclimb.reporting import genera_report_esperimento
from play2learn_hillclimb.storage import create_run_directory, save_run_metadata


class LabResult(dict):
    """Oggetto risultato arricchito con metodi didattici rapidi e retrocompatibilita completa."""

    def __init__(self, data: dict[str, Any]):
        super().__init__(data)
        self.model = data.get("model")
        self.run_dir = Path(data.get("checkpoint_dir", "."))

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'LabResult' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def genera_report(
        self,
        autore: str = "Studente",
        titolo: str = "Report Hill Climb",
        note: str = "",
    ) -> dict[str, Path]:
        """Genera istantaneamente la scheda tecnica in PDF, HTML e Markdown."""
        return genera_report_esperimento(run_or_fav=self, autore=autore, titolo=titolo, note=note)

    report = genera_report

    def salva_preferito(
        self,
        nome_personalizzato: str | None = None,
        note: str = "",
        tag: str = "Mio Preferito",
        nome: str | None = None,
    ) -> dict[str, Any]:
        """Salva questo esperimento nei preferiti personali."""
        from play2learn_hillclimb.favorites import salva_nei_preferiti
        chosen = nome or nome_personalizzato or self.get("mission", "campagna_standard")
        return salva_nei_preferiti(self, nome=chosen, note=note, tag=tag)

    preferito = salva_preferito

    def mostra_evoluzione(self, force: bool = False) -> Any:
        """
        Visualizza (e genera su richiesta se non ancora presente) il timelapse video/GIF
        dell'evoluzione dell'auto collinare nel corso del training.
        """
        existing = self.get("gif_evoluzione_path")
        run_dir = Path(self.get("checkpoint_dir", "."))

        valid_path = None
        candidates = [
            Path(existing) if existing else None,
            run_dir / "evoluzione_apprendimento.mp4",
            run_dir / "evoluzione_apprendimento.gif",
        ]
        for c in candidates:
            if c and c.exists():
                valid_path = c
                break

        if not force and valid_path:
            self["gif_evoluzione_path"] = str(valid_path)
            return self._display_timelapse(valid_path)

        milestones = self.get("milestone_snapshots")
        model = self.get("model")
        if not milestones or model is None:
            print("[play2learn] Nessun checkpoint o milestone di addestramento disponibile per generare l'evoluzione.")
            return None

        print("🎬 Generazione su richiesta del timelapse dell'evoluzione dell'apprendimento...")
        try:
            target_path = run_dir / "evoluzione_apprendimento.mp4"
            out = crea_gif_evoluzione_apprendimento(
                checkpoints=milestones,
                output_gif_path=target_path,
                env_kwargs=self.get("env_kwargs"),
                reward_fn=self.get("reward_fn"),
                base_model=model,
                seed=self.get("seed", 42),
            )
            if out and out.exists():
                self["gif_evoluzione_path"] = str(out)
                print(f"🎬 Timelapse salvato: runs/{run_dir.name}/{out.name}")
                return self._display_timelapse(out)
            else:
                print("[play2learn] Impossibile generare il timelapse.")
                return None
        except Exception as e:
            print(f"[play2learn] Avviso durante la generazione del timelapse: {e}")
            return None

    def _display_timelapse(self, media_path: Path) -> Any:
        try:
            import base64
            from IPython.display import HTML, display
            b64_data = base64.b64encode(media_path.read_bytes()).decode("ascii")
            if media_path.suffix.lower() == ".mp4":
                media_tag = (
                    f'<video width="480" controls autoplay loop muted style="border-radius:6px; box-shadow:0 4px 12px rgba(0,0,0,0.3);">'
                    f'<source src="data:video/mp4;base64,{b64_data}" type="video/mp4">'
                    f'</video>'
                )
            else:
                media_tag = f'<img src="data:image/gif;base64,{b64_data}" style="max-width:480px; border-radius:6px; box-shadow:0 4px 12px rgba(0,0,0,0.3);" />'

            html_code = (
                f'<div style="margin-top:10px; margin-bottom:10px; text-align:center; background:#1e1e2f; padding:14px; border-radius:8px; border:1.5px solid #2e7d32;">'
                f'<h4 style="color:#a5d6a7; margin-top:2px; margin-bottom:6px;">🎞️ Evoluzione dell\'Apprendimento nel Tempo (Hill Climb)</h4>'
                f'<p style="color:#c8e6c9; font-size:12px; margin-bottom:12px;">Avanzamento del veicolo sulle colline nelle varie fasi di addestramento:</p>'
                f'{media_tag}'
                f'</div>'
            )
            display(HTML(html_code))
            return media_path
        except Exception:
            print(f"🎞️ Timelapse evoluzione salvato in: {media_path}")
            return media_path

    genera_evoluzione = mostra_evoluzione


def run_lab(
    reward_fn: Callable | None = None,
    mission: str = "campagna_standard",
    total_timesteps: int = DEFAULT_TIMESTEPS,
    n_envs: int = DEFAULT_N_ENVS,
    learning_rate: float = 3e-4,
    crea_gif_evoluzione: bool = False,
    seed: int = 42,
    show_progress: bool = True,
) -> LabResult:
    """
    Funzione didattica principale per avviare l'addestramento dell'auto in Hill Climb Racing.
    """
    configure_cpu_threads()

    mission_cfg = MISSIONS.get(mission, MISSIONS["campagna_standard"])
    env_kwargs = dict(mission_cfg["env_kwargs"])

    run_dir = create_run_directory(mission=mission)
    checkpoints_dir = run_dir / "checkpoints"

    resolved_envs, use_subproc = get_optimal_vec_config(
        n_envs=n_envs,
        total_timesteps=total_timesteps,
    )

    print("=" * 70)
    print(" 🏎️  PLAY2LEARN — LABORATORIO DI REINFORCEMENT LEARNING")
    print("     Ambiente: Hill Climb Racing (Farama Box2D + Perlin Noise)")
    print(f"     Tracciato: {mission_cfg['title']}")
    print(f"     Timesteps previsti: {total_timesteps:,} | Ambienti paralleli: {resolved_envs}")
    print("=" * 70)

    # 1. Creazione ambienti paralleli con Monitor e render_mode=None per massima velocita
    def _make_single_env(rank: int) -> Callable[[], gym.Env]:
        def _thunk() -> gym.Env:
            env = gym.make(ENV_ID, render_mode=None, **env_kwargs)
            env = HillClimbRewardWrapper(env, reward_fn=reward_fn)
            env = Monitor(env, info_keywords=("is_success",))
            env.reset(seed=seed + rank * 100)
            return env
        return _thunk

    env_fns = [_make_single_env(i) for i in range(resolved_envs)]
    vec_env = make_parallel_envs(env_fns, use_subproc=use_subproc, vecnorm_kwargs=VECNORM_PARAMS)

    callback = HillClimbRewardTrackerCallback(
        total_timesteps=total_timesteps,
        show_progress=show_progress,
        save_checkpoints=True,
        checkpoint_dir=checkpoints_dir,
        milestone_freq=max(10_000, total_timesteps // 4),
    )

    # 2. Creazione modello PPO con feature extraction invariante
    model = build_hill_agent(
        env=vec_env,
        learning_rate=learning_rate,
        seed=seed,
    )
    attach_jit_rollout_forward(model)

    adaptive_cb = AdaptiveBatchSizeCallback(
        total_timesteps=total_timesteps,
        switch_fraction=0.5,
        target_batch_size=256,
    )

    t0 = time.time()
    try:
        model.learn(total_timesteps=total_timesteps, callback=[callback, adaptive_cb])
    finally:
        vec_env.close()

    elapsed = time.time() - t0
    print(f"\n✅ Addestramento completato in {elapsed:.1f} s ({total_timesteps/max(0.1, elapsed):.0f} step/s)!")

    # 3. Valutazione deterministica e registrazione video streaming
    print("\n🎥 Avvio sessione di verifica deterministica su 20 scenari collinari...")
    eval_stats = evaluate_and_record(
        model=model,
        output_dir=run_dir,
        env_kwargs=env_kwargs,
        reward_fn=reward_fn,
    )

    # 4. GIF evoluzione apprendimento
    gif_path = None
    if crea_gif_evoluzione and callback.milestone_snapshots:
        print("🎬 Generazione timelapse MP4 dell'evoluzione dell'apprendimento...")
        gif_path = crea_gif_evoluzione_apprendimento(
            checkpoints=callback.milestone_snapshots,
            output_gif_path=run_dir / "evoluzione_apprendimento.mp4",
            env_kwargs=env_kwargs,
            reward_fn=reward_fn,
            base_model=model,
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
        "milestone_snapshots": callback.milestone_snapshots,
        "reward_fn": reward_fn,
        "seed": seed,
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

    # Se siamo su Linux headless (es. Colab senza DISPLAY), avvisa l'utente
    is_headless = (os.name != "nt") and (not os.environ.get("DISPLAY")) and (sys.platform != "darwin")
    if is_headless:
        print("[play2learn] Ambiente headless rilevato (Google Colab / Linux server).")
        print("💡 La visualizzazione Pygame su schermo non e supportata in headless. I video MP4 sono disponibili nella cartella run.")
        return

    try:
        env = gym.make(ENV_ID, render_mode="human", **mission_cfg["env_kwargs"])
        obs, _ = env.reset(seed=seed)

        print("🚗 Dimostrazione live avviata... Premi ESC nella finestra Pygame per uscire.")
        for _ in range(1200):
            action, _ = model.predict(obs, deterministic=True)
            act_item = int(action.item()) if hasattr(action, "item") and not hasattr(action, "__len__") else int(action)
            obs, reward, terminated, truncated, info = env.step(act_item)
            env.render()
            if terminated or truncated:
                break

        env.close()
    except Exception as e:
        print(f"[play2learn] Impossibile avviare la finestra Pygame ({e}).")


def mostra_evoluzione(risultati_o_path: Any = None, force: bool = False) -> Any:
    """Funzione helper didattica per mostrare/generare il timelapse dell'evoluzione dell'apprendimento."""
    if risultati_o_path is None:
        print("[play2learn] Passa l'oggetto 'risultati' restituito da run_lab() per visualizzare l'evoluzione.")
        return None
    if isinstance(risultati_o_path, LabResult):
        return risultati_o_path.mostra_evoluzione(force=force)
    p = Path(risultati_o_path)
    if p.is_dir():
        cands = [p / "evoluzione_apprendimento.mp4", p / "evoluzione_apprendimento.gif"]
        for c in cands:
            if c.exists():
                p = c
                break
    if p.is_file():
        lr = LabResult({"checkpoint_dir": str(p.parent), "gif_evoluzione_path": str(p)})
        return lr._display_timelapse(p)
    print(f"[play2learn] Nessun timelapse trovato per: {risultati_o_path}")
    return None


genera_evoluzione = mostra_evoluzione
