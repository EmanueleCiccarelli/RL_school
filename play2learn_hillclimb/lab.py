"""
Modulo Orchestratore Principale per Play2Learn Hill Climb Racing.
Fornisce l'interfaccia didattica unificata `run_lab()` e le funzioni di supporto
per gli esperimenti, la visualizzazione e il confronto nei notebook degli studenti.
"""

from __future__ import annotations

import copy
import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Sequence
import numpy as np

import gymnasium as gym

try:
    from stable_baselines3 import PPO
    _SB3_AVAILABLE = True
except ImportError:
    PPO = Any  # type: ignore
    _SB3_AVAILABLE = False

from play2learn_hillclimb.config import (
    ENV_ID,
    MISSIONS,
    DEFAULT_TIMESTEPS,
    TARGET_DISTANCE_SUCCESS,
)
from play2learn_hillclimb.environment import (
    make_parallel_envs,
    _make_single_env,
    HillClimbRewardWrapper,
    classify_hill_climb_outcome,
)
from play2learn_hillclimb.agent import (
    create_ppo_agent,
    HillClimbRewardTrackerCallback,
)
from play2learn_hillclimb.evaluator import (
    evaluate_and_record,
    crea_gif_evoluzione_apprendimento,
)
from play2learn_hillclimb.diagnostics import diagnose_hill_climb_behavior
from play2learn_hillclimb.visualizer import (
    mostra_mappa_ambiente,
    plot_hill_telemetry,
)
from play2learn_hillclimb.storage import (
    create_run_directory,
    save_run_metadata,
    comprimi_archivio,
)
from play2learn_hillclimb.favorites import (
    salva_nei_preferiti,
    mostra_preferiti,
    carica_preferito,
    LoadedFavorite,
)
from play2learn_hillclimb.reporting import (
    genera_report_esperimento,
    genera_report_da_timestamp,
    mostra_cronologia,
)


class LabResult(dict):
    """
    Oggetto risultato didattico restituito da `run_lab`.
    Permette l'accesso sia come dizionario che come attributi (es. res.eval_metrics).
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'LabResult' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def mostra_evoluzione(self) -> None:
        """Visualizza la GIF animata dell'evoluzione dell'apprendimento su Colab o Jupyter."""
        gif_path = self.get("gif_path")
        if gif_path is not None and Path(gif_path).exists():
            try:
                from IPython.display import display, Image  # type: ignore
                display(Image(filename=str(gif_path)))
            except Exception:
                print(f"[play2learn] GIF salvata in: {gif_path}")
        else:
            print("[play2learn] Nessuna animazione GIF registrata per questa run.")

    def salva_preferito(self, nome_personalizzato: str, note: str = "") -> dict[str, Any]:
        """Salva la run nel catalogo dei preferiti personali."""
        run_dir = Path(self.get("dir_path", self.get("run_dir", ".")))
        return salva_nei_preferiti(run_dir=run_dir, nome=nome_personalizzato, note=note)

    def genera_report(
        self,
        titolo: str = "Esperimento Hill Climb Racing",
        autore: str = "Studente Play2Learn",
        note: str = "",
    ) -> dict[str, Path]:
        """Rigenera la scheda tecnica dell'esperimento."""
        run_dir = Path(self.get("dir_path", self.get("run_dir", ".")))
        return genera_report_esperimento(
            run_dir=run_dir,
            metadata=dict(self),
            titolo=titolo,
            autore=autore,
            note=note,
        )


def run_lab(
    reward_fn: Optional[Callable] = None,
    mission: str = "campagna_standard",
    total_timesteps: int = DEFAULT_TIMESTEPS,
    num_envs: int = 1,
    learning_rate: float = 3e-4,
    crea_gif_evoluzione: bool = True,
    seed: int = 42,
    device: str = "auto",
) -> LabResult:
    """
    Avvia l'esperimento didattico completo:
      1. Configura l'ambiente vettoriale con la custom_reward dello studente.
      2. Allena l'agente PPO visualizzando la barra di avanzamento e la telemetria live.
      3. Valuta le prestazioni su 20 seed deterministici.
      4. Registra video MP4 in streaming per le migliori scalate e i fallimenti.
      5. Genera la Scheda Tecnica (PDF, HTML5, Markdown) e il timelapse GIF.
    """
    print("═════════════════════════════════════════════════════════════════════")
    print(f" 🏎️  AVVIO ESPERIMENTO DIDATTICO: HILL CLIMB RACING ({mission})")
    print(f" • Timesteps Totali: {total_timesteps:,}")
    print(f" • Learning Rate:    {learning_rate}")
    print(f" • Seed Casuale:     {seed}")
    print("═════════════════════════════════════════════════════════════════════\n")

    run_dir = create_run_directory(prefix="run_hillclimb")

    # 1. Creazione ambiente vettoriale
    vec_env = make_parallel_envs(
        num_envs=num_envs,
        mission=mission,
        reward_fn=reward_fn,
        target_distance=TARGET_DISTANCE_SUCCESS,
        seed=seed,
    )

    # 2. Inizializzazione agente PPO
    model = create_ppo_agent(
        env=vec_env,
        learning_rate=learning_rate,
        total_timesteps=total_timesteps,
        seed=seed,
        device=device,
    )

    # 3. Tracciamento pietre miliari per il timelapse GIF (usando gli snapshot in RAM del callback)
    checkpoint_interval = max(10_000, total_timesteps // 5)
    cb = HillClimbRewardTrackerCallback(
        total_timesteps=total_timesteps,
        window_size=20,
        save_checkpoints=True,
        checkpoint_dir=run_dir,
        milestone_freq=checkpoint_interval,
    )

    # 4. Addestramento
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=cb,
            progress_bar=False,
        )
    finally:
        vec_env.close()

    # 5. Salvataggio checkpoint modello finale e best
    final_model_path = run_dir / "final_model.zip"
    model.save(str(final_model_path))

    best_model_path = run_dir / "best_model.zip"
    if not best_model_path.exists():
        model.save(str(best_model_path))

    # 6. Valutazione e Registrazione Video
    print("\n🔍 Valutazione finale su 20 scenari di collaudo...")
    eval_metrics = evaluate_and_record(
        model=model,
        mission=mission,
        reward_fn=reward_fn,
        output_dir=run_dir,
        max_steps=1600,
        frame_skip=3,
    )

    # 7. Generazione GIF Evoluzione
    gif_path = None
    snapshots = getattr(cb, "milestone_snapshots", [])
    if crea_gif_evoluzione and snapshots:
        gif_out = run_dir / "evoluzione_apprendimento.gif"
        try:
            gif_path = crea_gif_evoluzione_apprendimento(
                milestones=snapshots,
                model=model,
                output_path=gif_out,
                mission=mission,
                seed=seed,
            )
            if gif_path:
                print(f"🎬 Timelapse GIF generato con successo: {gif_out.name}")
        except Exception as e:
            print(f"[play2learn] Avviso generazione GIF: {e}")

    # 8. Salvataggio Metadati e Report Tecnico
    mission_title = MISSIONS.get(mission, {}).get("nome", mission)
    metadata = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mission": {"key": mission, "title": mission_title},
        "mission_name": mission,
        "total_timesteps": total_timesteps,
        "learning_rate": learning_rate,
        "seed": seed,
        "eval_metrics": eval_metrics,
        "history": eval_metrics.get("history", []),
        "run_dir": str(run_dir),
        "dir_path": str(run_dir),
        "gif_path": str(gif_path) if gif_path else None,
    }
    save_run_metadata(run_dir, metadata)
    genera_report_esperimento(run_dir, metadata=metadata)

    # 9. Stampa diagnosi finale
    print(diagnose_hill_climb_behavior(eval_metrics))

    res = LabResult(metadata)
    res["model"] = model
    res["eval_metrics"] = eval_metrics
    res["dir_path"] = run_dir
    res["run_dir"] = run_dir
    return res


def collauda_missione(
    model: Any,
    mission: str = "deserto_dune",
    eval_episodes: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    """Collauda un modello già addestrato su una missione specifica estrema."""
    print(f"\n🚗 Collaudo Modello su Missione Sfida: '{mission}'...")
    results = evaluate_and_record(
        model=model,
        mission=mission,
        eval_seeds=[seed + i * 17 for i in range(eval_episodes)],
        max_steps=1600,
        frame_skip=3,
    )
    print(diagnose_hill_climb_behavior(results))
    return results


def confronta_run(lista_risultati: Sequence[LabResult | dict[str, Any]]) -> None:
    """Mette a confronto le metriche e la telemetria di più sessioni di addestramento."""
    if not lista_risultati:
        print("[play2learn] Fornisci almeno un risultato da confrontare.")
        return

    print("══════════════════════════════════════════════════════════════════════════════════")
    print(f"{'RUN / MISSIONE':<30} | {'SUCCESSO %':<10} | {'DIST MEDIA':<12} | {'DIST MAX'}")
    print("──────────────────────────────────────────────────────────────────────────────────")
    for idx, r in enumerate(lista_risultati):
        meta = r if isinstance(r, dict) else dict(r)
        metrics = meta.get("eval_metrics", {})
        miss = str(meta.get("mission", f"Run {idx+1}"))[:28]
        succ = f"{metrics.get('success_rate', metrics.get('eval_success_rate', 0.0)):.1f}%"
        dmean = f"{metrics.get('mean_distance', 0.0):.1f}m"
        dmax = f"{metrics.get('max_distance', 0.0):.1f}m"
        print(f"{miss:<30} | {succ:<10} | {dmean:<12} | {dmax}")
    print("══════════════════════════════════════════════════════════════════════════════════")
