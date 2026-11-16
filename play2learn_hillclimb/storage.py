"""
Modulo di Storage e Persistenza Esperimenti per Hill Climb Racing (Play2Learn).
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Callable

from play2learn_core.storage import (
    atomic_write_json,
    atomic_write_text,
    elenca_cronologia_run as _core_elenca,
    find_run_by_timestamp as _core_find,
    save_generic_run_metadata,
    unique_dir,
)


def get_default_runs_dir() -> Path:
    """Restituisce il percorso della cartella principale delle run per Hill Climb."""
    cwd = Path.cwd()
    if cwd.name in ("play2learn_hillclimb", "play2learn_lunarlander"):
        base = cwd.parent
    else:
        base = cwd
    p = base / "runs_hillclimb"
    p.mkdir(parents=True, exist_ok=True)
    return p


def find_run_by_timestamp(identifier: str, runs_dir: Path | str | None = None) -> Path | None:
    """Risolve un identificatore nella cartella runs_hillclimb."""
    actual_dir = Path(runs_dir) if runs_dir else get_default_runs_dir()
    return _core_find(identifier=identifier, runs_dir=actual_dir)


def elenca_cronologia_run(runs_dir: Path | str | None = None) -> list[dict[str, Any]]:
    """Restituisce la cronologia delle run per Hill Climb."""
    actual_dir = Path(runs_dir) if runs_dir else get_default_runs_dir()
    return _core_elenca(runs_dir=actual_dir)


def create_run_directory(mission: str = "campagna_standard", base_dir: Path | str | None = None) -> Path:
    """Crea una nuova directory univoca per la sessione di addestramento."""
    parent = Path(base_dir) if base_dir else get_default_runs_dir()
    parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"run_{timestamp}_{mission}"
    target = unique_dir(parent, dir_name)
    target.mkdir(parents=True, exist_ok=True)
    (target / "videos").mkdir(parents=True, exist_ok=True)
    (target / "checkpoints").mkdir(parents=True, exist_ok=True)
    return target


def save_run_metadata(
    run_dir: Path,
    mission: str,
    mission_cfg: dict[str, Any],
    env_kwargs: dict[str, Any],
    total_timesteps: int,
    elapsed_seconds: float,
    reward_fn: Callable | None,
    eval_stats: dict[str, Any],
    diagnostics: dict[str, Any],
    notes: str = "",
) -> Path:
    """Salva i metadati della sessione in formato JSON atomico."""
    return save_generic_run_metadata(
        run_dir=run_dir,
        mission=mission,
        mission_cfg=mission_cfg,
        env_kwargs=env_kwargs,
        total_timesteps=total_timesteps,
        elapsed_seconds=elapsed_seconds,
        reward_fn=reward_fn,
        eval_stats=eval_stats,
        diagnostics=diagnostics,
        notes=notes,
    )


def comprimi_runs(
    runs_dir: Path | str | None = None,
    nome_zip: str = "runs_hillclimb.zip",
    scarica_su_colab: bool = True,
) -> Path | None:
    """
    Comprime l'intera cartella degli esperimenti Hill Climb Racing in un file ZIP al livello del notebook
    e ne avvia il download automatico se eseguito su Google Colab.
    """
    from play2learn_core.storage import comprimi_runs as _core_comprimi
    actual_dir = Path(runs_dir) if runs_dir else get_default_runs_dir()
    if not actual_dir.exists() and Path("runs").exists():
        actual_dir = Path("runs")
    return _core_comprimi(runs_dir=actual_dir, nome_zip=nome_zip, scarica_su_colab=scarica_su_colab)
