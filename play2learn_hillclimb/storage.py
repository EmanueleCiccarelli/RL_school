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
    elenca_cronologia_run,
    find_run_by_timestamp,
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
