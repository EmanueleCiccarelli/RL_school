"""
Modulo per la gestione della persistenza, salvataggio dei checkpoint dei modelli,
delle metriche, del codice della funzione di reward e dell'alberatura 'runs/'.
"""

from __future__ import annotations

import json
import inspect
import datetime
from pathlib import Path
from typing import Callable, Any


def get_default_runs_dir() -> Path:
    """Restituisce il percorso della cartella principale 'runs/'."""
    # Cerchiamo la cartella 'runs/' a livello di root del laboratorio
    runs_dir = Path("runs").resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / "favorites").mkdir(parents=True, exist_ok=True)
    return runs_dir


def create_run_directory(
    mission: str,
    base_dir: Path | str | None = None,
    custom_name: str | None = None
) -> Path:
    """
    Crea una nuova directory timestampata per la sessione corrente in 'runs/'.
    Esempio: runs/run_20261114_154500_luna_standard/
    """
    base = Path(base_dir).resolve() if base_dir else get_default_runs_dir()
    base.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if custom_name:
        clean_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in custom_name)
        dir_name = f"run_{timestamp}_{clean_name}"
    else:
        dir_name = f"run_{timestamp}_{mission}"

    run_dir = base / dir_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "videos").mkdir(parents=True, exist_ok=True)
    return run_dir


def extract_reward_function_code(reward_fn: Callable) -> str:
    """Estrae in modo sicuro il codice sorgente della funzione scritta dallo studente."""
    try:
        source = inspect.getsource(reward_fn)
        return source.strip()
    except Exception:
        return "# Codice sorgente non disponibile tramite inspect."


def save_run_metadata(
    run_dir: Path,
    mission: str,
    mission_cfg: dict[str, Any],
    env_kwargs: dict[str, Any],
    total_timesteps: int,
    elapsed_seconds: float,
    reward_fn: Callable,
    eval_stats: dict[str, Any],
    diagnostics: dict[str, Any],
    tag: str = "Standard",
    notes: str = "",
) -> Path:
    """
    Salva il file JSON con tutti i metadati e il file Python con il codice sorgente esatto
    della custom_reward utilizzata per questa run.
    """
    source_code = extract_reward_function_code(reward_fn)

    # 1. Salva il codice Python autonomo per consultazione rapida
    code_path = run_dir / "reward_function.py"
    try:
        code_path.write_text(
            f'"""\nFunzione di Reward registrata per la run: {run_dir.name}\nData: {datetime.datetime.now().isoformat()}\n"""\n\n'
            f"{source_code}\n",
            encoding="utf-8"
        )
    except Exception:
        pass

    metadata = {
        "run_id": run_dir.name,
        "timestamp": datetime.datetime.now().isoformat(),
        "tag": tag,
        "notes": notes,
        "mission": {
            "key": mission,
            "title": mission_cfg.get("title", mission),
            "description": mission_cfg.get("description", ""),
            "goal": mission_cfg.get("goal", ""),
            "env_kwargs": env_kwargs,
        },
        "training": {
            "total_timesteps": total_timesteps,
            "elapsed_seconds": round(elapsed_seconds, 2),
            "fps_average": int(total_timesteps / max(1.0, elapsed_seconds)),
        },
        "eval_metrics": {
            "eval_success_rate": eval_stats.get("eval_success_rate", 0.0),
            "eval_successes": eval_stats.get("eval_successes", 0),
            "eval_crashes": eval_stats.get("eval_crashes", 0),
            "eval_timeouts": eval_stats.get("eval_timeouts", 0),
            "total_evaluated_episodes": eval_stats.get("total_evaluated_episodes", 0),
            "mean_reward": eval_stats.get("mean_reward", 0.0),
            "std_reward": eval_stats.get("std_reward", 0.0),
            "mean_length": eval_stats.get("mean_length", 0.0),
            "mean_fuel": eval_stats.get("mean_fuel", 0.0),
            "mean_fuel_success": eval_stats.get("mean_fuel_success", 0.0),
            "mean_fuel_failure": eval_stats.get("mean_fuel_failure", 0.0),
        },
        "diagnostics": {
            "status": diagnostics.get("status", "unknown"),
            "title": diagnostics.get("title", ""),
            "suggestion": diagnostics.get("suggestion", ""),
        },
        "reward_source_code": source_code,
    }

    meta_file = run_dir / "run_metadata.json"
    meta_file.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta_file


def list_all_runs(runs_dir: Path | str | None = None) -> list[dict[str, Any]]:
    """Restituisce l'elenco di tutte le run salvate con i relativi metadati sintetici."""
    base = Path(runs_dir).resolve() if runs_dir else get_default_runs_dir()
    if not base.exists():
        return []

    runs = []
    for sub in sorted(base.iterdir(), reverse=True):
        if not sub.is_dir() or sub.name == "favorites":
            continue
        meta_file = sub / "run_metadata.json"
        if meta_file.exists():
            try:
                data = json.loads(meta_file.read_text(encoding="utf-8"))
                data["path"] = str(sub)
                runs.append(data)
            except Exception:
                continue
    return runs
