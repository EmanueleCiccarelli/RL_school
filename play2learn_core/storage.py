"""
Modulo Core di Storage e Persistenza per Play2Learn.
Gestisce scritture atomiche, risoluzione timestamp e indicizzazione metadata.
"""

from __future__ import annotations

import datetime
import inspect
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable


def atomic_write_text(path: Path | str, content: str, encoding: str = "utf-8") -> None:
    """Scrive testo in modo atomico usando un file temporaneo e sostituzione atomica."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = p.parent
    with tempfile.NamedTemporaryFile("w", dir=temp_dir, delete=False, encoding=encoding) as tf:
        tf.write(content)
        tf.flush()
        os.fsync(tf.fileno())
        temp_name = tf.name
    os.replace(temp_name, str(p))


def atomic_write_json(path: Path | str, data: Any, encoding: str = "utf-8", indent: int = 2) -> None:
    """Scrive JSON in modo atomico usando un file temporaneo e sostituzione atomica."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = p.parent
    with tempfile.NamedTemporaryFile("w", dir=temp_dir, delete=False, encoding=encoding) as tf:
        json.dump(data, tf, indent=indent, default=str)
        tf.flush()
        os.fsync(tf.fileno())
        temp_name = tf.name
    os.replace(temp_name, str(p))


def unique_dir(parent: Path, base_name: str) -> Path:
    """Crea una cartella garantendo che non collida con nomi esistenti."""
    target = parent / base_name
    if not target.exists():
        return target
    counter = 1
    while True:
        candidate = parent / f"{base_name}_{counter}"
        if not candidate.exists():
            return candidate
        counter += 1


def find_run_by_timestamp(
    identifier: str,
    runs_dir: Path | str | None = None
) -> Path | None:
    """
    Risolve un identificatore di run (timestamp 'YYYYMMDD_HHMMSS', ISO, o 'latest')
    nella directory corrispondente cercata in `runs_dir`.
    """
    if runs_dir is None:
        runs_dir = Path("runs")
    else:
        runs_dir = Path(runs_dir)

    if not runs_dir.exists():
        return None

    clean_id = str(identifier).strip().replace(":", "").replace("-", "").replace(" ", "_")

    # Scansione di tutte le directory che contengono run_metadata.json
    candidates: list[tuple[str, Path]] = []
    for d in runs_dir.iterdir():
        if d.is_dir() and d.name not in ("favorites", "preferiti"):
            meta = d / "run_metadata.json"
            if meta.exists():
                candidates.append((d.name, d))

    if not candidates:
        return None

    # Ordina cronologicamente
    candidates.sort(key=lambda x: x[0], reverse=True)

    if clean_id.lower() in ("latest", "last", "ultimo", "ultima"):
        return candidates[0][1]

    # Ricerca esatta o per prefisso/suffisso
    for name, path in candidates:
        norm_name = name.replace(":", "").replace("-", "").replace(" ", "_")
        if clean_id in norm_name:
            return path

    return None


def elenca_cronologia_run(runs_dir: Path | str | None = None) -> list[dict[str, Any]]:
    """Restituisce la lista di tutte le run completate indicizzate cronologicamente."""
    if runs_dir is None:
        runs_dir = Path("runs")
    else:
        runs_dir = Path(runs_dir)

    if not runs_dir.exists():
        return []

    cronologia = []
    for d in sorted(runs_dir.iterdir(), reverse=True):
        if not d.is_dir() or d.name in ("favorites", "preferiti"):
            continue
        meta_path = d / "run_metadata.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            raw_ts = meta.get("timestamp", d.name)
            try:
                dt = datetime.datetime.fromisoformat(raw_ts)
                fmt_data = dt.strftime("%Y-%m-%d")
                fmt_ora = dt.strftime("%H:%M:%S")
            except (ValueError, TypeError):
                clean_ts = raw_ts.replace(":", "").replace("-", "").replace("_", "")
                if len(clean_ts) >= 14:
                    fmt_data = f"{clean_ts[:4]}-{clean_ts[4:6]}-{clean_ts[6:8]}"
                    fmt_ora = f"{clean_ts[8:10]}:{clean_ts[10:12]}:{clean_ts[12:14]}"
                else:
                    fmt_data = raw_ts
                    fmt_ora = ""

            cronologia.append({
                "dir_name": d.name,
                "dir_path": str(d),
                "timestamp": raw_ts,
                "data": fmt_data,
                "ora": fmt_ora,
                "mission_title": meta.get("mission", {}).get("title", meta.get("mission", "Missione")),
                "success_rate": meta.get("eval_metrics", {}).get("eval_success_rate", 0.0),
                "mean_reward": meta.get("eval_metrics", {}).get("mean_reward", 0.0),
                "videos_count": len(meta.get("eval_metrics", {}).get("video_filenames", [])),
                "has_gif": meta.get("has_gif", False),
            })
        except Exception:
            continue

    return cronologia


def save_generic_run_metadata(
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
    extra: dict[str, Any] | None = None,
) -> Path:
    """Salva i metadati completi della sessione di addestramento in formato JSON atomico."""
    run_dir.mkdir(parents=True, exist_ok=True)
    meta_path = run_dir / "run_metadata.json"

    # Estrazione codice sorgente della reward function
    reward_code = "# Funzione non disponibile"
    if reward_fn is not None:
        try:
            reward_code = inspect.getsource(reward_fn)
        except Exception:
            reward_code = str(reward_fn)

    # Pulizia metadati di valutazione
    clean_eval = {}
    for k, v in eval_stats.items():
        if isinstance(v, (int, float, str, bool)):
            clean_eval[k] = v
        elif isinstance(v, list) and all(isinstance(x, (int, float, str, bool)) for x in v):
            clean_eval[k] = v

    data: dict[str, Any] = {
        "timestamp": datetime.datetime.now().isoformat(),
        "mission": {
            "key": mission,
            "title": mission_cfg.get("title", mission),
            "description": mission_cfg.get("description", ""),
            "env_kwargs": env_kwargs,
        },
        "training": {
            "total_timesteps": total_timesteps,
            "elapsed_seconds": round(elapsed_seconds, 2),
            "fps_average": round(total_timesteps / max(0.1, elapsed_seconds), 1),
        },
        "reward_source_code": reward_code,
        "eval_metrics": clean_eval,
        "diagnostics": diagnostics,
        "notes": notes,
    }

    if extra:
        data.update(extra)

    atomic_write_json(meta_path, data)
    return meta_path
