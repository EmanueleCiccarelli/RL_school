"""
Modulo Core di Storage e Persistenza per Play2Learn.
Gestisce scritture atomiche crash-safe, risoluzione timestamp e indicizzazione metadata.
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
    """Scrive testo in modo atomico con cleanup garantito del file temporaneo su errore."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=str(p.parent), prefix=f".{p.name}.", suffix=".tmp")
    tmp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as tf:
            tf.write(content)
            tf.flush()
            os.fsync(tf.fileno())
        os.replace(str(tmp_path), str(p))
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path | str, data: Any, encoding: str = "utf-8", indent: int = 2) -> None:
    """Serializza JSON (UTF-8 leggibile) e lo scrive atomicamente su disco."""
    payload = json.dumps(data, indent=indent, ensure_ascii=False, default=str)
    atomic_write_text(path, payload, encoding=encoding)


def _sanitize_json_value(val: Any) -> Any:
    """Converte ricorsivamente oggetti in strutture JSON-compatibili, serializzando i Path."""
    if isinstance(val, (int, float, str, bool)) or val is None:
        return val
    if isinstance(val, Path):
        return str(val)
    if isinstance(val, (list, tuple)):
        return [_sanitize_json_value(x) for x in val]
    if isinstance(val, dict):
        return {str(k): _sanitize_json_value(v) for k, v in val.items()}
    return str(val)


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
    Risolve un identificatore di run (timestamp 'YYYYMMDD_HHMMSS', ISO, 'latest', 'previous')
    nella directory corrispondente cercata in `runs_dir`.
    """
    if runs_dir is None:
        runs_dir = Path("runs")
    else:
        runs_dir = Path(runs_dir)

    if not runs_dir.exists():
        return None

    clean_id = str(identifier).strip()

    # Separa run standard da archivi (favorites, successi) per evitare che un salvataggio falsi 'latest'
    standard_dirs: list[Path] = []
    all_dirs: list[Path] = []
    for d in runs_dir.iterdir():
        if d.is_dir() and (d / "run_metadata.json").exists():
            all_dirs.append(d)
            if d.name not in ("favorites", "preferiti", "successi"):
                standard_dirs.append(d)

    if not all_dirs:
        return None

    standard_dirs.sort(key=lambda p: p.name, reverse=True)
    all_dirs.sort(key=lambda p: p.name, reverse=True)

    q_low = clean_id.lower()
    if q_low in ("latest", "ultima", "ultimo", "last", "recents", "recente"):
        return standard_dirs[0] if standard_dirs else all_dirs[0]

    if q_low in ("previous", "precedente", "penultima", "penultimo", "prev"):
        pool = standard_dirs if len(standard_dirs) >= 2 else all_dirs
        return pool[1] if len(pool) >= 2 else None

    # Ricerca per match esatto o parziale del timestamp
    norm_query = clean_id.replace(":", "").replace("-", "").replace(" ", "_").lower()
    for p in all_dirs:
        norm_name = p.name.replace(":", "").replace("-", "").replace(" ", "_").lower()
        if norm_query in norm_name:
            return p

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
        if not d.is_dir() or d.name in ("favorites", "preferiti", "successi"):
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
    """Salva i metadati completi (inclusa la history per i grafici) in formato JSON atomico."""
    run_dir.mkdir(parents=True, exist_ok=True)
    meta_path = run_dir / "run_metadata.json"

    reward_code = "# Funzione non disponibile"
    if reward_fn is not None:
        try:
            reward_code = inspect.getsource(reward_fn).strip()
        except Exception:
            reward_code = str(reward_fn)

    clean_eval = _sanitize_json_value(eval_stats)
    data: dict[str, Any] = {
        "timestamp": datetime.datetime.now().isoformat(),
        "mission": {
            "key": mission,
            "title": mission_cfg.get("title", mission),
            "description": mission_cfg.get("description", ""),
            "env_kwargs": _sanitize_json_value(env_kwargs),
        },
        "training": {
            "total_timesteps": int(total_timesteps),
            "elapsed_seconds": round(float(elapsed_seconds), 2),
            "fps_average": round(total_timesteps / max(0.1, float(elapsed_seconds)), 1),
        },
        "reward_source_code": reward_code,
        "eval_metrics": clean_eval,
        "diagnostics": _sanitize_json_value(diagnostics),
        "notes": notes,
    }
    if extra:
        data.update(_sanitize_json_value(extra))

    atomic_write_json(meta_path, data)
    return meta_path


def comprimi_runs(
    runs_dir: Path | str = "runs",
    nome_zip: str = "runs_archivio.zip",
    scarica_su_colab: bool = True,
) -> Path | None:
    """
    Comprime l'intera cartella 'runs/' in un file ZIP al livello del notebook
    e, se in esecuzione su Google Colab, avvia automaticamente il download nel browser.
    """
    import zipfile

    r_dir = Path(runs_dir)
    if not r_dir.exists():
        print(f"⚠️ Cartella '{runs_dir}' non trovata. Nessun esperimento da comprimere.")
        return None

    zip_path = Path(nome_zip)
    if not zip_path.is_absolute():
        zip_path = Path.cwd() / zip_path

    print(f"📦 Compressione della cartella '{r_dir}' in corso...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(r_dir):
            for file in files:
                file_p = Path(root) / file
                arcname = file_p.relative_to(r_dir.parent)
                zipf.write(file_p, arcname=str(arcname))

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"✅ Archivio ZIP creato con successo: {zip_path.name} ({size_mb:.2f} MB)")
    print(f"📍 Percorso: {zip_path}")

    if scarica_su_colab:
        try:
            from google.colab import files
            print("🚀 Avvio download automatico nel browser tramite Google Colab...")
            files.download(str(zip_path))
        except Exception:
            pass

    return zip_path
