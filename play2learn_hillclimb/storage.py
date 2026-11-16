"""
Modulo di Gestione Storage e Persistenza per Play2Learn Hill Climb Racing.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Optional

from play2learn_core.storage import (
    atomic_write_json,
    unique_dir,
    find_run_by_timestamp,
    elenca_cronologia_run,
    comprimi_runs,
)

BASE_RUNS_DIR = Path("runs")


def create_run_directory(prefix: str = "run_hillclimb", base_dir: Path = BASE_RUNS_DIR) -> Path:
    """Crea una cartella di esecuzione univoca con timestamp."""
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = unique_dir(base_dir, f"{prefix}_{ts}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run_metadata(run_dir: Path, metadata: dict[str, Any]) -> Path:
    """Salva i metadati della sessione di addestramento in formato JSON atomico."""
    run_dir = Path(run_dir)
    meta_path = run_dir / "run_metadata.json"
    atomic_write_json(meta_path, metadata)
    return meta_path


def load_run_metadata(run_dir: Path) -> dict[str, Any]:
    """Carica i metadati di una sessione."""
    meta_path = Path(run_dir) / "run_metadata.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def comprimi_archivio(source_dir: Path = BASE_RUNS_DIR, output_zip: Optional[Path] = None) -> Path:
    """Comprime la cartella delle run in formato ZIP per il download immediato su Google Colab."""
    source_dir = Path(source_dir)
    if output_zip is None:
        output_zip = Path("archivio_esperimenti_hillclimb.zip")
    else:
        output_zip = Path(output_zip)

    comprimi_runs(source_dir=source_dir, output_zip=output_zip)
    try:
        from google.colab import files  # type: ignore
        files.download(str(output_zip))
    except Exception:
        pass
    return output_zip
