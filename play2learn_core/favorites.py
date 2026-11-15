"""
Modulo Core di Gestione Esperimenti Preferiti (Play2Learn).
Archiviazione, indicizzazione e caricamento portabile tra Colab e Windows.
"""

from __future__ import annotations

import datetime
import json
import shutil
from pathlib import Path
from typing import Any

from play2learn_core.storage import atomic_write_json, unique_dir


class BaseFavoritesManager:
    """Gestore degli esperimenti salvati tra i preferiti."""

    def __init__(self, favorites_dir: Path | str):
        self.favorites_dir = Path(favorites_dir)
        self.favorites_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.favorites_dir / "index.json"

    def read_index(self) -> list[dict[str, Any]]:
        if not self.index_file.exists():
            return []
        try:
            return json.loads(self.index_file.read_text(encoding="utf-8"))
        except Exception:
            return []

    def write_index(self, entries: list[dict[str, Any]]) -> None:
        atomic_write_json(self.index_file, entries)

    def salva_preferito(
        self,
        source_dir: Path,
        nome: str,
        note: str = "",
        tag: str = "didattico",
    ) -> dict[str, Any]:
        """Salva una run nei preferiti copiando i file necessari e aggiornando index.json."""
        entries = self.read_index()
        new_id = len(entries) + 1
        safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in nome)[:40]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dest_dir_name = f"fav_{new_id:03d}_{safe_name}_{timestamp}"
        dest_dir = unique_dir(self.favorites_dir, dest_dir_name)
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Copia i file salienti
        for item in source_dir.iterdir():
            if item.is_file():
                try:
                    shutil.copy2(item, dest_dir / item.name)
                except Exception:
                    pass
            elif item.is_dir() and item.name in ("videos", "checkpoints"):
                try:
                    shutil.copytree(item, dest_dir / item.name, dirs_exist_ok=True)
                except Exception:
                    pass

        # Lettura metadati esistenti per arricchire l'entry
        meta = {}
        meta_file = source_dir / "run_metadata.json"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        fav_entry = {
            "id": new_id,
            "dir_name": dest_dir.name,
            "nome": nome,
            "timestamp": timestamp,
            "note": note,
            "tag": tag,
            "mission": meta.get("mission", {}).get("title", "Missione"),
            "success_rate": meta.get("eval_metrics", {}).get("eval_success_rate", 0.0),
            "mean_reward": meta.get("eval_metrics", {}).get("mean_reward", 0.0),
        }

        entries.append(fav_entry)
        self.write_index(entries)
        return fav_entry

    def carica_preferito(self, fav_id_or_name: int | str) -> dict[str, Any]:
        """Risolve un preferito e restituisce il percorso della sua cartella e i metadati."""
        entries = self.read_index()
        if not entries:
            raise ValueError("[Play2Learn] Nessun esperimento preferito salvato.")

        selected = None
        if isinstance(fav_id_or_name, int) or str(fav_id_or_name).isdigit():
            fid = int(fav_id_or_name)
            for e in entries:
                if e.get("id") == fid:
                    selected = e
                    break
        else:
            name_lower = str(fav_id_or_name).lower()
            for e in entries:
                if name_lower in str(e.get("nome", "")).lower() or name_lower in str(e.get("dir_name", "")).lower():
                    selected = e
                    break

        if not selected:
            raise ValueError(f"[Play2Learn] Nessun preferito trovato per: {fav_id_or_name}")

        dest_dir = self.favorites_dir / selected["dir_name"]
        if not dest_dir.exists():
            # Fallback ricerca per prefisso id
            for d in self.favorites_dir.iterdir():
                if d.is_dir() and d.name.startswith(f"fav_{selected.get('id', 0):03d}"):
                    dest_dir = d
                    break

        meta = {}
        meta_file = dest_dir / "run_metadata.json"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        return {
            "entry": selected,
            "dir_path": dest_dir,
            "metadata": meta,
        }
