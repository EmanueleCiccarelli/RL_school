"""
Modulo Gestione Esperimenti Preferiti per Hill Climb Racing (Play2Learn).
Archiviazione, indicizzazione, sanitizzazione HTML e caricamento portabile del modello PPO.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO

from play2learn_core.favorites import BaseFavoritesManager
from play2learn_hillclimb.storage import get_default_runs_dir


class LoadedFavorite(tuple):
    """
    Oggetto ibrido tuple/dict che permette sia l'unpacking didattico `model, meta = carica_preferito(0)`
    sia l'accesso a campi `.model`, `.metadata`, `.dir_path` o `fav['metadata']`.
    """

    def __new__(cls, model: Any, metadata: dict[str, Any], dir_path: Path, entry: dict[str, Any]):
        return super().__new__(cls, (model, metadata))

    def __init__(self, model: Any, metadata: dict[str, Any], dir_path: Path, entry: dict[str, Any]):
        self.model = model
        self.metadata = metadata
        self.dir_path = dir_path
        self.entry = entry

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, int):
            return super().__getitem__(item)
        if item == "model":
            return self.model
        if item in ("metadata", "meta"):
            return self.metadata
        if item in ("dir_path", "path"):
            return self.dir_path
        if item == "entry":
            return self.entry
        if item in self.metadata:
            return self.metadata[item]
        if item in self.entry:
            return self.entry[item]
        raise KeyError(item)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


def _get_manager() -> BaseFavoritesManager:
    fav_dir = get_default_runs_dir() / "favorites"
    return BaseFavoritesManager(fav_dir)


def salva_nei_preferiti(
    run_or_dir: Any,
    nome: str,
    note: str = "",
    tag: str = "didattico",
) -> dict[str, Any]:
    """Salva la run nei preferiti personali."""
    source_dir = None
    if isinstance(run_or_dir, dict):
        for k in ("checkpoint_dir", "run_dir", "path", "dir_path"):
            if k in run_or_dir:
                source_dir = Path(run_or_dir[k])
                break
    elif isinstance(run_or_dir, (str, Path)):
        source_dir = Path(run_or_dir)

    if not source_dir or not source_dir.exists():
        raise ValueError(f"[Play2Learn] Directory di run non trovata: {run_or_dir}")

    mgr = _get_manager()
    entry = mgr.salva_preferito(source_dir=source_dir, nome=nome, note=note, tag=tag)
    print(f"⭐ Esperimento salvato nei preferiti: #{entry['id']} - {entry['nome']}")
    return entry


salva_preferito = salva_nei_preferiti


def carica_preferito(fav_id_or_name: int | str = 0) -> LoadedFavorite:
    """
    Ricarica il modello PPO e i relativi metadati da un elemento salvato nei Preferiti.
    Supporta unpacking: `model, meta = carica_preferito(0)`.
    """
    mgr = _get_manager()
    fav_info = mgr.carica_preferito(fav_id_or_name)
    dest_dir = Path(fav_info["dir_path"])

    # Risoluzione checkpoint del modello
    model_path = dest_dir / "best_model.zip"
    if not model_path.exists():
        model_path = dest_dir / "final_model.zip"
    if not model_path.exists():
        zips = list(dest_dir.glob("*.zip"))
        if zips:
            model_path = zips[0]
        else:
            raise FileNotFoundError(f"[Play2Learn] Nessun modello .zip trovato in: {dest_dir}")

    model = PPO.load(str(model_path), device="cpu")
    metadata = fav_info.get("metadata", {})
    entry = fav_info.get("entry", {})

    return LoadedFavorite(model=model, metadata=metadata, dir_path=dest_dir, entry=entry)


def mostra_preferiti() -> list[dict[str, Any]]:
    """Stampa e restituisce l'elenco di tutti gli esperimenti preferiti salvati."""
    mgr = _get_manager()
    entries = mgr.read_index()
    if not entries:
        print("[play2learn] Nessun esperimento preferito salvato.")
        return []

    try:
        from IPython.display import HTML, display
        rows = []
        for e in entries:
            safe_nome = html.escape(str(e.get("nome", "")))
            safe_tag = html.escape(str(e.get("tag", "Preferito")))
            safe_mission = html.escape(str(e.get("mission", "Campagna")))
            safe_notes = html.escape(str(e.get("note", "")))
            dist_val = float(e.get("max_distance", 0.0))
            succ_val = float(e.get("success_rate", 0.0))
            badge_color = "#10b981" if succ_val >= 80 or dist_val >= 800 else "#f59e0b"

            rows.append(
                f"<tr>"
                f"<td style='padding:8px 12px; font-weight:bold;'>🏎️ {safe_nome}</td>"
                f"<td style='padding:8px 12px;'><span style='background:#e0f2fe; color:#0369a1; padding:2px 8px; border-radius:12px; font-size:12px;'>{safe_tag}</span></td>"
                f"<td style='padding:8px 12px;'>{safe_mission}</td>"
                f"<td style='padding:8px 12px; font-weight:bold;'>{dist_val:.1f} m</td>"
                f"<td style='padding:8px 12px;'><span style='font-weight:bold; color:{badge_color};'>{succ_val:.0f}%</span></td>"
                f"<td style='padding:8px 12px; color:#64748b; font-size:12px;'>{e.get('timestamp', '')}</td>"
                f"<td style='padding:8px 12px; font-size:12px;'>{safe_notes}</td>"
                f"</tr>"
            )

        html_table = f"""
        <div style="font-family:'Segoe UI',system-ui,sans-serif; margin:16px 0;">
            <h3 style="color:#0284c7; margin-bottom:8px;">⭐ I Tuoi Esperimenti Preferiti — Hill Climb Racing</h3>
            <table style="width:100%; border-collapse:collapse; background:#ffffff; border-radius:8px; overflow:hidden; box-shadow:0 1px 3px rgba(0,0,0,0.1);">
                <thead>
                    <tr style="background:#0f172a; color:#f8fafc; text-align:left; font-size:13px;">
                        <th style="padding:10px 12px;">Nome</th>
                        <th style="padding:10px 12px;">Tag</th>
                        <th style="padding:10px 12px;">Missione</th>
                        <th style="padding:10px 12px;">Distanza</th>
                        <th style="padding:10px 12px;">Successo</th>
                        <th style="padding:10px 12px;">Data</th>
                        <th style="padding:10px 12px;">Note</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(rows)}
                </tbody>
            </table>
        </div>
        """
        display(HTML(html_table))
    except Exception:
        print("=" * 80)
        print(" ⭐ GALLERIA PREFERITI - HILL CLIMB RACING")
        print("=" * 80)
        for e in entries:
            dist = float(e.get("max_distance", 0.0))
            succ = float(e.get("success_rate", 0.0))
            print(f"#{e['id']:02d} | {e['nome'][:20]:20} | Distanza: {dist:.1f} m | Successo: {succ:.0f}% | Data: {e['timestamp']}")
            if e.get("note"):
                print(f"     Note: {e['note']}")
        print("=" * 80)

    return entries
