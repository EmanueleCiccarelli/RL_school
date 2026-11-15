"""
Modulo per la gestione della sezione 'Preferiti' / 'Migliori Run'.
Permette agli studenti di salvare, organizzare, commentare e ricaricare le run
di maggior successo per analizzare e confrontare i risultati ottenuti.
"""

from __future__ import annotations

import json
import shutil
import datetime
from pathlib import Path
from typing import Any

from play2learn_lunarlander.storage import (
    get_default_runs_dir,
    atomic_write_json,
    _unique_dir,
)


def _get_favorites_dir() -> Path:
    fav_dir = get_default_runs_dir() / "favorites"
    fav_dir.mkdir(parents=True, exist_ok=True)
    return fav_dir


def _get_index_file() -> Path:
    fav_dir = _get_favorites_dir()
    index_file = fav_dir / "index.json"
    if not index_file.exists():
        atomic_write_json(index_file, [])
    return index_file


def _load_favorites_index() -> list[dict[str, Any]]:
    index_file = _get_index_file()
    try:
        return json.loads(index_file.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_favorites_index(data: list[dict[str, Any]]) -> None:
    index_file = _get_index_file()
    atomic_write_json(index_file, data)


def salva_nei_preferiti(
    run_or_result: Any,
    nome_personalizzato: str | None = None,
    note: str = "",
    tag: str = "Mio Preferito",
) -> dict[str, Any]:
    """
    Salva una run o un video nell'archivio dei Preferiti Personali dello studente.
    Copia in modo sicuro modello, grafici, video e codice in 'runs/favorites/'.

    Parametri:
    - run_or_result: Risultato restituito da `run_lab(...)`, percorso cartella run, o percorso a un file .mp4.
    - nome_personalizzato: Nome descrittivo (es. "Allunaggio_Stabile_v2").
    - note: Breve commento sulle scelte matematiche effettuate.
    - tag: Categoria (default: "Mio Preferito").
    """
    # Se è un file video .mp4, delega a salva_video_preferito
    if isinstance(run_or_result, (str, Path)) and str(run_or_result).lower().endswith(".mp4"):
        return salva_video_preferito(run_or_result, nome=nome_personalizzato, note=note, tag=tag)

    fav_dir = _get_favorites_dir()

    # 1. Risoluzione della cartella sorgente della run
    source_dir: Path | None = None
    if isinstance(run_or_result, dict):
        if "checkpoint_dir" in run_or_result:
            source_dir = Path(run_or_result["checkpoint_dir"])
        elif "path" in run_or_result:
            source_dir = Path(run_or_result["path"])
        elif "run_dir" in run_or_result:
            source_dir = Path(run_or_result["run_dir"])
    elif isinstance(run_or_result, (str, Path)):
        source_dir = Path(run_or_result)

    if not source_dir or not source_dir.exists():
        raise ValueError(f"[Play2Learn] Impossibile trovare la cartella della run: {run_or_result}")

    # 2. Lettura metadati esistenti
    meta_path = source_dir / "run_metadata.json"
    if meta_path.exists():
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {
                "run_id": source_dir.name,
                "mission": {"title": "Missione sconosciuta", "key": "luna_standard"},
                "eval_metrics": {},
            }
    else:
        metadata = {
            "run_id": source_dir.name,
            "mission": {"title": "Missione sconosciuta", "key": "luna_standard"},
            "eval_metrics": {},
        }

    # 3. Creazione destinazione nel folder favorites (anti-collisione)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_title = nome_personalizzato or metadata.get("mission", {}).get("key", "modello")
    clean_title = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in clean_title)
    dest_name = f"fav_{clean_title}_{timestamp}"
    dest_dir = _unique_dir(fav_dir, dest_name)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 4. Copia dei file essenziali (modello, grafici, codice, video/gif)
    for item in source_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, dest_dir / item.name)
        elif item.is_dir() and item.name == "videos":
            shutil.copytree(item, dest_dir / "videos", dirs_exist_ok=True)

    # 5. Aggiornamento metadati preferito
    metadata["favorite_info"] = {
        "display_name": nome_personalizzato or dest_dir.name,
        "saved_at": datetime.datetime.now().isoformat(),
        "tag": tag,
        "notes": note,
        "favorite_dir": str(dest_dir),
    }
    atomic_write_json(dest_dir / "run_metadata.json", metadata)

    # 6. Aggiornamento indice globale preferiti
    index_data = _load_favorites_index()
    entry = {
        "id": dest_dir.name,
        "type": "run",
        "name": nome_personalizzato or dest_dir.name,
        "tag": tag,
        "notes": note,
        "saved_at": datetime.datetime.now().strftime("%d/%m/%Y %H:%M"),
        "mission_title": metadata.get("mission", {}).get("title", "Lunar Lander"),
        "success_rate": metadata.get("eval_metrics", {}).get("eval_success_rate", 0.0),
        "mean_fuel": metadata.get("eval_metrics", {}).get("mean_fuel", 0.0),
        "mean_length": metadata.get("eval_metrics", {}).get("mean_length", 0.0),
        "dir_path": str(dest_dir),
    }
    index_data.insert(0, entry)
    _save_favorites_index(index_data)

    print("\n" + "=" * 70)
    print(f"  ⭐ MODELLO SALVATO NEI PREFERITI PERSONALI!")
    print(f"  🏷️  Nome: {entry['name']}")
    print(f"  🏆 Allunaggi Riusciti: {entry['success_rate']:.0f}% | Carburante Medio: {entry['mean_fuel']:.1f}")
    if note:
        print(f"  📝 Note: {note}")
    print(f"  📁 Cartella: runs/favorites/{dest_dir.name}")
    print("=" * 70 + "\n")

    return entry


def salva_video_preferito(
    video_ref: Any,
    nome: str | None = None,
    note: str = "",
    tag: str = "Video Preferito",
) -> dict[str, Any]:
    """
    Salva un singolo video tra i preferiti personali dell'utente.

    Parametri:
    - video_ref: percorso a un file .mp4, oppure oggetto LabResult (prende il primo video di successo).
    - nome: titolo descrittivo per il video (es. 'Mio Allunaggio Spettacolare').
    - note: annotazione didattica dello studente.
    - tag: categoria/etichetta (default: 'Video Preferito').
    """
    fav_dir = _get_favorites_dir()
    video_store = fav_dir / "videos"
    video_store.mkdir(parents=True, exist_ok=True)

    v_path: Path | None = None
    if isinstance(video_ref, (str, Path)):
        p = Path(video_ref)
        if p.exists() and p.suffix.lower() == ".mp4":
            v_path = p
    elif isinstance(video_ref, dict):
        succ_vids = video_ref.get("success_videos", []) or video_ref.get("recorded_videos", [])
        if succ_vids and Path(succ_vids[0]).exists():
            v_path = Path(succ_vids[0])

    if not v_path or not v_path.exists():
        raise ValueError(f"[Play2Learn] Video non trovato o non valido: {video_ref}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_name = nome or v_path.stem
    clean_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in clean_name)
    dest_video_name = f"fav_vid_{clean_name}_{timestamp}.mp4"
    dest_video_path = video_store / dest_video_name
    shutil.copy2(v_path, dest_video_path)

    # Aggiorna l'indice globale dei preferiti in modo atomico
    index_data = _load_favorites_index()
    entry = {
        "id": dest_video_name,
        "type": "video",
        "name": nome or clean_name,
        "tag": tag,
        "notes": note,
        "saved_at": datetime.datetime.now().strftime("%d/%m/%Y %H:%M"),
        "mission_title": "Video Allunaggio",
        "video_path": str(dest_video_path),
        "dir_path": str(video_store),
    }
    index_data.insert(0, entry)
    _save_favorites_index(index_data)

    print("\n" + "=" * 70)
    print(f"  🎬 VIDEO SALVATO NEI PREFERITI PERSONALI!")
    print(f"  🏷️  Nome: {entry['name']}")
    if note:
        print(f"  📝 Note: {note}")
    print(f"  📁 Salvato in: runs/favorites/videos/{dest_video_name}")
    print("=" * 70 + "\n")

    return entry


def elenca_preferiti() -> list[dict[str, Any]]:
    """Restituisce la lista di tutte le run salvate nei Preferiti."""
    return _load_favorites_index()


def mostra_preferiti() -> None:
    """
    Visualizza la galleria dei Preferiti personali dell'utente (modelli e video salvati esplicitamente).
    """
    entries = _load_favorites_index()
    if not entries:
        print("\n[play2learn] Nessun elemento salvato nei preferiti personali.")
        print("💡 Usa 'risultati.salva_preferito(\"Mio Nome\")' oppure 'salva_video_preferito(\"video.mp4\")' per aggiungerne uno!\n")
        return

    try:
        from IPython.display import HTML, display
        rows = []
        for e in entries:
            is_vid = e.get("type") == "video"
            icon = "🎬" if is_vid else "🏆"
            if is_vid:
                succ = "<span style='color:#0d47a1;'>🎬 Video</span>"
                fuel_str = "-"
            else:
                s_val = e.get("success_rate", 0)
                badge_color = "#2e7d32" if s_val >= 80 else "#ef6c00"
                succ = f"<span style='font-weight:bold; color:{badge_color};'>{s_val:.0f}%</span>"
                fuel_str = f"{e.get('mean_fuel', 0):.1f} un."

            rows.append(
                f"<tr>"
                f"<td style='padding:8px 12px; font-weight:bold;'>{icon} {e.get('name')}</td>"
                f"<td style='padding:8px 12px;'><span style='background:#e3f2fd; color:#0d47a1; padding:2px 8px; border-radius:12px; font-size:12px;'>{e.get('tag', 'Preferito')}</span></td>"
                f"<td style='padding:8px 12px;'>{e.get('mission_title', '')}</td>"
                f"<td style='padding:8px 12px;'>{succ}</td>"
                f"<td style='padding:8px 12px;'>{fuel_str}</td>"
                f"<td style='padding:8px 12px;'>{e.get('saved_at', '')}</td>"
                f"<td style='padding:8px 12px; font-style:italic; color:#555;'>{e.get('notes', '')}</td>"
                f"</tr>"
            )

        html = f"""
        <div style="font-family:sans-serif; margin:16px 0;">
            <h3 style="color:#1b5e20; margin-bottom:8px;">⭐ I Tuoi Preferiti (Modelli e Video Salvati dall'Utente)</h3>
            <table style="border-collapse:collapse; width:100%; border:1px solid #ccc; text-align:left; font-size:13.5px;">
                <thead style="background:#fff9c4; color:#5d4037;">
                    <tr>
                        <th style="padding:10px 12px;">Nome Preferito</th>
                        <th style="padding:10px 12px;">Tipo / Tag</th>
                        <th style="padding:10px 12px;">Missione</th>
                        <th style="padding:10px 12px;">Esito / Tipo</th>
                        <th style="padding:10px 12px;">Carburante</th>
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
        display(HTML(html))
    except Exception:
        print("\n" + "=" * 80)
        print("  ⭐ I TUOI PREFERITI (MODELLI E VIDEO SALVATI DALL'UTENTE)")
        print("=" * 80)
        print(f"{'NOME':<25} | {'MISSIONE':<20} | {'TIPO/ESITO':<12} | {'CARBURANTE':<12} | {'DATA':<15}")
        print("-" * 80)
        for e in entries:
            is_vid = e.get("type") == "video"
            succ = "Video" if is_vid else f"{e.get('success_rate', 0):.0f}%"
            fuel = "-" if is_vid else f"{e.get('mean_fuel', 0):.1f}"
            print(f"{e.get('name', ''):<25} | {e.get('mission_title', ''):<20} | {succ:<12} | {fuel:<12} | {e.get('saved_at', ''):<15}")
        print("=" * 80 + "\n")


def carica_preferito(identifier: str | int = 0) -> tuple[PPO, dict[str, Any]]:
    """
    Ricarica il modello PPO e i relativi metadati da un elemento salvato nei Preferiti.
    Può essere specificato l'indice numerico (0 = il più recente) o il nome.
    Supporta percorsi portabili tra Windows e Linux/Colab.
    """
    entries = _load_favorites_index()
    if not entries:
        raise ValueError("[Play2Learn] La lista dei preferiti è vuota.")

    selected_entry: dict[str, Any] | None = None
    if isinstance(identifier, int):
        if 0 <= identifier < len(entries):
            selected_entry = entries[identifier]
        else:
            raise IndexError(f"Indice preferito non valido: {identifier} (disponibili 0..{len(entries)-1})")
    else:
        for e in entries:
            if e.get("id") == identifier or e.get("name") == identifier:
                selected_entry = e
                break

    if not selected_entry:
        raise ValueError(f"Nessun preferito trovato con identificativo: {identifier}")

    if selected_entry.get("type") == "video":
        raise ValueError(
            f"[Play2Learn] L'elemento '{selected_entry['name']}' è un video salvato nei preferiti, non una rete neurale.\n"
            f"File video: {selected_entry.get('video_path')}"
        )

    # Risoluzione portabile del percorso cartella preferito
    dest_dir = Path(selected_entry.get("dir_path", ""))
    if not dest_dir.exists() and selected_entry.get("id"):
        dest_dir = _get_favorites_dir() / str(selected_entry["id"])

    # Cerca il file modello
    model_path = dest_dir / "best_model.zip"
    if not model_path.exists():
        model_path = dest_dir / "ppo_lunar_lander.zip"
    if not model_path.exists():
        zip_files = list(dest_dir.glob("*.zip"))
        if zip_files:
            model_path = zip_files[0]
        else:
            raise FileNotFoundError(f"Nessun checkpoint modello .zip trovato in: {dest_dir}")

    try:
        from stable_baselines3 import PPO
    except ImportError as e:
        raise ImportError(
            "❌ stable-baselines3 non trovato! Assicurati di aver eseguito la Cella di installazione:\n"
            "!pip install -q stable-baselines3"
        ) from e

    model = PPO.load(str(model_path))
    meta_path = dest_dir / "run_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else selected_entry
    return model, meta
