"""
Modulo per la gestione della sezione 'Preferiti' / 'Hall of Fame' per l'Esame di Maturità.
Permette agli studenti di salvare, organizzare, commentare e ricaricare le run
di maggior successo da mostrare durante il colloquio orale.
"""

from __future__ import annotations

import json
import shutil
import datetime
from pathlib import Path
from typing import Any

from play2learn_lunarlander.storage import get_default_runs_dir


def _get_favorites_dir() -> Path:
    fav_dir = get_default_runs_dir() / "favorites"
    fav_dir.mkdir(parents=True, exist_ok=True)
    return fav_dir


def _get_index_file() -> Path:
    fav_dir = _get_favorites_dir()
    index_file = fav_dir / "index.json"
    if not index_file.exists():
        index_file.write_text(json.dumps([], indent=2), encoding="utf-8")
    return index_file


def _load_favorites_index() -> list[dict[str, Any]]:
    index_file = _get_index_file()
    try:
        return json.loads(index_file.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_favorites_index(data: list[dict[str, Any]]) -> None:
    index_file = _get_index_file()
    index_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def salva_nei_preferiti(
    run_or_result: Any,
    nome_personalizzato: str | None = None,
    note: str = "",
    tag: str = "Top Run",
) -> dict[str, Any]:
    """
    Salva una run nell'archivio dei modelli preferiti per confrontarla e riutilizzarla.
    Copia in modo sicuro modello, grafici, video e codice in 'runs/favorites/'.

    Parametri:
    - run_or_result: Risultato restituito da `run_lab(...)` o percorso cartella.
    - nome_personalizzato: Nome descrittivo (es. "Allunaggio_Stabile_v2").
    - note: Breve commento sulle scelte matematiche effettuate.
    - tag: Categoria (default: "Top Run").
    """
    fav_dir = _get_favorites_dir()

    # 1. Risoluzione della cartella sorgente della run
    source_dir: Path | None = None
    if isinstance(run_or_result, dict):
        if "checkpoint_dir" in run_or_result:
            source_dir = Path(run_or_result["checkpoint_dir"])
        elif "path" in run_or_result:
            source_dir = Path(run_or_result["path"])
    elif isinstance(run_or_result, (str, Path)):
        source_dir = Path(run_or_result)

    if not source_dir or not source_dir.exists():
        raise ValueError(f"[Play2Learn] Impossibile trovare la cartella della run: {run_or_result}")

    # 2. Lettura metadati esistenti
    meta_path = source_dir / "run_metadata.json"
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        metadata = {
            "run_id": source_dir.name,
            "mission": {"title": "Missione sconosciuta", "key": "luna_standard"},
            "eval_metrics": {},
        }

    # 3. Creazione destinazione nel folder favorites
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_title = nome_personalizzato or metadata.get("mission", {}).get("key", "modello")
    clean_title = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in clean_title)
    dest_name = f"fav_{clean_title}_{timestamp}"
    dest_dir = fav_dir / dest_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 4. Copia dei file essenziali (modello, grafici, codice, video/gif)
    for item in source_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, dest_dir / item.name)
        elif item.is_dir() and item.name == "videos":
            shutil.copytree(item, dest_dir / "videos", dirs_exist_ok=True)

    # 5. Aggiornamento metadati preferito
    metadata["favorite_info"] = {
        "display_name": nome_personalizzato or dest_name,
        "saved_at": datetime.datetime.now().isoformat(),
        "tag": tag,
        "notes": note,
        "favorite_dir": str(dest_dir),
    }
    (dest_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    # 6. Aggiornamento indice globale preferiti
    index_data = _load_favorites_index()
    entry = {
        "id": dest_name,
        "name": nome_personalizzato or dest_name,
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
    print(f"  ⭐ MODELLO SALVATO NEI PREFERITI!")
    print(f"  🏷️  Nome: {entry['name']}")
    print(f"  🏆 Allunaggi Riusciti: {entry['success_rate']:.0f}% | Carburante Medio: {entry['mean_fuel']:.1f}")
    if note:
        print(f"  📝 Note: {note}")
    print(f"  📁 Cartella: {dest_dir.name}")
    print("=" * 70 + "\n")

    return entry


def elenca_preferiti() -> list[dict[str, Any]]:
    """Restituisce la lista di tutte le run salvate nei Preferiti."""
    return _load_favorites_index()


def mostra_preferiti() -> None:
    """
    Visualizza la 'Hall of Fame' dei modelli preferiti in una tabella formattata
    elegante adatta per l'output di Jupyter Notebook / Google Colab.
    """
    entries = _load_favorites_index()
    if not entries:
        print("\n[play2learn] Nessun modello salvato nei preferiti. Usa 'risultati.salva_preferito()' per aggiungerne uno!\n")
        return

    try:
        from IPython.display import HTML, display
        rows = []
        for e in entries:
            succ = f"{e.get('success_rate', 0):.0f}%"
            badge_color = "#2e7d32" if e.get("success_rate", 0) >= 80 else "#ef6c00"
            rows.append(
                f"<tr>"
                f"<td style='padding:8px 12px; font-weight:bold;'>{e.get('name')}</td>"
                f"<td style='padding:8px 12px;'><span style='background:#e3f2fd; color:#0d47a1; padding:2px 8px; border-radius:12px; font-size:12px;'>{e.get('tag', 'Maturità')}</span></td>"
                f"<td style='padding:8px 12px;'>{e.get('mission_title', '')}</td>"
                f"<td style='padding:8px 12px; font-weight:bold; color:{badge_color};'>{succ}</td>"
                f"<td style='padding:8px 12px;'>{e.get('mean_fuel', 0):.1f} un.</td>"
                f"<td style='padding:8px 12px;'>{e.get('saved_at', '')}</td>"
                f"<td style='padding:8px 12px; font-style:italic; color:#555;'>{e.get('notes', '')}</td>"
                f"</tr>"
            )

        html = f"""
        <div style="font-family:sans-serif; margin:16px 0;">
            <h3 style="color:#1b5e20; margin-bottom:8px;">🏆 Modelli Selezionati / Archivio Run Migliori</h3>
            <table style="border-collapse:collapse; width:100%; border:1px solid #ccc; text-align:left; font-size:13.5px;">
                <thead style="background:#f0f4c3; color:#33691e;">
                    <tr>
                        <th style="padding:10px 12px;">Nome Modello</th>
                        <th style="padding:10px 12px;">Tag</th>
                        <th style="padding:10px 12px;">Missione</th>
                        <th style="padding:10px 12px;">Successo</th>
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
        # Fallback a testo console formattato
        print("\n" + "=" * 80)
        print("  🏆 MODELLI SELEZIONATI / ARCHIVIO RUN MIGLIORI")
        print("=" * 80)
        print(f"{'NOME':<25} | {'MISSIONE':<20} | {'SUCCESSO':<10} | {'CARBURANTE':<12} | {'DATA':<15}")
        print("-" * 80)
        for e in entries:
            succ = f"{e.get('success_rate', 0):.0f}%"
            print(f"{e.get('name', ''):<25} | {e.get('mission_title', ''):<20} | {succ:<10} | {e.get('mean_fuel', 0):<12.1f} | {e.get('saved_at', ''):<15}")
        print("=" * 80 + "\n")


def carica_preferito(identifier: str | int = 0) -> tuple[PPO, dict[str, Any]]:
    """
    Ricarica il modello PPO e i relativi metadati da un elemento salvato nei Preferiti.
    Può essere specificato l'indice numerico (0 = il più recente) o il nome.
    Ideale per una dimostrazione dal vivo durante il colloquio orale!
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

    dest_dir = Path(selected_entry["dir_path"])
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

    print(f"✅ Modello '{selected_entry['name']}' caricato con successo da: {dest_dir.name}")
    return model, meta
