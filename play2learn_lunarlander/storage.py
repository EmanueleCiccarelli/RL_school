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
    (runs_dir / "successi").mkdir(parents=True, exist_ok=True)
    return runs_dir


def get_default_success_dir() -> Path:
    """Restituisce il percorso della cartella 'runs/successi/' per le run con esito >= 80%."""
    succ_dir = get_default_runs_dir() / "successi"
    succ_dir.mkdir(parents=True, exist_ok=True)
    idx_file = succ_dir / "index.json"
    if not idx_file.exists():
        idx_file.write_text("[]\n", encoding="utf-8")
    return succ_dir


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
        if not sub.is_dir() or sub.name in ("favorites", "successi"):
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


def archivia_esperimento_successo(
    run_dir: Path | str,
    nome_run: str = "successo",
    succ_rate: float = 80.0,
    eval_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Archivia automaticamente un esperimento con tasso di successo >= 80%
    nella cartella dedicata 'runs/successi/', separata dai preferiti dell'utente.
    Copia checkpoint, video, grafici, codice sorgente e genera la Scheda Tecnica PDF.
    """
    import shutil
    run_dir = Path(run_dir)
    eval_stats = eval_stats or {}
    succ_base = get_default_success_dir()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in nome_run)
    dest_name = f"successo_{clean_name}_{timestamp}"
    dest_dir = succ_base / dest_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Copia i file generati dalla run
    for item in run_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, dest_dir / item.name)
        elif item.is_dir() and item.name == "videos":
            shutil.copytree(item, dest_dir / "videos", dirs_exist_ok=True)

    # Genera la Scheda Tecnica nella cartella di destinazione
    from play2learn_lunarlander.reporting import genera_report_esperimento
    rep_files = genera_report_esperimento(
        run_or_fav=dest_dir,
        output_dir=dest_dir,
    )

    # Aggiorna l'indice dei successi
    idx_file = succ_base / "index.json"
    index_data = []
    if idx_file.exists():
        try:
            index_data = json.loads(idx_file.read_text(encoding="utf-8"))
        except Exception:
            index_data = []

    entry = {
        "id": dest_name,
        "name": nome_run,
        "success_rate": round(succ_rate, 1),
        "mean_fuel": round(float(eval_stats.get("mean_fuel", 0.0)), 1),
        "mean_length": round(float(eval_stats.get("mean_length", 0.0)), 1),
        "archived_at": datetime.datetime.now().strftime("%d/%m/%Y %H:%M"),
        "dir_path": str(dest_dir),
        "pdf_report": str(rep_files.get("pdf_path", "")),
        "video_count": len(list((dest_dir / "videos").glob("*.mp4"))) if (dest_dir / "videos").exists() else 0,
    }
    index_data.insert(0, entry)
    idx_file.write_text(json.dumps(index_data, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "dest_dir": dest_dir,
        "rep_files": rep_files,
        "entry": entry,
    }


def elenca_successi() -> list[dict[str, Any]]:
    """Restituisce l'elenco di tutti gli esperimenti con successo >= 80% archiviati in 'runs/successi/'."""
    idx_file = get_default_success_dir() / "index.json"
    if idx_file.exists():
        try:
            return json.loads(idx_file.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def mostra_successi() -> None:
    """Visualizza la tabella riassuntiva di tutti gli esperimenti con successo >= 80% archiviati."""
    entries = elenca_successi()
    if not entries:
        print("\n[play2learn] Nessun esperimento >= 80% attualmente archiviato in 'runs/successi/'.\n")
        return

    try:
        from IPython.display import HTML, display
        rows = []
        for e in entries:
            rows.append(
                f"<tr>"
                f"<td style='padding:8px 12px; font-weight:bold;'>{e.get('name')}</td>"
                f"<td style='padding:8px 12px; font-weight:bold; color:#2e7d32;'>{e.get('success_rate', 0):.1f}%</td>"
                f"<td style='padding:8px 12px;'>{e.get('mean_fuel', 0):.1f} un.</td>"
                f"<td style='padding:8px 12px;'>{e.get('archived_at', '')}</td>"
                f"<td style='padding:8px 12px;'><code>runs/successi/{e.get('id')}</code></td>"
                f"</tr>"
            )
        html = f"""
        <div style="font-family:sans-serif; margin:16px 0;">
            <h3 style="color:#1b5e20; margin-bottom:8px;">🎯 Archivio Esperimenti Riusciti (&ge; 80%)</h3>
            <table style="border-collapse:collapse; width:100%; border:1px solid #ccc; text-align:left; font-size:13.5px;">
                <thead style="background:#e8f5e9; color:#1b5e20;">
                    <tr>
                        <th style="padding:10px 12px;">Nome Esperimento</th>
                        <th style="padding:10px 12px;">Successo</th>
                        <th style="padding:10px 12px;">Carburante Medio</th>
                        <th style="padding:10px 12px;">Data</th>
                        <th style="padding:10px 12px;">Percorso</th>
                    </tr>
                </thead>
                <tbody>{''.join(rows)}</tbody>
            </table>
        </div>
        """
        display(HTML(html))
    except Exception:
        print("\n" + "=" * 75)
        print("  🎯 ARCHIVIO ESPERIMENTI RIUSCITI (>= 80%)")
        print("=" * 75)
        print(f"{'NOME':<25} | {'SUCCESSO':<10} | {'CARBURANTE':<12} | {'DATA':<15}")
        print("-" * 75)
        for e in entries:
            print(f"{e.get('name', ''):<25} | {e.get('success_rate', 0):.1f}%    | {e.get('mean_fuel', 0):<12.1f} | {e.get('archived_at', ''):<15}")
        print("=" * 75 + "\n")


def find_run_by_timestamp(
    timestamp_or_id: str | Path | None = None,
    runs_dir: Path | str | None = None,
) -> Path | None:
    """
    Individua la cartella di una run a partire dal suo timestamp (es. '20261115_210530'),
    da una frazione di esso o dal nome della cartella (anche di più esperimenti fa).
    Se il parametro è vuoto o 'latest', restituisce la run più recente.
    """
    base = Path(runs_dir).resolve() if runs_dir else get_default_runs_dir()
    if not base.exists():
        return None

    # Se è già un Path valido su disco
    if timestamp_or_id is not None:
        p = Path(timestamp_or_id)
        if p.exists() and p.is_dir() and (p / "run_metadata.json").exists():
            return p.resolve()
        if (base / p).exists() and (base / p).is_dir() and ((base / p) / "run_metadata.json").exists():
            return (base / p).resolve()
        if (base / "successi" / p).exists() and ((base / "successi" / p) / "run_metadata.json").exists():
            return (base / "successi" / p).resolve()

    query = str(timestamp_or_id or "").strip()
    is_latest = query.lower() in ("", "latest", "ultima", "ultimo", "now")

    # Raccoglie tutte le directory candidate da runs/, runs/successi/ e runs/favorites/
    all_dirs: list[Path] = []

    # 1. runs standard dirette
    for sub in base.iterdir():
        if sub.is_dir() and sub.name not in ("favorites", "successi"):
            if (sub / "run_metadata.json").exists():
                all_dirs.append(sub)

    # 2. runs in successi/
    succ_dir = base / "successi"
    if succ_dir.exists():
        for sub in succ_dir.iterdir():
            if sub.is_dir() and (sub / "run_metadata.json").exists():
                all_dirs.append(sub)

    # 3. runs in favorites/
    fav_dir = base / "favorites"
    if fav_dir.exists():
        for sub in fav_dir.iterdir():
            if sub.is_dir() and (sub / "run_metadata.json").exists():
                all_dirs.append(sub)

    if not all_dirs:
        return None

    # Ordina per data di modifica più recente
    all_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)

    if is_latest:
        return all_dirs[0]

    # Pulizia query: rimozione caratteri non alfanumerici per confronto flessibile
    clean_query = query.lower().replace("-", "").replace(":", "").replace(" ", "_")

    # Ricerca esatta o parziale
    for d in all_dirs:
        d_name = d.name.lower()
        if clean_query in d_name or query.lower() in d_name:
            return d

        # Controlla nei metadati se il timestamp o run_id corrispondono
        meta_file = d / "run_metadata.json"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                ts_meta = meta.get("timestamp", "").lower().replace("-", "").replace(":", "")
                run_id = meta.get("run_id", "").lower()
                if clean_query in ts_meta or clean_query in run_id or query.lower() in run_id:
                    return d
            except Exception:
                continue

    return None


def elenca_cronologia_run(runs_dir: Path | str | None = None) -> list[dict[str, Any]]:
    """
    Restituisce l'elenco cronologico di tutte le run salvate (standard, successi e preferiti),
    estraendo per ciascuna il timestamp esatto, il nome, l'esito percentuale e il consumo.
    """
    import re
    base = Path(runs_dir).resolve() if runs_dir else get_default_runs_dir()
    if not base.exists():
        return []

    results: list[dict[str, Any]] = []

    def _process_dir(d: Path, category: str) -> None:
        meta_file = d / "run_metadata.json"
        if not meta_file.exists():
            return
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            return

        ts_match = re.search(r"(\d{8}_\d{6})", d.name)
        if ts_match:
            ts_str = ts_match.group(1)
        else:
            raw_ts = meta.get("timestamp", "")
            ts_str = raw_ts.replace("-", "")[:8] + "_" + raw_ts.replace(":", "")[9:15] if len(raw_ts) >= 19 else d.name

        metrics = meta.get("eval_metrics", {})
        mission = meta.get("mission", {})
        succ_rate = float(metrics.get("eval_success_rate", 0.0))

        results.append({
            "timestamp": ts_str,
            "run_id": d.name,
            "name": meta.get("tag") or mission.get("title") or d.name,
            "mission": mission.get("key", ""),
            "success_rate": succ_rate,
            "mean_fuel": float(metrics.get("mean_fuel", 0.0)),
            "mean_length": float(metrics.get("mean_length", 0.0)),
            "category": category,
            "dir_path": str(d),
            "mtime": d.stat().st_mtime,
            "has_model": (d / "best_model.zip").exists() or (d / "ppo_lunar_lander.zip").exists(),
            "video_count": len(list((d / "videos").glob("*.mp4"))) if (d / "videos").exists() else 0,
        })

    # Standard runs
    for sub in base.iterdir():
        if sub.is_dir() and sub.name not in ("favorites", "successi"):
            _process_dir(sub, "Standard")

    # Successi
    succ_dir = base / "successi"
    if succ_dir.exists():
        for sub in succ_dir.iterdir():
            if sub.is_dir():
                _process_dir(sub, "Successo (>=80%)")

    # Preferiti
    fav_dir = base / "favorites"
    if fav_dir.exists():
        for sub in fav_dir.iterdir():
            if sub.is_dir():
                _process_dir(sub, "Preferito")

    # Ordina per mtime decrescente (le più recenti in cima)
    results.sort(key=lambda r: r["mtime"], reverse=True)
    return results


def mostra_cronologia_run(runs_dir: Path | str | None = None) -> None:
    """
    Visualizza la tabella della cronologia di tutte le run passate,
    evidenziando chiaramente il timestamp da inserire nella cella di reportistica.
    """
    entries = elenca_cronologia_run(runs_dir)
    if not entries:
        print("\n[play2learn] Nessuna run attualmente presente nella cartella 'runs/'. Esegui prima un addestramento!\n")
        return

    try:
        from IPython.display import HTML, display
        rows = []
        for e in entries:
            succ = e['success_rate']
            badge_color = "#2e7d32" if succ >= 80.0 else ("#f57c00" if succ >= 50.0 else "#c62828")
            bg_badge = "#e8f5e9" if succ >= 80.0 else ("#fff3e0" if succ >= 50.0 else "#ffebee")
            has_mod_str = "✅ Sì" if e["has_model"] else "❌ No"

            rows.append(
                f"<tr>"
                f"<td style='padding:8px 12px; font-family:monospace; font-weight:bold; color:#1a237e; background:#f5f5f5;'>{e['timestamp']}</td>"
                f"<td style='padding:8px 12px;'><b>{e['name']}</b> <span style='font-size:11px; color:#666;'>({e['category']})</span></td>"
                f"<td style='padding:8px 12px; text-align:center;'><span style='background:{bg_badge}; color:{badge_color}; padding:3px 8px; border-radius:12px; font-weight:bold;'>{succ:.1f}%</span></td>"
                f"<td style='padding:8px 12px; text-align:center;'>{e['mean_fuel']:.1f} un.</td>"
                f"<td style='padding:8px 12px; text-align:center;'>{e['video_count']} video</td>"
                f"<td style='padding:8px 12px; text-align:center;'>{has_mod_str}</td>"
                f"<td style='padding:8px 12px; font-size:11.5px; color:#666;'><code>{e['run_id']}</code></td>"
                f"</tr>"
            )

        html = f"""
        <div style="font-family:'Segoe UI', sans-serif; margin:16px 0;">
            <h3 style="color:#1a237e; margin-bottom:6px;">🕒 Cronologia Esperimenti e Timestamp Disponibili</h3>
            <p style="color:#555; margin-bottom:12px; font-size:13px;">
                💡 <b>Come usare:</b> copia uno dei <b>Timestamp</b> della prima colonna (o scrivi <code>latest</code>) nella cella di generazione report sottostante.
            </p>
            <table style="border-collapse:collapse; width:100%; border:1px solid #dcdcdc; text-align:left; font-size:13px;">
                <thead style="background:#e8eaf6; color:#1a237e;">
                    <tr>
                        <th style="padding:10px 12px;">Timestamp (da copiare)</th>
                        <th style="padding:10px 12px;">Esperimento / Categoria</th>
                        <th style="padding:10px 12px; text-align:center;">Allunaggio</th>
                        <th style="padding:10px 12px; text-align:center;">Carburante</th>
                        <th style="padding:10px 12px; text-align:center;">Video</th>
                        <th style="padding:10px 12px; text-align:center;">Modello salvato</th>
                        <th style="padding:10px 12px;">Cartella Run</th>
                    </tr>
                </thead>
                <tbody>{''.join(rows)}</tbody>
            </table>
        </div>
        """
        display(HTML(html))
    except Exception:
        print("\n" + "=" * 80)
        print("  🕒 CRONOLOGIA ESPERIMENTI PASSATI (TIMESTAMP DISPONIBILI)")
        print("=" * 80)
        print(f"{'TIMESTAMP':<18} | {'NOME RUN':<25} | {'SUCCESSO':<10} | {'CARBURANTE':<12} | {'MODELLO':<8}")
        print("-" * 80)
        for e in entries:
            mod_str = "Sì" if e["has_model"] else "No"
            print(f"{e['timestamp']:<18} | {e['name'][:24]:<25} | {e['success_rate']:<9.1f}% | {e['mean_fuel']:<12.1f} | {mod_str:<8}")
        print("=" * 80)
        print("💡 Copia il Timestamp desiderato nella cella di reportistica per rigenerare la scheda tecnica!\n")
