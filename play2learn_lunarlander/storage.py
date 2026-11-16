"""
Modulo per la gestione della persistenza, salvataggio atomico dei checkpoint dei modelli,
delle metriche, del codice della funzione di reward e dell'alberatura 'runs/'.
"""

from __future__ import annotations

import os
import re
import html
import json
import inspect
import datetime
import tempfile
from pathlib import Path
from typing import Callable, Any


def atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Scrittura atomica a prova di crash/interruzione tramite file temporaneo e rename os.replace()."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp_path), str(path))
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, data: Any) -> None:
    """Serializza dati in JSON e li scrive su disco in modo atomico."""
    atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False, default=str))


def get_default_runs_dir() -> Path:
    """
    Restituisce il percorso della cartella 'runs/' specifica per Lunar Lander
    all'interno di 'play2learn_lunarlander/runs/'.
    """
    env_override = os.environ.get("PLAY2LEARN_LUNARLANDER_RUNS_DIR") or os.environ.get("PLAY2LEARN_RUNS_DIR")
    if env_override:
        runs_dir = Path(env_override).resolve()
    else:
        pkg_dir = Path(__file__).resolve().parent
        runs_dir = pkg_dir / "runs"

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
        atomic_write_text(idx_file, "[]\n")
    return succ_dir


def _unique_dir(base: Path, name: str) -> Path:
    """Evita collisioni di cartelle timestampate generate nello stesso secondo."""
    candidate = base / name
    suffix = 1
    while candidate.exists():
        candidate = base / f"{name}_{suffix}"
        suffix += 1
    return candidate


def create_run_directory(
    mission: str,
    base_dir: Path | str | None = None,
    custom_name: str | None = None
) -> Path:
    """
    Crea una nuova directory timestampata per la sessione corrente in 'play2learn_lunarlander/runs/'.
    Esempio: play2learn_lunarlander/runs/run_20261114_154500_luna_standard/
    """
    if base_dir is None or str(base_dir) in ("runs", "runs_lunarlander"):
        base = get_default_runs_dir()
    else:
        base = Path(base_dir).resolve()
    base.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if custom_name:
        clean_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in custom_name)
        dir_name = f"run_{timestamp}_{clean_name}"
    else:
        dir_name = f"run_{timestamp}_{mission}"

    run_dir = _unique_dir(base, dir_name)
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
    della custom_reward utilizzata per questa run in modalità atomica.
    """
    source_code = extract_reward_function_code(reward_fn)

    # 1. Salva il codice Python autonomo per consultazione rapida
    code_path = run_dir / "reward_function.py"
    try:
        atomic_write_text(
            code_path,
            f'"""\nFunzione di Reward registrata per la run: {run_dir.name}\nData: {datetime.datetime.now().isoformat()}\n"""\n\n'
            f"{source_code}\n",
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
    atomic_write_json(meta_file, metadata)
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
    dest_dir = _unique_dir(succ_base, dest_name)
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

    # Aggiorna l'indice dei successi in modo atomico
    idx_file = succ_base / "index.json"
    index_data = []
    if idx_file.exists():
        try:
            index_data = json.loads(idx_file.read_text(encoding="utf-8"))
        except Exception:
            index_data = []

    entry = {
        "id": dest_dir.name,
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
    atomic_write_json(idx_file, index_data)

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
            pdf_link = f"<a href='{e.get('pdf_report', '')}' target='_blank' style='color:#1565c0; font-weight:bold;'>📄 Scarica PDF</a>" if e.get("pdf_report") else "-"
            safe_name = html.escape(str(e.get('name', '')))
            rows.append(
                f"<tr>"
                f"<td style='padding:8px 12px; font-weight:bold;'>🏆 {safe_name}</td>"
                f"<td style='padding:8px 12px; text-align:center;'><span style='background:#e8f5e9; color:#2e7d32; padding:3px 8px; border-radius:12px; font-weight:bold;'>{e.get('success_rate', 0)}%</span></td>"
                f"<td style='padding:8px 12px; text-align:center;'>{e.get('mean_fuel', 0)} un.</td>"
                f"<td style='padding:8px 12px; text-align:center;'>{e.get('mean_length', 0)} step</td>"
                f"<td style='padding:8px 12px; text-align:center;'>{e.get('video_count', 0)} video</td>"
                f"<td style='padding:8px 12px;'>{e.get('archived_at', '')}</td>"
                f"<td style='padding:8px 12px;'>{pdf_link}</td>"
                f"</tr>"
            )

        html_str = f"""
        <div style="font-family:sans-serif; margin:16px 0;">
            <h3 style="color:#2e7d32; margin-bottom:8px;">🎯 Archivio Esperimenti Riusciti (Successo &ge; 80%)</h3>
            <p style="color:#555; margin-bottom:12px; font-size:13px;">Tutti i modelli che hanno superato l'80% di allunaggio con le relative Schede Tecniche PDF vettoriali.</p>
            <table style="border-collapse:collapse; width:100%; border:1px solid #c8e6c9; text-align:left; font-size:13.5px;">
                <thead style="background:#e8f5e9; color:#1b5e20;">
                    <tr>
                        <th style="padding:10px 12px;">Esperimento</th>
                        <th style="padding:10px 12px; text-align:center;">Allunaggio</th>
                        <th style="padding:10px 12px; text-align:center;">Carburante Medio</th>
                        <th style="padding:10px 12px; text-align:center;">Durata Media</th>
                        <th style="padding:10px 12px; text-align:center;">Video</th>
                        <th style="padding:10px 12px;">Data</th>
                        <th style="padding:10px 12px;">Report</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(rows)}
                </tbody>
            </table>
        </div>
        """
        display(HTML(html_str))
    except Exception:
        print("\n" + "=" * 80)
        print("  🎯 ARCHIVIO ESPERIMENTI RIUSCITI (SUCCESSO >= 80%)")
        print("=" * 80)
        print(f"{'NOME':<25} | {'SUCCESSO':<10} | {'CARBURANTE':<12} | {'DATA':<15}")
        print("-" * 80)
        for e in entries:
            print(f"{e.get('name', ''):<25} | {e.get('success_rate', 0):<9.1f}% | {e.get('mean_fuel', 0):<12.1f} | {e.get('archived_at', ''):<15}")
        print("=" * 80 + "\n")


def find_run_by_timestamp(timestamp_query: str, runs_dir: Path | str | None = None) -> Path | None:
    """
    Individua in modo robusto la cartella di una run passata a partire da un timestamp o parola chiave.
    Supporta:
    - 'latest' o 'ultima': la run cronologicamente più recente in assoluto
    - Timestamp completo (es. '20261114_154500' o '2026-11-14T15:45:00')
    - Porzione di data/ora (es. '154500' o '20261114')
    - Nome o ID parziale della run (es. 'luna_standard', 'eco_lander')
    Cerca in 'runs/', 'runs/successi/' e 'runs/favorites/'.
    """
    base = Path(runs_dir).resolve() if runs_dir else get_default_runs_dir()
    if not base.exists():
        return None

    query = timestamp_query.strip()
    all_dirs: list[Path] = []

    # Separa run standard da archivi per evitare che un salvataggio in favorites falsi "latest"
    standard_dirs: list[Path] = []
    all_dirs: list[Path] = []

    # 1. Cartelle dirette in runs/
    for sub in base.iterdir():
        if sub.is_dir() and sub.name not in ("favorites", "successi"):
            if (sub / "run_metadata.json").exists():
                standard_dirs.append(sub)
                all_dirs.append(sub)

    # 2. Cartelle in runs/successi/
    succ_dir = base / "successi"
    if succ_dir.exists():
        for sub in succ_dir.iterdir():
            if sub.is_dir() and (sub / "run_metadata.json").exists():
                all_dirs.append(sub)

    # 3. Cartelle in runs/favorites/
    fav_dir = base / "favorites"
    if fav_dir.exists():
        for sub in fav_dir.iterdir():
            if sub.is_dir() and (sub / "run_metadata.json").exists():
                all_dirs.append(sub)

    if not all_dirs:
        return None

    # Ordina cronologicamente per nome della directory (formato run_YYYYMMDD_HHMMSS_...)
    standard_dirs.sort(key=lambda p: p.name, reverse=True)
    all_dirs.sort(key=lambda p: p.name, reverse=True)

    q_low = query.lower()
    # Caso speciale: latest / ultima
    if q_low in ("latest", "ultima", "ultimo", "last", "recents", "recente"):
        return standard_dirs[0] if standard_dirs else all_dirs[0]

    # Caso speciale: previous / precedente
    if q_low in ("previous", "precedente", "penultima", "penultimo", "prev"):
        pool = standard_dirs if len(standard_dirs) >= 2 else all_dirs
        return pool[1] if len(pool) >= 2 else None

    clean_query = query.lower().replace("-", "").replace(":", "").replace(" ", "_")

    # Ricerca esatta o parziale
    for d in all_dirs:
        d_name = d.name.lower()
        if clean_query in d_name or query.lower() in d_name:
            return d

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
            if raw_ts:
                try:
                    ts_str = datetime.datetime.fromisoformat(raw_ts).strftime("%Y%m%d_%H%M%S")
                except ValueError:
                    ts_str = d.name
            else:
                ts_str = d.name

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

            safe_ts = html.escape(str(e.get('timestamp', '')))
            safe_name = html.escape(str(e.get('name', '')))
            safe_cat = html.escape(str(e.get('category', '')))
            safe_id = html.escape(str(e.get('run_id', '')))

            rows.append(
                f"<tr>"
                f"<td style='padding:8px 12px; font-family:monospace; font-weight:bold; color:#1a237e; background:#f5f5f5;'>{safe_ts}</td>"
                f"<td style='padding:8px 12px;'><b>{safe_name}</b> <span style='font-size:11px; color:#666;'>({safe_cat})</span></td>"
                f"<td style='padding:8px 12px; text-align:center;'><span style='background:{bg_badge}; color:{badge_color}; padding:3px 8px; border-radius:12px; font-weight:bold;'>{succ:.1f}%</span></td>"
                f"<td style='padding:8px 12px; text-align:center;'>{e['mean_fuel']:.1f} un.</td>"
                f"<td style='padding:8px 12px; text-align:center;'>{e['video_count']} video</td>"
                f"<td style='padding:8px 12px; text-align:center;'>{has_mod_str}</td>"
                f"<td style='padding:8px 12px; font-size:11.5px; color:#666;'><code>{safe_id}</code></td>"
                f"</tr>"
            )

        html_str = f"""
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
        display(HTML(html_str))
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


def comprimi_runs(
    runs_dir: Path | str | None = None,
    nome_zip: str = "runs_lunarlander.zip",
    scarica_su_colab: bool = True,
) -> Path | None:
    """
    Comprime l'intera cartella 'runs/' degli esperimenti Lunar Lander in un file ZIP al livello del notebook
    e ne avvia il download automatico se eseguito su Google Colab.
    """
    from play2learn_core.storage import comprimi_runs as _core_comprimi
    actual_dir = Path(runs_dir) if runs_dir else get_default_runs_dir()
    return _core_comprimi(runs_dir=actual_dir, nome_zip=nome_zip, scarica_su_colab=scarica_su_colab)
