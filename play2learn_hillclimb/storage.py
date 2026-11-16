"""
Modulo di Storage e Persistenza Esperimenti per Hill Climb Racing (Play2Learn).
"""

from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import Any, Callable

from play2learn_core.storage import (
    atomic_write_json,
    atomic_write_text,
    elenca_cronologia_run as _core_elenca,
    find_run_by_timestamp as _core_find,
    save_generic_run_metadata,
    unique_dir,
)


def get_default_runs_dir() -> Path:
    """Restituisce il percorso della cartella principale 'runs/' specifica per Hill Climb Racing."""
    env_override = os.environ.get("PLAY2LEARN_HILLCLIMB_RUNS_DIR") or os.environ.get("PLAY2LEARN_RUNS_DIR")
    if env_override:
        p = Path(env_override).resolve()
    else:
        pkg_dir = Path(__file__).resolve().parent
        p = pkg_dir / "runs"
    p.mkdir(parents=True, exist_ok=True)
    (p / "favorites").mkdir(parents=True, exist_ok=True)
    return p


def find_run_by_timestamp(identifier: str, runs_dir: Path | str | None = None) -> Path | None:
    """Risolve un identificatore nella cartella runs di Hill Climb."""
    actual_dir = Path(runs_dir) if runs_dir else get_default_runs_dir()
    found = _core_find(identifier=identifier, runs_dir=actual_dir)
    if found is None and runs_dir is None:
        legacy_dir = Path.cwd() / "runs_hillclimb"
        if legacy_dir.exists():
            found = _core_find(identifier=identifier, runs_dir=legacy_dir)
    return found


def elenca_cronologia_run(runs_dir: Path | str | None = None) -> list[dict[str, Any]]:
    """Restituisce la cronologia delle run per Hill Climb."""
    actual_dir = Path(runs_dir) if runs_dir else get_default_runs_dir()
    return _core_elenca(runs_dir=actual_dir)


def mostra_cronologia_run(runs_dir: Path | str | None = None) -> None:
    """
    Visualizza la tabella della cronologia di tutte le run passate per Hill Climb,
    evidenziando chiaramente il timestamp da inserire nella cella di reportistica.
    """
    import html
    entries = elenca_cronologia_run(runs_dir)
    if not entries:
        print("\n[play2learn] Nessuna run attualmente presente nella cartella delle run. Esegui prima un addestramento!\n")
        return

    try:
        from IPython.display import HTML, display
        rows = []
        for e in entries:
            succ = float(e.get("success_rate", 0.0))
            badge_color = "#2e7d32" if succ >= 80.0 else ("#f57c00" if succ >= 50.0 else "#c62828")
            bg_badge = "#e8f5e9" if succ >= 80.0 else ("#fff3e0" if succ >= 50.0 else "#ffebee")
            has_mod = Path(e.get("dir_path", "")).joinpath("final_model.zip").exists()
            has_mod_str = "✅ Sì" if has_mod else "❌ No"

            safe_ts = html.escape(str(e.get("timestamp", "")))
            safe_name = html.escape(str(e.get("mission_title", e.get("dir_name", ""))))
            safe_dir = html.escape(str(e.get("dir_name", "")))
            vid_count = int(e.get("videos_count", 0))

            rows.append(
                f"<tr>"
                f"<td style='padding:8px 12px; font-family:monospace; font-weight:bold; color:#0284c7; background:#f0f9ff;'>{safe_ts}</td>"
                f"<td style='padding:8px 12px;'><b>{safe_name}</b></td>"
                f"<td style='padding:8px 12px; text-align:center;'><span style='background:{bg_badge}; color:{badge_color}; padding:3px 8px; border-radius:12px; font-weight:bold;'>{succ:.1f}%</span></td>"
                f"<td style='padding:8px 12px; text-align:center;'>{vid_count} video</td>"
                f"<td style='padding:8px 12px; text-align:center;'>{has_mod_str}</td>"
                f"<td style='padding:8px 12px; font-size:11.5px; color:#666;'><code>{safe_dir}</code></td>"
                f"</tr>"
            )

        html_str = f"""
        <div style="font-family:'Segoe UI', sans-serif; margin:16px 0;">
            <h3 style="color:#0369a1; margin-bottom:6px;">🕒 Cronologia Esperimenti e Timestamp Disponibili</h3>
            <p style="color:#555; margin-bottom:12px; font-size:13px;">
                💡 <b>Come usare:</b> copia uno dei <b>Timestamp</b> della prima colonna (o scrivi <code>latest</code>) nella cella di generazione report sottostante.
            </p>
            <table style="border-collapse:collapse; width:100%; border:1px solid #bae6fd; text-align:left; font-size:13px;">
                <thead style="background:#e0f2fe; color:#0369a1;">
                    <tr>
                        <th style="padding:10px 12px;">Timestamp (da copiare)</th>
                        <th style="padding:10px 12px;">Tracciato / Percorso</th>
                        <th style="padding:10px 12px; text-align:center;">Tasso Successo (≥800m)</th>
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
        print(f"{'TIMESTAMP':<18} | {'TRACCIATO':<25} | {'SUCCESSO':<10} | {'MODELLO':<8}")
        print("-" * 80)
        for e in entries:
            has_mod = Path(e.get("dir_path", "")).joinpath("final_model.zip").exists()
            mod_str = "Sì" if has_mod else "No"
            print(f"{e.get('timestamp',''):<18} | {str(e.get('mission_title',''))[:24]:<25} | {float(e.get('success_rate',0)):<9.1f}% | {mod_str:<8}")
        print("=" * 80)
        print("💡 Copia il Timestamp desiderato nella cella di reportistica per rigenerare la scheda tecnica!\n")


def create_run_directory(mission: str = "campagna_standard", base_dir: Path | str | None = None) -> Path:
    """Crea una nuova directory univoca per la sessione di addestramento in play2learn_hillclimb/runs."""
    if base_dir is None or str(base_dir) in ("runs", "runs_hillclimb"):
        parent = get_default_runs_dir()
    else:
        parent = Path(base_dir).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"run_{timestamp}_{mission}"
    target = unique_dir(parent, dir_name)
    target.mkdir(parents=True, exist_ok=True)
    (target / "videos").mkdir(parents=True, exist_ok=True)
    (target / "checkpoints").mkdir(parents=True, exist_ok=True)
    return target


def save_run_metadata(
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
) -> Path:
    """Salva i metadati della sessione in formato JSON atomico."""
    return save_generic_run_metadata(
        run_dir=run_dir,
        mission=mission,
        mission_cfg=mission_cfg,
        env_kwargs=env_kwargs,
        total_timesteps=total_timesteps,
        elapsed_seconds=elapsed_seconds,
        reward_fn=reward_fn,
        eval_stats=eval_stats,
        diagnostics=diagnostics,
        notes=notes,
    )


def comprimi_runs(
    runs_dir: Path | str | None = None,
    nome_zip: str = "runs_hillclimb.zip",
    scarica_su_colab: bool = True,
) -> Path | None:
    """
    Comprime l'intera cartella degli esperimenti Hill Climb Racing in un file ZIP al livello del notebook
    e ne avvia il download automatico se eseguito su Google Colab.
    """
    from play2learn_core.storage import comprimi_runs as _core_comprimi
    actual_dir = Path(runs_dir) if runs_dir else get_default_runs_dir()
    if not actual_dir.exists() and Path("runs").exists():
        actual_dir = Path("runs")
    return _core_comprimi(runs_dir=actual_dir, nome_zip=nome_zip, scarica_su_colab=scarica_su_colab)
