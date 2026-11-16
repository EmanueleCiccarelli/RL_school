"""
Modulo di Generazione Report e Confronto Esperimenti per Hill Climb Racing.
Produce schede tecniche in PDF (vettoriale ad alta risoluzione), HTML standalone e Markdown,
con corretta separazione tra sanitizzazione HTML e formati PDF/MD, e supporto flessibile per confronta_run.
"""

from __future__ import annotations

import base64
import datetime
import html
import io
import json
from pathlib import Path
from typing import Any

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from play2learn_core.storage import atomic_write_json, atomic_write_text
from play2learn_hillclimb.storage import find_run_by_timestamp, get_default_runs_dir
from play2learn_hillclimb.visualizer import plot_hill_telemetry


def genera_report_esperimento(
    run_or_fav: Any,
    autore: str = "Studente",
    titolo: str = "Report Esperimento Hill Climb Racing",
    note: str = "",
    output_dir: Path | str | None = None,
) -> dict[str, Path]:
    """Genera la scheda tecnica completa (PDF, HTML, MD) dell'esperimento di guida."""
    source_dir: Path | None = None
    if isinstance(run_or_fav, dict):
        for k in ("checkpoint_dir", "path", "dir_path", "run_dir"):
            if k in run_or_fav:
                source_dir = Path(run_or_fav[k])
                break
    elif isinstance(run_or_fav, (str, Path)):
        p = Path(run_or_fav)
        if p.exists() and p.is_dir():
            source_dir = p
        else:
            source_dir = find_run_by_timestamp(str(run_or_fav), get_default_runs_dir())

    if not source_dir or not source_dir.exists():
        raise ValueError(f"[Play2Learn] Nessuna directory di run trovata per: {run_or_fav}")

    meta_file = source_dir / "run_metadata.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    target_dir = Path(output_dir) if output_dir else (source_dir / f"report_{timestamp}")
    target_dir.mkdir(parents=True, exist_ok=True)

    metrics = meta.get("eval_metrics", {})
    success_rate = float(metrics.get("eval_success_rate", 0.0))
    mean_dist = float(metrics.get("mean_distance", 0.0))
    max_dist = float(metrics.get("max_distance_achieved", mean_dist))
    mean_speed = float(metrics.get("mean_speed", 0.0))
    airtime_pct = float(metrics.get("mean_airtime_pct", 0.0))
    crashes = int(metrics.get("eval_crashes", 0))
    timeouts = int(metrics.get("eval_timeouts", 0))

    mission = meta.get("mission", {})
    mission_title = mission.get("title", "Campagna Standard")
    mission_desc = mission.get("description", "Colline ondulate.")

    code_source = meta.get("reward_source_code", "# Codice non disponibile.")
    raw_notes = note or meta.get("notes", "")

    # Sanitizzazione separata per contesto (HTML vs testo/PDF/MD)
    html_titolo = html.escape(titolo)
    html_autore = html.escape(autore)
    html_mission = html.escape(mission_title)
    html_note = html.escape(raw_notes)
    html_code = html.escape(code_source)

    # Generazione grafico telemetrico
    plot_path = target_dir / "telemetria_guida.png"
    history = meta.get("eval_metrics", {}).get("history", [])
    if not history:
        history = [{"distance": mean_dist, "max_speed": mean_speed, "airtime_pct": airtime_pct, "is_success": success_rate >= 80}] * 20
    plot_hill_telemetry(history, plot_path)

    # 1. Generazione PDF multipagina (senza entita HTML escape)
    pdf_path = target_dir / "scheda_esperimento.pdf"
    with PdfPages(str(pdf_path)) as pdf:
        # Pagina 1: Telemetria e KPI
        fig1 = Figure(figsize=(8.27, 11.69), dpi=160)
        FigureCanvasAgg(fig1)
        ax1 = fig1.subplots()
        ax1.axis("off")
        fig1.text(0.5, 0.94, titolo, ha="center", fontsize=15, fontweight="bold", color="#1e293b")
        fig1.text(0.5, 0.915, f"Missione: {mission_title} | Autore: {autore} | Data: {timestamp}", ha="center", fontsize=9.5, color="#64748b")

        # Tabella KPI
        table_data = [
            ["Distanza Massima", f"{max_dist:.1f} m", ">= 800.0 m", "Obiettivo" if max_dist >= 800 else "In Corso"],
            ["Distanza Media", f"{mean_dist:.1f} m", ">= 650.0 m", "Nominale"],
            ["Velocita Media", f"{mean_speed:.1f} m/s", "3.0 - 8.0 m/s", "Costante"],
            ["Airtime Medio", f"{airtime_pct:.1f}%", "< 35.0%", "Aderenza"],
            ["Schianti / Ribaltamenti", f"{crashes} su 20", "<= 3", "Sicurezza" if crashes <= 3 else "Criticita"],
            ["Blocchi in Salita", f"{timeouts} su 20", "< 2", "Trazione"],
        ]
        t = ax1.table(cellText=table_data, colLabels=["Parametro Telemetrico", "Valore Rilevato", "Riferimento Ideale", "Stato"],
                      loc="top", cellLoc="center", bbox=[0.05, 0.70, 0.90, 0.18])
        t.auto_set_font_size(False)
        t.set_fontsize(8.5)

        # Inserimento grafico telemetrico nel PDF
        if plot_path.exists():
            img = plt.imread(str(plot_path))
            ax_img = fig1.add_axes([0.05, 0.10, 0.90, 0.55])
            ax_img.imshow(img)
            ax_img.axis("off")

        pdf.savefig(fig1)
        fig1.clear()

        # Pagina 2: Codice Sorgente e Note
        fig2 = Figure(figsize=(8.27, 11.69), dpi=160)
        FigureCanvasAgg(fig2)
        ax2 = fig2.subplots()
        ax2.axis("off")
        fig2.text(0.08, 0.94, "2. Codice Sorgente della Reward Function:", fontsize=12, fontweight="bold", color="#1e293b")
        fig2.text(0.08, 0.91, code_source[:2200], fontsize=7.5, family="monospace", va="top",
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#cbd5e1"))

        fig2.text(0.08, 0.32, "3. Note di Laboratorio e Debug:", fontsize=12, fontweight="bold", color="#1e293b")
        notes_box = f"Note:\n{raw_notes}\n\nAnalisi didattica:\n• Punti di forza dell'assetto\n• Comportamento nelle salite ripide"
        fig2.text(0.08, 0.29, notes_box, fontsize=8.5, va="top",
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="#fffde7", edgecolor="#fbc02d"))

        pdf.savefig(fig2)
        fig2.clear()

    # 2. Generazione HTML e Markdown
    plot_b64 = base64.b64encode(plot_path.read_bytes()).decode("ascii") if plot_path.exists() else ""

    md_content = f"""# 🏎️ Scheda Tecnica Esperimento — Hill Climb Racing
**Titolo:** {titolo}  
**Autore:** {autore}  
**Missione:** {mission_title}  
**Data:** {timestamp}  

## 1. Indicatori di Performance Telemetrica
| Parametro Telemetrico | Valore Rilevato | Obiettivo Ideale | Esito |
| :--- | :--- | :--- | :--- |
| **Distanza Massima** | {max_dist:.1f} m | $\\ge 800.0$ m | {'✅ Raggiunto' if max_dist >= 800 else '⚠️ In corso'} |
| **Distanza Media** | {mean_dist:.1f} m | $\\ge 650.0$ m | Nominale |
| **Velocità Media** | {mean_speed:.1f} m/s | $3.0 - 8.0$ m/s | Trazione |
| **Airtime Medio** | {airtime_pct:.1f}% | $< 35.0\\%$ | Assetto |
| **Ribaltamenti / Schianti** | {crashes} / 20 | $\\le 3$ | {'Nominale' if crashes <= 3 else 'Criticità'} |

## 2. Codice della Custom Reward
```python
{code_source}
```

## 3. Note di Laboratorio
{raw_notes}
"""

    html_content = f"""<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <title>{html_titolo}</title>
    <style>
        body {{ font-family:'Segoe UI',system-ui,sans-serif; background:#0b0f19; color:#f1f5f9; padding:24px; }}
        .card {{ background:#1e293b; border-radius:12px; padding:20px; max-width:860px; margin:auto; border:1px solid #334155; }}
        h1 {{ color:#38bdf8; font-size:22px; margin-top:0; }}
        table {{ width:100%; border-collapse:collapse; margin-top:14px; font-size:13px; }}
        th, td {{ padding:10px; border-bottom:1px solid #334155; text-align:left; }}
        th {{ background:#0f172a; color:#94a3b8; }}
        .badge {{ background:#0284c7; color:#fff; padding:3px 8px; border-radius:4px; font-weight:bold; }}
        pre {{ background:#0f172a; padding:14px; border-radius:8px; border:1px solid #334155; overflow-x:auto; font-size:12px; }}
        img {{ max-width:100%; border-radius:8px; margin-top:14px; }}
    </style>
</head>
<body>
    <div class="card">
        <h1>🏎️ {html_titolo}</h1>
        <div>Autore: <strong>{html_autore}</strong> | Missione: <strong>{html_mission}</strong> | Data: <strong>{timestamp}</strong></div>

        <table>
            <thead><tr><th>Parametro</th><th>Rilevato</th><th>Obiettivo</th><th>Esito</th></tr></thead>
            <tbody>
                <tr><td><strong>Distanza Massima</strong></td><td>{max_dist:.1f} m</td><td>&ge; 800.0 m</td><td><span class="badge">{'Traguardo OK' if max_dist >= 800 else 'In Corso'}</span></td></tr>
                <tr><td><strong>Distanza Media</strong></td><td>{mean_dist:.1f} m</td><td>&ge; 650.0 m</td><td>Nominale</td></tr>
                <tr><td><strong>Velocità Media</strong></td><td>{mean_speed:.1f} m/s</td><td>3.0 - 8.0 m/s</td><td>Costante</td></tr>
                <tr><td><strong>Airtime Medio</strong></td><td>{airtime_pct:.1f}%</td><td>&lt; 35%</td><td>Aderenza</td></tr>
                <tr><td><strong>Schianti</strong></td><td>{crashes} / 20</td><td>&le; 3</td><td>Sicurezza</td></tr>
            </tbody>
        </table>

        {f'<img src="data:image/png;base64,{plot_b64}" alt="Telemetria">' if plot_b64 else ''}

        <h3>Codice Reward Function</h3>
        <pre><code>{html_code}</code></pre>

        <h3>Note di Laboratorio</h3>
        <div>{html_note}</div>
    </div>
</body>
</html>
"""

    md_path = target_dir / "report_esperimento.md"
    html_path = target_dir / "report_esperimento.html"
    atomic_write_text(md_path, md_content)
    atomic_write_text(html_path, html_content)

    print(f"✅ Report Hill Climb generato in: {target_dir.name}")
    print(f"   • PDF:      {pdf_path.name}")
    print(f"   • HTML:     {html_path.name}")
    print(f"   • Markdown: {md_path.name}")

    return {
        "pdf_path": pdf_path,
        "html_path": html_path,
        "md_path": md_path,
        "export_dir": target_dir,
    }


def confronta_run(runs_or_favs: Any) -> dict[str, Any]:
    """
    Genera un confronto sinottico tra piu sessioni di Hill Climb.
    Accetta una lista di run/preferiti, un oggetto LabResult, una cartella o una stringa separata da virgola
    (es. 'latest, previous' oppure ['run_1', 'run_2']).
    """
    runs_dir = get_default_runs_dir()

    # Normalizza l'input se e una stringa con virgole o singolo identificatore
    items_to_check: list[Any] = []
    if isinstance(runs_or_favs, str):
        if "," in runs_or_favs:
            items_to_check = [s.strip() for s in runs_or_favs.split(",") if s.strip()]
        else:
            items_to_check = [runs_or_favs.strip()]
    elif isinstance(runs_or_favs, (list, tuple)):
        items_to_check = list(runs_or_favs)
    elif isinstance(runs_or_favs, dict):
        items_to_check = [runs_or_favs]
    else:
        items_to_check = [runs_or_favs]

    resolved: list[Path] = []
    for item in items_to_check:
        if isinstance(item, Path) and item.exists():
            resolved.append(item)
        elif isinstance(item, dict):
            for k in ("checkpoint_dir", "path", "dir_path", "run_dir"):
                if k in item:
                    cand = Path(item[k])
                    if cand.exists():
                        resolved.append(cand)
                        break
        elif isinstance(item, str):
            p = find_run_by_timestamp(item, runs_dir)
            if p and p.exists():
                resolved.append(p)

    if not resolved:
        print("[play2learn] Nessuna run valida trovata per il confronto.")
        return {}

    labels = []
    distances = []
    speeds = []
    for r in resolved:
        meta_p = r / "run_metadata.json"
        if meta_p.exists():
            try:
                meta = json.loads(meta_p.read_text(encoding="utf-8"))
                labels.append(meta.get("mission", {}).get("title", r.name[:16]))
                distances.append(float(meta.get("eval_metrics", {}).get("mean_distance", 0.0)))
                speeds.append(float(meta.get("eval_metrics", {}).get("mean_speed", 0.0)))
            except Exception:
                labels.append(r.name[:16])
                distances.append(0.0)
                speeds.append(0.0)
        else:
            labels.append(r.name[:16])
            distances.append(0.0)
            speeds.append(0.0)

    fig = Figure(figsize=(8, 4), dpi=120)
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    ax.bar(labels, distances, color="#38bdf8", alpha=0.85)
    ax.set_title("Confronto Distanza Media tra Run", fontweight="bold")
    ax.set_ylabel("Distanza Media (m)")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()

    try:
        from IPython.display import Image, display
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        fig.clear()
        display(Image(data=buf.getvalue()))
    except Exception:
        pass

    return {"resolved_runs": resolved, "mean_distances": distances}


def genera_report_da_timestamp(
    timestamp_or_id: str = "latest",
    livello_report: str = "Base",
    autore: str = "Studente",
    note: str = "",
) -> dict[str, Path]:
    """Risolve la run da timestamp e genera il report tecnico."""
    runs_dir = get_default_runs_dir()
    run_path = find_run_by_timestamp(timestamp_or_id, runs_dir)
    if not run_path:
        raise ValueError(f"[Play2Learn] Nessuna run trovata per: {timestamp_or_id}")

    return genera_report_esperimento(
        run_or_fav=run_path,
        autore=autore,
        titolo=f"Report Esperimento {run_path.name}",
        note=note,
    )
