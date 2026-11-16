"""
Modulo di Reportistica Tecnica e Documentazione per Play2Learn Hill Climb Racing.
Genera Schede Tecniche in formato PDF, HTML5 standalone e Markdown per analisi e maturità.
"""

from __future__ import annotations

import html
import json
import datetime
from pathlib import Path
from typing import Any, Optional

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

from play2learn_core.reporting import serialize_eval_metrics
from play2learn_hillclimb.diagnostics import diagnose_hill_climb_behavior
from play2learn_hillclimb.visualizer import plot_hill_telemetry
from play2learn_hillclimb.storage import BASE_RUNS_DIR, load_run_metadata


def genera_report_esperimento(
    run_dir: Path | str,
    metadata: Optional[dict[str, Any]] = None,
    titolo: str = "Esperimento Hill Climb Racing",
    autore: str = "Studente Play2Learn",
    note: str = "",
) -> dict[str, Path]:
    """Genera la suite completa di report (PDF, HTML5, Markdown) per una run completata."""
    run_dir = Path(run_dir)
    if metadata is None:
        metadata = load_run_metadata(run_dir)

    metrics = metadata.get("eval_metrics", {})
    history = metadata.get("history", metrics.get("history", []))

    # 1. Generazione grafico telemetria
    telemetry_img = run_dir / "telemetria_esperimento.png"
    if history and _MPL_AVAILABLE:
        try:
            plot_hill_telemetry(history, output_path=telemetry_img)
        except Exception as e:
            print(f"[play2learn] Errore plot telemetria: {e}")

    # 2. Diagnostica testuale
    diag_text = diagnose_hill_climb_behavior(metrics)

    # 3. Report Markdown
    md_path = run_dir / "report_esperimento.md"
    md_content = f"""# 🏎️ Scheda Tecnica: {titolo}

* **Data e Ora:** {metadata.get('timestamp', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}
* **Autore:** {autore}
* **Missione:** {metadata.get('mission', 'campagna_standard')}
* **Timesteps Addestramento:** {metadata.get('total_timesteps', 300_000):,}

---

## 📊 Prestazioni e Metriche di Guida

| Metrica | Risultato |
| :--- | :--- |
| **Percentuale di Successo (>= 800m)** | **{metrics.get('success_rate', metrics.get('eval_success_rate', 0.0)):.1f}%** |
| **Tasso di Schianto (Ribaltamento)** | {metrics.get('crash_rate', 0.0):.1f}% |
| **Tasso di Timeout (Blocco)** | {metrics.get('timeout_rate', 0.0):.1f}% |
| **Distanza Media Percorsa** | {metrics.get('mean_distance', 0.0):.1f} m |
| **Distanza Massima (Record)** | **{metrics.get('max_distance', 0.0):.1f} m** |
| **Velocità Media di Punta** | {metrics.get('mean_speed', 0.0):.2f} m/s |
| **Tempo Trascorso in Volo (Airtime)** | {metrics.get('mean_airtime_pct', 0.0):.1f}% |

---

## 📋 Diagnostica del Coach

```text
{diag_text}
```

---

## 💡 Note dello Studente
{note if note else "Nessuna nota aggiuntiva."}

---
*Generato automaticamente da Play2Learn Hill Climb Racing Engine.*
"""
    md_path.write_text(md_content, encoding="utf-8")

    # 4. Report HTML5 Standalone
    html_path = run_dir / "report_esperimento.html"
    html_content = f"""<!DOCTYPE html>
<html lang="it">
<head>
  <meta charset="UTF-8">
  <title>{html.escape(titolo)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 30px; background: #f8f9fa; color: #212529; }}
    .container {{ max-width: 900px; margin: auto; background: #ffffff; padding: 30px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
    h1 {{ color: #0d6efd; border-bottom: 2px solid #e9ecef; padding-bottom: 12px; }}
    table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
    th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #dee2e6; }}
    th {{ background: #f1f3f5; }}
    .badge-success {{ color: #198754; font-weight: bold; }}
    pre {{ background: #212529; color: #f8f9fa; padding: 15px; border-radius: 6px; overflow-x: auto; }}
  </style>
</head>
<body>
<div class="container">
  <h1>🏎️ Scheda Tecnica Esperimento</h1>
  <p><strong>Titolo:</strong> {html.escape(titolo)} | <strong>Autore:</strong> {html.escape(autore)}</p>
  <p><strong>Missione:</strong> {html.escape(str(metadata.get('mission', 'campagna_standard')))}</p>
  <table>
    <tr><th>Metrica</th><th>Valore</th></tr>
    <tr><td>Percentuale di Successo (&ge; 800m)</td><td class="badge-success">{metrics.get('success_rate', metrics.get('eval_success_rate', 0.0)):.1f}%</td></tr>
    <tr><td>Schianti / Ribaltamenti</td><td>{metrics.get('crash_rate', 0.0):.1f}%</td></tr>
    <tr><td>Blocchi in Salita / Timeout</td><td>{metrics.get('timeout_rate', 0.0):.1f}%</td></tr>
    <tr><td>Distanza Media Percorsa</td><td>{metrics.get('mean_distance', 0.0):.1f} m</td></tr>
    <tr><td>Distanza Massima Raggiunta</td><td><strong>{metrics.get('max_distance', 0.0):.1f} m</strong></td></tr>
    <tr><td>Velocità Media di Punta</td><td>{metrics.get('mean_speed', 0.0):.2f} m/s</td></tr>
    <tr><td>Tempo in Volo (Airtime)</td><td>{metrics.get('mean_airtime_pct', 0.0):.1f}%</td></tr>
  </table>
  <h3>📋 Diagnostica del Coach</h3>
  <pre>{html.escape(diag_text)}</pre>
</div>
</body>
</html>
"""
    html_path.write_text(html_content, encoding="utf-8")

    # 5. Scheda Tecnica PDF
    pdf_path = run_dir / "scheda_esperimento.pdf"
    if _MPL_AVAILABLE:
        try:
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis("off")
            pdf_text = f"SCHEDA TECNICA: {titolo}\nAutore: {autore}\nData: {metadata.get('timestamp')}\n\n"
            pdf_text += f"Successo: {metrics.get('success_rate', 0.0):.1f}%\n"
            pdf_text += f"Distanza Media: {metrics.get('mean_distance', 0.0):.1f}m (Max: {metrics.get('max_distance', 0.0):.1f}m)\n"
            pdf_text += f"Schianti: {metrics.get('crash_rate', 0.0):.1f}%\n\n"
            import re
            clean_diag = re.sub(r"[^\x00-\x7F]+", " ", diag_text)
            pdf_text += clean_diag
            ax.text(0.05, 0.95, pdf_text, fontsize=10, verticalalignment="top", fontfamily="monospace")
            with PdfPages(str(pdf_path)) as pdf:
                pdf.savefig(fig)
            plt.close(fig)
        except Exception as e:
            print(f"[play2learn] Avviso generazione PDF: {e}")

    print(f"✅ Report tecnico generato in: {run_dir.name}")
    print("   • PDF:      scheda_esperimento.pdf")
    print("   • HTML:     report_esperimento.html")
    print("   • Markdown: report_esperimento.md")

    return {
        "md": md_path,
        "html": html_path,
        "pdf": pdf_path,
    }


def genera_report_da_timestamp(timestamp: str, base_dir: Path = BASE_RUNS_DIR) -> dict[str, Path]:
    """Rigenera la scheda tecnica di una run passata cercando per timestamp o nome cartella."""
    base_dir = Path(base_dir)
    matches = list(base_dir.glob(f"*{timestamp}*"))
    if not matches:
        raise FileNotFoundError(f"Nessuna cartella trovata corrispondente al timestamp: {timestamp}")

    target_dir = matches[0]
    return genera_report_esperimento(target_dir)


def mostra_cronologia(base_dir: Path = BASE_RUNS_DIR) -> None:
    """Mostra la tabella di tutte le sessioni di addestramento eseguite."""
    base_dir = Path(base_dir)
    if not base_dir.exists():
        print("[play2learn] Nessun esperimento registrato.")
        return

    runs = sorted(list(base_dir.glob("run_*")), reverse=True)
    if not runs:
        print("[play2learn] Nessuna run trovata in runs/.")
        return

    print("══════════════════════════════════════════════════════════════════════════════════")
    print(f"{'TIMESTAMP / DIRECTORY':<35} | {'MISSIONE':<20} | {'SUCCESSO %':<10} | {'DIST MAX'}")
    print("──────────────────────────────────────────────────────────────────────────────────")
    for r in runs:
        meta = load_run_metadata(r)
        metrics = meta.get("eval_metrics", {})
        ts = r.name
        miss = str(meta.get("mission", "standard"))[:18]
        succ = f"{metrics.get('success_rate', 0.0):.1f}%"
        dmax = f"{metrics.get('max_distance', 0.0):.1f}m"
        print(f"{ts:<35} | {miss:<20} | {succ:<10} | {dmax}")
    print("══════════════════════════════════════════════════════════════════════════════════")
