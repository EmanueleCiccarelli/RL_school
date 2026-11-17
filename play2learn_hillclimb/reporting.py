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
from play2learn_hillclimb.storage import BASE_RUNS_DIR, find_run_by_timestamp, load_run_metadata


def genera_report_esperimento(
    run_dir: Path | str,
    metadata: Optional[dict[str, Any]] = None,
    titolo: str = "Esperimento Hill Climb Racing",
    autore: str = "Studente Play2Learn",
    note: str = "",
    **kwargs: Any,
) -> dict[str, Path]:
    """Genera la suite completa di report (PDF, HTML5, Markdown) per una run completata."""
    run_dir = Path(run_dir)
    if metadata is None:
        metadata = load_run_metadata(run_dir)

    metrics = metadata.get("eval_metrics", {})
    history = metadata.get("history", metrics.get("history", []))

    # 1. Generazione grafico telemetria
    telemetry_img = run_dir / "telemetria_esperimento.png"
    if history and _MPL_AVAILABLE and not telemetry_img.exists():
        try:
            plot_hill_telemetry(history, output_path=telemetry_img)
        except Exception as e:
            print(f"[play2learn] Errore plot telemetria: {e}")

    # 2. Diagnostica testuale
    diag_text = diagnose_hill_climb_behavior(metrics)
    code_source = metadata.get("reward_source_code", "")

    # 3. Report Markdown
    code_md_block = f"\n## 💻 Codice Sorgente Custom Reward\n```python\n{code_source}\n```\n" if code_source else ""
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
{code_md_block}
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

    # 4. Report HTML5 Standalone con telemetria incorporata
    plot_b64_tag = ""
    if telemetry_img.exists():
        try:
            import base64
            img_b64 = base64.b64encode(telemetry_img.read_bytes()).decode("ascii")
            plot_b64_tag = f'<div style="text-align:center; margin:22px 0;"><img src="data:image/png;base64,{img_b64}" style="max-width:100%; border-radius:8px; box-shadow:0 3px 12px rgba(0,0,0,0.12);" alt="Dashboard Telemetrica"/></div>'
        except Exception:
            plot_b64_tag = f'<div style="text-align:center; margin:22px 0;"><img src="{telemetry_img.name}" style="max-width:100%;" alt="Dashboard Telemetrica"/></div>'

    code_html_block = ""
    if code_source:
        code_html_block = f"""
  <h3>💻 Codice Sorgente Custom Reward</h3>
  <pre style="background:#1e1e1e; color:#d4d4d4; padding:14px; border-radius:6px; font-size:13px; overflow-x:auto;"><code>{html.escape(code_source)}</code></pre>
"""

    html_path = run_dir / "report_esperimento.html"
    html_content = f"""<!DOCTYPE html>
<html lang="it">
<head>
  <meta charset="UTF-8">
  <title>{html.escape(titolo)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 30px; background: #f8f9fa; color: #212529; }}
    .container {{ max-width: 960px; margin: auto; background: #ffffff; padding: 32px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.08); }}
    h1 {{ color: #0d47a1; border-bottom: 2px solid #e9ecef; padding-bottom: 12px; margin-top: 0; }}
    .meta-box {{ background: #e3f2fd; padding: 12px 18px; border-radius: 6px; margin-bottom: 20px; font-size: 14px; }}
    table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
    th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #dee2e6; }}
    th {{ background: #f1f3f5; font-weight: bold; }}
    .badge-success {{ color: #1b5e20; font-weight: bold; font-size: 15px; }}
    pre {{ background: #212529; color: #f8f9fa; padding: 15px; border-radius: 6px; overflow-x: auto; font-size: 13px; }}
    .notes-box {{ background: #fffde7; padding: 14px 18px; border-left: 4px solid #fbc02d; border-radius: 4px; margin-top: 16px; font-size: 13.5px; }}
  </style>
</head>
<body>
<div class="container">
  <h1>🏎️ Scheda Tecnica Esperimento: Hill Climb Racing</h1>
  <div class="meta-box">
    <div><strong>Titolo:</strong> {html.escape(titolo)} | <strong>Autore:</strong> {html.escape(autore)}</div>
    <div><strong>Missione:</strong> {html.escape(str(metadata.get('mission', 'campagna_standard')))} | <strong>Data:</strong> {html.escape(str(metadata.get('timestamp', '')))}</div>
  </div>

  <h2>1. Prestazioni e Metriche di Guida</h2>
  <table>
    <tr><th>Parametro di Guida</th><th>Valore Misurato</th><th>Riferimento Obiettivo</th></tr>
    <tr><td><strong>Percentuale di Successo (&ge; 800m)</strong></td><td class="badge-success">{metrics.get('success_rate', metrics.get('eval_success_rate', 0.0)):.1f}%</td><td>&ge; 80.0%</td></tr>
    <tr><td>Schianti / Ribaltamenti</td><td>{metrics.get('crash_rate', 0.0):.1f}%</td><td>Minimizzare</td></tr>
    <tr><td>Blocchi in Salita / Timeout</td><td>{metrics.get('timeout_rate', 0.0):.1f}%</td><td>Minimizzare</td></tr>
    <tr><td>Distanza Media Percorsa</td><td>{metrics.get('mean_distance', 0.0):.1f} m</td><td>&ge; 800.0 m</td></tr>
    <tr><td>Distanza Massima (Record)</td><td><strong>{metrics.get('max_distance', 0.0):.1f} m</strong></td><td>&ge; 800.0 m</td></tr>
    <tr><td>Velocità Media di Punta</td><td>{metrics.get('mean_speed', 0.0):.2f} m/s</td><td>Dinamica sostenuta</td></tr>
    <tr><td>Tempo in Volo (Airtime)</td><td>{metrics.get('mean_airtime_pct', 0.0):.1f}%</td><td>Equilibrato</td></tr>
  </table>

  {plot_b64_tag}
  {code_html_block}

  <h2>2. Diagnostica del Coach</h2>
  <pre>{html.escape(diag_text)}</pre>

  <h2>3. Note di Debug e Considerazioni dello Studente</h2>
  <div class="notes-box">
    {html.escape(note) if note else "<em>Nessuna annotazione inserita. Puoi modificare questo report per aggiungere le tue osservazioni.</em>"}
  </div>
</div>
</body>
</html>
"""
    html_path.write_text(html_content, encoding="utf-8")

    # 5. Scheda Tecnica PDF
    pdf_path = run_dir / "scheda_esperimento.pdf"
    if _MPL_AVAILABLE:
        fig = None
        try:
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis("off")
            pdf_text = f"SCHEDA TECNICA: {titolo}\nAutore: {autore}\nData: {metadata.get('timestamp')}\n\n"
            pdf_text += f"Successo: {metrics.get('success_rate', 0.0):.1f}%\n"
            pdf_text += f"Distanza Media: {metrics.get('mean_distance', 0.0):.1f}m (Max: {metrics.get('max_distance', 0.0):.1f}m)\n"
            pdf_text += f"Schianti: {metrics.get('crash_rate', 0.0):.1f}%\n\n"
            import re
            clean_diag = re.sub(r"[^\x00-\xFF]+", "", diag_text)
            pdf_text += clean_diag
            ax.text(0.05, 0.95, pdf_text, fontsize=10, verticalalignment="top", fontfamily="monospace")
            with PdfPages(str(pdf_path)) as pdf:
                pdf.savefig(fig)
        except Exception as e:
            print(f"[play2learn] Avviso generazione PDF: {e}")
        finally:
            if fig is not None:
                plt.close(fig)

    print(f"✅ Report tecnico generato in: {run_dir.name}")
    print(f"   • PDF:      {pdf_path.name}")
    print(f"   • HTML:     {html_path.name}")
    print(f"   • Markdown: {md_path.name}")

    return {
        "md": md_path,
        "html": html_path,
        "pdf": pdf_path,
        "md_path": md_path,
        "html_path": html_path,
        "pdf_path": pdf_path,
        "telemetry_img": telemetry_img,
    }


def genera_report_da_timestamp(
    timestamp: str = "latest",
    tipo: str = "Base",
    autore: str = "Studente",
    titolo: Optional[str] = None,
    note: str = "",
    base_dir: Path = BASE_RUNS_DIR,
    **kwargs: Any,
) -> dict[str, Path]:
    """Rigenera la scheda tecnica di una run passata cercando per timestamp o nome cartella."""
    base_dir = Path(base_dir)
    target_dir = find_run_by_timestamp(timestamp, base_dir)
    if target_dir is None:
        matches = list(base_dir.glob(f"*{timestamp}*"))
        if not matches:
            raise FileNotFoundError(f"Nessuna cartella trovata corrispondente al timestamp: {timestamp}")
        target_dir = matches[0]

    title_str = titolo if titolo is not None else "Esperimento Hill Climb Racing"
    return genera_report_esperimento(
        run_dir=target_dir,
        titolo=title_str,
        autore=autore,
        note=note,
    )


def mostra_cronologia(base_dir: Path = BASE_RUNS_DIR) -> None:
    """Mostra la tabella di tutte le sessioni di addestramento eseguite."""
    base_dir = Path(base_dir)
    if not base_dir.exists():
        print("[play2learn] Nessun esperimento registrato.")
        return

    runs = sorted(
        (
            d
            for d in base_dir.iterdir()
            if d.is_dir()
            and d.name not in ("favorites", "preferiti", "successi")
            and (d / "run_metadata.json").exists()
        ),
        reverse=True,
    )
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
        miss_data = meta.get("mission", "standard")
        if isinstance(miss_data, dict):
            miss = miss_data.get("title", miss_data.get("key", "standard"))[:18]
        else:
            miss = str(miss_data)[:18]
        succ = f"{metrics.get('success_rate', 0.0):.1f}%"
        dmax = f"{metrics.get('max_distance', 0.0):.1f}m"
        print(f"{ts:<35} | {miss:<20} | {succ:<10} | {dmax}")
    print("══════════════════════════════════════════════════════════════════════════════════")


mostra_cronologia_run = mostra_cronologia
