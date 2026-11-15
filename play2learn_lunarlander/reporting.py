"""
Modulo di Reportistica Tecnica e Diagnostica.
Genera schede riassuntive dell'esperimento in PDF, HTML e Markdown per analizzare,
fare debug e documentare i risultati delle diverse funzioni di reward.
"""

from __future__ import annotations

import json
import base64
import datetime
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from play2learn_lunarlander.visualizer import plot_comparison
from play2learn_lunarlander.storage import get_default_runs_dir


def genera_pdf_report(
    pdf_path: Path | str,
    meta: dict[str, Any],
    plot_path: Path | str | None = None,
    autore: str = "Studente",
    titolo_esperimento: str = "Esperimento Reinforcement Learning",
    note_studente: str = "",
) -> Path:
    """
    Genera una scheda tecnica PDF vettoriale (2 pagine A4 ad alta risoluzione 300 DPI)
    con telemetria di volo, confronto tra successi e fallimenti, codice sorgente
    integrale della custom reward e note di debug.
    """
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    mission = meta.get("mission", {})
    mission_title = mission.get("title", "Lunar Lander")
    env_kwargs = mission.get("env_kwargs", {})
    metrics = meta.get("eval_metrics", {})
    success_rate = metrics.get("eval_success_rate", 0.0)
    fuel_used = metrics.get("mean_fuel", 0.0)
    fuel_succ = metrics.get("mean_fuel_success", fuel_used)
    fuel_fail = metrics.get("mean_fuel_failure", 0.0)
    succ_count = metrics.get("eval_successes", 0)
    fail_count = metrics.get("eval_crashes", 0) + metrics.get("eval_timeouts", 0)
    flight_steps = metrics.get("mean_length", 0.0)
    code_source = meta.get("reward_source_code", "# Codice reward non disponibile.")

    if not note_studente:
        note_studente = meta.get("notes") or meta.get("favorite_info", {}).get("notes") or "Nessuna annotazione inserita."

    with PdfPages(str(pdf_path)) as pdf:
        # =========================================================================
        # PAGINA 1: Intestazione, Tabella Telemetrica e Grafico di Apprendimento 2x2
        # =========================================================================
        fig1 = Figure(figsize=(8.27, 11.69), dpi=300)
        FigureCanvasAgg(fig1)

        # Header
        fig1.text(0.5, 0.955, "SCHEDA TECNICA ESPERIMENTO — REINFORCEMENT LEARNING",
                  ha="center", va="top", fontsize=13.5, fontweight="bold", color="#1a237e")
        fig1.text(0.5, 0.932, f"{titolo_esperimento}   |   Autore: {autore}   |   Data: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}",
                  ha="center", va="top", fontsize=10.0, color="#333333")
        fig1.text(0.5, 0.912, f"Missione: {mission_title}  (Gravità: {env_kwargs.get('gravity', -10.0)}, Vento: {env_kwargs.get('enable_wind', False)})",
                  ha="center", va="top", fontsize=9.2, style="italic", color="#555555")

        # Linea orizzontale
        ax_line = fig1.add_axes([0.06, 0.898, 0.88, 0.002])
        ax_line.axhline(0, color="#1a237e", lw=2)
        ax_line.axis("off")

        # 1. Tabella Telemetrica Valutazione
        fig1.text(0.06, 0.880, "1. Risultati della Valutazione (Analisi Successi e Fallimenti)",
                  fontsize=11.0, fontweight="bold", color="#283593")

        col_labels = ["Metrica Telemetrica", "Valore Sperimentale", "Riferimento Ideale", "Stato"]
        fuel_fail_val = f"{fuel_fail:.1f} unità" if fail_count > 0 else "0 fallimenti"
        fuel_fail_ref = "Nessun fallimento (0 un.)"
        fuel_fail_stato = "OTTIMO" if fail_count == 0 else "DIAGNOSI"

        table_vals = [
            ["Tasso di Allunaggio Riuscito", f"{success_rate:.1f}%", ">= 80.0%", "OBIETTIVO RAGGIUNTO" if success_rate >= 80 else "IN OTTIMIZZAZIONE"],
            ["Prove di Valutazione", f"{succ_count} successi / {fail_count} fallimenti", "Campione misto", "CAMPIONATO"],
            ["Carburante Medio Voli Riusciti", f"{fuel_succ:.1f} unità", "< 60.0 unità", "OTTIMO" if fuel_succ <= 60 else "STANDARD"],
            ["Carburante Medio Voli Falliti", fuel_fail_val, fuel_fail_ref, fuel_fail_stato],
            ["Durata Media Volo", f"{flight_steps:.0f} step ({flight_steps/50:.1f} s)", "150 - 350 step", "NOMINALE"],
        ]

        ax_table = fig1.add_axes([0.06, 0.705, 0.88, 0.165])
        ax_table.axis("off")
        col_widths = [0.31, 0.24, 0.25, 0.20]
        tab = ax_table.table(
            cellText=table_vals,
            colLabels=col_labels,
            colWidths=col_widths,
            loc="center",
            cellLoc="center",
            colColours=["#e8eaf6"] * 4,
        )
        tab.auto_set_font_size(False)
        tab.set_fontsize(8.5)
        tab.scale(1.0, 1.45)

        for (row, col), cell in tab.get_celld().items():
            cell.set_edgecolor("#c5cae9")
            if row == 0:
                cell.set_text_props(weight="bold", color="#1a237e")
            if col == 0 and row > 0:
                cell.set_text_props(horizontalalignment="left")
            else:
                cell.set_text_props(horizontalalignment="center")

        # 2. Inserimento Immagine Grafico 2x2 ad alta definizione
        fig1.text(0.06, 0.675, "2. Curve di Apprendimento e Telemetria di Volo",
                  fontsize=11.0, fontweight="bold", color="#283593")

        if plot_path and Path(plot_path).exists():
            img = plt.imread(str(plot_path))
            ax_img = fig1.add_axes([0.06, 0.04, 0.88, 0.62])
            ax_img.imshow(img, interpolation="antialiased")
            ax_img.axis("off")

        pdf.savefig(fig1)
        fig1.clear()

        # =========================================================================
        # PAGINA 2: Codice Sorgente Integrale della Custom Reward e Note di Debug
        # =========================================================================
        fig2 = Figure(figsize=(8.27, 11.69), dpi=300)
        FigureCanvasAgg(fig2)

        fig2.text(0.5, 0.955, "FORMULAZIONE MATEMATICA E NOTE DI DEBUG",
                  ha="center", va="top", fontsize=13.5, fontweight="bold", color="#1a237e")

        fig2.text(0.06, 0.915, "3. Codice Sorgente Completo della Funzione di Reward (Reward Shaping):",
                  fontsize=11.0, fontweight="bold", color="#283593")

        # Pulizia righe e calcolo altezza
        clean_code = "\n".join(line.rstrip() for line in code_source.strip().splitlines())
        code_lines = clean_code.splitlines()
        n_lines = len(code_lines)

        # Dimensionamento dinamico del font per visualizzare tutto il codice senza troncamenti
        if n_lines <= 42:
            font_size = 7.8
            line_spacing = 0.0120
        elif n_lines <= 62:
            font_size = 7.1
            line_spacing = 0.0102
        elif n_lines <= 85:
            font_size = 6.4
            line_spacing = 0.0088
        else:
            font_size = 5.8
            line_spacing = 0.0075

        code_height = min(0.64, max(0.20, n_lines * line_spacing + 0.03))
        y_code_top = 0.895
        y_code_bottom = y_code_top - code_height

        fig2.text(
            0.06, y_code_top, clean_code,
            fontsize=font_size, family="monospace", va="top",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#90caf9", lw=1.2)
        )

        # Sezione Note e Osservazioni di Debug dello Studente
        y_notes = max(0.035, y_code_bottom - 0.035)
        fig2.text(0.06, y_notes, "4. Note di Debug e Considerazioni Sperimentali:",
                  fontsize=10.5, fontweight="bold", color="#283593")

        debug_box_content = (
            f"NOTE INSERITE DALL'AUTORE:\n{note_studente}\n\n"
            "SPAZIO PER ANNOTAZIONI MANUALI / MODIFICHE FUTURE:\n"
            "• Quali termini matematici hanno migliorato la stabilità?\n"
            "• Quali comportamenti indesiderati sono stati osservati nei fallimenti?\n"
            "• Prossimi tentativi da sperimentare (pesi, velocità, soglie quota):"
        )

        fig2.text(
            0.06, y_notes - 0.02, debug_box_content,
            fontsize=8.4, va="top", family="sans-serif",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fffde7", edgecolor="#fbc02d", lw=1.3)
        )

        pdf.savefig(fig2)
        fig2.clear()

    return pdf_path


def genera_report_esperimento(
    run_or_fav: Any,
    autore: str = "Studente",
    titolo: str = "Report Esperimento Lunar Lander",
    note: str = "",
    output_dir: Path | str | None = None
) -> dict[str, Path]:
    """
    Genera un report tecnico completo in formato PDF, HTML e Markdown.
    Ogni esportazione viene salvata in una cartella univoca e dedicata (nessuna sovrascrittura),
    in modo che lo studente possa archiviare, consultare e modificare il materiale prodotto.
    """
    # 1. Risoluzione della directory della run
    source_dir: Path | None = None
    if isinstance(run_or_fav, dict):
        if "checkpoint_dir" in run_or_fav:
            source_dir = Path(run_or_fav["checkpoint_dir"])
        elif "path" in run_or_fav:
            source_dir = Path(run_or_fav["path"])
        elif "dir_path" in run_or_fav:
            source_dir = Path(run_or_fav["dir_path"])
        elif "run_dir" in run_or_fav:
            source_dir = Path(run_or_fav["run_dir"])
    elif isinstance(run_or_fav, (str, Path)):
        p = Path(run_or_fav)
        if p.exists() and p.is_dir():
            source_dir = p
        else:
            from play2learn_lunarlander.storage import find_run_by_timestamp
            source_dir = find_run_by_timestamp(str(run_or_fav))

    if not source_dir or not source_dir.exists():
        raise ValueError(f"[Play2Learn] Nessuna directory di run trovata per: {run_or_fav}")

    meta_file = source_dir / "run_metadata.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    else:
        meta = {}

    # Genera SEMPRE una cartella dedicata e univoca per evitare sovrascritture
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir:
        target_dir = Path(output_dir)
        if target_dir.name in ("runs", "favorites"):
            target_dir = target_dir / f"report_{timestamp}"
    else:
        target_dir = source_dir / f"report_{timestamp}"

    target_dir.mkdir(parents=True, exist_ok=True)

    mission = meta.get("mission", {})
    mission_title = mission.get("title", "Lunar Lander")
    mission_desc = mission.get("description", "Condizioni lunari standard.")
    env_kwargs = mission.get("env_kwargs", {})

    metrics = meta.get("eval_metrics", {})
    success_rate = metrics.get("eval_success_rate", 0.0)
    fuel_used = metrics.get("mean_fuel", 0.0)
    fuel_succ = metrics.get("mean_fuel_success", fuel_used)
    fuel_fail = metrics.get("mean_fuel_failure", 0.0)
    succ_count = metrics.get("eval_successes", 0)
    fail_count = metrics.get("eval_crashes", 0) + metrics.get("eval_timeouts", 0)
    flight_steps = metrics.get("mean_length", 0.0)

    train_info = meta.get("training", {})
    timesteps = train_info.get("total_timesteps", 0)
    elapsed = train_info.get("elapsed_seconds", 0.0)
    fps = train_info.get("fps_average", 0)

    code_source = meta.get("reward_source_code", "# Codice reward non disponibile.")

    if not note:
        note = meta.get("notes") or meta.get("favorite_info", {}).get("notes") or ""

    # Trova il grafico plot 2x2
    plot_path = source_dir / "reward_plot.png"
    if not plot_path.exists() and "plot_path" in meta:
        plot_path = Path(meta["plot_path"])

    # 2. Generazione del PDF vettoriale
    pdf_path = target_dir / "scheda_esperimento.pdf"
    try:
        genera_pdf_report(
            pdf_path=pdf_path,
            meta=meta,
            plot_path=plot_path if plot_path.exists() else None,
            autore=autore,
            titolo_esperimento=titolo,
            note_studente=note,
        )
    except Exception as e:
        print(f"[play2learn] Avviso generazione PDF: {e}")

    # Verifica se ci sono immagini per embeddarle in base64 nell'HTML
    plot_img_b64 = ""
    if plot_path.exists():
        plot_img_b64 = base64.b64encode(plot_path.read_bytes()).decode("ascii")

    fail_fuel_val = f"{fuel_fail:.1f} unità" if fail_count > 0 else "0 fallimenti"
    fail_fuel_ref = "Nessun fallimento (0 un.)"
    fail_fuel_esito = "🔍 Analisi criticità" if fail_count > 0 else "✅ Nessun fallimento"
    fail_fuel_badge = "Criticità" if fail_count > 0 else "Ottimo (0 fall.)"

    # 3. Generazione Markdown
    success_symbol = "\\ge"
    md_content = f"""# 📊 Scheda Tecnica Esperimento — Reinforcement Learning
**Titolo:** {titolo}  
**Autore:** {autore}  
**Ambiente:** Gymnasium `LunarLander-v3` | **Algoritmo:** PPO  
**Missione:** {mission_title}  
**Data:** {datetime.datetime.now().strftime("%d/%m/%Y %H:%M")}  

---

## 1. Configurazione e Parametri Fisici
* **Descrizione missione:** {mission_desc}
* **Fisica:** Gravità = `{env_kwargs.get('gravity', -10.0)}` | Vento = `{env_kwargs.get('enable_wind', False)}` (Potenza: `{env_kwargs.get('wind_power', 0.0)}`, Turbolenza: `{env_kwargs.get('turbulence_power', 0.0)}`)
* **Training:** {timesteps:,} step completati in {elapsed:.1f}s ({fps:,} step/s)

---

## 2. Telemetria di Valutazione (Successi e Fallimenti)
| Metrica di Valutazione | Risultato Sperimentale | Riferimento Nominale | Esito |
|---|---|---|---|
| **Percentuale Allunaggi Riusciti** | **{success_rate:.1f}%** | ${success_symbol} 80.0\\%$ | {'✅ Raggiunto' if success_rate >= 80 else '⚠️ In ottimizzazione'} |
| **Allunaggi Riusciti / Falliti** | **{succ_count} successi / {fail_count} fallimenti** | Campione misto | 🎯 Diagnostica completa |
| **Consumo Carburante (Voli Riusciti)** | **{fuel_succ:.1f} unità** | $< 60.0$ unità | {'✅ Ottimo' if fuel_succ <= 60 else 'ℹ️ Standard'} |
| **Consumo Carburante (Voli Falliti)** | **{fail_fuel_val}** | {fail_fuel_ref} | {fail_fuel_esito} |
| **Durata Media Volo** | **{flight_steps:.0f} step ({flight_steps/50:.1f} s)** | $150 - 350$ step | {'✅ Nominale' if 150 <= flight_steps <= 350 else 'ℹ️ Accettabile'} |

---

## 3. Codice Sorgente della Reward Function
```python
{code_source}
```

---

## 4. Note di Debug e Considerazioni Personali
{note if note else "*(Nessuna annotazione inserita. Puoi modificare questo file per aggiungere le tue osservazioni)*"}
"""

    img_tag = f'<div style="text-align:center; margin:20px 0;"><img src="data:image/png;base64,{plot_img_b64}" style="max-width:100%; border-radius:8px; box-shadow:0 2px 10px rgba(0,0,0,0.12);" alt="Dashboard Telemetrica"/></div>' if plot_img_b64 else ''

    html_content = f"""<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <title>{titolo}</title>
    <style>
        body {{
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
            line-height: 1.6;
            color: #222;
            max-width: 900px;
            margin: 30px auto;
            padding: 0 20px;
            background: #fafafa;
        }}
        .card {{
            background: #ffffff;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            margin-bottom: 20px;
        }}
        h1 {{ color: #1a237e; border-bottom: 2px solid #3f51b5; padding-bottom: 8px; margin-top: 0; font-size: 24px; }}
        h2 {{ color: #283593; margin-top: 24px; border-bottom: 1px solid #e0e0e0; padding-bottom: 4px; font-size: 18px; }}
        .header-meta {{
            display: flex;
            justify-content: space-between;
            background: #e8eaf6;
            padding: 12px 16px;
            border-radius: 6px;
            font-size: 13.5px;
            margin-bottom: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 16px 0;
            font-size: 13.5px;
        }}
        th, td {{
            padding: 9px 12px;
            border: 1px solid #e0e0e0;
            text-align: left;
        }}
        th {{ background: #f5f5f5; color: #333; }}
        .highlight {{ font-weight: bold; color: #2e7d32; }}
        pre {{
            background: #272822;
            color: #f8f8f2;
            padding: 16px;
            border-radius: 6px;
            overflow-x: auto;
            font-size: 13px;
        }}
        .badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 11.5px;
            font-weight: bold;
            background: #c8e6c9;
            color: #1b5e20;
        }}
        .notes-box {{
            background: #fffde7;
            padding: 14px 18px;
            border-left: 4px solid #fbc02d;
            border-radius: 4px;
            margin-top: 16px;
            font-size: 13.5px;
        }}
        @media print {{
            body {{ margin: 0; padding: 0; background: #fff; }}
            .card {{ box-shadow: none; padding: 0; }}
        }}
    </style>
</head>
<body>
    <div class="card">
        <h1>📊 Scheda Tecnica Esperimento: Reinforcement Learning</h1>
        <div class="header-meta">
            <div><strong>Titolo:</strong> {titolo} | <strong>Autore:</strong> {autore}</div>
            <div><strong>Missione:</strong> {mission_title} | <strong>Data:</strong> {datetime.datetime.now().strftime("%d/%m/%Y %H:%M")}</div>
        </div>

        <h2>1. Risultati della Valutazione (Successi e Fallimenti)</h2>
        <table>
            <thead>
                <tr>
                    <th>Parametro Telemetrico</th>
                    <th>Valore Misurato</th>
                    <th>Riferimento</th>
                    <th>Stato</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Tasso di Allunaggio Riuscito</strong></td>
                    <td class="highlight">{success_rate:.1f}%</td>
                    <td>&ge; 80.0%</td>
                    <td><span class="badge">{'Obiettivo Raggiunto' if success_rate >= 80 else 'In Ottimizzazione'}</span></td>
                </tr>
                <tr>
                    <td><strong>Prove di Verifica</strong></td>
                    <td>{succ_count} successi / {fail_count} fallimenti</td>
                    <td>Mix bilanciato</td>
                    <td><span class="badge">Diagnostica</span></td>
                </tr>
                <tr>
                    <td><strong>Carburante Voli Riusciti</strong></td>
                    <td>{fuel_succ:.1f} unità</td>
                    <td>&lt; 60.0 unità</td>
                    <td><span class="badge">Efficienza</span></td>
                </tr>
                <tr>
                    <td><strong>Carburante Voli Falliti</strong></td>
                    <td>{fail_fuel_val}</td>
                    <td>{fail_fuel_ref}</td>
                    <td><span class="badge">{fail_fuel_badge}</span></td>
                </tr>
                <tr>
                    <td><strong>Durata Media Volo</strong></td>
                    <td>{flight_steps:.0f} step ({flight_steps/50:.1f} s)</td>
                    <td>150 - 350 step</td>
                    <td><span class="badge">Nominale</span></td>
                </tr>
            </tbody>
        </table>

        {img_tag}

        <h2>2. Codice Sorgente della Reward Function</h2>
        <pre><code>{code_source}</code></pre>

        <h2>3. Note di Debug e Considerazioni Sperimentali</h2>
        <div class="notes-box">
            {note if note else "<em>Nessuna nota inserita. Modifica questo report per documentare i risultati dell'esperimento.</em>"}
        </div>
    </div>
</body>
</html>
"""

    md_path = target_dir / "report_esperimento.md"
    html_path = target_dir / "report_esperimento.html"

    md_path.write_text(md_content, encoding="utf-8")
    html_path.write_text(html_content, encoding="utf-8")

    print(f"✅ Report tecnico generato in: {target_dir.name}")
    print(f"   • PDF:      {pdf_path.name}")
    print(f"   • HTML:     {html_path.name}")
    print(f"   • Markdown: {md_path.name}")

    return {
        "pdf_path": pdf_path,
        "markdown_path": md_path,
        "md_path": md_path,
        "html_path": html_path,
        "export_dir": target_dir,
    }


# Alias retrocompatibile
genera_report_maturita = genera_report_esperimento


def confronta_run(
    runs_or_favs: list[Any] | tuple[Any, ...] | str,
    output_image_path: Path | str | None = None
) -> dict[str, Any]:
    """
    Confronta 2 o più run generando un grafico comparativo e una tabella di sintesi a colonne.
    Funziona con:
      - Oggetti LabResult restituite da run_lab(...)
      - Stringhe o liste di timestamp (es. '20261115_210530, 20261115_212000')
      - Parole chiave come 'latest, previous' per confrontare al volo le ultime due run
      - Percorsi a cartelle di run
    """
    from play2learn_lunarlander.storage import find_run_by_timestamp, mostra_cronologia_run

    if not runs_or_favs:
        print("[play2learn] Nessuna run fornita per il confronto.")
        print("💡 Suggerimento: inserisci due timestamp separati da virgola (es. 'latest, previous').")
        return {}

    # Se passata come stringa separata da virgola o punto e virgola
    if isinstance(runs_or_favs, str):
        raw_items = [x.strip() for x in runs_or_favs.replace(";", ",").split(",") if x.strip()]
    elif isinstance(runs_or_favs, (list, tuple)):
        raw_items = list(runs_or_favs)
    else:
        raw_items = [runs_or_favs]

    runs_data = []
    unresolved = []

    for item in raw_items:
        source_dir = None
        if isinstance(item, dict):
            if "checkpoint_dir" in item:
                source_dir = Path(item["checkpoint_dir"])
            elif "dir_path" in item:
                source_dir = Path(item["dir_path"])
            elif "path" in item:
                source_dir = Path(item["path"])
            elif "run_dir" in item:
                source_dir = Path(item["run_dir"])
            else:
                runs_data.append(item)
                continue
        elif isinstance(item, (str, Path)):
            p = Path(item)
            if p.exists() and p.is_dir() and (p / "run_metadata.json").exists():
                source_dir = p
            else:
                # Risoluzione automatica da timestamp o alias (es. "latest", "previous")
                resolved = find_run_by_timestamp(str(item))
                if resolved and resolved.exists():
                    source_dir = resolved
                else:
                    unresolved.append(str(item))

        if source_dir and source_dir.exists():
            meta_path = source_dir / "run_metadata.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    metrics = meta.get("eval_metrics", {})
                    mission = meta.get("mission", {})

                    import re
                    ts_m = re.search(r"(\d{8}_\d{6})", source_dir.name)
                    ts_display = ts_m.group(1) if ts_m else source_dir.name
                    run_name = meta.get("favorite_info", {}).get("display_name") or meta.get("tag") or mission.get("title") or source_dir.name

                    runs_data.append({
                        "name": run_name,
                        "timestamp": ts_display,
                        "display_label": f"{run_name}\n({ts_display})",
                        "eval_success_rate": float(metrics.get("eval_success_rate", 0.0)),
                        "eval_mean_fuel": float(metrics.get("mean_fuel", 0.0)),
                        "eval_mean_length": float(metrics.get("mean_length", 0.0)),
                        "mission": mission.get("title", ""),
                        "dir_name": source_dir.name,
                    })
                except Exception:
                    continue

    if unresolved:
        print(f"⚠️ Attenzione: impossibile trovare le seguenti run dal timestamp inserito: {', '.join(unresolved)}")
        print("💡 Ecco l'elenco delle run disponibili con i relativi timestamp da copiare:")
        mostra_cronologia_run()

    if not runs_data:
        print("[play2learn] Impossibile estrarre i dati per il confronto. Verifica i timestamp forniti.")
        return {}

    save_img = Path(output_image_path) if output_image_path else get_default_runs_dir() / "confronto_run.png"
    plot_comparison(runs_data, save_img)

    # Visualizzazione tabella comparativa arricchita con colonna Timestamp
    try:
        from IPython.display import HTML, Image, display
        rows = []
        for r in runs_data:
            succ = r.get("eval_success_rate", 0.0)
            badge_color = "#2e7d32" if succ >= 80.0 else ("#f57c00" if succ >= 50.0 else "#c62828")
            bg_badge = "#e8f5e9" if succ >= 80.0 else ("#fff3e0" if succ >= 50.0 else "#ffebee")
            rows.append(
                f"<tr>"
                f"<td style='padding:8px 12px; font-family:monospace; font-weight:bold; color:#1a237e; background:#f8f9fa;'>{r.get('timestamp', '')}</td>"
                f"<td style='padding:8px 12px; font-weight:bold;'>{r.get('name')}</td>"
                f"<td style='padding:8px 12px;'>{r.get('mission', '')}</td>"
                f"<td style='padding:8px 12px; text-align:center;'><span style='background:{bg_badge}; color:{badge_color}; padding:3px 8px; border-radius:12px; font-weight:bold;'>{succ:.1f}%</span></td>"
                f"<td style='padding:8px 12px; text-align:center;'>{r.get('eval_mean_fuel', 0):.1f} un.</td>"
                f"<td style='padding:8px 12px; text-align:center;'>{r.get('eval_mean_length', 0):.0f} step</td>"
                f"</tr>"
            )

        html = f"""
        <div style="font-family:'Segoe UI', sans-serif; margin:16px 0;">
            <h3 style="color:#0d47a1; margin-bottom:8px;">📊 Confronto Esperimenti di Apprendimento (Analisi Comparativa)</h3>
            <table style="border-collapse:collapse; width:100%; border:1px solid #ccc; text-align:left; font-size:13.5px;">
                <thead style="background:#e3f2fd; color:#0d47a1;">
                    <tr>
                        <th style="padding:10px 12px;">Timestamp</th>
                        <th style="padding:10px 12px;">Esperimento / Versione</th>
                        <th style="padding:10px 12px;">Missione</th>
                        <th style="padding:10px 12px; text-align:center;">% Allunaggio</th>
                        <th style="padding:10px 12px; text-align:center;">Carburante Medio</th>
                        <th style="padding:10px 12px; text-align:center;">Durata Media</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(rows)}
                </tbody>
            </table>
        </div>
        """
        display(HTML(html))
        if save_img.exists():
            display(Image(filename=str(save_img)))
    except Exception:
        print("\n" + "=" * 78)
        print("  📊 CONFRONTO ESPERIMENTI (TABELLA COMPARATIVA)")
        print("=" * 78)
        print(f"{'TIMESTAMP':<18} | {'NOME RUN':<24} | {'SUCCESSO':<10} | {'CARBURANTE':<12} | {'DURATA':<10}")
        print("-" * 78)
        for r in runs_data:
            print(f"{r.get('timestamp', ''):<18} | {r.get('name', '')[:23]:<24} | {r.get('eval_success_rate', 0):<9.1f}% | {r.get('eval_mean_fuel', 0):<12.1f} | {r.get('eval_mean_length', 0):<10.0f}")
        print("=" * 78 + "\n")

    return {"runs": runs_data, "comparison_plot": save_img}


def genera_report_da_timestamp(
    timestamp: str = "latest",
    tipo: str = "Base",
    autore: str = "Studente",
    titolo: str | None = None,
    note: str = "",
    n_extra_episodes: int = 35,
    target_success_videos: int = 4,
    target_failure_videos: int = 4,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """
    Genera la reportistica tecnica completa (Scheda Tecnica PDF 300 DPI, HTML, Markdown)
    a partire dal timestamp (es. '20261115_210530') o dal nome di una run passata.

    Parametri:
      - timestamp: stringa del timestamp (es. '20261115_210530') oppure 'latest' per la run più recente.
      - tipo: 'Base' (genera il report con i dati archiviati) oppure
              'Avanzata' (ricarica il modello, esegue nuove run di verifica e registra più video di test).
      - autore: nome dello studente o gruppo.
      - titolo: titolo personalizzato dell'esperimento (opzionale).
      - note: note personali e considerazioni di debug.
      - n_extra_episodes: numero di episodi di sweep per la valutazione avanzata (default 35).
      - target_success_videos: video di successo da registrare in modalità avanzata (default 4).
      - target_failure_videos: video di fallimento da registrare in modalità avanzata (default 4).
      - output_dir: cartella di salvataggio (opzionale, default sottocartella timestampata della run).
    """
    from play2learn_lunarlander.storage import find_run_by_timestamp, mostra_cronologia_run
    from play2learn_lunarlander.visualizer import display_notebook_summary

    print("\n" + "=" * 78)
    print("  🕒 GENERATORE DI REPORT DA TIMESTAMP — LUNAR LANDER")
    print("=" * 78)

    # 1. Ricerca della directory della run dal timestamp
    run_dir = find_run_by_timestamp(timestamp)
    if run_dir is None:
        print(f"❌ Nessuna run trovata corrispondente a: '{timestamp}'")
        print("💡 Ecco l'elenco delle run disponibili con i relativi timestamp da copiare:\n")
        mostra_cronologia_run()
        return {}

    print(f"🔍 Run individuata con successo: runs/{run_dir.name}")

    meta_file = run_dir / "run_metadata.json"
    if meta_file.exists():
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    else:
        meta = {}

    mission = meta.get("mission", {})
    mission_title = mission.get("title", "Lunar Lander")
    env_kwargs = mission.get("env_kwargs", {})

    if not titolo:
        titolo = f"Esperimento {mission_title} — Reinforcement Learning"
    if not note:
        note = meta.get("notes") or meta.get("favorite_info", {}).get("notes") or ""

    is_avanzata = "avanzat" in str(tipo).lower()

    # 2. Se modalità Avanzata: esegui nuove run di test e registra più video
    if is_avanzata:
        print("\n🚀 [Modalità AVANZATA attiva] Avvio sessione di verifica estesa per registrare e salvare più video...")
        model_path = None
        for cand in ("best_model.zip", "ppo_lunar_lander.zip"):
            if (run_dir / cand).exists():
                model_path = run_dir / cand
                break
        if model_path is None:
            zips = list(run_dir.glob("*.zip"))
            if zips:
                model_path = zips[0]

        if model_path is not None:
            try:
                from stable_baselines3 import PPO
                from play2learn_lunarlander.evaluator import evaluate_and_record
                from play2learn_lunarlander.config import ENV_ID

                print(f"📦 Caricamento modello salvato da: {model_path.name}")
                model = PPO.load(str(model_path))

                # Ricostruzione della funzione custom_reward dalla run
                reward_fn = None
                reward_file = run_dir / "reward_function.py"
                if reward_file.exists():
                    try:
                        code_txt = reward_file.read_text(encoding="utf-8")
                        scope = {}
                        exec(code_txt, scope)
                        reward_fn = scope.get("custom_reward")
                    except Exception:
                        reward_fn = None

                if reward_fn is None and "reward_source_code" in meta:
                    try:
                        scope = {}
                        exec(meta["reward_source_code"], scope)
                        reward_fn = scope.get("custom_reward")
                    except Exception:
                        reward_fn = None

                if reward_fn is None:
                    reward_fn = lambda obs, act, s, c, t: 0.0

                env_id = meta.get("env_id") or ENV_ID
                videos_dir = run_dir / "videos"
                videos_dir.mkdir(parents=True, exist_ok=True)

                print(f"🎬 Registrazione di nuovi video di test in: runs/{run_dir.name}/videos/")
                adv_stats = evaluate_and_record(
                    model=model,
                    env_id=env_id,
                    reward_fn=reward_fn,
                    video_dir=videos_dir,
                    target_successes=target_success_videos,
                    target_failures=target_failure_videos,
                    sweep_timeout_seconds=8.0,
                    min_eval_episodes=n_extra_episodes,
                    max_eval_episodes=n_extra_episodes + 30,
                    seed=int(datetime.datetime.now().timestamp()) % 10000,
                    env_kwargs=env_kwargs,
                    create_gif=False,
                    filename_prefix="avanzato",
                )

                meta["eval_metrics_avanzata"] = adv_stats
                meta["eval_metrics"] = adv_stats
                meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"✅ Sessione avanzata completata: {adv_stats.get('eval_success_rate', 0.0):.1f}% di successo su {adv_stats.get('total_evaluated_episodes', 0)} voli!")
            except Exception as e:
                print(f"⚠️ Impossibile completare la sessione estesa ({e}). Utilizzo dati pregressi.")
        else:
            print(f"ℹ️ File pesi del modello (.zip) non trovato in '{run_dir.name}'. Verranno utilizzati i video e i dati esistenti.")
    else:
        print("📄 [Modalità Base] Generazione report dai dati e video archiviati per la run.")

    # 3. Generazione report (PDF 300 DPI, HTML, MD)
    rep_files = genera_report_esperimento(
        run_or_fav=run_dir,
        autore=autore,
        titolo=titolo,
        note=note,
        output_dir=output_dir,
    )

    # 4. Visualizzazione inline dei video e del grafico nel notebook
    plot_path = run_dir / "reward_plot.png"
    vids_dir = run_dir / "videos"
    all_videos = sorted(vids_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True) if vids_dir.exists() else []

    print("\n" + "=" * 78)
    print("  📊 TELEMETRIA DI VOLO E VIDEO SALVATI:")
    print("=" * 78)
    display_notebook_summary(
        plot_path=plot_path if plot_path.exists() else "",
        videos=all_videos,
        max_videos=6 if is_avanzata else 3,
    )

    print("\n" + "=" * 78)
    print(f"  ✨ REPORT ({tipo.upper()}) PRONTO E SALVATO CON SUCCESSO!")
    print(f"  • Scheda Tecnica PDF (300 DPI): {rep_files.get('pdf_path')}")
    print(f"  • Report HTML Interattivo:     {rep_files.get('html_path')}")
    print(f"  • Documento Markdown:          {rep_files.get('md_path')}")
    print("=" * 78 + "\n")

    return {
        "run_dir": run_dir,
        "meta": meta,
        "rep_files": rep_files,
        "videos": all_videos,
    }
