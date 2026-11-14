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
    Genera una scheda tecnica PDF vettoriale (2 pagine A4) con telemetria di volo,
    confronto tra successi e fallimenti, codice sorgente della reward e note di debug.
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
        # PAGINA 1: Intestazione, Tabella Telemetrica Successi/Fallimenti e Grafico di Apprendimento
        fig1 = plt.figure(figsize=(8.27, 11.69), dpi=150)

        # Header
        fig1.text(0.5, 0.95, "SCHEDA TECNICA ESPERIMENTO — REINFORCEMENT LEARNING",
                  ha="center", va="top", fontsize=14, fontweight="bold", color="#1a237e")
        fig1.text(0.5, 0.925, f"{titolo_esperimento}   |   Autore: {autore}   |   Data: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}",
                  ha="center", va="top", fontsize=10.5, color="#333333")
        fig1.text(0.5, 0.905, f"Missione: {mission_title}  (Gravità: {env_kwargs.get('gravity', -10.0)}, Vento: {env_kwargs.get('enable_wind', False)})",
                  ha="center", va="top", fontsize=9.5, style="italic", color="#555555")

        # Linea orizzontale
        ax_line = fig1.add_axes([0.08, 0.89, 0.84, 0.002])
        ax_line.axhline(0, color="#1a237e", lw=2)
        ax_line.axis("off")

        # Tabella Telemetrica Valutazione
        fig1.text(0.08, 0.875, "1. Risultati della Valutazione (Analisi Successi e Fallimenti)",
                  fontsize=11.5, fontweight="bold", color="#283593")

        col_labels = ["Metrica Telemetrica", "Valore Sperimentale", "Riferimento Ideale", "Stato"]
        table_vals = [
            ["Tasso di Allunaggio Riuscito", f"{success_rate:.1f}%", ">= 80.0%", "OBIETTIVO RAGGIUNTO" if success_rate >= 80 else "IN OTTIMIZZAZIONE"],
            ["Prove di Valutazione", f"{succ_count} successi / {fail_count} fallimenti", "Campione misto", "CAMPIONATO"],
            ["Carburante Medio Voli Riusciti", f"{fuel_succ:.1f} unità", "< 60.0 unità", "OTTIMO" if fuel_succ <= 60 else "STANDARD"],
            ["Carburante Medio Voli Falliti", f"{fuel_fail:.1f} unità", "N/A", "DIAGNOSI"],
            ["Durata Media Volo", f"{flight_steps:.0f} step ({flight_steps/50:.1f} s)", "150 - 350 step", "NOMINALE"],
        ]

        ax_table = fig1.add_axes([0.08, 0.69, 0.84, 0.17])
        ax_table.axis("off")
        tab = ax_table.table(
            cellText=table_vals,
            colLabels=col_labels,
            loc="center",
            cellLoc="left",
            colColours=["#e8eaf6"] * 4
        )
        tab.auto_set_font_size(False)
        tab.set_fontsize(9.5)
        tab.scale(1.0, 1.4)

        # Inserimento Immagine Grafico 2x2
        fig1.text(0.08, 0.67, "2. Curve di Apprendimento e Telemetria di Volo",
                  fontsize=11.5, fontweight="bold", color="#283593")

        if plot_path and Path(plot_path).exists():
            img = plt.imread(str(plot_path))
            ax_img = fig1.add_axes([0.08, 0.08, 0.84, 0.57])
            ax_img.imshow(img)
            ax_img.axis("off")

        pdf.savefig(fig1)
        plt.close(fig1)

        # PAGINA 2: Codice Sorgente Reward Function e Note di Debug dello Studente
        fig2 = plt.figure(figsize=(8.27, 11.69), dpi=150)

        fig2.text(0.5, 0.95, "FORMULAZIONE MATEMATICA E NOTE DI DEBUG",
                  ha="center", va="top", fontsize=13.5, fontweight="bold", color="#1a237e")

        fig2.text(0.08, 0.91, "3. Codice Sorgente della Funzione di Reward (Reward Shaping):",
                  fontsize=11, fontweight="bold", color="#283593")

        # Box per il codice sorgente
        lines_code = code_source.split("\n")
        if len(lines_code) > 42:
            display_code = "\n".join(lines_code[:42]) + "\n# ... (codice continuo nel file reward_function.py)"
        else:
            display_code = code_source

        fig2.text(
            0.08, 0.89, display_code,
            fontsize=8.2, family="monospace", va="top",
            bbox=dict(boxstyle="round,pad=0.7", facecolor="#f5f5f5", edgecolor="#bbbbbb", lw=1.2)
        )

        # Sezione Note e Osservazioni di Debug dello Studente
        y_pos = 0.35
        fig2.text(0.08, y_pos, "4. Note di Debug e Considerazioni Sperimentali:",
                  fontsize=11, fontweight="bold", color="#283593")

        debug_box_content = (
            f"NOTE INSERITE DALL'AUTORE:\n{note_studente}\n\n"
            "SPAZIO PER ANNOTAZIONI MANUALI / MODIFICHE FUTURE:\n"
            "• Quali termini matematici hanno migliorato la stabilità?\n"
            "• Quali comportamenti indesiderati sono stati osservati nei fallimenti?\n"
            "• Prossimi tentativi da sperimentare (pesi, velocità, soglie quota):"
        )

        fig2.text(
            0.08, y_pos - 0.02, debug_box_content,
            fontsize=9.0, va="top", family="sans-serif",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#fffde7", edgecolor="#fbc02d", lw=1.3)
        )

        pdf.savefig(fig2)
        plt.close(fig2)

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
    elif isinstance(run_or_fav, (str, Path)):
        source_dir = Path(run_or_fav)

    if not source_dir or not source_dir.exists():
        raise ValueError(f"[Play2Learn] Directory della run non valida: {run_or_fav}")

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
| **Consumo Carburante (Voli Falliti)** | **{fuel_fail:.1f} unità** | N/A | 🔍 Analisi criticità |
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
                    <td>{fuel_fail:.1f} unità</td>
                    <td>N/A</td>
                    <td><span class="badge">Criticità</span></td>
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
        "html_path": html_path,
        "export_dir": target_dir,
    }


# Alias retrocompatibile
genera_report_maturita = genera_report_esperimento


def confronta_run(
    runs_or_favs: list[Any],
    output_image_path: Path | str | None = None
) -> dict[str, Any]:
    """
    Confronta 2 o più run generando un grafico comparativo e una tabella
    di sintesi a colonne. Strumento di debug per verificare se le modifiche alla reward
    hanno migliorato il successo o ridotto i consumi.
    """
    if not runs_or_favs:
        print("[play2learn] Nessuna run fornita per il confronto.")
        return {}

    runs_data = []
    for item in runs_or_favs:
        source_dir = None
        if isinstance(item, dict):
            if "checkpoint_dir" in item:
                source_dir = Path(item["checkpoint_dir"])
            elif "dir_path" in item:
                source_dir = Path(item["dir_path"])
            elif "path" in item:
                source_dir = Path(item["path"])
            else:
                runs_data.append(item)
                continue
        elif isinstance(item, (str, Path)):
            source_dir = Path(item)

        if source_dir and source_dir.exists():
            meta_path = source_dir / "run_metadata.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                runs_data.append({
                    "name": meta.get("favorite_info", {}).get("display_name") or meta.get("run_id", source_dir.name),
                    "eval_success_rate": meta.get("eval_metrics", {}).get("eval_success_rate", 0.0),
                    "eval_mean_fuel": meta.get("eval_metrics", {}).get("mean_fuel", 0.0),
                    "eval_mean_length": meta.get("eval_metrics", {}).get("mean_length", 0.0),
                    "mission": meta.get("mission", {}).get("title", ""),
                })

    if not runs_data:
        print("[play2learn] Impossibile estrarre i dati per il confronto.")
        return {}

    save_img = Path(output_image_path) if output_image_path else get_default_runs_dir() / "confronto_run.png"
    plot_comparison(runs_data, save_img)

    # Visualizzazione tabella
    try:
        from IPython.display import HTML, Image, display
        rows = []
        for r in runs_data:
            rows.append(
                f"<tr>"
                f"<td style='padding:8px 12px; font-weight:bold;'>{r.get('name')}</td>"
                f"<td style='padding:8px 12px;'>{r.get('mission', '')}</td>"
                f"<td style='padding:8px 12px; font-weight:bold; color:#2e7d32;'>{r.get('eval_success_rate', 0):.0f}%</td>"
                f"<td style='padding:8px 12px;'>{r.get('eval_mean_fuel', 0):.1f} un.</td>"
                f"<td style='padding:8px 12px;'>{r.get('eval_mean_length', 0):.0f} step</td>"
                f"</tr>"
            )

        html = f"""
        <div style="font-family:sans-serif; margin:16px 0;">
            <h3 style="color:#0d47a1; margin-bottom:8px;">📊 Confronto Esperimenti di Apprendimento</h3>
            <table style="border-collapse:collapse; width:100%; border:1px solid #ccc; text-align:left; font-size:13.5px;">
                <thead style="background:#e3f2fd; color:#0d47a1;">
                    <tr>
                        <th style="padding:10px 12px;">Esperimento / Versione</th>
                        <th style="padding:10px 12px;">Missione</th>
                        <th style="padding:10px 12px;">% Successo</th>
                        <th style="padding:10px 12px;">Carburante Medio</th>
                        <th style="padding:10px 12px;">Durata Media</th>
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
        print("\n" + "=" * 70)
        print("  📊 CONFRONTO ESPERIMENTI")
        print("=" * 70)
        for r in runs_data:
            print(f"{r.get('name', ''):<25} | {r.get('eval_success_rate', 0):.0f}% succ | {r.get('eval_mean_fuel', 0):.1f} fuel | {r.get('eval_mean_length', 0):.0f} step")
        print("=" * 70 + "\n")

    return {"runs": runs_data, "comparison_plot": save_img}
