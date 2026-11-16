"""
Modulo Diagnostica di Guida per Play2Learn Hill Climb Racing.
Fornisce feedback pedagogico strutturato e suggerimenti mirati ("Flight Coach")
per guidare lo studente nel perfezionamento della funzione di reward.
"""

from __future__ import annotations

from typing import Any


def diagnose_hill_climb_behavior(eval_metrics: dict[str, Any]) -> str:
    """
    Analizza i risultati della sessione di test e genera una diagnosi dettagliata
    in italiano con raccomandazioni pratiche di reward shaping.
    """
    succ = float(eval_metrics.get("success_rate", eval_metrics.get("eval_success_rate", 0.0)))
    crash = float(eval_metrics.get("crash_rate", 0.0))
    tout = float(eval_metrics.get("timeout_rate", 0.0))
    mean_dist = float(eval_metrics.get("mean_distance", 0.0))
    max_dist = float(eval_metrics.get("max_distance", 0.0))
    mean_spd = float(eval_metrics.get("mean_speed", 0.0))
    airtime = float(eval_metrics.get("mean_airtime_pct", 0.0))

    lines: list[str] = [
        "═════════════════════════════════════════════════════════════════════",
        " 🏎️  DIAGNOSTICA DEL COACH DI GUIDA (HILL CLIMB RACING)",
        "═════════════════════════════════════════════════════════════════════",
        f" • Percentuale di Successo: {succ:.1f}%",
        f" • Schianti / Ribaltamenti: {crash:.1f}%",
        f" • Blocchi / Tempo Scaduto:  {tout:.1f}%",
        f" • Distanza Media Percorsa: {mean_dist:.1f} m  (Record: {max_dist:.1f} m)",
        f" • Velocità Media Massima:  {mean_spd:.2f} m/s",
        f" • Tempo Trascorso in Volo: {airtime:.1f}%",
        "─────────────────────────────────────────────────────────────────────",
        " 📋 ANALISI E SUGGERIMENTI PEDAGOGICI:",
    ]

    if succ >= 80.0:
        lines.append(
            " 🌟 ECCELLENTE! Il pilota ha compreso perfettamente come modulare il gas,\n"
            "    mantenendo la stabilità sulle creste e raggiungendo la cima nella quasi totalità dei tracciati."
        )
    elif crash > 45.0:
        lines.append(
            " ⚠️ TROPPI RIBALTAMENTI (Schianto Testa-Suolo):\n"
            "    L'agente preme troppo il gas e fa impennare la vettura sui dossi o sulle salite ripide.\n"
            "    💡 SUGGERIMENTO: Aggiungi una penalità quadratica sull'inclinazione quando abs(angle) > 0.4,\n"
            "       oppure penalizza la velocità angolare eccessiva quando le ruote perdono contatto col suolo."
        )
    elif tout > 45.0:
        lines.append(
            " ⚠️ VETTURA BLOCCATA NELLE VALLI O IN SALITA (Timeout):\n"
            "    Il veicolo non accumula sufficiente velocità per superare le rampe e finisce per slittare sul posto.\n"
            "    💡 SUGGERIMENTO: Aumenta il premio proporzionale alla velocità in avanti (vx > 0)\n"
            "       e incentiva la continuità della distanza percorsa rispetto al massimo precedente."
        )
    elif mean_dist < 400.0:
        lines.append(
            " 📈 AVANZAMENTO INIZIALE MODESTO:\n"
            "    L'agente impara a sopravvivere ma non spinge abbastanza a fondo per raggiungere il traguardo.\n"
            "    💡 SUGGERIMENTO: Assegna un premio progressivo all'avanzamento (es. dist_x * 0.1)\n"
            "       e un bonus cospicuo (+100) quando is_success è True."
        )
    else:
        lines.append(
            " 👍 BUONA GUIDA DI BASE:\n"
            "    L'auto supera la prima metà della collina ma fatica nei passaggi più ripidi.\n"
            "    💡 SUGGERIMENTO: Affina il bilanciamento tra velocità di salita e stabilità dell'assetto."
        )

    lines.append("═════════════════════════════════════════════════════════════════════")
    return "\n".join(lines)
