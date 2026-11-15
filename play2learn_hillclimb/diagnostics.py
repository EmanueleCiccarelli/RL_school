"""
Modulo di Diagnostica Comportamentale Didattica per Hill Climb Racing.
Analizza distanza percorsa, velocità, airtime, ribaltamenti e blocchi in salita.
"""

from __future__ import annotations

from typing import Any


def diagnose_hill_climb_behavior(eval_stats: dict[str, Any]) -> dict[str, Any]:
    """Analizza le metriche aggregate di valutazione e produce consigli didattici mirati."""
    success_rate = float(eval_stats.get("eval_success_rate", 0.0))
    mean_dist = float(eval_stats.get("mean_distance", 0.0))
    max_dist = float(eval_stats.get("max_distance_achieved", mean_dist))
    crashes = int(eval_stats.get("eval_crashes", 0))
    timeouts = int(eval_stats.get("eval_timeouts", 0))
    total_ep = max(1, int(eval_stats.get("episodes", 20)))
    airtime_pct = float(eval_stats.get("mean_airtime_pct", 0.0))

    crash_rate = (crashes / total_ep) * 100.0
    timeout_rate = (timeouts / total_ep) * 100.0

    consigli = []

    if success_rate >= 80.0:
        valutazione = "ECCELLENTE (Obiettivo Raggiunto)"
        sintesi = (
            f"L'agente ha completato con successo l'allungo collinare nell'{success_rate:.0f}% dei casi, "
            f"raggiungendo una distanza massima di {max_dist:.1f} metri con traiettorie veloci e stabili."
        )
    elif max_dist >= 600.0:
        valutazione = "BUONO (Quasi al Traguardo)"
        sintesi = (
            f"L'agente guida con buona velocità e raggiunge {max_dist:.1f} m, ma alcuni salti improvvisi o "
            "avvallamenti ripidi provocano ancora perdite di assetto."
        )
    else:
        valutazione = "IN CORSO DI APPRENDIMENTO"
        sintesi = (
            f"L'agente si ferma a una media di {mean_dist:.1f} m. "
            "Occorre bilanciare la propensione ad accelerare con il mantenimento dell'aderenza al suolo."
        )

    if crash_rate > 35.0:
        consigli.append(
            "⚠️ ATTENZIONE AI RIBALTAMENTI: Il veicolo subisce troppi impatti della testa al suolo. "
            "Aggiungi una penalità per angoli elevati (|angle| > 0.6 rad) o quando la vettura si impenna."
        )
    if timeout_rate > 30.0:
        consigli.append(
            "⏳ BLOCCO NELLE SALITE: Il veicolo perde slancio e resta bloccato sul fondo delle valli. "
            "Premia la velocità in avanti (vx > 2.0 m/s) o penalizza la stasi con ruote a terra."
        )
    if airtime_pct > 40.0:
        consigli.append(
            "🚀 TROPPO TEMPO IN ARIA: Volare a lungo riduce la trazione motrice delle ruote. "
            "Premia l'avanzamento con entrambe le ruote a contatto con il terreno (on_ground)."
        )

    if not consigli:
        consigli.append("✅ Assetto e velocità nominali. Nessuna criticità rilevata.")

    formatted_text = f"DIAGNOSI GUIDA: {valutazione}\n{sintesi}\n\nSUGGERIMENTI:\n" + "\n".join(consigli)

    return {
        "valutazione": valutazione,
        "sintesi": sintesi,
        "consigli": consigli,
        "crash_rate": crash_rate,
        "timeout_rate": timeout_rate,
        "formatted_text": formatted_text,
    }
