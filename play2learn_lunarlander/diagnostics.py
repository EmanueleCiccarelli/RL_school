"""
Modulo 'Flight Coach' - Diagnostica didattica intelligente del comportamento dell'agente.
Fornisce spiegazioni logico-matematiche chiare sulle anomalie di apprendimento (Reward Hacking,
trappole di hovering o kamikaze) e suggerisce come correggere la formula di reward.
"""

from __future__ import annotations
from typing import Any
import numpy as np

from play2learn_lunarlander.agent import RewardTrackerCallback
from play2learn_lunarlander.config import HELIPAD_X_MAX


def diagnose_agent_behavior(tracker: RewardTrackerCallback) -> dict[str, Any]:
    """
    Analizza gli ultimi episodi completati dall'agente per rilevare eventuali
    comportamenti anomali o subottimali e formulare consigli didattici mirati.

    Restituisce un dizionario strutturato con metriche e testo descrittivo.
    """
    n = len(tracker.outcomes)
    if n == 0:
        return {
            "status": "nessun_dato",
            "title": "Nessun dato disponibile",
            "message": "Nessun episodio completato.",
            "suggestion": "Verifica che l'addestramento non sia stato interrotto subito.",
            "formatted_text": "Nessun episodio completato per la diagnostica.",
            "metrics": {},
        }

    tail = min(50, n)
    recent_outcomes = tracker.outcomes[-tail:]
    succ_rate = (recent_outcomes.count("success") / tail) * 100.0
    crash_rate = (recent_outcomes.count("crash") / tail) * 100.0
    time_rate = (recent_outcomes.count("timeout") / tail) * 100.0

    mean_len = float(np.mean(tracker.episode_lengths[-tail:]))
    mean_fuel = float(np.mean(tracker.fuel_used_list[-tail:]))
    mean_ground = float(np.mean(tracker.ground_steps_list[-tail:]))

    # Calcola le medie fisiche relative agli schianti strettamente sugli episodi falliti (per evitare masking con i successi)
    crash_indices = [i for i, o in enumerate(recent_outcomes) if o == "crash"]
    recent_x = tracker.final_x_list[-tail:]
    recent_vy = tracker.final_vy_list[-tail:]
    recent_ang = tracker.final_angle_list[-tail:] if tracker.final_angle_list else []

    if crash_indices:
        mean_x = float(np.mean([recent_x[i] for i in crash_indices]))
        mean_vy = float(np.mean([recent_vy[i] for i in crash_indices]))
        mean_angle = float(np.mean([recent_ang[i] for i in crash_indices])) if recent_ang else 0.0
    else:
        mean_x = float(np.mean(recent_x))
        mean_vy = float(np.mean(recent_vy))
        mean_angle = float(np.mean(recent_ang)) if recent_ang else 0.0

    metrics = {
        "tail": tail,
        "success_rate": succ_rate,
        "crash_rate": crash_rate,
        "timeout_rate": time_rate,
        "mean_length": mean_len,
        "mean_fuel": mean_fuel,
        "mean_ground_steps": mean_ground,
        "mean_abs_x": mean_x,
        "mean_abs_vy": mean_vy,
        "mean_abs_angle": mean_angle,
    }

    # Determinazione diagnosi principale
    if succ_rate >= 75.0:
        status = "ottimo"
        title = "ALLUNAGGIO PERFETTO! 🎯"
        msg = "L'agente ha imparato ad atterrare con sicurezza e centrare la piazzola."
        sugg = "Ottimo lavoro! Puoi provare ora ad ottimizzare il consumo di carburante o passare a una missione con vento/gravità estrema."
    elif time_rate >= 40.0 and mean_ground > 80:
        status = "motore_acceso_a_terra"
        title = "TRAPPOLA: MOTORI ACCESI A TERRA 🔥"
        msg = "L'agente tocca la piazzola ma non spegne i motori, continuando a consumare carburante fino al timeout."
        sugg = "Quando entrambe le zampe toccano terra (left_leg==1 e right_leg==1), dai un premio speciale all'azione 0 (action == 0)."
    elif time_rate >= 40.0:
        status = "hovering_infinito"
        title = "TRAPPOLA: HOVERING INFINITO (PAURA DI SCENDERE) 🪂"
        msg = "L'agente galleggia in aria a mezza altezza per evitare di schiantarsi, consumando tutti i 1000 step."
        sugg = "Riduci o rimuovi qualsiasi premio positivo assegnato 'in volo'; penalizza l'altitudine 'abs(y)' e dai una forte punizione a 'is_timeout'."
    elif crash_rate >= 70.0 and mean_len < 95:
        status = "kamikaze"
        title = "TRAPPOLA: STRATEGIA KAMIKAZE 💥"
        msg = "L'agente si schianta deliberatamente al suolo in pochi istanti per interrompere subito l'episodio."
        sugg = "Le penalità che assegni ad ogni step in volo sono troppo severe: all'agente conviene morire subito piuttosto che accumulare penalità continue! Riduci le penalità di step o aumenta la punizione per 'is_crashed'."
    elif crash_rate >= 50.0 and mean_vy > 0.35:
        status = "impatto_veloce"
        title = "IMPATTO TROPPO VIOLENTO 🏎️"
        msg = "L'agente scende dritto sulla piazzola ma con una velocità verticale eccessiva (|vy| > 0.35), distruggendo le zampe."
        sugg = "Aumenta la penalità sulla velocità di discesa abs(vy), specialmente alle quote basse (ad esempio quando y < 0.5 moltiplica la penalità per 1.5 - y)."
    elif crash_rate >= 50.0 and mean_x > HELIPAD_X_MAX:
        status = "fuori_bersaglio"
        title = "ATTERRAGGIO FUORI BERSAGLIO 📍"
        msg = f"L'agente tocca il terreno roccioso fuori dalle bandierine di segnalazione (|x| > {HELIPAD_X_MAX:.2f})."
        sugg = "Aumenta la penalità sulla distanza orizzontale abs(x) e sulla velocità orizzontale abs(vx) per forzare il modulo verso il centro x = 0.0."
    elif crash_rate >= 50.0 and mean_angle > 0.25:
        status = "assetto_inclinato"
        title = "ASSETTO TROPPO INCLINATO 📐"
        msg = "L'astronave si rovescia prima del contatto a causa di un'inclinazione eccessiva (|angle| > 0.25 rad)."
        sugg = "Aumenta il coefficiente negativo su abs(angle) e su abs(vangle) per stabilizzare la verticale."
    else:
        status = "in_apprendimento"
        title = "APPRENDIMENTO IN CORSO ⚙️"
        msg = "L'agente sta iniziando a comprendere la dinamica, ma la combinazione dei pesi è ancora instabile."
        sugg = "Controlla l'equilibrio tra i termini: dai priorità prima alla stabilizzazione dell'angolo e della discesa verticale, poi alla centratura orizzontale."

    lines = [
        f"ANALISI DEGLI ULTIMI {tail} EPISODI:",
        f"• Successi: {succ_rate:.1f}% | Schianti: {crash_rate:.1f}% | Timeout: {time_rate:.1f}%",
        f"• Durata media: {mean_len:.0f} step ({mean_len/50:.1f}s) | Carburante medio: {mean_fuel:.1f} unità",
        f"• Posizione finale media: |x| = {mean_x:.2f} | Velocità impatto: |vy| = {mean_vy:.2f}",
        f"\nDIAGNOSI FLIGHT COACH: {title}",
        f"{msg}",
        f"💡 CONSIGLIO: {sugg}",
    ]

    return {
        "status": status,
        "title": title,
        "message": msg,
        "suggestion": sugg,
        "formatted_text": "\n".join(lines),
        "metrics": metrics,
    }
