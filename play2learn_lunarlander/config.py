"""
Modulo di configurazione per Play2Learn - Lunar Lander.
Definisce le missioni disponibili, le costanti fisiche dell'ambiente
e gli iperparametri ottimizzati per Google Colab.
"""

from __future__ import annotations
from typing import Any

# =============================================================================
# 1. COSTANTI FISICHE E DI SIMULAZIONE
# =============================================================================

ENV_ID: str = "LunarLander-v3"

# Frequenza del ciclo di controllo fisico di Gymnasium (50 Hz -> 0.02s per step)
SIMULATION_FPS: int = 50

# Coordinate geometriche della piazzola di allunaggio (normalizzate Gymnasium)
PAD_X_MIN: float = -0.20
PAD_X_MAX: float = 0.20
PAD_Y: float = 0.00
BOUNDARY_X: float = 1.00  # Confine laterale dello schermo (|x| >= 1.0 -> crash)

# Soglie fisiche di sicurezza per atterraggio morbido
SAFE_MAX_VY: float = 0.20       # Velocità verticale massima consigliata
CRITICAL_VY: float = 0.50       # Sopra questo valore le zampe cedono
SAFE_MAX_VX: float = 0.15       # Velocità orizzontale massima consigliata
SAFE_MAX_ANGLE: float = 0.20    # Angolo massimo consigliato (in radianti, ~11°)

# Consumo nominale di carburante per tipo di motore
FUEL_MAIN_ENGINE: float = 1.0   # Azione 2
FUEL_SIDE_ENGINE: float = 0.3   # Azione 1 (sinistra) e Azione 3 (destra)
FUEL_OFF: float = 0.0           # Azione 0 (motori spenti)


# =============================================================================
# 2. PRESET DELLE MISSIONI (LIVELLI DI DIFFICOLTÀ DIDATTICA)
# =============================================================================

MISSIONS: dict[str, dict[str, Any]] = {
    "luna_standard": {
        "title": "Livello Base: Allunaggio Lunare Standard",
        "description": "Gravità lunare standard (-10.0), assenza di vento e turbolenza.",
        "goal": "Atterrare sulla piazzola (|x| < 0.2) ad assetto verticale e spegnere i motori.",
        "env_kwargs": {
            "gravity": -10.0,
            "enable_wind": False,
            "wind_power": 0.0,
            "turbulence_power": 0.0,
        },
    },
    "eco_lander": {
        "title": "Sfida 1: Eco-Lander (Risparmio Carburante)",
        "description": "Condizioni standard, ma l'obiettivo è completare l'allunaggio con meno di 60 unità di carburante!",
        "goal": "Consumo medio carburante < 60 unità con successo >= 80%.",
        "env_kwargs": {
            "gravity": -10.0,
            "enable_wind": False,
            "wind_power": 0.0,
            "turbulence_power": 0.0,
        },
    },
    "speedrun": {
        "title": "Sfida 2: Speedrun (Atterraggio Rapido)",
        "description": "Condizioni standard, ma l'obiettivo è completare l'allunaggio in meno di 160 step (3.2 secondi)!",
        "goal": "Durata media del volo < 160 step con successo >= 80%.",
        "env_kwargs": {
            "gravity": -10.0,
            "enable_wind": False,
            "wind_power": 0.0,
            "turbulence_power": 0.0,
        },
    },
    "tempesta_solare": {
        "title": "Sfida 3: Tempesta Solare (Vento e Turbolenza)",
        "description": "Vento trasversale attivo (potenza 15.0) e raffiche di turbolenza (1.5) che deviano e ruotano il modulo.",
        "goal": "Resistere alle perturbazioni laterali e allineare l'angolo prima del contatto.",
        "env_kwargs": {
            "gravity": -10.0,
            "enable_wind": True,
            "wind_power": 15.0,
            "turbulence_power": 1.5,
        },
    },
    "pianeta_pesante": {
        "title": "Sfida 4: Pianeta Pesante (Alta Gravità)",
        "description": "Gravità estrema (-11.8) con leggero vento. Il motore principale deve frenare una caduta violenta!",
        "goal": "Frenare la caduta verticale accelerata senza andare in stallo o esaurire il tempo.",
        "env_kwargs": {
            "gravity": -11.8,
            "enable_wind": True,
            "wind_power": 8.0,
            "turbulence_power": 0.8,
        },
    },
    "marte_leggero": {
        "title": "Sfida 5: Micro-Gravità (Gravità Debole + Turbolenza)",
        "description": "Gravità ridotta (-4.5) e turbolenza (1.2): basta poca spinta per rimbalzare fuori dallo schermo!",
        "goal": "Gestire spinte minime senza rimbalzare verso l'alto (vy > 0).",
        "env_kwargs": {
            "gravity": -4.5,
            "enable_wind": True,
            "wind_power": 10.0,
            "turbulence_power": 1.2,
        },
    },
}


# =============================================================================
# 3. IPERPARAMETRI PPO OTTIMIZZATI PER GOOGLE COLAB (2 vCPU)
# =============================================================================

DEFAULT_PPO_PARAMS: dict[str, Any] = {
    "policy": "MlpPolicy",
    "n_steps": 512,            # Orizzonte per ambiente prima dell'aggiornamento
    "batch_size": 256,         # Dimensione minibatch
    "n_epochs": 4,             # Epoche di ottimizzazione SGD per iterazione
    "gamma": 0.99,             # Fattore di sconto ricompense future
    "gae_lambda": 0.95,        # Stima GAE per varianza e bias
    "ent_coef": 0.01,          # Coefficiente di entropia per esplorazione
    "learning_rate": 8e-4,     # Passo di apprendimento Adam
}

DEFAULT_N_ENVS: int = 8
DEFAULT_TIMESTEPS: int = 500_000
