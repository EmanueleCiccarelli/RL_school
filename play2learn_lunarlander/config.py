"""
Modulo di configurazione per Play2Learn - Lunar Lander.
Definisce le missioni disponibili, le costanti fisiche dell'ambiente
e gli iperparametri PPO ottimizzati per convergenza rapida (150k-200k step)
su Google Colab (2 vCPU) e PC locali Windows.
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
HELIPAD_X_MIN: float = -0.20
HELIPAD_X_MAX: float = 0.20
PAD_Y: float = 0.00
BOUNDARY_X: float = 1.00  # Confine laterale dello schermo (|x| >= 1.0 -> crash)

# Soglie fisiche di sicurezza per atterraggio morbido
SAFE_MAX_VY: float = 0.20       # Velocità verticale massima consigliata
CRITICAL_VY: float = 0.50       # Sopra questo valore le zampe cedono
SAFE_MAX_VX: float = 0.15       # Velocità orizzontale massima consigliata
SAFE_MAX_ANGLE: float = 0.20    # Angolo massimo consigliato (in radianti, ~11°)
SAFE_LANDING_ANGLE: float = 0.25 # Angolo massimo tollerato per considerare l'atterraggio valido (~14°)

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
# 3. IPERPARAMETRI PPO OTTIMIZZATI PER SAMPLE-EFFICIENCY E CPU
# =============================================================================

# Dimensione target complessiva del rollout buffer (n_envs * n_steps = 2048)
# Mantiene invariante la dinamica di PPO al variare di n_envs.
TARGET_ROLLOUT_SIZE: int = 2048

# Schedule lineare del learning rate (evita oscillazioni tardive e fissa la policy)
DEFAULT_LR_START: float = 5e-4
DEFAULT_LR_END: float = 5e-5

DEFAULT_PPO_PARAMS: dict[str, Any] = {
    "policy": "MlpPolicy",
    "n_steps": 256,            # Con n_envs=8 -> 2048 step per rollout
    "batch_size": 128,         # 16 minibatch/epoca -> equilibrio ottimale gradienti/CPU throughput
    "n_epochs": 4,             # 64 update SGD per rollout (2.2x-2.7x piu veloce su Colab 2 vCPU)
    "gamma": 0.995,            # Orizzonte ~200 step: propaga +100/-100 senza esplosione di varianza
    "gae_lambda": 0.98,        # Riduce bias su ricompense terminali sparse
    "clip_range": 0.2,         # Trust region PPO standard
    "ent_coef": 0.008,         # Esplorazione sufficiente senza impedire lo spegnimento a terra
    "vf_coef": 0.5,            # Peso della Value Function loss
    "max_grad_norm": 0.5,      # Gradient clipping di sicurezza su reward studentesche rumorose
    "target_kl": 0.03,         # Early-stopping delle epoche SGD in caso di spike della reward
    "learning_rate": DEFAULT_LR_START,
    "policy_kwargs": {
        # Rete leggera e disaccoppiata: 2-3x più veloce su CPU e meno overfitting
        "net_arch": {"pi": [64, 64], "vf": [64, 64]},
        "ortho_init": True,
    },
}

# Normalizzazione solo delle reward (scale-invariant per formule studentesche).
# norm_obs=False garantisce piena compatibilità a inferenza con obs grezze di Gym.
VECNORM_PARAMS: dict[str, Any] = {
    "norm_obs": False,
    "norm_reward": True,
    "clip_reward": 10.0,
    "gamma": DEFAULT_PPO_PARAMS["gamma"],
}

DEFAULT_N_ENVS: int = 8
DEFAULT_TIMESTEPS: int = 300_000
