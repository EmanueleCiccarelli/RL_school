"""
Configurazione e Parametri di Default per Hill Climb Racing (Play2Learn).
"""

from __future__ import annotations

ENV_ID = "HillClimbRacing-v0"

# Scenari di Missione e Tracciati Speciali
MISSIONS = {
    "campagna_standard": {
        "title": "Campagna Standard (Colline Verdi)",
        "description": "Colline ondulate con pendenze moderate. Ottimo banco di prova iniziale per imparare a bilanciare acceleratore e stabilità.",
        "goal": "Percorrere almeno 800 metri senza ribaltarsi né finire bloccati.",
        "env_kwargs": {
            "gravity": 10.0,
            "difficulty": -150.0,
            "original_noise": False,
        },
    },
    "deserto_dune": {
        "title": "Deserto delle Dune (Pendenze Sabbiose)",
        "description": "Dune ripide e valli profonde. Richiede di prendere rincorsa e dosare la velocità in cresta per non cappottare in discesa.",
        "goal": "Superare le dune e raggiungere 800 metri con trazione continua.",
        "env_kwargs": {
            "gravity": 10.5,
            "difficulty": -100.0,
            "original_noise": False,
        },
    },
    "montagna_estrema": {
        "title": "Montagna Estrema (Picchi Rocciosi)",
        "description": "Pendenze scoscese e salti improvvisi. Richiede forte controllo del rollio in aria e freno reattivo.",
        "goal": "Scalare i picchi rocciosi mantenendo il pilota sempre al sicuro.",
        "env_kwargs": {
            "gravity": 11.0,
            "difficulty": -50.0,
            "original_noise": False,
        },
    },
    "luna_bassa_gravita": {
        "title": "Superficie Lunare (Bassa Gravità)",
        "description": "Gravità ridotta a 2.5 m/s². I salti durano tantissimo: ogni colpo di gas inclina il veicolo all'indietro per reazione di coppia!",
        "goal": "Sfruttare l'airtime lunare atterrando sempre dolcemente su entrambe le ruote.",
        "env_kwargs": {
            "gravity": 2.5,
            "difficulty": -120.0,
            "original_noise": False,
        },
    },
    "autostrada_veloce": {
        "title": "Autostrada Veloce (Time Attack)",
        "description": "Profilo quasi pianeggiante con dolci avvallamenti. Sfida di pura velocità e accelerazione massima.",
        "goal": "Raggiungere il traguardo di 1000 metri nel minor numero di step possibile.",
        "env_kwargs": {
            "gravity": 9.8,
            "difficulty": -220.0,
            "original_noise": True,
        },
    },
}

# Iperparametri ottimali PPO per Hill Climb Racing
DEFAULT_PPO_PARAMS = {
    "n_steps": 256,
    "batch_size": 128,
    "n_epochs": 4,
    "gamma": 0.995,
    "gae_lambda": 0.98,
    "clip_range": 0.2,
    "ent_coef": 0.005,
    "learning_rate": 3e-4,
    "max_grad_norm": 0.5,
    "target_kl": 0.03,
}

VECNORM_PARAMS = {
    "norm_obs": False,
    "norm_reward": True,
    "clip_reward": 10.0,
    "clip_obs": 10.0,
    "gamma": 0.995,
}

DEFAULT_N_ENVS = 8
DEFAULT_TIMESTEPS = 150_000
MIN_EVAL_EPISODES = 20
EVAL_SEEDS = [100 + i * 37 for i in range(MIN_EVAL_EPISODES)]

TARGET_DISTANCE_SUCCESS = 800.0
MAX_TRACK_DISTANCE = 1000.0
