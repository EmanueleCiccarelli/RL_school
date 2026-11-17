"""
Configurazione e parametri di default per Play2Learn Hill Climb Racing.
"""

from __future__ import annotations

from typing import Any

ENV_ID = "hill_racing_env/HillRacing-v0"

TARGET_DISTANCE_SUCCESS = 800.0  # Metri minimi per considerare la scalata un successo
MAX_DISTANCE_GOAL = 1000.0      # Traguardo massimo della collina
HELIPAD_MIN_X = 800.0

DEFAULT_TIMESTEPS = 300_000
DEFAULT_EVAL_EPISODES = 20

# Scenari collinari configurabili tramite parametri di difficoltà e rumore Perlin dell'engine originale
MISSIONS: dict[str, dict[str, Any]] = {
    "campagna_standard": {
        "nome": "Campagna Standard (Difficoltà Normale)",
        "difficulty": -150.0,
        "original_noise": False,
        "descrizione": "Colline ondulate classiche con pendenze moderate. Ottimo banco di prova per imparare a dosare il gas.",
    },
    "deserto_dune": {
        "nome": "Deserto delle Dune (Rumore Originale)",
        "difficulty": -100.0,
        "original_noise": True,
        "descrizione": "Dune sabbiose ripide e irregolari generate con l'algoritmo Perlin classico. Rischio frequente di impennate.",
    },
    "montagna_estrema": {
        "nome": "Montagna Estrema (Pendenze Ripide)",
        "difficulty": -50.0,
        "original_noise": False,
        "descrizione": "Salite molto accentuate e creste affilate. Richiede una guida precisa per evitare ribaltamenti all'indietro.",
    },
    "colline_ripide": {
        "nome": "Colline Ripide (Massima Difficoltà)",
        "difficulty": 0.0,
        "original_noise": False,
        "descrizione": "Scoscendimenti vertiginosi e salite al limite della trazione. Sfida estrema per modelli esperti.",
    },
    "autostrada_veloce": {
        "nome": "Autostrada Veloce (Terreno Dolce)",
        "difficulty": -220.0,
        "original_noise": False,
        "descrizione": "Tracciato pianeggiante con leggere ondulazioni. Permette velocità elevate e test rapidi.",
    },
}

DEFAULT_PPO_PARAMS: dict[str, Any] = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 128,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}

VECNORM_PARAMS: dict[str, Any] = {
    "norm_obs": False,
    "norm_reward": True,
    "clip_obs": 10.0,
    "clip_reward": 10.0,
    "gamma": 0.99,
}
