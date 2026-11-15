"""
Motore Fisico e Grafico per Hill Climb Racing.
"""

import gymnasium as gym
from .hill_racing import HillRacingEnv

# Registrazione dell'ambiente in Gymnasium
try:
    gym.register(
        id="HillClimbRacing-v0",
        entry_point="play2learn_hillclimb.engine.hill_racing:HillRacingEnv",
        max_episode_steps=1800,
    )
except Exception:
    pass

__all__ = ["HillRacingEnv"]
