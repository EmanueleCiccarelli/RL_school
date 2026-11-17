import numpy as np


def generate_perlin_1d(
    x: np.ndarray,
    seed: int,
    octaves: int = 4,
    persistence: float = 0.5,
) -> np.ndarray:
    """
    Generatore di Perlin Noise 1D completamente deterministico e vettorizzato in NumPy.
    Elimina la dipendenza C esterna `noise` e la tabella globale non seedata.
    Restituisce valori in [0, 1] compatibili con le curve di pendenza di HillRacingEnv.
    """
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    perm_size = 4096
    mask = perm_size - 1
    table = rng.random(perm_size, dtype=np.float64)

    coords = np.abs(np.asarray(x, dtype=np.float64))
    total = np.zeros_like(coords, dtype=np.float64)
    amplitude = 0.5
    max_amp = 0.0
    freq = 1.0

    for _ in range(octaves):
        scaled = coords * freq
        xi = np.floor(scaled).astype(np.int64)
        xf = scaled - xi
        u = 0.5 * (1.0 - np.cos(xf * np.pi))

        n0 = table[xi & mask]
        n1 = table[(xi + 1) & mask]
        total += (n0 + u * (n1 - n0)) * amplitude

        max_amp += amplitude
        amplitude *= persistence
        freq *= 2.0

    return total / max(max_amp, 1e-9)


def original_pnoise(x, y=0, z=0, seed: int = 42) -> float:
    arr = np.array([x], dtype=np.float64)
    return float(generate_perlin_1d(arr, seed=seed)[0])
