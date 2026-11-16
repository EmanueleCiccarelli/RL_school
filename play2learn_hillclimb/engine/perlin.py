"""
Modulo Perlin Noise in puro Python.
Porting da Processing (https://processingfoundation.org/) e implementazione autonoma
a 4 ottave per garantire funzionamento 100% cross-platform senza dipendenze C.
"""

import math
import random

PERLIN_YWRAPB = 4
PERLIN_YWRAP = 1 << PERLIN_YWRAPB
PERLIN_ZWRAPB = 8
PERLIN_ZWRAP = 1 << PERLIN_ZWRAPB
PERLIN_SIZE = 4095

perlin_octaves = 4
perlin_amp_falloff = 0.5


def scaled_cosine(i: float) -> float:
    return 0.5 * (1.0 - math.cos(i * math.pi))


# Tabella di permutazione pseudo-casuale deterministica a seed fisso (100% cross-platform)
_PERLIN_RNG = random.Random(1337)
_PERLIN_TABLE = [_PERLIN_RNG.random() for _ in range(PERLIN_SIZE + 1)]


def seed_perlin(seed: int) -> None:
    """Inizializza la tabella di rumore di Perlin in modo deterministico."""
    global _PERLIN_TABLE
    rng = random.Random(int(seed))
    _PERLIN_TABLE = [rng.random() for _ in range(PERLIN_SIZE + 1)]


def original_pnoise(x: float, y: float = 0.0, z: float = 0.0) -> float:
    """Implementazione Perlin Noise porting da Processing."""
    if x < 0:
        x = -x
    if y < 0:
        y = -y
    if z < 0:
        z = -z

    xi = math.floor(x)
    yi = math.floor(y)
    zi = math.floor(z)
    xf = x - xi
    yf = y - yi
    zf = z - zi
    r = 0.0
    ampl = 0.5

    for _ in range(perlin_octaves):
        of = xi + (yi << PERLIN_YWRAPB) + (zi << PERLIN_ZWRAPB)

        rxf = scaled_cosine(xf)
        ryf = scaled_cosine(yf)

        n1 = _PERLIN_TABLE[of & PERLIN_SIZE]
        n1 += rxf * (_PERLIN_TABLE[(of + 1) & PERLIN_SIZE] - n1)
        n2 = _PERLIN_TABLE[(of + PERLIN_YWRAP) & PERLIN_SIZE]
        n2 += rxf * (_PERLIN_TABLE[(of + PERLIN_YWRAP + 1) & PERLIN_SIZE] - n2)
        n1 += ryf * (n2 - n1)

        of += PERLIN_ZWRAP
        n2 = _PERLIN_TABLE[of & PERLIN_SIZE]
        n2 += rxf * (_PERLIN_TABLE[(of + 1) & PERLIN_SIZE] - n2)
        n3 = _PERLIN_TABLE[(of + PERLIN_YWRAP) & PERLIN_SIZE]
        n3 += rxf * (_PERLIN_TABLE[(of + PERLIN_YWRAP + 1) & PERLIN_SIZE] - n3)
        n2 += ryf * (n3 - n2)

        n1 += scaled_cosine(zf) * (n2 - n1)

        r += n1 * ampl
        ampl *= perlin_amp_falloff
        xi <<= 1
        xf *= 2
        yi <<= 1
        yf *= 2
        zi <<= 1
        zf *= 2

        if xf >= 1.0:
            xi += 1
            xf -= 1
        if yf >= 1.0:
            yi += 1
            yf -= 1
        if zf >= 1.0:
            zi += 1
            zf -= 1

    return r


def pnoise1(x: float, octaves: int = 4) -> float:
    """Equivalente puro Python di noise.pnoise1() a 4 ottave nell'intervallo [-1, 1]."""
    # Mappa l'output da [0, 1] a [-1, 1]
    val = original_pnoise(x)
    return (val - 0.5) * 2.0
