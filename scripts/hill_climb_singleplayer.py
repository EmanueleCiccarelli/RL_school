#!/usr/bin/env python3
"""Modalita' single-player locale di Hill Climb Racing.

Script standalone (da terminale, non dai notebook) che replica ESATTAMENTE lo
scenario che vede l'agente durante gli episodi di training PPO, ma comandabile
da tastiera per testare l'ambiente fisico e scovare eventuali bug.

La catena degli ambienti e' identica a quella del training (_make_single_env):

    raw HillRacingEnv(action_space="discrete_3", reward_type="aggressive",
                     difficulty=<missione>, original_noise=<missione>)
      -> HillClimbRewardWrapper      # osservazione 7D didattica + esito episodio
      -> Monitor                     # statistiche episodio stile stable-baselines3

con un'unica differenza voluta e documentata: render_mode="rgb_array" al posto
di None, cosi' lo script puo' disegnare la propria finestra pygame con l'HUD di
diagnostica sovrapposto senza alterare alcun parametro fisico.

Uso tipici:

    # Finestra interattiva con i controlli da tastiera
    python scripts/hill_climb_singleplayer.py

    # Stessa missione del training esplicitamente richiesta
    python scripts/hill_climb_singleplayer.py --mission campagna_standard --seed 42

    # Smoke test headless senza finestra (per CI / server senza display)
    SDL_VIDEODRIVER=dummy python scripts/hill_climb_singleplayer.py --autoplay 300

    # Limita il numero di step simulati anche in modalita' finestra (debug)
    python scripts/hill_climb_singleplayer.py --max-steps 400 --scale 0.75

Controlli da tastiera:

    GAS                  freccia SU / D / freccia DESTRA / W   -> azione 1
    FRENO-RETROMARCIA    freccia GIU / A / freccia SINISTRA / S -> azione 2
    FOLLE                nessun tasto premuto                   -> azione 0
    R                    reset immediato dell'episodio corrente
    P                    pausa / riprendi la simulazione fisica
    H                    mostra / nasconde l'HUD di diagnostica
    M                    ristampa su console lo snapshot delle masse fisiche
    ESC oppure Q         chiude lo script con il riepilogo finale su console

La diagnostica e pensata proprio come caccia ai bug della fisica:

    - osservazioni NaN o Inf segnalate subito su console e marcate nell'HUD;
    - contatori di slittamento ruota-superficie rispetto alla velocita' del telaio;
    - allarme "trazione persa": entrambe le ruote in aria oltre soglia con gas tenuto;
    - allarme "ruote girano a vuoto": slip elevato sostenuto con vettura quasi ferma;
    - dump delle masse dei corpi Box2D all'avvio e su richiesta (tasto M);
    - log opzionale CSV per-step (--log-csv FILE.csv) da analizzare fuori dal gioco.

A fine episodio viene stampato su console lo stesso tipo di report prodotto dal
wrapper durante il training tramite info["telemetry"] (successo/crash/timeout,
distanza massima, percentuale di aria e numero di ribaltamenti).

NOTA sul comportamento atteso (non e' un bug): a inizio episodio l'engine
originale lascia cadere il telaio sul terreno, come nel gioco arcade originale;
per i primi ~40 step le ruote sono in aria e i contatti compaiono solo
dopo l'atterraggio (verificabile con la sequenza "solo folle" dell'autoplay).
"""

from __future__ import annotations

import argparse
import csv
from collections import deque
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Rende lo script eseguibile da qualsiasi cartella garantendo che venga importata
# la copia locale del progetto e non un eventuale pacchetto installato altrove.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np           # noqa: E402

# Import postponed dopo il setup del path cosi' fallisce con un messaggio chiaro.


_MSG_NO_PYGAME_OR_ENV = (
    "Dipendenza mancante per lo script single-player.\n"
    "Serve l'ambiente del progetto completo:\n"
    f"   - repo locale disponibile sotto {_REPO_ROOT}\n"
    "   - pygame installato nell'interprete Python corrente"
)


# ---------------------------------------------------------------------------
# Costanti di presentazione della finestra (l'engine resta quello dell'ambiente)
# ---------------------------------------------------------------------------
_FRAME_W = 1280            # risoluzione del frame rgb_array dell'engine originale
_FRAME_H = 720             # SCREEN_HEIGHT dell'engine originale hill_racing.py
_HUD_H = 200               # altezza della fascia HUD sotto al gameplay visibile
_WINDOW_TITLE = "Hill Climb Racing - modalita' single player (ambiente identico al training)"
_ACTION_NAMES = {0: "FOLLE", 1: "GAS", 2: "FRENO"}

_LOG_FIELDS = [
    "step",
    "episodio",
    "azione",
    "dist_x_m",
    "quota_y_m",
    "assetto_rad",
    "omega_posteriore_rad_s",
    "omega_anteriore_rad_s",
    "contatto_posteriore",
    "contatto_anteriore",
    "vel_x_m_s",
    "vel_y_m_s",
    "slip_posteriore_m_s",
    "slip_anteriore_m_s",
    "step_stuck",
    "oss_nan_inf",
]              # ordine stabile richiesto da csv.DictWriter


# ---------------------------------------------------------------------------
# Utilita' generiche senza dipendenze pygame
# ---------------------------------------------------------------------------


def _resolve_mission(mission: str) -> Dict[str, Any]:
    """Ritorna la configurazione effettiva della missione (stessa logica del training)."""
    from play2learn_hillclimb.config import MISSIONS

    return MISSIONS.get(mission, MISSIONS["campagna_standard"])


def _wrap_angle(angle_rad: float) -> float:
    """Normalizza un angolo in radianti in [-pi, +pi] (stessa convenzione del wrapper)."""
    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


def _fmt(value: Optional[float], spec: str = "{:.1f}", dash: str = "n/d") -> str:
    """Formatta un float o 'n/d' se non disponibile (None o NaN)."""
    if value is None:
        return dash
    try:
        if math.isnan(value):
            return dash
    except TypeError:
        return dash
    return spec.format(value)


# ---------------------------------------------------------------------------
# Costruzione dell'ambiente: identico al training, render modo a parte
# ---------------------------------------------------------------------------


def _build_env(mission: str, seed: Optional[int], windowed: bool):
    """Crea la stessa catena di ambienti usata dal training PPO.

    _make_single_env e' la stessa fabbrica chiamata da make_parallel_envs per
    gli episodi di addestramento: usarla qui garantisce difficolta', rumore
    Perlin, reward di motore "aggressive", wrapper 7D e Monitor identici.
    Unica differenza voluta: render_mode="rgb_array" in modalita' finestra per
    disegnare la nostra HUD (la fisica non viene toccata dal rendering).
    """
    try:
        from play2learn_hillclimb.environment import _make_single_env
        from play2learn_hillclimb.config import TARGET_DISTANCE_SUCCESS
    except Exception as exc:  # pragma: no cover - messaggio d'aiuto per l'utente
        raise SystemExit(_MSG_NO_PYGAME_OR_ENV + f"\n\nDettaglio errore: {exc}")

    return _make_single_env(
        mission=mission,
        reward_fn=None,  # gioco umano: resta la reward originale dell'engine
        render_mode="rgb_array" if windowed else None,
        target_distance=TARGET_DISTANCE_SUCCESS,
        frame_skip=1,
        seed=seed,
    )


def _core(env):
    """Ritorna l'istanza HillRacingEnv sottostante (salta Monitor/wrapper/gym.make)."""
    return env.unwrapped


# ---------------------------------------------------------------------------
# Diagnostica fisica per-step (il cuore "caccia ai bug" dello script)
# ---------------------------------------------------------------------------


def _obs_has_nan_inf(obs: np.ndarray) -> bool:
    """True se l'osservazione 7D contiene NaN o Inf."""
    arr = np.asarray(obs, dtype=np.float64)
    return bool(not np.all(np.isfinite(arr)))


def _physics_snapshot(core) -> Dict[str, Any]:
    """Estrae grandezze fisiche grezze dai corpi Box2D dell'engine.

    Tutto letture in sola lettura: lo stato della simulazione non viene alterato.
    """
    from hill_racing_env.envs import hill_racing as engine

    car = core.agent.car
    chassis = car.chassis_body
    wheel_r = car.wheels[0]
    wheel_f = car.wheels[1]

    vx = float(chassis.linearVelocity.x)
    vy = float(chassis.linearVelocity.y)

    # Raggio in metri dei cerchi (pixel / SCALE dell'engine).
    r_m_post = float(wheel_r.radius) / float(engine.SCALE)
    r_m_ant = float(wheel_f.radius) / float(engine.SCALE)

    # L'osservazione didattica nega joint.speed: > 0 rotazione in avanti.
    omega_post = -float(wheel_r.joint.speed) if wheel_r.joint is not None else 0.0
    omega_ant = -float(wheel_f.joint.speed) if wheel_f.joint is not None else 0.0

    # Velocita' del battistrada rispetto al terreno vs velocita' del telaio:
    # se la gomma aderisce i due valori coincidono; se girano a vuoto divergono.
    slip_post = abs(omega_post * r_m_post - vx)
    slip_ant = abs(omega_ant * r_m_ant - vx)

    return {
        "vx": vx,
        "vy": vy,
        "omega_post": omega_post,
        "omega_ant": omega_ant,
        "slip_post": slip_post,
        "slip_ant": slip_ant,
        "contact_post": int(bool(wheel_r.on_ground)),
        "contact_ant": int(bool(wheel_f.on_ground)),
        "contact_count_post": int(wheel_r.ground_contacts),
        "contact_count_ant": int(wheel_f.ground_contacts),
        "mass_chassis": float(chassis.mass),
        "mass_wheels": float(wheel_r.body.mass + wheel_r.rim_body.mass +
                             wheel_f.body.mass + wheel_f.rim_body.mass),
        "mass_person": float(car.person.head.body.mass + car.person.torso.body.mass),
    }


# ---------------------------------------------------------------------------
# Stato dell'episodio corrente + allarmi automatici anti-regressione fisica
# ---------------------------------------------------------------------------

_AIR_ALERT_STEPS = 45        # ~0.75 s a 60 Hz di volo con gas tenuto -> sospetto bug trazione
_SPIN_ALERT_STEPS = 45       # ~0.75 s di slittamento sostenuto -> ruote che girano a vuoto
_SPIN_SLIP_MPS = 8.0         # soglia di slip battistrada-telaio (m/s)
_SPIN_MAX_VX = 0.5           # sotto questa velocita' del telaio lo slip alto e' sospetto


class _EpisodeState:
    """Contatori incrementali dell'episodio corrente + gestione allarmi una-tantum."""

    def __init__(self, index: int) -> None:
        self.index = index
        self.reward = 0.0
        self.max_distance = 0.0
        self.max_speed = 0.0
        self.steps = 0
        self.air_steps = 0            # step con entrambe le ruote in aria
        self.flip_steps = 0           # step con |assetto| > ~80 gradi
        self.traction_air_steps = 0   # consecutivi in aria mentre si tiene GAS
        self.spin_steps = 0           # consecutivi con slip alto e vettura quasi ferma
        self.alerts_shown: set[str] = set()

    def update(self, obs7: Tuple[float, ...], phys: Dict[str, Any], action: int,
               step_reward: float) -> None:
        """Aggiorna i contatori dopo uno step e stampa gli allarmi solo la prima volta."""
        dist_x, _quota_y, angle_rad = float(obs7[0]), float(obs7[1]), float(obs7[2])
        r_gnd, f_gnd = float(obs7[5]), float(obs7[6])

        self.steps += 1
        self.reward += step_reward
        self.max_distance = max(self.max_distance, dist_x)
        self.max_speed = max(self.max_speed, abs(phys["vx"]))

        airborne = r_gnd == 0.0 and f_gnd == 0.0
        if airborne:
            self.air_steps += 1
        if abs(angle_rad) > 1.4:
            self.flip_steps += 1

        # Allarme trazione persa: volo prolungato mentre l'utente tiene premuto GAS.
        if airborne and action == 1:
            self.traction_air_steps += 1
            if (self.traction_air_steps == _AIR_ALERT_STEPS and
                    "traction_air" not in self.alerts_shown):
                self.alerts_shown.add("traction_air")
                print(f"[!] episodio {self.index}: trazione persa per {_AIR_ALERT_STEPS} "
                      "step consecutivi con GAS tenuto (possibile contatto ruota-suolo assente)")
        else:
            self.traction_air_steps = 0

        # Allarme ruote che girano a vuoto: slip alto sostenuto a vettura ferma.
        spin_now = ((phys["slip_post"] > _SPIN_SLIP_MPS or phys["slip_ant"] > _SPIN_SLIP_MPS)
                    and abs(phys["vx"]) < _SPIN_MAX_VX)
        if spin_now:
            self.spin_steps += 1
            if (self.spin_steps == _SPIN_ALERT_STEPS and
                    "spin" not in self.alerts_shown):
                self.alerts_shown.add("spin")
                print(f"[!] episodio {self.index}: slip > {_SPIN_SLIP_MPS:.0f} m/s per "
                      f"{_SPIN_ALERT_STEPS} step con vettura quasi ferma "
                      "(ruote che girano a vuoto senza trazione?)")
        else:
            self.spin_steps = 0


_OUTCOME_LABELS = {"success": "SUCCESSO", "crash": "CRASH", "timeout": "TIMEOUT/BLOCCO",
                   "unknown": "FINE"}


def _classify(info: Dict[str, Any], terminated: bool, truncated: bool) -> Tuple[str, str]:
    """Classifica l'esito usando la stessa fonte di verita' del training (wrapper)."""
    telem = info.get("telemetry", {}) or {}
    if telem.get("is_success"):
        return "success", f"raggiunti {_fmt(telem.get('max_distance'), '{:.1f}')} m"
    if telem.get("is_crashed"):
        return "crash", "testa del pilota a terra"
    if telem.get("is_timeout"):
        return "timeout", "blocco o tempo scaduto senza morte ne' traguardo"
    if terminated or truncated:
        return "unknown", "terminazione dell'engine non classificata"
    return "", ""     # episodio ancora in corso


def _print_episode_report(st: _EpisodeState, info: Dict[str, Any],
                          terminated: bool, truncated: bool) -> None:
    """Stampa su console lo stesso tipo di report telemetrico del training."""
    kind, why = _classify(info, terminated, truncated)
    telem = info.get("telemetry", {}) or {}

    print("=" * 66)
    print(f"FINE EPISODIO #{st.index}   esito: {_OUTCOME_LABELS.get(kind, kind)} ({why})")
    print(f"   distanza max : {_fmt(telem.get('max_distance', st.max_distance), '{:.1f}')} m"
          f"   velocita' max : {_fmt(st.max_speed, '{:.2f}')} m/s")
    print(f"   step totali  : {telem.get('steps', st.steps)}"
          f"   aria : {_fmt(telem.get('airtime_pct'), '{:.1f}')}%"
          f"   ribaltamenti (>80 gradi) : {telem.get('flips', '-')}")
    print(f"   reward cumulata : {st.reward:+.2f}")


class _CsvLogger:
    """Log per-step su CSV (colonne stabili) per analisi offline di eventuali bug."""

    def __init__(self, path: str) -> None:
        self._fh = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=_LOG_FIELDS)
        self._writer.writeheader()

    def write(self, row: Dict[str, Any]) -> None:
        self._writer.writerow({key: row.get(key, "") for key in _LOG_FIELDS})

    def close(self) -> None:
        self._fh.close()


# ---------------------------------------------------------------------------
# Runner headless (--autoplay): smoke test riproducibile della fisica
# ---------------------------------------------------------------------------

_AUTOPLAY_LOG_EVERY = 60      # step tra due righe di log su console


def _autoplay_policy(obs7: Tuple[float, ...], phys: Dict[str, Any]) -> int:
    """Politica deterministica a feedback per l'autoplay.

    Più realistica di un gas costante (che per fisica intrinseca dell'engine
    provoca sempre il wheelie): agisce solo a contatto con il suolo, dosa il
    gas in funzione della velocita' e rilascia quando il telaio cabra troppo.
    Serve a esercitare l'ambiente in modo riproducibile per i test di fisica.
    """
    grounded = obs7[5] > 0.0 or obs7[6] > 0.0
    if not grounded:
        return 0                                  # in volo il gas non spinge comunque
    if _wrap_angle(obs7[2]) > 0.45:
        return 0                                  # assetto cabrato: rilascia il gas
    if abs(phys["vx"]) < 9.0:
        return 1                                  # gas modulato fino alla velocita' target
    return 0                                      # velocita' raggiunta: mantieni


_AUTOPLAY_LABEL = "politica a feedback deterministica (gas solo a contatto e assetto controllato)"


def _autoplay_run(env, st: _EpisodeState, max_steps: int, csv_logger: Optional[_CsvLogger],
                  sink: List[Dict[str, Any]], seed: Optional[int]) -> None:
    """Esegue una corsa headless senza finestra con la politica a feedback e appende gli esiti."""
    obs, _info = env.reset(seed=seed)

    for step in range(max_steps):
        obs7_prev = tuple(float(v) for v in obs)
        phys_prev = _physics_snapshot(_core(env))
        action = _autoplay_policy(obs7_prev, phys_prev)
        t0 = time.perf_counter()
        obs, reward, terminated, truncated, info = env.step(action)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        obs7 = tuple(float(v) for v in obs)
        phys = _physics_snapshot(_core(env))
        nan_flag = _obs_has_nan_inf(obs)
        if nan_flag and "nan" not in st.alerts_shown:
            st.alerts_shown.add("nan")
            print(f"[!] episodio {st.index}: osservazione NaN/Inf allo step {st.steps}")

        st.update(obs7, phys, action, float(reward))

        if csv_logger is not None:
            csv_logger.write({
                "step": st.steps,
                "episodio": st.index,
                "azione": action,
                "dist_x_m": obs7[0], "quota_y_m": obs7[1], "assetto_rad": obs7[2],
                "omega_posteriore_rad_s": obs7[3], "omega_anteriore_rad_s": obs7[4],
                "contatto_posteriore": obs7[5], "contatto_anteriore": obs7[6],
                "vel_x_m_s": phys["vx"], "vel_y_m_s": phys["vy"],
                "slip_posteriore_m_s": phys["slip_post"],
                "slip_anteriore_m_s": phys["slip_ant"],
                "step_stuck": int(info.get("steps_stuck", 0)),
                "oss_nan_inf": int(nan_flag),
            })

        if step % _AUTOPLAY_LOG_EVERY == 0:
            deg = math.degrees(_wrap_angle(obs7[2]))
            print(f"[auto] ep{st.index} step={st.steps:4d} azione={_ACTION_NAMES[action]:5s} "
                  f"dist={obs7[0]:7.1f} m  quota={obs7[1]:6.1f} m  assetto={deg:+6.1f} gradi  "
                  f"vx={phys['vx']:+5.1f} m/s  slip P/A={phys['slip_post']:.1f}/{phys['slip_ant']:.1f} "
                  f"terra P/A={obs7[5]:.0f}/{obs7[6]:.0f}  stuck={int(info.get('steps_stuck', 0))}"
                  f"  {dt_ms:.1f} ms/step")

        if terminated or truncated:
            _print_episode_report(st, info, terminated, truncated)
            sink.append({"episodio": st.index,
                         "esito": _classify(info, terminated, truncated)[0],
                         "distanza_max": float(info.get("telemetry", {}).get("max_distance",
                                                                             st.max_distance))})
            obs, _info = env.reset()
            st = _EpisodeState(st.index + 1)


    # Nota finale se il limite di step chiude la sessione a episodio non ancora finito.
    if st.steps > 0:
        print(f"[i] autoplay terminato a episodio #{st.index} non ancora concluso "
              f"(distanza corrente max {st.max_distance:.1f} m dopo {st.steps} step)")


# ---------------------------------------------------------------------------
# Modalita' interattiva con finestra e HUD di diagnostica
# ---------------------------------------------------------------------------

_GAS_KEYS_NAME = "SU/W/D/DESTRA"
_BRAKE_KEYS_NAME = "GIU/S/A/SINISTRA"
_TARGET_LINE_WIDTH = 42          # larghezza della barra di avanzamento in caratteri
_STEP_TIMING_WINDOW = 120        # step usati per la media dei tempi di simulazione


def _make_font(size: int):
    """Carica un font monospace se disponibile, con fallback sul font di default."""
    import pygame

    pygame.font.init()
    for name in ("dejavusansmono", "dejavusansmono-bold", "consolas", "couriernew"):
        try:
            return pygame.font.SysFont(name, size)
        except Exception:
            continue
    return pygame.font.Font(None, size)


def _progress_bar(dist_x: float, target_m: float) -> str:
    """Barra ASCII di avanzamento verso il traguardo del training."""
    frac = max(0.0, min(1.0, dist_x / target_m))
    filled = int(round(frac * _TARGET_LINE_WIDTH))
    return "#" * filled + "-" * (_TARGET_LINE_WIDTH - filled)


def _print_engine_banner(core) -> None:
    """Stampa la configurazione effettiva dell'engine per verifica visiva a runtime."""
    from play2learn_hillclimb.config import TARGET_DISTANCE_SUCCESS

    print(f"[INFO engine] difficulty={core.difficulty}  original_noise={core.original_noise}  "
          f"max_steps={core.max_steps}  fps engine={core.metadata['render_fps']}  "
          f"solver=(180 vel / 60 pos)  gravita=10")
    print(f"[INFO engine] traguardo wrapper (successo) = {TARGET_DISTANCE_SUCCESS:.0f} m  "
          f"azione discreta = 3 (0 folle, 1 gas, 2 freno)")


def _print_mass_snapshot(phys: Optional[Dict[str, Any]]) -> None:
    """Stampa su console le masse Box2D lette dall'ultimo snapshot fisico."""
    if phys is None:
        print("[INFO masse] snapshot non ancora disponibile (nessuno step simulato)")
        return
    totale = phys["mass_chassis"] + phys["mass_wheels"] + phys["mass_person"]
    print(f"[INFO masse Box2D] telaio={phys['mass_chassis']:.2f} kg  "
          f"ruote+cerchi={phys['mass_wheels']:.2f} kg  "
          f"pilota={phys['mass_person']:.2f} kg  totale={totale:.2f} kg")


def _windowed_run(env, args, csv_logger: Optional[_CsvLogger],
                  sink: List[Dict[str, Any]], seed: Optional[int]) -> None:
    """Ciclo principale con finestra pygame, controlli da tastiera e HUD."""
    import pygame

    pygame.init()
    try:
        win_w = int(round(_FRAME_W * args.scale))
        win_h = int(round((_FRAME_H + _HUD_H) * args.scale))
        screen = pygame.display.set_mode((win_w, win_h))
        pygame.display.set_caption(_WINDOW_TITLE)
        canvas = pygame.Surface((_FRAME_W, _FRAME_H + _HUD_H))
        clock = pygame.time.Clock()
        font_small = _make_font(17)
        font_med = _make_font(19)
        font_big = _make_font(26)

        core = _core(env)
        mission_cfg = _resolve_mission(args.mission)

        st = _EpisodeState(1)
        obs, _info = env.reset(seed=seed)
        obs7 = tuple(float(v) for v in obs)
        phys: Optional[Dict[str, Any]] = _physics_snapshot(core)
        info: Dict[str, Any] = {}

        _print_engine_banner(core)
        _print_mass_snapshot(phys)
        print(f"[INFO] missione attiva: {mission_cfg['nome']}  "
              f"(controlli: {_GAS_KEYS_NAME} gas, {_BRAKE_KEYS_NAME} freno)")

        running = True
        paused = False
        hud_on = True
        last_result = "esito episodio precedente: n/d (prima corsa in corso)"
        step_times: deque = deque(maxlen=_STEP_TIMING_WINDOW)
        session_steps = 0

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_r:
                        obs, _info = env.reset()
                        st = _EpisodeState(st.index + 1)
                        last_result = f"reset manuale -> episodio #{st.index}"
                        print(f"[i] reset manuale richiesto -> episodio #{st.index}")
                    elif event.key == pygame.K_p:
                        paused = not paused
                        print("[i] PAUSA attivata" if paused else "[i] pausa disattivata")
                    elif event.key == pygame.K_h:
                        hud_on = not hud_on
                    elif event.key == pygame.K_m:
                        _print_mass_snapshot(phys)

            keys = pygame.key.get_pressed()
            gas_held = keys[pygame.K_UP] or keys[pygame.K_w] or keys[pygame.K_d] or keys[pygame.K_RIGHT]
            brake_held = keys[pygame.K_DOWN] or keys[pygame.K_s] or keys[pygame.K_a] or keys[pygame.K_LEFT]
            action = 1 if gas_held else (2 if brake_held else 0)

            if not paused:
                t0 = time.perf_counter()
                obs, reward, terminated, truncated, info = env.step(action)
                step_times.append((time.perf_counter() - t0) * 1000.0)

                obs7 = tuple(float(v) for v in obs)
                phys = _physics_snapshot(core)

                if _obs_has_nan_inf(obs) and "nan" not in st.alerts_shown:
                    st.alerts_shown.add("nan")
                    print(f"[!] episodio {st.index}: osservazione NaN/Inf allo step {st.steps}")

                st.update(obs7, phys, action, float(reward))

                if csv_logger is not None:
                    csv_logger.write({
                        "step": st.steps,
                        "episodio": st.index,
                        "azione": action,
                        "dist_x_m": obs7[0], "quota_y_m": obs7[1], "assetto_rad": obs7[2],
                        "omega_posteriore_rad_s": obs7[3], "omega_anteriore_rad_s": obs7[4],
                        "vel_x_m_s": phys["vx"], "vel_y_m_s": phys["vy"],
                        "slip_posteriore_m_s": phys["slip_post"],
                        "slip_anteriore_m_s": phys["slip_ant"],
                        "step_stuck": int(info.get("steps_stuck", 0)),
                        "oss_nan_inf": int(_obs_has_nan_inf(obs)),
                    })

                session_steps += 1

                if terminated or truncated:
                    _print_episode_report(st, info, terminated, truncated)
                    kind = _classify(info, terminated, truncated)[0]
                    telem = info.get("telemetry", {}) or {}
                    last_result = (f"EP#{st.index} {_OUTCOME_LABELS.get(kind, kind)} a "
                                   f"{_fmt(telem.get('max_distance'), '{:.1f}')} m "
                                   f"in {telem.get('steps', st.steps)} step")
                    sink.append({"episodio": st.index,
                                 "esito": kind,
                                 "distanza_max": float(telem.get("max_distance", st.max_distance))})
                    # Reset immediato: stessa semantica di continuazione degli episodi di training.
                    obs, _info = env.reset()
                    st = _EpisodeState(st.index + 1)

            frame = env.render()
            if frame is not None:
                canvas.blit(pygame.image.frombuffer(frame.tobytes(), (_FRAME_W, _FRAME_H), "RGB"),
                            (0, 0))

            if hud_on:
                deg = math.degrees(_wrap_angle(obs7[2]))
                stuck = int(info.get("steps_stuck", 0))
                avg_ms = (sum(step_times) / len(step_times)) if step_times else float("nan")
                target_m = 800.0     # TARGET_DISTANCE_SUCCESS (stessa soglia del training)

                rows = [
                    (f"{mission_cfg['nome']}  |  terreno seed={seed if seed is not None else 'default(42)'}"
                     f"  |  ep#{st.index} step {st.steps}  |  {clock.get_fps():4.0f} fps reali  |  "
                     f"{avg_ms:.1f} ms/step", (230, 230, 230), font_med),
                    (f"obs[ dist={obs7[0]:7.1f} m   quota={obs7[1]:6.1f} m   assetto={deg:+6.1f} gradi"
                     f"   omegaP={obs7[3]:+6.2f} rad/s   omegaA={obs7[4]:+6.2f} rad/s   "
                     f"terra P/A={obs7[5]:.0f}/{obs7[6]:.0f} ]", (140, 210, 255), font_small),
                    (f"distanza max={st.max_distance:6.1f} m   "
                     f"[{_progress_bar(obs7[0], target_m)}]   "
                     f"v={phys['vx']:+5.1f} m/s   aria={100.0 * st.air_steps / max(1, st.steps):4.1f}%   "
                     f"ribaltamenti={st.flip_steps}   stuck={stuck}/{core.max_steps}",
                     (120, 230, 120), font_small),
                    (f"slip P={phys['slip_post']:5.1f} A={phys['slip_ant']:5.1f} m/s   "
                     f"contatti P={phys['contact_count_post']} A={phys['contact_count_ant']}   "
                     f"masse tot={phys['mass_chassis'] + phys['mass_wheels'] + phys['mass_person']:.1f} kg   "
                     f"azione={_ACTION_NAMES[action]}", (255, 210, 60), font_small),
                    (last_result, (160, 160, 160), font_small),
                ]

                y = _FRAME_H + 8
                for text, color, font in rows:
                    canvas.blit(font.render(text, True, color), (14, y))
                    y += int(_HUD_H / 5) - 4

                if st.traction_air_steps > 0:
                    color = (255, 90, 90) if st.traction_air_steps >= _AIR_ALERT_STEPS else (255, 210, 60)
                    canvas.blit(font_small.render(
                        f"trazione persa da {st.traction_air_steps} step (entrambe le ruote in aria col GAS)",
                        True, color), (14, y + 2))
                elif st.spin_steps > 0:
                    color = (255, 90, 90) if st.spin_steps >= _SPIN_ALERT_STEPS else (255, 210, 60)
                    canvas.blit(font_small.render(
                        f"possibile spin a vuoto da {st.spin_steps} step (slip alto a vettura ferma)",
                        True, color), (14, y + 2))

            if paused:
                canvas.blit(font_big.render("PAUSA - premi P per riprendere", True, (255, 90, 90)),
                            (_FRAME_W // 2 - 220, _FRAME_H // 2 - 16))

            if args.scale == 1.0:
                screen.blit(canvas, (0, 0))
            else:
                screen.blit(pygame.transform.smoothscale(canvas, (win_w, win_h)), (0, 0))
            pygame.display.flip()
            clock.tick(args.fps_limit)

            if args.max_steps is not None and session_steps >= args.max_steps:
                print(f"[i] raggiunto il limite di {args.max_steps} step simulati -> chiusura")
                running = False

    finally:
        pygame.quit()


def _print_session_summary(sink: List[Dict[str, Any]]) -> None:
    """Riepilogo aggregato di tutta la sessione di gioco/test."""
    print("=" * 66)
    if not sink:
        print("SESSIONE TERMINATA senza episodi completati (nessun confine episodio).")
        return

    conteggi: Dict[str, int] = {}
    for entry in sink:
        conteggi[entry["esito"] or "unknown"] = conteggi.get(entry["esito"] or "unknown", 0) + 1

    distanze = [entry["distanza_max"] for entry in sink]
    labels = ", ".join(f"{_OUTCOME_LABELS.get(k, k)}={v}" for k, v in sorted(conteggi.items()))
    print(f"SESSIONE TERMINATA - episodi completati: {len(sink)}  [{labels}]")
    print(f"   distanza max media : {sum(distanze) / len(distanze):.1f} m   "
          f"migliore : {max(distanze):.1f} m")


def _list_missions() -> None:
    """Stampa le missioni disponibili con i loro parametri di terreno."""
    from play2learn_hillclimb.config import MISSIONS

    print("Missioni disponibili (--mission NOME):")
    for key, cfg in MISSIONS.items():
        print(f"  {key:18s} difficulty={cfg['difficulty']:7.1f}  "
              f"original_noise={str(cfg['original_noise']):5s}  {cfg['nome']}")


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    from play2learn_hillclimb.config import MISSIONS

    parser = argparse.ArgumentParser(
        description="Single player locale di Hill Climb Racing sull'ambiente identico al training PPO.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mission", default="campagna_standard", choices=sorted(MISSIONS.keys()),
                        help="missione/terreno da replicare (stessi parametri del training)")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed del terreno del primo episodio (42 = terreno del training)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="fattore di scala della finestra (0.3-2.0)")
    parser.add_argument("--fps-limit", type=int, default=60,
                        help="limite di frame al secondo del ciclo pygame")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="limite totale di step simulati prima della chiusura automatica")
    parser.add_argument("--autoplay", type=int, default=None, metavar="N",
                        help="esegue N step headless senza finestra con la politica a feedback "
                             "deterministica (smoke test riproducibile della fisica)")
    parser.add_argument("--log-csv", default=None, metavar="FILE.csv",
                        help="scrive un log CSV per-step della telemetria per analisi offline")
    parser.add_argument("--list-missions", action="store_true",
                        help="elenca le missioni disponibili ed esce")

    args = parser.parse_args(argv)
    if args.autoplay is not None and args.autoplay <= 0:
        parser.error("--autoplay richiede un numero di step positivo")
    if not (0.3 <= args.scale <= 2.0):
        parser.error("--scale deve stare nell'intervallo 0.3-2.0")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps deve essere positivo")
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    if args.list_missions:
        _list_missions()
        return 0

    autoplay = args.autoplay is not None

    # In modalita' finestra verifichiamo che non ci sia il driver video dummy
    # attivo per sbaglio (utile per capire subito perche' non appare la finestra).
    if not autoplay and os.environ.get("SDL_VIDEODRIVER", "").lower() == "dummy":
        print("[!] SDL_VIDEODRIVER=dummy attivo: la finestra non sara' visibile "
              "(modalita' smoke test). Deseleziona la variabile per giocare davvero.")

    env = _build_env(args.mission, seed=args.seed, windowed=not autoplay)

    csv_logger = _CsvLogger(args.log_csv) if args.log_csv else None
    sink: List[Dict[str, Any]] = []

    try:
        if autoplay:
            print(f"[i] autoplay headless: {args.autoplay} step con {_AUTOPLAY_LABEL}")
            _autoplay_run(env, _EpisodeState(1), args.autoplay, csv_logger, sink, args.seed)
        else:
            _windowed_run(env, args, csv_logger, sink, args.seed)
    except KeyboardInterrupt:
        print("\n[i] interrotto da tastiera (Ctrl+C)")
    finally:
        if csv_logger is not None:
            csv_logger.close()
        env.close()

    _print_session_summary(sink)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())