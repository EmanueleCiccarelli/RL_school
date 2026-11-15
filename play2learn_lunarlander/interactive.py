"""
Modulo di Simulazione e Gioco Interattivo (Play2Learn - Lunar Lander).
Permette agli studenti di pilotare manualmente il modulo lunare direttamente
in Google Colab o Jupyter Notebook (HTML5 Canvas + Web Audio API):
  1. Modalità Single Player: per comprendere la difficoltà del controllo manuale.
  2. Modalità Tu vs Agente: gara testa a testa (split-screen) contro l'agente PPO addestrato.
"""

from __future__ import annotations

import json
import random
import uuid
from pathlib import Path
from typing import Any

from play2learn_lunarlander.config import (
    MISSIONS,
    ENV_ID,
    FUEL_MAIN_ENGINE,
    FUEL_SIDE_ENGINE,
)
from play2learn_lunarlander.environment import classify_lunar_lander_outcome
from play2learn_lunarlander.favorites import carica_preferito
from play2learn_lunarlander.storage import find_run_by_timestamp, get_default_runs_dir


def _display_or_open_html(html_code: str, filename: str = "lunar_lander") -> Any:
    """Mostra l'HTML inline nel notebook oppure lo apre nel browser se non in ambiente IPython."""
    try:
        from IPython.display import HTML, display
        res = HTML(html_code)
        display(res)
        return res
    except ImportError:
        import webbrowser
        import tempfile
        tmp = Path(tempfile.gettempdir()) / f"{filename}.html"
        tmp.write_text(html_code, encoding="utf-8")
        print(f"[play2learn] Apertura del simulatore nel browser: {tmp}")
        webbrowser.open(tmp.as_uri())
        return html_code


def _extract_terrain_and_helipad(env: Any) -> tuple[list[list[float]], list[float]]:
    """Estrae i vertici della superficie lunare e le coordinate della piazzola da Gymnasium."""
    un = getattr(env, "unwrapped", env)
    fixtures = getattr(getattr(un, "moon", None), "fixtures", [])

    segments: list[tuple[list[float], list[float]]] = []
    for fix in fixtures:
        shape = getattr(fix, "shape", None)
        verts = getattr(shape, "vertices", None)
        if not verts or len(verts) != 2:
            continue
        p1 = [round(float(verts[0][0]), 3), round(float(verts[0][1]), 3)]
        p2 = [round(float(verts[1][0]), 3), round(float(verts[1][1]), 3)]
        # Salta il bordo inferiore del box [(0, 0), (W, 0)]
        if p1[1] == 0.0 and p2[1] == 0.0:
            continue
        if p1[0] > p2[0]:
            p1, p2 = p2, p1
        segments.append((p1, p2))

    points: list[list[float]] = []
    if segments:
        segments.sort(key=lambda seg: seg[0][0])
        points.append(segments[0][0])
        for _, p2 in segments:
            points.append(p2)
    else:
        points = [
            [0.0, 4.5], [2.0, 4.3], [4.0, 3.8], [6.0, 3.9],
            [8.0, 3.3], [10.0, 3.3], [12.0, 3.3],
            [14.0, 2.5], [16.0, 2.4], [18.0, 2.2], [20.0, 3.8]
        ]

    helipad = [
        round(float(getattr(un, "helipad_x1", 8.0)), 3),
        round(float(getattr(un, "helipad_x2", 12.0)), 3),
        round(float(getattr(un, "helipad_y", 3.333)), 3),
    ]
    return points, helipad


def _find_model_zip_in_dir(run_dir: Path) -> Path | None:
    """Trova il file checkpoint .zip in una cartella di run."""
    for fname in ("best_model.zip", "ppo_lunar_lander.zip", "final_model.zip"):
        cand = run_dir / fname
        if cand.exists():
            return cand
    zips = list(run_dir.glob("*.zip"))
    return zips[0] if zips else None


def _resolve_model(model_or_result: Any) -> Any:
    """
    Risolve l'agente PPO a partire da un oggetto LabResult, un'istanza PPO,
    un nome nei Preferiti o un timestamp di una run passata.
    """
    if model_or_result is None:
        try:
            model, _ = carica_preferito(0)
            return model
        except Exception:
            pass

        runs_dir = get_default_runs_dir()
        if runs_dir.exists():
            runs = sorted(
                [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
                key=lambda x: x.name,
                reverse=True,
            )
            for r in runs:
                tgt = _find_model_zip_in_dir(r)
                if tgt:
                    from stable_baselines3 import PPO
                    return PPO.load(str(tgt))

        raise ValueError(
            "❌ Nessun modello PPO fornito o trovato in memoria!\n"
            "Passa l'oggetto restituito da run_lab (es. `gioca_vs_agente(risultati)`),\n"
            "oppure specifica il nome di un preferito o un timestamp."
        )

    # 1. È già un LabResult o dizionario
    if hasattr(model_or_result, "get"):
        mod = model_or_result.get("model")
        if mod is not None:
            return mod
        bmp = model_or_result.get("best_model_path") or model_or_result.get("model_path")
        if bmp and Path(bmp).exists():
            from stable_baselines3 import PPO
            return PPO.load(str(bmp))

    # 2. È direttamente un modello PPO
    if hasattr(model_or_result, "predict"):
        return model_or_result

    # 3. È una stringa (preferito, timestamp o path)
    if isinstance(model_or_result, (str, Path)):
        p = Path(model_or_result)
        if p.exists() and p.suffix == ".zip":
            from stable_baselines3 import PPO
            return PPO.load(str(p))

        # Prova come preferito
        try:
            model, _ = carica_preferito(str(model_or_result))
            return model
        except Exception:
            pass

        # Prova come timestamp
        try:
            r_dir = find_run_by_timestamp(str(model_or_result))
            if r_dir:
                tgt = _find_model_zip_in_dir(r_dir)
                if tgt:
                    from stable_baselines3 import PPO
                    return PPO.load(str(tgt))
        except Exception:
            pass

    raise ValueError(f"Impossibile trovare o caricare il modello specificato: {model_or_result}")


def registra_traiettoria_agente(
    model: Any,
    seed: int = 42,
    mission: str = "luna_standard",
    env_kwargs: dict[str, Any] | None = None,
    max_steps: int = 800,
) -> dict[str, Any]:
    """
    Esegue 1 episodio deterministico dell'agente PPO nello scenario specificato
    e ne estrae la telemetria completa e la superficie per il confronto interattivo.
    """
    import gymnasium as gym

    mission_cfg = MISSIONS.get(mission, MISSIONS["luna_standard"])
    final_kwargs = dict(mission_cfg["env_kwargs"])
    if env_kwargs:
        final_kwargs.update(env_kwargs)

    try:
        env = gym.make(ENV_ID, disable_env_checker=True, **final_kwargs)
    except TypeError:
        env = gym.make(ENV_ID, **final_kwargs)

    try:
        obs, _ = env.reset(seed=seed)
        un = env.unwrapped
        terrain_pts, helipad = _extract_terrain_and_helipad(env)

        steps_data: list[dict[str, Any]] = []
        total_fuel = 0.0
        done = False
        step_count = 0
        is_success = False
        is_crashed = False

        while not done and step_count < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            act_int = int(action)

            if act_int == 2:
                total_fuel += FUEL_MAIN_ENGINE
            elif act_int in (1, 3):
                total_fuel += FUEL_SIDE_ENGINE

            obs, rew, term, trunc, _ = env.step(act_int)
            step_count += 1
            done = bool(term or trunc)

            pos = un.lander.position
            vel = un.lander.linearVelocity
            ang = un.lander.angle

            steps_data.append({
                "x": round(float(pos.x), 3),
                "y": round(float(pos.y), 3),
                "angle": round(float(ang), 3),
                "vx": round(float(vel.x), 3),
                "vy": round(float(vel.y), 3),
                "action": act_int,
                "fuel": round(float(total_fuel), 2),
            })

            if done:
                is_success, is_crashed, is_timeout, _ = classify_lunar_lander_outcome(
                    obs=obs,
                    original_reward=float(rew),
                    terminated=bool(term),
                    truncated=bool(trunc),
                    unwrapped=un,
                )
                if is_timeout:
                    is_crashed = True

        if not done:
            is_success = False
            is_crashed = True
    finally:
        env.close()

    final_vy = steps_data[-1]["vy"] if steps_data else 0.0
    final_angle = steps_data[-1]["angle"] if steps_data else 0.0

    return {
        "terrain": terrain_pts,
        "helipad": helipad,
        "steps": steps_data,
        "stats": {
            "success": is_success,
            "crashed": is_crashed,
            "fuel_used": round(total_fuel, 1),
            "steps": step_count,
            "time_sec": round(step_count / 50.0, 1),
            "final_vy": final_vy,
            "final_angle_deg": round(final_angle * 180.0 / 3.14159, 1),
        },
    }


def _get_shared_game_js() -> str:
    """Restituisce il codice JavaScript essenziale per la fisica e il rendering del modulo lunare."""
    return r"""
    // Audio sintetico con Web Audio API (zero file esterni)
    let audioCtx = null;
    let soundEnabled = true;
    function getAudioCtx() {
        if (!audioCtx) {
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        }
        if (audioCtx.state === 'suspended') {
            audioCtx.resume();
        }
        return audioCtx;
    }

    function playThrusterSound(isMain) {
        if (!soundEnabled) return;
        try {
            const ctx = getAudioCtx();
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = isMain ? 'sawtooth' : 'triangle';
            osc.frequency.setValueAtTime(isMain ? 65 : 120, ctx.currentTime);
            gain.gain.setValueAtTime(0.04, ctx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.08);
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.start();
            osc.stop(ctx.currentTime + 0.08);
        } catch(e) {}
    }

    function playLandingSound() {
        if (!soundEnabled) return;
        try {
            const ctx = getAudioCtx();
            const notes = [261.63, 329.63, 392.00, 523.25]; // C E G C
            notes.forEach((freq, idx) => {
                const osc = ctx.createOscillator();
                const gain = ctx.createGain();
                osc.type = 'sine';
                osc.frequency.setValueAtTime(freq, ctx.currentTime + idx * 0.1);
                gain.gain.setValueAtTime(0.08, ctx.currentTime + idx * 0.1);
                gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + idx * 0.1 + 0.25);
                osc.connect(gain);
                gain.connect(ctx.destination);
                osc.start(ctx.currentTime + idx * 0.1);
                osc.stop(ctx.currentTime + idx * 0.1 + 0.25);
            });
        } catch(e) {}
    }

    function playCrashSound() {
        if (!soundEnabled) return;
        try {
            const ctx = getAudioCtx();
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = 'sawtooth';
            osc.frequency.setValueAtTime(140, ctx.currentTime);
            osc.frequency.exponentialRampToValueAtTime(30, ctx.currentTime + 0.4);
            gain.gain.setValueAtTime(0.12, ctx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.4);
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.start();
            osc.stop(ctx.currentTime + 0.4);
        } catch(e) {}
    }

    // Disegno vettoriale dettagliato del modulo lunare stile Apollo
    function drawLander(ctx, x, y, angle, isMain, isLeft, isRight, isAi, isLanded, isCrashed) {
        ctx.save();
        ctx.translate(x, y);
        // Correzione asse di rotazione Canvas: l'angolo di Box2D e' antiorario (y-up), il canvas e' orario (y-down)
        ctx.rotate(-angle);

        if (isCrashed) {
            ctx.fillStyle = '#ef4444';
            ctx.beginPath();
            ctx.arc(0, 0, 16, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = '#991b1b';
            ctx.lineWidth = 2;
            ctx.stroke();
            ctx.restore();
            return;
        }

        const primaryCol = isAi ? '#10b981' : '#3b82f6';
        const accentCol = isAi ? '#047857' : '#1d4ed8';
        const foilCol = isAi ? '#d1fae5' : '#f59e0b';

        // Fiamme motore principale
        if (isMain && !isLanded) {
            ctx.save();
            ctx.fillStyle = '#f97316';
            ctx.beginPath();
            ctx.moveTo(-7, 12);
            ctx.lineTo(0, 24 + Math.random() * 12);
            ctx.lineTo(7, 12);
            ctx.closePath();
            ctx.fill();

            ctx.fillStyle = '#fef08a';
            ctx.beginPath();
            ctx.moveTo(-4, 12);
            ctx.lineTo(0, 19 + Math.random() * 6);
            ctx.lineTo(4, 12);
            ctx.closePath();
            ctx.fill();
            ctx.restore();
        }

        // Getti laterali RCS
        if (isLeft && !isLanded) {
            ctx.save();
            ctx.fillStyle = '#38bdf8';
            ctx.beginPath();
            ctx.moveTo(11, -2);
            ctx.lineTo(21 + Math.random() * 6, -2);
            ctx.lineTo(11, 2);
            ctx.closePath();
            ctx.fill();
            ctx.restore();
        }
        if (isRight && !isLanded) {
            ctx.save();
            ctx.fillStyle = '#38bdf8';
            ctx.beginPath();
            ctx.moveTo(-11, -2);
            ctx.lineTo(-21 - Math.random() * 6, -2);
            ctx.lineTo(-11, 2);
            ctx.closePath();
            ctx.fill();
            ctx.restore();
        }

        // Zampe di atterraggio e pattini ammortizzati
        ctx.strokeStyle = '#94a3b8';
        ctx.lineWidth = 2.4;
        ctx.beginPath();
        ctx.moveTo(-9, 4);
        ctx.lineTo(-16, 14);
        ctx.moveTo(9, 4);
        ctx.lineTo(16, 14);
        ctx.stroke();

        ctx.fillStyle = '#64748b';
        ctx.fillRect(-20, 13, 8, 2.5);
        ctx.fillRect(12, 13, 8, 2.5);

        // Corpo inferiore dorato
        ctx.fillStyle = foilCol;
        ctx.fillRect(-11, 2, 22, 10);
        ctx.strokeStyle = '#78350f';
        ctx.lineWidth = 1;
        ctx.strokeRect(-11, 2, 22, 10);

        // Ugello motore centrale
        ctx.fillStyle = '#334155';
        ctx.beginPath();
        ctx.moveTo(-6, 12);
        ctx.lineTo(-9, 16);
        ctx.lineTo(9, 16);
        ctx.lineTo(6, 12);
        ctx.closePath();
        ctx.fill();

        // Modulo superiore (Cabina di pilotaggio)
        ctx.fillStyle = primaryCol;
        ctx.beginPath();
        ctx.moveTo(-12, 2);
        ctx.lineTo(-10, -10);
        ctx.lineTo(0, -14);
        ctx.lineTo(10, -10);
        ctx.lineTo(12, 2);
        ctx.closePath();
        ctx.fill();
        ctx.strokeStyle = accentCol;
        ctx.lineWidth = 1.8;
        ctx.stroke();

        // Oblò frontale con riflesso
        ctx.fillStyle = '#0f172a';
        ctx.beginPath();
        ctx.arc(0, -5, 4.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = '#38bdf8';
        ctx.beginPath();
        ctx.arc(-1, -6, 2, 0, Math.PI * 2);
        ctx.fill();

        // Antenna superiore
        ctx.strokeStyle = '#cbd5e1';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(0, -14);
        ctx.lineTo(0, -19);
        ctx.stroke();
        ctx.fillStyle = '#ef4444';
        ctx.beginPath();
        ctx.arc(0, -19, 1.8, 0, Math.PI * 2);
        ctx.fill();

        ctx.restore();
    }

    // Disegno superficie lunare, piazzola e bandierine
    function drawTerrain(ctx, terrain, helipad, W, H) {
        const sx = W / 20.0;
        const sy = H / 13.333;

        ctx.fillStyle = '#050814';
        ctx.fillRect(0, 0, W, H);

        ctx.fillStyle = '#1e293b';
        ctx.beginPath();
        ctx.moveTo(0, H);
        terrain.forEach((pt, i) => {
            const px = pt[0] * sx;
            const py = H - pt[1] * sy;
            if (i === 0) ctx.lineTo(px, py);
            else ctx.lineTo(px, py);
        });
        ctx.lineTo(W, H);
        ctx.closePath();
        ctx.fill();

        ctx.strokeStyle = '#475569';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        terrain.forEach((pt, i) => {
            const px = pt[0] * sx;
            const py = H - pt[1] * sy;
            if (i === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
        });
        ctx.stroke();

        const hx1 = helipad[0] * sx;
        const hx2 = helipad[1] * sx;
        const hy = H - helipad[2] * sy;
        const hw = hx2 - hx1;

        ctx.fillStyle = '#334155';
        ctx.fillRect(hx1, hy - 3, hw, 7);
        ctx.strokeStyle = '#fbbf24';
        ctx.lineWidth = 2;
        ctx.strokeRect(hx1, hy - 3, hw, 7);

        ctx.save();
        ctx.beginPath();
        ctx.rect(hx1, hy - 3, hw, 7);
        ctx.clip();
        ctx.strokeStyle = '#fbbf24';
        ctx.lineWidth = 3;
        for (let x = hx1 - 10; x < hx2 + 10; x += 8) {
            ctx.beginPath();
            ctx.moveTo(x, hy + 4);
            ctx.lineTo(x + 6, hy - 4);
            ctx.stroke();
        }
        ctx.restore();

        drawFlag(ctx, hx1 + 3, hy - 3, '#ef4444');
        drawFlag(ctx, hx2 - 3, hy - 3, '#ef4444');
    }

    function drawFlag(ctx, x, y, col) {
        ctx.strokeStyle = '#e2e8f0';
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x, y - 16);
        ctx.stroke();

        ctx.fillStyle = col;
        ctx.beginPath();
        ctx.moveTo(x, y - 16);
        ctx.lineTo(x + 8, y - 12);
        ctx.lineTo(x, y - 8);
        ctx.closePath();
        ctx.fill();
    }
    """


def gioca_single_player(
    seed: int | None = None,
    mission: str = "luna_standard",
    gravity: float | None = None,
    wind: bool | None = None,
    env_kwargs: dict[str, Any] | None = None,
) -> Any:
    """
    Avvia la modalità arcade Single Player interattiva in Google Colab o Jupyter Notebook.
    Permette allo studente di pilotare manualmente il modulo lunare con frecce da tastiera
    o pulsanti touch a schermo.
    """
    import gymnasium as gym

    actual_seed = seed if seed is not None else random.randint(100, 99999)
    mission_cfg = MISSIONS.get(mission, MISSIONS["luna_standard"])
    final_env_kwargs = dict(mission_cfg["env_kwargs"])
    if env_kwargs:
        final_env_kwargs.update(env_kwargs)
    if gravity is not None:
        final_env_kwargs["gravity"] = float(gravity)
    if wind is not None:
        final_env_kwargs["enable_wind"] = bool(wind)

    try:
        env = gym.make(ENV_ID, disable_env_checker=True, **final_env_kwargs)
    except TypeError:
        env = gym.make(ENV_ID, **final_env_kwargs)

    env.reset(seed=actual_seed)
    terrain_pts, helipad = _extract_terrain_and_helipad(env)
    env.close()

    terrain_json = json.dumps(terrain_pts)
    helipad_json = json.dumps(helipad)
    shared_js = _get_shared_game_js()
    uid = uuid.uuid4().hex[:6]

    html_code = f"""
    <div id="sp_root_{uid}" style="font-family:'Segoe UI',sans-serif; background:#0b0f19; color:#f8fafc; padding:18px; border-radius:12px; max-width:720px; margin:10px auto; box-shadow:0 6px 20px rgba(0,0,0,0.5);">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; border-bottom:1px solid #1e293b; padding-bottom:8px;">
            <div>
                <h3 style="margin:0; color:#38bdf8; font-size:18px;">🚀 Simulatore Manuale: Pilota il Modulo Lunare!</h3>
                <span style="font-size:12px; color:#94a3b8;">Missione: {mission_cfg['title']} (Seed: {actual_seed})</span>
            </div>
            <div>
                <button id="btn_sound_{uid}" style="background:#1e293b; color:#cbd5e1; border:1px solid #475569; padding:4px 10px; border-radius:6px; font-size:12px; cursor:pointer;">🔊 Audio: ON</button>
            </div>
        </div>

        <div style="position:relative; width:100%; text-align:center;">
            <canvas id="sp_canvas_{uid}" width="640" height="400" tabindex="0" style="background:#050814; border:1.5px solid #334155; border-radius:8px; outline:none; max-width:100%; cursor:pointer;"></canvas>
            <div id="sp_overlay_{uid}" style="position:absolute; top:45%; left:50%; transform:translate(-50%,-50%); background:rgba(15,23,42,0.92); border:1.5px solid #38bdf8; padding:14px 22px; border-radius:8px; text-align:center;">
                <div style="font-size:15px; font-weight:bold; color:#f8fafc; margin-bottom:6px;">Clicca qui o premi [SPAZIO] per decollare!</div>
                <div style="font-size:12px; color:#94a3b8;">Comandi: Frecce ⬆️ ⬅️ ➡️ oppure tasti W / A / D</div>
            </div>
        </div>

        <div style="display:grid; grid-template-columns:repeat(4, 1fr); gap:8px; margin-top:12px; font-size:12px; background:#111827; padding:10px; border-radius:8px; border:1px solid #1f2937;">
            <div><span style="color:#94a3b8;">Carburante:</span> <strong id="hud_fuel_{uid}" style="color:#10b981;">100%</strong></div>
            <div><span style="color:#94a3b8;">Quota Y:</span> <strong id="hud_y_{uid}">13.3 m</strong></div>
            <div><span style="color:#94a3b8;">Velocità VY:</span> <strong id="hud_vy_{uid}" style="color:#10b981;">0.0 m/s</strong></div>
            <div><span style="color:#94a3b8;">Assetto / Tilt:</span> <strong id="hud_ang_{uid}" style="color:#10b981;">0°</strong></div>
        </div>

        <div style="display:flex; justify-content:center; gap:12px; margin-top:14px;">
            <button id="btn_left_{uid}" style="background:#1e293b; color:#38bdf8; border:1.5px solid #0284c7; padding:10px 18px; border-radius:8px; font-weight:bold; cursor:pointer; font-size:14px; user-select:none;">⬅️ Razzo Dx</button>
            <button id="btn_main_{uid}" style="background:#0369a1; color:#fff; border:1.5px solid #38bdf8; padding:10px 24px; border-radius:8px; font-weight:bold; cursor:pointer; font-size:14px; user-select:none;">🚀 MOTORE PRINCIPALE</button>
            <button id="btn_right_{uid}" style="background:#1e293b; color:#38bdf8; border:1.5px solid #0284c7; padding:10px 18px; border-radius:8px; font-weight:bold; cursor:pointer; font-size:14px; user-select:none;">➡️ Razzo Sx</button>
        </div>

        <div style="text-align:center; margin-top:12px;">
            <button id="btn_reset_{uid}" style="background:#334155; color:#f1f5f9; border:none; padding:6px 16px; border-radius:6px; font-size:12px; cursor:pointer;">🔄 Riavvia Volo (Stesso Terreno)</button>
        </div>
    </div>

    <script>
    (function() {{
        {shared_js}

        const terrain = {terrain_json};
        const helipad = {helipad_json};
        const canvas = document.getElementById('sp_canvas_{uid}');
        const ctx = canvas.getContext('2d');
        const overlay = document.getElementById('sp_overlay_{uid}');
        const W = canvas.width;
        const H = canvas.height;

        let x = 10.0, y = 13.2, vx = 0.0, vy = 0.0, angle = 0.0, vangle = 0.0;
        let fuel = 100.0, fuelUsed = 0.0;
        let gameActive = false;
        let isLanded = false;
        let isCrashed = false;
        let keyMain = false, keyLeft = false, keyRight = false;
        let animId = null;

        function getTerrainY(px) {{
            for (let i = 0; i < terrain.length - 1; i++) {{
                const p1 = terrain[i], p2 = terrain[i+1];
                if (px >= p1[0] && px <= p2[0]) {{
                    const t = (px - p1[0]) / (p2[0] - p1[0]);
                    return p1[1] + t * (p2[1] - p1[1]);
                }}
            }}
            return 3.0;
        }}

        function resetGame() {{
            x = 10.0 + (Math.random() - 0.5) * 1.5;
            y = 13.2;
            vx = (Math.random() - 0.5) * 0.4;
            vy = 0.0;
            angle = (Math.random() - 0.5) * 0.08;
            vangle = 0.0;
            fuel = 100.0;
            fuelUsed = 0.0;
            isLanded = false;
            isCrashed = false;
            gameActive = true;
            overlay.style.display = 'none';
            canvas.focus();
        }}

        function updatePhysics(dt) {{
            if (!gameActive || isLanded || isCrashed) return;

            const g = {float(final_env_kwargs.get('gravity', -10.0))};
            let ax = 0, ay = g;

            if (keyMain && fuel > 0) {{
                const fMain = 14.5;
                ax += -Math.sin(angle) * fMain;
                ay += Math.cos(angle) * fMain;
                fuel = Math.max(0, fuel - 0.4);
                fuelUsed += 1.0;
                playThrusterSound(true);
            }}

            if (keyLeft && fuel > 0) {{
                // Razzo destro -> spinge verso sinistra (-x)
                ax -= Math.cos(angle) * 1.6;
                ay -= Math.sin(angle) * 1.6;
                vangle += 2.2 * dt;
                fuel = Math.max(0, fuel - 0.15);
                fuelUsed += 0.3;
                playThrusterSound(false);
            }}
            if (keyRight && fuel > 0) {{
                // Razzo sinistro -> spinge verso destra (+x)
                ax += Math.cos(angle) * 1.6;
                ay += Math.sin(angle) * 1.6;
                vangle -= 2.2 * dt;
                fuel = Math.max(0, fuel - 0.15);
                fuelUsed += 0.3;
                playThrusterSound(false);
            }}

            vangle *= 0.985;
            vx += ax * dt;
            vy += ay * dt;
            x += vx * dt;
            y += vy * dt;
            angle += vangle * dt;

            if (x < 0.6) {{ x = 0.6; vx = 0; }}
            if (x > 19.4) {{ x = 19.4; vx = 0; }}

            const groundY = getTerrainY(x);
            if (y - 0.5 <= groundY) {{
                y = groundY + 0.5;
                gameActive = false;
                const onPad = (x >= helipad[0] && x <= helipad[1]);
                const softY = Math.abs(vy) < 2.2;
                const softX = Math.abs(vx) < 1.3;
                const upright = Math.abs(angle) < 0.28;

                if (onPad && softY && softX && upright) {{
                    isLanded = true;
                    playLandingSound();
                    overlay.innerHTML = '<div style="color:#10b981; font-size:16px; font-weight:bold;">🎉 ALLUNAGGIO RIUSCITO!</div><div style="font-size:12px; color:#cbd5e1; margin-top:4px;">Carburante usato: ' + fuelUsed.toFixed(1) + ' un. | VY: ' + Math.abs(vy).toFixed(1) + ' m/s</div><button id="btn_retry_' + '{uid}' + '" style="margin-top:8px; background:#10b981; color:#fff; border:none; padding:4px 12px; border-radius:4px; cursor:pointer;">Rigioca</button>';
                }} else {{
                    isCrashed = true;
                    playCrashSound();
                    let causa = !onPad ? "Atterraggio fuori dalla piazzola!" : (!softY ? "Impatto troppo violento (VY > 2.2 m/s)!" : "Modulo troppo inclinato (> 15°)!");
                    overlay.innerHTML = '<div style="color:#ef4444; font-size:16px; font-weight:bold;">💥 SCHIANTO AL SUOLO!</div><div style="font-size:12px; color:#fca5a5; margin-top:4px;">' + causa + '</div><button id="btn_retry_' + '{uid}' + '" style="margin-top:8px; background:#ef4444; color:#fff; border:none; padding:4px 12px; border-radius:4px; cursor:pointer;">Riprova</button>';
                }}
                overlay.style.display = 'block';
                const retryBtn = document.getElementById('btn_retry_' + '{uid}');
                if (retryBtn) retryBtn.addEventListener('click', resetGame);
            }}
        }}

        function drawScene() {{
            drawTerrain(ctx, terrain, helipad, W, H);
            const sx = W / 20.0;
            const sy = H / 13.333;
            const screenX = x * sx;
            const screenY = H - y * sy;

            drawLander(ctx, screenX, screenY, angle, keyMain && fuel > 0, keyLeft && fuel > 0, keyRight && fuel > 0, false, isLanded, isCrashed);

            const fuelElem = document.getElementById('hud_fuel_{uid}');
            if (fuelElem) {{
                fuelElem.innerText = fuel.toFixed(0) + '%';
                fuelElem.style.color = fuel > 40 ? '#10b981' : (fuel > 15 ? '#f59e0b' : '#ef4444');
            }}
            const yElem = document.getElementById('hud_y_{uid}');
            if (yElem) yElem.innerText = Math.max(0, y - getTerrainY(x)).toFixed(1) + ' m';
            const hudVy = document.getElementById('hud_vy_{uid}');
            if (hudVy) {{
                hudVy.innerText = vy.toFixed(1) + ' m/s';
                hudVy.style.color = Math.abs(vy) < 2.2 ? '#10b981' : '#ef4444';
            }}
            const hudAng = document.getElementById('hud_ang_{uid}');
            if (hudAng) {{
                const angDeg = (angle * 180 / Math.PI).toFixed(0);
                hudAng.innerText = angDeg + '°';
                hudAng.style.color = Math.abs(angle) < 0.28 ? '#10b981' : '#ef4444';
            }}
        }}

        // Accumulatore a 50 Hz fisso: indipendente dal refresh rate del monitor (60Hz, 120Hz, 144Hz)
        const FIXED_DT = 0.02;
        let lastFrameTs = performance.now();
        let accumulator = 0.0;

        function loop(now) {{
            const frameSec = Math.min(0.1, (now - lastFrameTs) / 1000.0);
            lastFrameTs = now;
            accumulator += frameSec;
            while (accumulator >= FIXED_DT) {{
                updatePhysics(FIXED_DT);
                accumulator -= FIXED_DT;
            }}
            drawScene();
            animId = requestAnimationFrame(loop);
        }}

        canvas.addEventListener('keydown', function(e) {{
            if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', ' '].includes(e.key)) {{
                e.preventDefault();
            }}
            if (!gameActive && (e.key === ' ' || e.key === 'ArrowUp')) {{
                resetGame();
                return;
            }}
            if (e.key === 'ArrowUp' || e.key === 'w' || e.key === 'W' || e.key === ' ') keyMain = true;
            if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') keyLeft = true;
            if (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') keyRight = true;
        }});

        canvas.addEventListener('keyup', function(e) {{
            if (e.key === 'ArrowUp' || e.key === 'w' || e.key === 'W' || e.key === ' ') keyMain = false;
            if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') keyLeft = false;
            if (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') keyRight = false;
        }});

        overlay.addEventListener('click', resetGame);
        document.getElementById('btn_reset_{uid}').addEventListener('click', resetGame);

        function bindBtn(btnId, setFlag) {{
            const btn = document.getElementById(btnId);
            if (!btn) return;
            const onDown = (e) => {{ e.preventDefault(); if (!gameActive) resetGame(); setFlag(true); }};
            const onUp = (e) => {{ e.preventDefault(); setFlag(false); }};
            btn.addEventListener('mousedown', onDown);
            btn.addEventListener('mouseup', onUp);
            btn.addEventListener('mouseleave', onUp);
            btn.addEventListener('touchstart', onDown);
            btn.addEventListener('touchend', onUp);
            btn.addEventListener('touchcancel', onUp);
        }}
        bindBtn('btn_main_{uid}', (v) => keyMain = v);
        bindBtn('btn_left_{uid}', (v) => keyLeft = v);
        bindBtn('btn_right_{uid}', (v) => keyRight = v);

        const sndBtn = document.getElementById('btn_sound_{uid}');
        sndBtn.addEventListener('click', function() {{
            soundEnabled = !soundEnabled;
            sndBtn.innerText = soundEnabled ? '🔊 Audio: ON' : '🔇 Audio: OFF';
        }});

        loop(performance.now());
    }})();
    </script>
    """
    return _display_or_open_html(html_code, "lunar_lander_single_player")


def gioca_vs_agente(
    model_or_result: Any = None,
    seed: int = 42,
    mission: str = "luna_standard",
    gravity: float | None = None,
    wind: bool | None = None,
    model: Any = None,
    env_kwargs: dict[str, Any] | None = None,
) -> Any:
    """
    Avvia la modalità testa a testa 'Tu vs Agente' (Split-Screen interattivo)
    in Google Colab o Jupyter Notebook.
    Permette allo studente di gareggiare in tempo reale sullo stesso identico terreno lunare
    contro la policy PPO addestrata, con telemetria simultanea e verdetto finale!
    """
    chosen_model = model if model is not None else model_or_result
    resolved_model = _resolve_model(chosen_model)
    mission_cfg = MISSIONS.get(mission, MISSIONS["luna_standard"])
    final_env_kwargs = dict(mission_cfg["env_kwargs"])
    if env_kwargs:
        final_env_kwargs.update(env_kwargs)
    if gravity is not None:
        final_env_kwargs["gravity"] = float(gravity)
    if wind is not None:
        final_env_kwargs["enable_wind"] = bool(wind)

    print("[play2learn] Calcolo della traiettoria dell'Agente AI sullo scenario lunare...")
    agent_data = registra_traiettoria_agente(
        model=resolved_model,
        seed=seed,
        mission=mission,
        env_kwargs=final_env_kwargs,
    )

    terrain_json = json.dumps(agent_data["terrain"])
    helipad_json = json.dumps(agent_data["helipad"])
    steps_json = json.dumps(agent_data["steps"])
    stats_json = json.dumps(agent_data["stats"])
    shared_js = _get_shared_game_js()
    uid = uuid.uuid4().hex[:6]

    html_code = f"""
    <div id="vs_root_{uid}" style="font-family:'Segoe UI',sans-serif; background:#0b0f19; color:#f8fafc; padding:18px; border-radius:12px; max-width:960px; margin:10px auto; box-shadow:0 6px 20px rgba(0,0,0,0.5);">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; border-bottom:1px solid #1e293b; padding-bottom:8px;">
            <div>
                <h3 style="margin:0; color:#38bdf8; font-size:19px;">⚔️ Sfida Allunaggio: Tu (Pilota Umano) vs Agente AI (PPO)</h3>
                <span style="font-size:12px; color:#94a3b8;">Gara simultanea sullo stesso terreno (Seed: {seed} | Missione: {mission_cfg['title']})</span>
            </div>
            <div>
                <button id="vs_btn_sound_{uid}" style="background:#1e293b; color:#cbd5e1; border:1px solid #475569; padding:4px 10px; border-radius:6px; font-size:12px; cursor:pointer;">🔊 Audio: ON</button>
            </div>
        </div>

        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:14px; position:relative;">
            <div style="background:#111827; padding:10px; border-radius:8px; border:1.5px solid #0284c7; text-align:center;">
                <div style="font-weight:bold; color:#38bdf8; font-size:14px; margin-bottom:6px;">🧑 TU (Pilota Umano) — Lander Blu</div>
                <canvas id="canvas_human_{uid}" width="440" height="300" tabindex="0" style="background:#050814; border-radius:6px; outline:none; max-width:100%; cursor:pointer;"></canvas>
                <div style="display:flex; justify-content:space-between; font-size:11.5px; color:#cbd5e1; margin-top:6px; background:#1e293b; padding:6px 10px; border-radius:6px;">
                    <span>Carb Speso: <strong id="h_hud_fuel_{uid}" style="color:#10b981;">0.0</strong></span>
                    <span>Quota: <strong id="h_hud_y_{uid}">13.2m</strong></span>
                    <span>VY: <strong id="h_hud_vy_{uid}" style="color:#10b981;">0.0m/s</strong></span>
                    <span>Stato: <strong id="h_hud_status_{uid}" style="color:#38bdf8;">Pronto</strong></span>
                </div>
            </div>

            <div style="background:#111827; padding:10px; border-radius:8px; border:1.5px solid #059669; text-align:center;">
                <div style="font-weight:bold; color:#10b981; font-size:14px; margin-bottom:6px;">🤖 AGENTE AI (PPO) — Lander Verde</div>
                <canvas id="canvas_ai_{uid}" width="440" height="300" style="background:#050814; border-radius:6px; max-width:100%;"></canvas>
                <div style="display:flex; justify-content:space-between; font-size:11.5px; color:#cbd5e1; margin-top:6px; background:#1e293b; padding:6px 10px; border-radius:6px;">
                    <span>Carb Speso: <strong id="ai_hud_fuel_{uid}" style="color:#10b981;">0.0</strong></span>
                    <span>Quota: <strong id="ai_hud_y_{uid}">13.2m</strong></span>
                    <span>VY: <strong id="ai_hud_vy_{uid}" style="color:#10b981;">0.0m/s</strong></span>
                    <span>Stato: <strong id="ai_hud_status_{uid}" style="color:#10b981;">Pronto</strong></span>
                </div>
            </div>

            <div id="vs_overlay_{uid}" style="position:absolute; top:45%; left:50%; transform:translate(-50%,-50%); background:rgba(15,23,42,0.95); border:2px solid #38bdf8; padding:16px 28px; border-radius:10px; text-align:center; z-index:20; box-shadow:0 8px 30px rgba(0,0,0,0.7);">
                <div style="font-size:16px; font-weight:bold; color:#f8fafc; margin-bottom:8px;">Pronto per la sfida Uomo vs Macchina?</div>
                <div style="font-size:12px; color:#94a3b8; margin-bottom:12px;">Decolla con il lander di sinistra e cerca di allunare consumando meno carburante dell'agente!</div>
                <button id="vs_btn_start_{uid}" style="background:#0284c7; color:#fff; border:none; padding:8px 20px; border-radius:6px; font-weight:bold; font-size:14px; cursor:pointer;">🚀 INIZIA LA SFIDA!</button>
            </div>
        </div>

        <div style="display:flex; justify-content:center; gap:12px; margin-top:14px;">
            <button id="vs_btn_left_{uid}" style="background:#1e293b; color:#38bdf8; border:1.5px solid #0284c7; padding:9px 18px; border-radius:8px; font-weight:bold; cursor:pointer; font-size:13px; user-select:none;">⬅️ Razzo Dx</button>
            <button id="vs_btn_main_{uid}" style="background:#0369a1; color:#fff; border:1.5px solid #38bdf8; padding:9px 24px; border-radius:8px; font-weight:bold; cursor:pointer; font-size:13px; user-select:none;">🚀 MOTORE PRINCIPALE</button>
            <button id="vs_btn_right_{uid}" style="background:#1e293b; color:#38bdf8; border:1.5px solid #0284c7; padding:9px 18px; border-radius:8px; font-weight:bold; cursor:pointer; font-size:13px; user-select:none;">➡️ Razzo Sx</button>
        </div>

        <div id="vs_scoreboard_{uid}" style="display:none; margin-top:14px; background:#1e293b; border:1.5px solid #38bdf8; border-radius:8px; padding:14px; text-align:center;">
            <h4 id="vs_verdetto_titolo_{uid}" style="margin:0 0 8px 0; font-size:16px; color:#f59e0b;">🏆 VERDETTO DELLA SFIDA</h4>
            <div id="vs_verdetto_desc_{uid}" style="font-size:13px; color:#e2e8f0; margin-bottom:10px;"></div>
            <button id="vs_btn_replay_{uid}" style="background:#0284c7; color:#fff; border:none; padding:6px 16px; border-radius:6px; font-size:12px; cursor:pointer; font-weight:bold;">🔄 Rigioca Sfida</button>
        </div>
    </div>

    <script>
    (function() {{
        {shared_js}

        const terrain = {terrain_json};
        const helipad = {helipad_json};
        const aiSteps = {steps_json};
        const aiStats = {stats_json};

        const canvasH = document.getElementById('canvas_human_{uid}');
        const ctxH = canvasH.getContext('2d');
        const canvasA = document.getElementById('canvas_ai_{uid}');
        const ctxA = canvasA.getContext('2d');
        const overlay = document.getElementById('vs_overlay_{uid}');
        const scoreboard = document.getElementById('vs_scoreboard_{uid}');

        const W = canvasH.width;
        const H = canvasH.height;

        let gameRunning = false;
        let hX = 10.0, hY = 13.2, hVx = 0.0, hVy = 0.0, hAngle = 0.0, hVangle = 0.0;
        let hFuel = 100.0, hFuelUsed = 0.0;
        let hLanded = false, hCrashed = false;
        let hKeyMain = false, hKeyLeft = false, hKeyRight = false;
        let hStepsCount = 0;

        let aiIdx = 0;
        let aiFinished = false;
        let animId = null;

        function getTerrainY(px) {{
            for (let i = 0; i < terrain.length - 1; i++) {{
                const p1 = terrain[i], p2 = terrain[i+1];
                if (px >= p1[0] && px <= p2[0]) {{
                    const t = (px - p1[0]) / (p2[0] - p1[0]);
                    return p1[1] + t * (p2[1] - p1[1]);
                }}
            }}
            return 3.0;
        }}

        function startChallenge() {{
            hX = (aiSteps.length > 0 ? aiSteps[0].x : 10.0);
            hY = (aiSteps.length > 0 ? aiSteps[0].y : 13.2);
            hVx = 0.0;
            hVy = 0.0;
            hAngle = 0.0;
            hVangle = 0.0;
            hFuel = 100.0;
            hFuelUsed = 0.0;
            hLanded = false;
            hCrashed = false;
            hStepsCount = 0;

            aiIdx = 0;
            aiFinished = false;
            gameRunning = true;

            overlay.style.display = 'none';
            scoreboard.style.display = 'none';
            canvasH.focus();
        }}

        function updateHumanPhysics(dt) {{
            if (!gameRunning || hLanded || hCrashed) return;

            const g = {float(final_env_kwargs.get('gravity', -10.0))};
            let ax = 0, ay = g;

            if (hKeyMain && hFuel > 0) {{
                const fMain = 14.5;
                ax += -Math.sin(hAngle) * fMain;
                ay += Math.cos(hAngle) * fMain;
                hFuel = Math.max(0, hFuel - 0.4);
                hFuelUsed += 1.0;
                playThrusterSound(true);
            }}
            if (hKeyLeft && hFuel > 0) {{
                // Razzo destro -> spinta verso sinistra (-x)
                ax -= Math.cos(hAngle) * 1.6;
                ay -= Math.sin(hAngle) * 1.6;
                hVangle += 2.2 * dt;
                hFuel = Math.max(0, hFuel - 0.15);
                hFuelUsed += 0.3;
                playThrusterSound(false);
            }}
            if (hKeyRight && hFuel > 0) {{
                // Razzo sinistro -> spinta verso destra (+x)
                ax += Math.cos(hAngle) * 1.6;
                ay += Math.sin(hAngle) * 1.6;
                hVangle -= 2.2 * dt;
                hFuel = Math.max(0, hFuel - 0.15);
                hFuelUsed += 0.3;
                playThrusterSound(false);
            }}

            hVangle *= 0.985;
            hVx += ax * dt;
            hVy += ay * dt;
            hX += hVx * dt;
            hY += hVy * dt;
            hAngle += hVangle * dt;
            hStepsCount++;

            if (hX < 0.6) {{ hX = 0.6; hVx = 0; }}
            if (hX > 19.4) {{ hX = 19.4; hVx = 0; }}

            const groundY = getTerrainY(hX);
            if (hY - 0.5 <= groundY) {{
                hY = groundY + 0.5;
                const onPad = (hX >= helipad[0] && hX <= helipad[1]);
                const softY = Math.abs(hVy) < 2.2;
                const softX = Math.abs(hVx) < 1.3;
                const upright = Math.abs(hAngle) < 0.28;

                if (onPad && softY && softX && upright) {{
                    hLanded = true;
                    playLandingSound();
                }} else {{
                    hCrashed = true;
                    playCrashSound();
                }}
                checkBothFinished();
            }}
        }}

        function checkBothFinished() {{
            const hDone = hLanded || hCrashed;
            const aDone = aiFinished;
            if (hDone && aDone) {{
                gameRunning = false;
                evaluateWinner();
            }}
        }}

        function evaluateWinner() {{
            const vTitolo = document.getElementById('vs_verdetto_titolo_{uid}');
            const vDesc = document.getElementById('vs_verdetto_desc_{uid}');

            let winnerMsg = "";
            let badgeCol = "#f59e0b";

            const hSucc = hLanded;
            const aiSucc = aiStats.success;

            if (hSucc && !aiSucc) {{
                winnerMsg = "👑 VITTORIA DEL PILOTA UMANO! Hai completato l'allunaggio mentre l'IA ha fallito!";
                badgeCol = "#10b981";
            }} else if (!hSucc && aiSucc) {{
                winnerMsg = "🤖 VITTORIA DELL'INTELLIGENZA ARTIFICIALE! L'agente autonomo ha effettuato un allunaggio perfetto!";
                badgeCol = "#3b82f6";
            }} else if (hSucc && aiSucc) {{
                if (hFuelUsed < aiStats.fuel_used) {{
                    winnerMsg = "🏆 VITTORIA DEL PILOTA UMANO PER EFFICIENZA! Hai consumato meno carburante dell'IA (" + hFuelUsed.toFixed(1) + " contro " + aiStats.fuel_used.toFixed(1) + " un.)!";
                    badgeCol = "#10b981";
                }} else {{
                    winnerMsg = "🤖 VITTORIA DELL'AGENTE AI PER EFFICIENZA! L'IA ha ottimizzato i consumi (" + aiStats.fuel_used.toFixed(1) + " contro " + hFuelUsed.toFixed(1) + " un.)!";
                    badgeCol = "#3b82f6";
                }}
            }} else {{
                winnerMsg = "💥 DOPPIO SCHIANTO! Nessun vincitore: la gravità ha sconfitto sia l'uomo che l'algoritmo!";
                badgeCol = "#ef4444";
            }}

            vTitolo.innerHTML = winnerMsg;
            vTitolo.style.color = badgeCol;

            vDesc.innerHTML = `
                <table style="width:100%; max-width:540px; margin:auto; border-collapse:collapse; font-size:12.5px;">
                    <tr style="border-bottom:1px solid #475569; color:#94a3b8;">
                        <th style="padding:4px; text-align:left;">Parametro</th>
                        <th style="padding:4px; text-align:center; color:#38bdf8;">Tu (Umano)</th>
                        <th style="padding:4px; text-align:center; color:#10b981;">Agente AI</th>
                    </tr>
                    <tr style="border-bottom:1px solid #334155;">
                        <td style="padding:5px; text-align:left;">Esito Volo</td>
                        <td style="padding:5px; text-align:center; font-weight:bold;">${{hSucc ? '🟢 Allunaggio' : '🔴 Schianto'}}</td>
                        <td style="padding:5px; text-align:center; font-weight:bold;">${{aiSucc ? '🟢 Allunaggio' : '🔴 Schianto'}}</td>
                    </tr>
                    <tr style="border-bottom:1px solid #334155;">
                        <td style="padding:5px; text-align:left;">Carburante Usato</td>
                        <td style="padding:5px; text-align:center;">${{hFuelUsed.toFixed(1)}} un.</td>
                        <td style="padding:5px; text-align:center;">${{aiStats.fuel_used.toFixed(1)}} un.</td>
                    </tr>
                    <tr>
                        <td style="padding:5px; text-align:left;">Durata Volo</td>
                        <td style="padding:5px; text-align:center;">${{(hStepsCount/50).toFixed(1)}} s</td>
                        <td style="padding:5px; text-align:center;">${{aiStats.time_sec.toFixed(1)}} s</td>
                    </tr>
                </table>
            `;
            scoreboard.style.display = 'block';
        }}

        function drawScenes() {{
            const sx = W / 20.0, sy = H / 13.333;

            // 1. Schermo Umano
            drawTerrain(ctxH, terrain, helipad, W, H);
            drawLander(ctxH, hX * sx, H - hY * sy, hAngle, hKeyMain && hFuel > 0, hKeyLeft && hFuel > 0, hKeyRight && hFuel > 0, false, hLanded, hCrashed);

            const hFuelEl = document.getElementById('h_hud_fuel_{uid}');
            if (hFuelEl) hFuelEl.innerText = hFuelUsed.toFixed(1) + ' un.';
            const hYEl = document.getElementById('h_hud_y_{uid}');
            if (hYEl) hYEl.innerText = Math.max(0, hY - getTerrainY(hX)).toFixed(1) + 'm';
            const hVyEl = document.getElementById('h_hud_vy_{uid}');
            if (hVyEl) {{
                hVyEl.innerText = hVy.toFixed(1) + 'm/s';
                hVyEl.style.color = Math.abs(hVy) < 2.2 ? '#10b981' : '#ef4444';
            }}
            const hStEl = document.getElementById('h_hud_status_{uid}');
            if (hStEl) {{
                hStEl.innerText = hLanded ? '🟢 Allunato' : (hCrashed ? '🔴 Schiantato' : 'In Volo');
                hStEl.style.color = hLanded ? '#10b981' : (hCrashed ? '#ef4444' : '#38bdf8');
            }}

            // 2. Schermo Agente AI
            drawTerrain(ctxA, terrain, helipad, W, H);
            let aiCur = (aiSteps.length > 0 ? aiSteps[Math.min(aiIdx, aiSteps.length - 1)] : {{ x: 10, y: 13.2, angle: 0, action: 0, fuel: 0, vy: 0 }});

            const aiIsMain = (aiCur.action === 2 && !aiFinished);
            const aiIsLeft = (aiCur.action === 1 && !aiFinished);
            const aiIsRight = (aiCur.action === 3 && !aiFinished);

            drawLander(ctxA, aiCur.x * sx, H - aiCur.y * sy, aiCur.angle, aiIsMain, aiIsLeft, aiIsRight, true, aiFinished && aiStats.success, aiFinished && aiStats.crashed);

            const aiFuelEl = document.getElementById('ai_hud_fuel_{uid}');
            if (aiFuelEl) aiFuelEl.innerText = aiCur.fuel.toFixed(1) + ' un.';
            const aiYEl = document.getElementById('ai_hud_y_{uid}');
            if (aiYEl) aiYEl.innerText = Math.max(0, aiCur.y - getTerrainY(aiCur.x)).toFixed(1) + 'm';
            const aiVyEl = document.getElementById('ai_hud_vy_{uid}');
            if (aiVyEl) aiVyEl.innerText = (aiCur.vy || 0).toFixed(1) + 'm/s';
            const aiStEl = document.getElementById('ai_hud_status_{uid}');
            if (aiStEl) {{
                aiStEl.innerText = aiFinished ? (aiStats.success ? '🟢 Allunato' : '🔴 Schiantato') : 'In Volo';
                aiStEl.style.color = aiFinished ? (aiStats.success ? '#10b981' : '#ef4444') : '#10b981';
            }}
        }}

        // Accumulatore 50 Hz per gara simultanea
        const FIXED_DT = 0.02;
        let lastFrameTs = performance.now();
        let accumulator = 0.0;

        function loop(now) {{
            const frameSec = Math.min(0.1, (now - lastFrameTs) / 1000.0);
            lastFrameTs = now;
            accumulator += frameSec;
            while (accumulator >= FIXED_DT) {{
                updateHumanPhysics(FIXED_DT);
                if (gameRunning && !aiFinished) {{
                    aiIdx++;
                    if (aiIdx >= aiSteps.length) {{
                        aiFinished = true;
                        checkBothFinished();
                    }}
                }}
                accumulator -= FIXED_DT;
            }}
            drawScenes();
            animId = requestAnimationFrame(loop);
        }}

        canvasH.addEventListener('keydown', function(e) {{
            if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', ' '].includes(e.key)) {{
                e.preventDefault();
            }}
            if (!gameRunning && (e.key === ' ' || e.key === 'ArrowUp')) {{
                startChallenge();
                return;
            }}
            if (e.key === 'ArrowUp' || e.key === 'w' || e.key === 'W' || e.key === ' ') hKeyMain = true;
            if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') hKeyLeft = true;
            if (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') hKeyRight = true;
        }});

        canvasH.addEventListener('keyup', function(e) {{
            if (e.key === 'ArrowUp' || e.key === 'w' || e.key === 'W' || e.key === ' ') hKeyMain = false;
            if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') hKeyLeft = false;
            if (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') hKeyRight = false;
        }});

        document.getElementById('vs_btn_start_{uid}').addEventListener('click', startChallenge);
        document.getElementById('vs_btn_replay_{uid}').addEventListener('click', startChallenge);

        function bindBtn(btnId, setFlag) {{
            const btn = document.getElementById(btnId);
            if (!btn) return;
            const onDown = (e) => {{ e.preventDefault(); if (!gameRunning) startChallenge(); setFlag(true); }};
            const onUp = (e) => {{ e.preventDefault(); setFlag(false); }};
            btn.addEventListener('mousedown', onDown);
            btn.addEventListener('mouseup', onUp);
            btn.addEventListener('mouseleave', onUp);
            btn.addEventListener('touchstart', onDown);
            btn.addEventListener('touchend', onUp);
            btn.addEventListener('touchcancel', onUp);
        }}
        bindBtn('vs_btn_main_{uid}', (v) => hKeyMain = v);
        bindBtn('vs_btn_left_{uid}', (v) => hKeyLeft = v);
        bindBtn('vs_btn_right_{uid}', (v) => hKeyRight = v);

        const sndBtn = document.getElementById('vs_btn_sound_{uid}');
        sndBtn.addEventListener('click', function() {{
            soundEnabled = !soundEnabled;
            sndBtn.innerText = soundEnabled ? '🔊 Audio: ON' : '🔇 Audio: OFF';
        }});

        loop(performance.now());
    }})();
    </script>
    """
    return _display_or_open_html(html_code, "lunar_lander_vs_agente")


def gioca(
    modalita: str = "vs_agente",
    model: Any = None,
    seed: int = 42,
    mission: str = "luna_standard",
    gravity: float | None = None,
    wind: bool | None = None,
    env_kwargs: dict[str, Any] | None = None,
) -> Any:
    """
    Punto di ingresso unificato per il simulatore interattivo.
    Supporta `modalita='single'` (oppure 'single_player') e `modalita='vs_agente'` (oppure 'sfida').
    """
    mod_clean = str(modalita).lower().strip()
    if "single" in mod_clean or "solo" in mod_clean:
        return gioca_single_player(
            seed=seed, mission=mission, gravity=gravity, wind=wind, env_kwargs=env_kwargs
        )
    else:
        return gioca_vs_agente(
            model_or_result=model,
            seed=seed,
            mission=mission,
            gravity=gravity,
            wind=wind,
            env_kwargs=env_kwargs,
        )
