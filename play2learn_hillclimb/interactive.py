"""
Modulo di Simulazione e Gioco Interattivo per Hill Climb Racing (Play2Learn).
Permette agli studenti di guidare manualmente il veicolo direttamente in Jupyter e Google Colab
(HTML5 Canvas + Web Audio API):
  1. Modalità Single Player: per comprendere la difficoltà del bilanciamento trazione/ribaltamento.
  2. Modalità Tu vs Agente: gara testa a testa (split-screen) contro l'agente PPO addestrato.
"""

from __future__ import annotations

import json
import random
import uuid
from pathlib import Path
from typing import Any

from play2learn_core.base_interactive import display_or_open_html, get_shared_audio_and_loop_js
from play2learn_hillclimb.config import MISSIONS
from play2learn_hillclimb.storage import find_run_by_timestamp, get_default_runs_dir


def registra_traiettoria_agente(
    model: Any,
    seed: int = 42,
    mission: str = "campagna_standard",
    max_steps: int = 1000,
    env_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Registra la telemetria passo-passo dell'agente PPO per il replay in split-screen."""
    import gymnasium as gym
    from play2learn_hillclimb.config import ENV_ID
    from play2learn_hillclimb.environment import HillClimbRewardWrapper

    final_kwargs = dict(env_kwargs or {})
    env = gym.make(ENV_ID, render_mode="rgb_array", **final_kwargs)
    env = HillClimbRewardWrapper(env)
    obs, _ = env.reset(seed=seed)

    steps_data = []
    tot_dist = 0.0

    for s_idx in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        act_val = int(action.item() if hasattr(action, "item") and not hasattr(action, "__len__") else action)

        x_pos = float(obs[0])
        y_pos = float(obs[1])
        angle = float(obs[4])
        vx = float(obs[2])

        steps_data.append({
            "step": s_idx,
            "x": round(x_pos, 2),
            "y": round(y_pos, 2),
            "angle": round(angle, 3),
            "vx": round(vx, 2),
            "act": act_val,
        })

        obs, reward, terminated, truncated, info = env.step(act_val)
        tot_dist = max(tot_dist, x_pos)
        if terminated or truncated:
            break

    env.close()

    return {
        "steps": steps_data,
        "final_distance": round(tot_dist, 1),
        "is_success": bool(tot_dist >= 800.0),
    }


def gioca_single_player(
    seed: int | None = None,
    mission: str = "campagna_standard",
    env_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Avvia il simulatore arcade Single Player in Google Colab o Jupyter Notebook."""
    actual_seed = seed if seed is not None else random.randint(100, 99999)
    mission_cfg = MISSIONS.get(mission, MISSIONS["campagna_standard"])
    uid = uuid.uuid4().hex[:6]
    shared_js = get_shared_audio_and_loop_js()

    html_code = f"""
    <div id="hcr_sp_root_{uid}" style="font-family:'Segoe UI',sans-serif; background:#0b0f19; color:#f8fafc; padding:18px; border-radius:12px; max-width:840px; margin:10px auto; box-shadow:0 6px 20px rgba(0,0,0,0.5);">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; border-bottom:1px solid #1e293b; padding-bottom:8px;">
            <div>
                <h3 style="margin:0; color:#38bdf8; font-size:18px;">🏎️ Simulatore Manuale: Guida su Hill Climb Racing!</h3>
                <span style="font-size:12px; color:#94a3b8;">Tracciato: {mission_cfg['title']} (Seed: {actual_seed})</span>
            </div>
            <div>
                <button id="btn_sound_{uid}" style="background:#1e293b; color:#cbd5e1; border:1px solid #475569; padding:4px 10px; border-radius:6px; font-size:12px; cursor:pointer;">🔊 Audio: ON</button>
            </div>
        </div>

        <div style="position:relative; width:100%; text-align:center;">
            <canvas id="canvas_{uid}" width="800" height="420" tabindex="0" style="background:#87CEEB; border:1.5px solid #334155; border-radius:8px; outline:none; max-width:100%; cursor:pointer;"></canvas>
            <div id="overlay_{uid}" style="position:absolute; top:45%; left:50%; transform:translate(-50%,-50%); background:rgba(15,23,42,0.92); border:1.5px solid #38bdf8; padding:14px 22px; border-radius:8px; text-align:center;">
                <div style="font-size:15px; font-weight:bold; color:#f8fafc; margin-bottom:6px;">Premi [SPAZIO] o Clicca per Partire!</div>
                <div style="font-size:12px; color:#94a3b8;">Comandi: ⬆️ / W = Gas | ⬇️ / S = Freno/Retro | ⬅️ ➡️ = Bilanciamento in Volo</div>
            </div>
        </div>

        <div style="display:grid; grid-template-columns:repeat(4, 1fr); gap:8px; margin-top:12px; font-size:12px; background:#111827; padding:10px; border-radius:8px; border:1px solid #1f2937;">
            <div><span style="color:#94a3b8;">Distanza:</span> <strong id="hud_dist_{uid}" style="color:#38bdf8;">0.0 m</strong></div>
            <div><span style="color:#94a3b8;">Velocità:</span> <strong id="hud_spd_{uid}" style="color:#10b981;">0.0 m/s</strong></div>
            <div><span style="color:#94a3b8;">Inclinazione:</span> <strong id="hud_ang_{uid}">0°</strong></div>
            <div><span style="color:#94a3b8;">Stato:</span> <strong id="hud_stat_{uid}" style="color:#10b981;">Pronto</strong></div>
        </div>
    </div>

    <script>
    (function() {{
        {shared_js}

        const canvas = document.getElementById('canvas_{uid}');
        const ctx = canvas.getContext('2d');
        const overlay = document.getElementById('overlay_{uid}');
        const W = canvas.width, H = canvas.height;

        let seedVal = {actual_seed};
        function pseudoNoise(x) {{
            return Math.sin(x * 0.008 + seedVal) * 60 + Math.cos(x * 0.02 + seedVal * 2) * 25 + Math.sin(x * 0.05) * 12;
        }}

        function getTerrainY(px) {{
            if (px < 150) return 320;
            return 320 + pseudoNoise(px - 150);
        }}

        let carX = 120, carY = 280, vx = 0, vy = 0, angle = 0, vangle = 0;
        let gameActive = false, dead = false, finished = false;
        let keyGas = false, keyBrake = false, keyLeft = false, keyRight = false;

        function resetGame() {{
            carX = 120; carY = 280; vx = 0; vy = 0; angle = 0; vangle = 0;
            dead = false; finished = false; gameActive = true;
            overlay.style.display = 'none';
            canvas.focus();
        }}

        function updatePhysics(dt) {{
            if (!gameActive || dead || finished) return;

            const g = 18.0;
            let ax = 0, ay = g;

            const groundY = getTerrainY(carX);
            const onGround = (carY >= groundY - 18);

            if (keyGas) {{
                if (onGround) {{
                    ax += Math.cos(angle) * 32.0;
                    ay += Math.sin(angle) * 32.0;
                    vangle -= 1.8 * dt;
                }}
                playEngineTone(140 + Math.abs(vx) * 12, 0.06);
            }}

            if (keyBrake) {{
                if (onGround) {{
                    ax -= Math.cos(angle) * 22.0;
                    vangle += 2.0 * dt;
                }}
                playEngineTone(90, 0.05);
            }}

            // Bilanciamento in aria
            if (keyLeft) vangle -= 4.5 * dt;
            if (keyRight) vangle += 4.5 * dt;

            vx = (vx + ax * dt) * (onGround ? 0.985 : 0.995);
            vy += ay * dt;
            carX += vx * dt;
            carY += vy * dt;
            vangle *= 0.96;
            angle += vangle * dt;

            // Collisione suolo
            if (carY > groundY - 16) {{
                carY = groundY - 16;
                vy = 0;
                // Allineamento naturale alla pendenza
                const slope = (getTerrainY(carX + 15) - getTerrainY(carX - 15)) / 30.0;
                const targetAngle = Math.atan2(slope, 1.0);
                angle += (targetAngle - angle) * 0.25;

                // Controllo cappottamento
                if (Math.abs(angle) > 1.45) {{
                    dead = true;
                    playCrashSound();
                    overlay.innerHTML = '<div style="color:#ef4444; font-size:16px; font-weight:bold;">💥 PILOTA SCHIANTATO! (Ribaltamento)</div><button id="btn_ret_{uid}" style="margin-top:8px; background:#ef4444; color:#fff; border:none; padding:6px 14px; border-radius:4px; cursor:pointer;">Riprova</button>';
                    overlay.style.display = 'block';
                    document.getElementById('btn_ret_{uid}').addEventListener('click', resetGame);
                }}
            }}

            if (carX >= 2400) {{
                finished = true;
                playVictoryChime();
                overlay.innerHTML = '<div style="color:#10b981; font-size:16px; font-weight:bold;">🎉 TRAGUARDO 800m COMPLETATO!</div><button id="btn_ret_{uid}" style="margin-top:8px; background:#10b981; color:#fff; border:none; padding:6px 14px; border-radius:4px; cursor:pointer;">Rigioca</button>';
                overlay.style.display = 'block';
                document.getElementById('btn_ret_{uid}').addEventListener('click', resetGame);
            }}
        }}

        function drawScene() {{
            ctx.fillStyle = '#87CEEB';
            ctx.fillRect(0, 0, W, H);

            const camX = carX - 160;

            // Disegna terreno
            ctx.fillStyle = '#582300';
            ctx.strokeStyle = '#22c55e';
            ctx.lineWidth = 6;
            ctx.beginPath();
            ctx.moveTo(0, H);
            for (let px = 0; px <= W; px += 8) {{
                const worldX = camX + px;
                const gy = getTerrainY(worldX);
                ctx.lineTo(px, gy);
            }}
            ctx.lineTo(W, H);
            ctx.closePath();
            ctx.fill();
            ctx.stroke();

            // Disegna vettura
            const sx = carX - camX;
            const sy = carY;
            ctx.save();
            ctx.translate(sx, sy);
            ctx.rotate(angle);

            // Telaio rosso
            ctx.fillStyle = '#ef4444';
            ctx.fillRect(-22, -10, 44, 14);
            // Cabina
            ctx.fillStyle = '#0284c7';
            ctx.fillRect(-10, -20, 20, 10);
            // Pilota
            ctx.fillStyle = '#fde047';
            ctx.beginPath();
            ctx.arc(-2, -24, 7, 0, Math.PI * 2);
            ctx.fill();

            // Ruote
            ctx.fillStyle = '#1e293b';
            ctx.strokeStyle = '#94a3b8';
            ctx.lineWidth = 2;
            ctx.beginPath(); ctx.arc(-16, 6, 8, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
            ctx.beginPath(); ctx.arc(16, 6, 8, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
            ctx.restore();

            // Aggiorna HUD
            const distM = Math.max(0, (carX - 120) / 3.0);
            document.getElementById('hud_dist_{uid}').innerText = distM.toFixed(1) + ' m';
            document.getElementById('hud_spd_{uid}').innerText = Math.abs(vx * 0.3).toFixed(1) + ' m/s';
            document.getElementById('hud_ang_{uid}').innerText = (angle * 180 / Math.PI).toFixed(0) + '°';
        }}

        const FIXED_DT = 0.02;
        let lastTime = performance.now();
        let accum = 0.0;

        function loop(now) {{
            const dt = Math.min(0.1, (now - lastTime) / 1000.0);
            lastTime = now;
            accum += dt;
            while (accum >= FIXED_DT) {{
                updatePhysics(FIXED_DT);
                accum -= FIXED_DT;
            }}
            drawScene();
            requestAnimationFrame(loop);
        }}

        canvas.addEventListener('keydown', function(e) {{
            initAudio();
            if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', ' '].includes(e.key)) e.preventDefault();
            if (!gameActive && (e.key === ' ' || e.key === 'ArrowUp')) {{ resetGame(); return; }}
            if (e.key === 'ArrowUp' || e.key === 'w' || e.key === 'W') keyGas = true;
            if (e.key === 'ArrowDown' || e.key === 's' || e.key === 'S') keyBrake = true;
            if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') keyLeft = true;
            if (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') keyRight = true;
        }});

        canvas.addEventListener('keyup', function(e) {{
            if (e.key === 'ArrowUp' || e.key === 'w' || e.key === 'W') keyGas = false;
            if (e.key === 'ArrowDown' || e.key === 's' || e.key === 'S') keyBrake = false;
            if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') keyLeft = false;
            if (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') keyRight = false;
        }});

        canvas.addEventListener('click', function() {{ initAudio(); if (!gameActive) resetGame(); }});
        loop(performance.now());
    }})();
    </script>
    """
    return display_or_open_html(html_code, "hill_climb_single_player")


def gioca_vs_agente(
    model_or_result: Any = None,
    seed: int = 42,
    mission: str = "campagna_standard",
    env_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Avvia la modalità testa a testa 'Tu vs Agente AI' in Split-Screen."""
    print("[play2learn] Calcolo della traiettoria dell'Agente AI sulle colline...")

    # Risoluzione modello
    resolved_model = None
    if hasattr(model_or_result, "model"):
        resolved_model = model_or_result.model
    elif model_or_result is not None:
        resolved_model = model_or_result

    agent_data = {"steps": [], "final_distance": 750.0}
    if resolved_model is not None:
        try:
            agent_data = registra_traiettoria_agente(
                model=resolved_model,
                seed=seed,
                mission=mission,
                env_kwargs=env_kwargs,
            )
        except Exception as exc:
            print(f"[play2learn] Nota: registrazione traiettoria agente fallita ({exc}), uso replay sintetico.")

    uid = uuid.uuid4().hex[:6]
    shared_js = get_shared_audio_and_loop_js()
    agent_steps_json = json.dumps(agent_data["steps"])

    html_code = f"""
    <div id="hcr_vs_root_{uid}" style="font-family:'Segoe UI',sans-serif; background:#0b0f19; color:#f8fafc; padding:18px; border-radius:12px; max-width:960px; margin:10px auto; box-shadow:0 6px 20px rgba(0,0,0,0.5);">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; border-bottom:1px solid #1e293b; padding-bottom:8px;">
            <div>
                <h3 style="margin:0; color:#38bdf8; font-size:19px;">⚔️ Sfida Hill Climb: Tu (Pilota Umano) vs Agente AI (PPO)</h3>
                <span style="font-size:12px; color:#94a3b8;">Gara simultanea sullo stesso tracciato collinare (Seed: {seed})</span>
            </div>
            <div>
                <button id="vs_btn_sound_{uid}" style="background:#1e293b; color:#cbd5e1; border:1px solid #475569; padding:4px 10px; border-radius:6px; font-size:12px; cursor:pointer;">🔊 Audio: ON</button>
            </div>
        </div>

        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:14px;">
            <div style="background:#111827; padding:10px; border-radius:8px; border:1.5px solid #0284c7; text-align:center;">
                <div style="font-weight:bold; color:#38bdf8; font-size:14px; margin-bottom:6px;">🧑 TU (Pilota Umano) — Auto Rossa</div>
                <canvas id="canvas_human_{uid}" width="440" height="300" tabindex="0" style="background:#87CEEB; border-radius:6px; outline:none; max-width:100%; cursor:pointer;"></canvas>
                <div style="display:flex; justify-content:space-between; font-size:11.5px; color:#cbd5e1; margin-top:6px; background:#1e293b; padding:6px 10px; border-radius:6px;">
                    <span>Distanza: <strong id="h_dist_{uid}" style="color:#38bdf8;">0.0 m</strong></span>
                    <span>Velocità: <strong id="h_spd_{uid}" style="color:#10b981;">0.0 m/s</strong></span>
                </div>
            </div>

            <div style="background:#111827; padding:10px; border-radius:8px; border:1.5px solid #059669; text-align:center;">
                <div style="font-weight:bold; color:#10b981; font-size:14px; margin-bottom:6px;">🤖 AGENTE AI (PPO) — Auto Verde</div>
                <canvas id="canvas_ai_{uid}" width="440" height="300" style="background:#87CEEB; border-radius:6px; max-width:100%;"></canvas>
                <div style="display:flex; justify-content:space-between; font-size:11.5px; color:#cbd5e1; margin-top:6px; background:#1e293b; padding:6px 10px; border-radius:6px;">
                    <span>Distanza: <strong id="ai_dist_{uid}" style="color:#10b981;">0.0 m</strong></span>
                    <span>Velocità: <strong id="ai_spd_{uid}" style="color:#38bdf8;">0.0 m/s</strong></span>
                </div>
            </div>
        </div>
    </div>

    <script>
    (function() {{
        {shared_js}
        const canvasH = document.getElementById('canvas_human_{uid}');
        const ctxH = canvasH.getContext('2d');
        const canvasA = document.getElementById('canvas_ai_{uid}');
        const ctxA = canvasA.getContext('2d');

        const aiSteps = {agent_steps_json};
        let aiIdx = 0;
        let seedVal = {seed};

        function pseudoNoise(x) {{
            return Math.sin(x * 0.008 + seedVal) * 60 + Math.cos(x * 0.02 + seedVal * 2) * 25 + Math.sin(x * 0.05) * 12;
        }}
        function getTerrainY(px) {{
            if (px < 150) return 220;
            return 220 + pseudoNoise(px - 150);
        }}

        let hX = 120, hY = 190, hVx = 0, hVy = 0, hAngle = 0, hVangle = 0;
        let keyGas = false, keyBrake = false;

        function update() {{
            // AI Replay
            if (aiIdx < aiSteps.length) {{
                const step = aiSteps[aiIdx];
                const aiDist = step.x;
                document.getElementById('ai_dist_{uid}').innerText = aiDist.toFixed(1) + ' m';
                document.getElementById('ai_spd_{uid}').innerText = Math.abs(step.vx).toFixed(1) + ' m/s';
                aiIdx++;
            }}

            // Umano
            const gy = getTerrainY(hX);
            const onGround = (hY >= gy - 16);
            if (keyGas && onGround) {{ hVx += 1.2; playEngineTone(150, 0.04); }}
            if (keyBrake && onGround) hVx -= 0.8;
            hVx *= 0.98;
            hX += hVx;
            hY = gy - 16;

            document.getElementById('h_dist_{uid}').innerText = Math.max(0, (hX - 120) / 3.0).toFixed(1) + ' m';
            document.getElementById('h_spd_{uid}').innerText = Math.abs(hVx * 0.4).toFixed(1) + ' m/s';

            // Draw Human
            ctxH.fillStyle = '#87CEEB'; ctxH.fillRect(0, 0, 440, 300);
            const camH = hX - 120;
            ctxH.fillStyle = '#582300';
            ctxH.beginPath(); ctxH.moveTo(0, 300);
            for (let p = 0; p <= 440; p += 10) ctxH.lineTo(p, getTerrainY(camH + p));
            ctxH.lineTo(440, 300); ctxH.fill();
            ctxH.fillStyle = '#ef4444'; ctxH.fillRect(100, hY - 8, 30, 10);

            // Draw AI
            ctxA.fillStyle = '#87CEEB'; ctxA.fillRect(0, 0, 440, 300);
            const aiCurX = (aiIdx < aiSteps.length ? aiSteps[aiIdx].x * 3.0 + 120 : hX);
            const camA = aiCurX - 120;
            ctxA.fillStyle = '#582300';
            ctxA.beginPath(); ctxA.moveTo(0, 300);
            for (let p = 0; p <= 440; p += 10) ctxA.lineTo(p, getTerrainY(camA + p));
            ctxA.lineTo(440, 300); ctxA.fill();
            ctxA.fillStyle = '#10b981'; ctxA.fillRect(100, getTerrainY(aiCurX) - 24, 30, 10);

            requestAnimationFrame(update);
        }}

        canvasH.addEventListener('keydown', function(e) {{
            initAudio();
            if (e.key === 'ArrowUp' || e.key === 'w') keyGas = true;
            if (e.key === 'ArrowDown' || e.key === 's') keyBrake = true;
        }});
        canvasH.addEventListener('keyup', function(e) {{
            if (e.key === 'ArrowUp' || e.key === 'w') keyGas = false;
            if (e.key === 'ArrowDown' || e.key === 's') keyBrake = false;
        }});

        update();
    }})();
    </script>
    """
    return display_or_open_html(html_code, "hill_climb_vs_agente")


def gioca(
    modalita: str = "vs_agente",
    model: Any = None,
    seed: int = 42,
    mission: str = "campagna_standard",
    env_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Punto di ingresso unificato per il gioco interattivo."""
    if "single" in str(modalita).lower():
        return gioca_single_player(seed=seed, mission=mission, env_kwargs=env_kwargs)
    return gioca_vs_agente(model_or_result=model, seed=seed, mission=mission, env_kwargs=env_kwargs)
