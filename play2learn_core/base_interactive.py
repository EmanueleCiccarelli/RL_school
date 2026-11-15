"""
Modulo Core per la Simulazione Interattiva nel Browser e Notebook (Play2Learn).
Fornisce sintetizzatori Web Audio API, display HTML e loop di fisica a 50 Hz fisso.
"""

from __future__ import annotations

import tempfile
import webbrowser
from pathlib import Path
from typing import Any


def display_or_open_html(html_code: str, filename: str = "play2learn_interactive") -> Any:
    """Mostra l'HTML inline nel notebook oppure lo apre nel browser predefinito."""
    try:
        from IPython.display import HTML, display
        res = HTML(html_code)
        display(res)
        return res
    except ImportError:
        tmp = Path(tempfile.gettempdir()) / f"{filename}.html"
        tmp.write_text(html_code, encoding="utf-8")
        print(f"[play2learn] Apertura del simulatore nel browser: {tmp}")
        webbrowser.open(tmp.as_uri())
        return html_code


def get_shared_audio_and_loop_js() -> str:
    """Restituisce il codice JavaScript comune per la sintesi sonora e il game loop a 50 Hz fisso."""
    return """
    let audioCtx = null;
    let soundEnabled = true;

    function initAudio() {
        if (!audioCtx) {
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            if (AudioContext) audioCtx = new AudioContext();
        }
        if (audioCtx && audioCtx.state === 'suspended') {
            audioCtx.resume();
        }
    }

    function playEngineTone(freq, duration) {
        if (!soundEnabled || !audioCtx) return;
        try {
            const osc = audioCtx.createOscillator();
            const gain = audioCtx.createGain();
            osc.type = 'sawtooth';
            osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
            gain.gain.setValueAtTime(0.08, audioCtx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);
            osc.connect(gain);
            gain.connect(audioCtx.destination);
            osc.start();
            osc.stop(audioCtx.currentTime + duration);
        } catch(e) {}
    }

    function playCrashSound() {
        if (!soundEnabled || !audioCtx) return;
        try {
            const osc = audioCtx.createOscillator();
            const gain = audioCtx.createGain();
            osc.type = 'square';
            osc.frequency.setValueAtTime(90, audioCtx.currentTime);
            osc.frequency.exponentialRampToValueAtTime(25, audioCtx.currentTime + 0.35);
            gain.gain.setValueAtTime(0.2, audioCtx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.35);
            osc.connect(gain);
            gain.connect(audioCtx.destination);
            osc.start();
            osc.stop(audioCtx.currentTime + 0.35);
        } catch(e) {}
    }

    function playVictoryChime() {
        if (!soundEnabled || !audioCtx) return;
        try {
            const notes = [261.63, 329.63, 392.00, 523.25];
            notes.forEach((freq, idx) => {
                const osc = audioCtx.createOscillator();
                const gain = audioCtx.createGain();
                osc.type = 'triangle';
                osc.frequency.setValueAtTime(freq, audioCtx.currentTime + idx * 0.1);
                gain.gain.setValueAtTime(0.15, audioCtx.currentTime + idx * 0.1);
                gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + idx * 0.1 + 0.28);
                osc.connect(gain);
                gain.connect(audioCtx.destination);
                osc.start(audioCtx.currentTime + idx * 0.1);
                osc.stop(audioCtx.currentTime + idx * 0.1 + 0.3);
            });
        } catch(e) {}
    }
    """
