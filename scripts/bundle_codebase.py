#!/usr/bin/env python3
"""
Script di utilita per generare il bundle completo del codice sorgente della repository
e il prompt di code review per modelli linguistici avanzati.

I file generati (es. PROMPT_CODE_REVIEW.txt e codice_completo_repo.txt) sono esclusi
dal versionamento git (presenti in .gitignore) e servono unicamente come strumenti locali
per interrogare modelli AI esterni.

Uso:
    python scripts/bundle_codebase.py [--root DIR] [--code-out FILE] [--prompt-out FILE]
"""

import argparse
import os
import sys
from pathlib import Path

DEFAULT_FILES_TO_BUNDLE = [
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "play2learn_core/__init__.py",
    "play2learn_core/storage.py",
    "play2learn_core/base_env.py",
    "play2learn_core/agent.py",
    "play2learn_core/evaluator.py",
    "play2learn_core/favorites.py",
    "play2learn_core/reporting.py",
    "play2learn_core/base_interactive.py",
    "play2learn_lunarlander/__init__.py",
    "play2learn_lunarlander/config.py",
    "play2learn_lunarlander/environment.py",
    "play2learn_lunarlander/agent.py",
    "play2learn_lunarlander/evaluator.py",
    "play2learn_lunarlander/diagnostics.py",
    "play2learn_lunarlander/visualizer.py",
    "play2learn_lunarlander/storage.py",
    "play2learn_lunarlander/favorites.py",
    "play2learn_lunarlander/reporting.py",
    "play2learn_lunarlander/interactive.py",
    "play2learn_lunarlander/reset.py",
    "play2learn_lunarlander/lab.py",
    "play2learn_hillclimb/__init__.py",
    "play2learn_hillclimb/config.py",
    "play2learn_hillclimb/environment.py",
    "play2learn_hillclimb/agent.py",
    "play2learn_hillclimb/evaluator.py",
    "play2learn_hillclimb/diagnostics.py",
    "play2learn_hillclimb/visualizer.py",
    "play2learn_hillclimb/storage.py",
    "play2learn_hillclimb/favorites.py",
    "play2learn_hillclimb/reporting.py",
    "play2learn_hillclimb/interactive.py",
    "play2learn_hillclimb/reset.py",
    "play2learn_hillclimb/lab.py",
    "play2learn_hillclimb/engine/__init__.py",
    "play2learn_hillclimb/engine/hill_racing.py",
    "play2learn_hillclimb/engine/car.py",
    "play2learn_hillclimb/engine/wheels.py",
    "play2learn_hillclimb/engine/person.py",
    "play2learn_hillclimb/engine/ground.py",
    "play2learn_hillclimb/engine/agent.py",
    "play2learn_hillclimb/engine/perlin.py",
    "tests/__init__.py",
    "tests/test_config.py",
    "tests/test_environment.py",
    "tests/test_agent.py",
    "tests/test_storage_favorites.py",
    "tests/test_interactive.py",
    "tests/test_reporting.py",
    "tests/test_reset.py",
    "tests/test_hillclimb.py",
]

REVIEW_PROMPT_TEMPLATE = """================================================================================
ROLE & OBJECTIVE:
You are an elite Principal AI / Reinforcement Learning Research Engineer, Senior Systems
Architect, and High-Performance Python Specialist.

Your objective is to conduct a rigorous, line-by-line, architectural, algorithmic,
and performance code review of the entire multi-environment `play2learn` repository:
including `play2learn_core`, `play2learn_lunarlander`, and `play2learn_hillclimb`.

NOTE: The complete source code of the entire repository is provided in the accompanying
file `codice_completo_repo.txt`. Please inspect and cross-reference all files.

================================================================================
PROJECT CONTEXT & OPERATIONAL CONSTRAINTS:
`play2learn` is an educational Reinforcement Learning framework built on top of Farama
Gymnasium, Box2D physics, and Stable-Baselines3 (PPO). It is designed for high school (5th year)
and introductory university students who practice Reward Shaping across two distinct physical domains:
  1. Lunar Lander (Gymnasium `LunarLander-v3`): 8D continuous observation, retro-rocket thrusters,
     soft landing on pad between flags.
  2. Hill Climb Racing (`HillClimbRacing-v0`, based on alexzh3/hillclimbracing & Code Bullet):
     2D car chassis, revolute and prismatic suspension joints, ragdoll driver (head impact = death),
     procedural terrain with Perlin noise, traction control and flip avoidance.

Key Operational Constraints:
1. Student Public Interface Invariance:
   The public API (`custom_reward`, `run_lab`, `carica_preferito`, `salva_preferito`,
   `mostra_preferiti`, `confronta_run`, `genera_report_da_timestamp`, `gioca_single_player`,
   `gioca_vs_agente`, `LabResult`) must remain 100% backward compatible for both environments.
2. Dual Runtime Environments:
   - Google Colab (2 vCPU, headless Linux, limited RAM, virtual framebuffer).
   - Local Windows / macOS (4-16 threads, Jupyter Notebook / VS Code).
3. Sample Efficiency Goals:
   - Lunar Lander: Consistent convergence (success rate >= 80%, soft landing on pad) within
     200,000 - 300,000 timesteps using PPO.
   - Hill Climb Racing: Consistent distance advancement (>= 800m) and flip avoidance within
     100,000 - 150,000 timesteps using PPO.
4. Zero Memory Leaks & Streaming Video:
   Videos must stream directly to disk with constant O(1) RAM footprint (< 1 MB) to avoid
   crashing headless Jupyter/Colab instances.

================================================================================
SPECIFIC AUDIT FOCUS AREAS:

1. ARCHITECTURAL DESIGN PATTERNS & MODULARITY:
   - Common Core Extraction (`play2learn_core`): Audit the inheritance hierarchy, template
     methods in `BaseRewardWrapper`, `BaseRewardTrackerCallback`, and `BaseFavoritesManager`.
   - Separation of Concerns: Are domain-specific mechanics (thrusters vs wheels/suspensions)
     properly encapsulated without leaky abstractions?
   - Re-exporting & Backward Compatibility: Verify that all student imports from either
     `play2learn_lunarlander` or `play2learn_hillclimb` work transparently.

2. ALGORITHMIC & REINFORCEMENT LEARNING SOUNDNESS:
   - Outcome Classification: Validate `classify_lunar_lander_outcome` and
     `classify_hill_climb_outcome`. Check edge cases (e.g., perched on one leg, hovering
     timeout, stuck in ditch, driver head contact, high angle flips).
   - VecNormalize Dynamics: Verify `norm_obs=False`, `norm_reward=True`, `clip_reward=10.0`.
     Are running variances protected from NaN/Inf poisoning? Does deterministic evaluation
     sweep isolate inference observations from training normalization updates?
   - Observation Space & Action Space Design: Review the 8D continuous observation space
     for Hill Climb Racing and ensure smooth learning with SB3's MlpPolicy.

3. PERFORMANCE, CPU CONCURRENCY & MEMORY FOOTPRINT:
   - Concurrency & Thread Pinning: Verify early setting of `OMP_NUM_THREADS=1`,
     `MKL_NUM_THREADS=1`, etc., to avoid thread oversubscription in `SubprocVecEnv`.
   - Streaming Video Recording: Inspect `record_episode_streaming` with `imageio` H.264
     pipe. Ensure zero RAM accumulation across 20 evaluation episodes.
   - Box2D & Pygame Memory Cleanup: Check `_destroy_world()`, joint destruction, and
     Pygame surface lifecycle in headless `rgb_array` mode. Are any Box2D bodies leaked?
   - Matplotlib Notebook Memory: Verify that all plotting routines call `plt.close(fig)`
     or `plt.close('all')` to prevent memory bloat in long notebook sessions.

4. CROSS-PLATFORM ROBUSTNESS & RESILIENCE:
   - Atomic File Operations: Review `atomic_write_text` and `atomic_write_json`. Ensure
     resilience against notebook interruption and Colab timeouts.
   - Pure-Python Fallbacks: Audit `perlin.py` (pure Python Perlin noise) vs the C-extension
     `noise` package. Does it run deterministically across Windows and Linux?
   - Interactive HTML5 Canvas & Web Audio API: Audit `interactive.py` in both environments.
     Check the 50 Hz fixed accumulator physics loop, keybindings, touch controls, and
     side-by-side trajectory playback in Jupyter/Colab.
   - HTML Sanitization: Verify `html.escape()` on all student inputs in generated reports.

================================================================================
OUTPUT FORMAT EXPECTED:

Please structure your review into the following sections:
1. EXECUTIVE SUMMARY: High-level architectural health check and critical highlights.
2. CRITICAL BUGS & LOGICAL FLAWS: Issues affecting RL convergence, false outcome classifications,
   crashes, or numerical instability.
3. PERFORMANCE & CONCURRENCY OPTIMIZATIONS: Concurrency bottlenecks, memory footprint reductions,
   and loop optimizations.
4. CROSS-PLATFORM & EDGE-CASE ENHANCEMENTS: Path handling, headless compatibility, and UI fixes.
5. CONCRETE DROP-IN CODE REPLACEMENTS: Provide complete, ready-to-use drop-in code
   snippets for any suggested changes, with zero regressions on existing student APIs.
================================================================================
"""

CODE_HEADER = """# ==============================================================================
# RL_SCHOOL - CODICE SORGENTE COMPLETO DELLA REPOSITORY
# Progetti: Lunar Lander (LunarLander-v3) & Hill Climb Racing (HillClimbRacing-v0)
# Architettura: play2learn_core + play2learn_lunarlander + play2learn_hillclimb
# ==============================================================================

"""


def bundle_repository(root_dir: Path, code_out: Path, prompt_out: Path) -> None:
    print(f"📦 Avvio bundling repository da: {root_dir}")
    
    # 1. Scrivi il prompt di review separato
    prompt_out.write_text(REVIEW_PROMPT_TEMPLATE.strip() + "\n", encoding="utf-8")
    print(f"  ✅ Prompt salvato in: {prompt_out.name} ({prompt_out.stat().st_size:,} bytes)")
    
    # 2. Raccogli e scrivi tutto il codice sorgente
    chunks = [CODE_HEADER]
    bundled_count = 0
    missing_count = 0
    
    for rel_path in DEFAULT_FILES_TO_BUNDLE:
        file_path = root_dir / rel_path
        if file_path.exists():
            chunks.append("#" * 80)
            chunks.append(f"### FILE: {rel_path}")
            chunks.append("#" * 80 + "\n")
            try:
                content = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = file_path.read_text(encoding="latin-1")
            chunks.append(content)
            chunks.append("\n\n")
            bundled_count += 1
        else:
            print(f"  ⚠️  File non trovato: {rel_path}")
            missing_count += 1
            
    full_code = "\n".join(chunks)
    code_out.write_text(full_code, encoding="utf-8")
    print(f"  ✅ Codice sorgente salvato in: {code_out.name} ({code_out.stat().st_size:,} bytes, {bundled_count} file inclusi)")
    if missing_count > 0:
        print(f"  ⚠️  {missing_count} file mancanti.")


def main():
    parser = argparse.ArgumentParser(description="Genera il bundle del codice e il prompt di review.")
    parser.add_argument("--root", type=str, default=".", help="Directory radice del repository (default: .)")
    parser.add_argument("--code-out", type=str, default="codice_completo_repo.txt", help="Nome file di output per il codice")
    parser.add_argument("--prompt-out", type=str, default="PROMPT_CODE_REVIEW.txt", help="Nome file di output per il prompt")
    args = parser.parse_args()

    root_path = Path(args.root).resolve()
    code_out_path = root_path / args.code_out
    prompt_out_path = root_path / args.prompt_out

    bundle_repository(root_path, code_out_path, prompt_out_path)


if __name__ == "__main__":
    main()
