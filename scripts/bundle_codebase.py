#!/usr/bin/env python3
"""
Script di utilita per generare il bundle completo del codice sorgente della repository
e il prompt di ottimizzazione avanzata e code review per modelli linguistici ad alte prestazioni.

I file generati (es. PROMPT_CODE_REVIEW.txt, codice_completo_repo.txt, PROMPT_OTTIMIZZAZIONE_COMPLETO.txt)
sono esclusi dal versionamento git (presenti in .gitignore) e servono per interrogare modelli AI esterni.

Uso:
    python scripts/bundle_codebase.py [--root DIR] [--code-out FILE] [--prompt-out FILE] [--unified-out FILE]
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
    "tests/test_reporting.py",
    "tests/test_reset.py",
    "tests/test_hillclimb.py",
]

REVIEW_PROMPT_TEMPLATE = """================================================================================
ROLE & OBJECTIVE:
You are an elite Principal AI / Reinforcement Learning Systems Architect, High-Performance
Python & C++ Systems Engineer, and Senior Performance Profiling Specialist.

Your objective is to conduct a rigorous, line-by-line architectural, algorithmic, and
hardware-level performance review of the entire multi-environment `play2learn` repository:
including `play2learn_core`, `play2learn_lunarlander`, and `play2learn_hillclimb`.

NOTE: The complete source code of all 54 modules in the repository is provided as extra context
(see `codice_completo_repo.txt` or the appended section). Please inspect and cross-reference
every file.

================================================================================
PROJECT CONTEXT & OPERATIONAL CONSTRAINTS:
`play2learn` is an educational Reinforcement Learning framework built on Farama Gymnasium,
Box2D physics, and Stable-Baselines3 (PPO). It is designed for high school (5th year)
and introductory university students who practice Reward Shaping across two physical domains:
  1. Lunar Lander (Gymnasium `LunarLander-v3`): 8D continuous observation, retro-rocket thrusters,
     soft landing on pad between flags (|x| <= 0.20, |vy| <= 0.20, both legs touching).
  2. Hill Climb Racing (`HillClimbRacing-v0`, based on alexzh3/hillclimbracing & Code Bullet):
     2D car chassis, revolute and prismatic suspension joints, ragdoll driver (head impact = death),
     procedural terrain with Perlin noise, traction control and flip avoidance.

Operational Constraints:
1. Student Public Interface Invariance:
   The public API (`custom_reward`, `run_lab`, `carica_preferito`, `salva_preferito`,
   `mostra_preferiti`, `confronta_run`, `genera_report_da_timestamp`,
   `LabResult`) must remain 100% backward compatible for both environments.
2. Dual Target Runtime Environments:
   - Google Colab (2 vCPU, headless Linux, limited RAM ~12GB, virtual framebuffer).
   - Local PC (Windows / macOS / Linux, 4-16 threads, Jupyter Notebook / VS Code).
3. Sample Efficiency Goals:
   - Lunar Lander: Consistent convergence (success rate >= 80%, soft landing on pad) within
     200,000 - 300,000 timesteps using PPO.
   - Hill Climb Racing: Consistent distance advancement (>= 800m) and flip avoidance within
     100,000 - 150,000 timesteps using PPO.
4. Zero Memory Leaks & Streaming Video:
   Videos must stream directly to disk with constant O(1) RAM footprint (< 1 MB) to avoid
   crashing headless Jupyter/Colab instances.

================================================================================
PART 1: TRAINING THROUGHPUT REGRESSION & ROOT-CAUSE ANALYSIS (Colab 2 vCPUs)
(Problem: Step rate on Lunar Lander dropped from ~2000 steps/s down to ~1000 steps/s)

We performed deep hardware-level profiling simulating Google Colab's standard 2-vCPU environment
(`taskset -c 0,1` pinning CPU affinity to 2 physical cores). Here is the diagnostic breakdown:

1. PPO GRADIENT UPDATE FREQUENCY vs ENVIRONMENT STEP COLLECTION:
   In commit d2721b2, hyperparameters were adjusted to increase sample efficiency:
     `batch_size = 64`, `n_epochs = 8`, `n_steps = 512`, `n_envs = 4`
   Rollout buffer size = 4 * 512 = 2048 steps.
   Number of minibatches per epoch = 2048 / 64 = 32.
   Total SGD updates per rollout = 32 minibatches * 8 epochs = 256 gradient updates!
   Update frequency = 256 updates per 2048 env steps = 1 gradient update every 8 env steps!
   On a 2-vCPU machine, executing 256 forward-backward passes with policy ratio calculations,
   value loss, entropy loss, gradient clipping, and Adam optimizer steps took >60% of total runtime!
   Environment step collection was fast, but PyTorch SGD optimization completely starved the CPU.

2. MULTIPROCESSING IPC BOTTLENECK ON 2-vCPU INSTANCES (SubprocVecEnv vs DummyVecEnv):
   On 2 vCPUs, running `SubprocVecEnv` creates 4-8 worker processes + 1 main process, all competing
   for 2 CPU cores. OS pipe serialization (pickle of obs, rewards, dones, infos) and context switching
   overhead cause severe cache thrashing:
     - `SubprocVecEnv` (4 envs): ~3140 FPS (host) -> ~1350 FPS (Colab)
     - `DummyVecEnv` (4 envs):   ~4270 FPS (host) -> ~1850 FPS (Colab)
     - `SubprocVecEnv` (2 envs): ~1860 FPS (host) -> ~780 FPS (Colab)
   `DummyVecEnv` runs sequentially in the main thread with zero IPC, zero serialization, and zero context switches.

3. POLICY INFERENCE CALL OVERHEAD DURING ROLLOUT (n_envs: 4 vs 8):
   With `n_envs = 4` and `n_steps = 512` (rollout 2048): PyTorch policy inference is called 512 times with batch size 4.
   With `n_envs = 8` and `n_steps = 256` (rollout 2048): PyTorch policy inference is called 256 times with batch size 8.
   Doubling `n_envs` in `DummyVecEnv` halves the number of Python-to-PyTorch forward call overheads without IPC penalty!

4. THREAD CONTENTION IN PYTORCH:
   Running PyTorch with multiple OpenMP/MKL threads on compact MLPs (64x64) induces futex locking overhead.
   `torch.set_num_threads(1)` and `torch.set_num_interop_threads(1)` consistently outperform multi-threading.

BENCHMARK MATRIX MEASURED ON 2 PINNED CPU CORES (Colab Simulation: taskset -c 0,1):

| Configuration                               | Envs | Backend        | Batch | Epochs | n_steps | FPS (Host 2 Cores) | Estimated Colab FPS |
|---------------------------------------------|------|----------------|-------|--------|---------|---------------------|---------------------|
| Regression baseline (bs=64, ep=8, ns=512)   | 4    | DummyVecEnv    | 64    | 8      | 512     | 2,506 st/s          | ~1,050 st/s         |
| Subproc 2 (bs=128, ep=4, ns=1024)           | 2    | SubprocVecEnv  | 128   | 4      | 1024    | 1,863 st/s          | ~780 st/s           |
| Subproc 4 (bs=128, ep=4, ns=512)            | 4    | SubprocVecEnv  | 128   | 4      | 512     | 3,143 st/s          | ~1,350 st/s         |
| Opt-A: Balanced (bs=128, ep=4, ns=512)      | 4    | DummyVecEnv    | 128   | 4      | 512     | 4,268 st/s          | ~1,850 st/s         |
| Opt-C: Large Batch (bs=256, ep=4, ns=512)   | 4    | DummyVecEnv    | 256   | 4      | 512     | 4,862 st/s          | ~2,100 st/s         |
| Opt-E: Fast Vectorized (bs=128, ep=4, ns=256)| 8   | DummyVecEnv    | 128   | 4      | 256     | 5,776 st/s          | ~2,500 st/s         |
| Opt-F: Ultra Vectorized (bs=256, ep=4, ns=256)| 8  | DummyVecEnv    | 256   | 4      | 256     | 6,825 st/s          | ~2,900 st/s         |

Training fixes already implemented:
- Re-tuned default PPO hyperparameters in `play2learn_lunarlander/config.py`:
  `DEFAULT_N_ENVS = 8`, `batch_size = 128`, `n_epochs = 4`, `n_steps = 256`.
- Updated `get_optimal_vec_config` in `play2learn_lunarlander/environment.py` and `play2learn_core/agent.py`:
  resolves 8 environments with `DummyVecEnv` on Colab and <= 4 CPUs.
- Restores Lunar Lander training throughput from ~1000 st/s to ~2500-2800 st/s on Colab.

================================================================================
PART 2: POST-TRAINING & REPORT GENERATION BOTTLENECK ANALYSIS
(Problem: The report generation after training is particularly slow compared to before)

In addition to training throughput, users reported that the post-training pipeline
(evaluation, video/GIF creation, archiving, and technical report generation) takes
excessive wall-clock time on Google Colab (often 20 to 35 seconds, which on a fast 40s
training run accounts for almost half of the total user waiting time!).

Here is the exact diagnostic breakdown of the post-training pipeline:

1. DUAL EVALUATION & VIDEO RENDERING PIPELINE (`evaluate_and_record`):
   - Step 1 (Sweep): Runs 20 to 80 deterministic episodes with fast PyTorch predictor to compute
     unbiased metrics (`eval_success_rate`, `mean_fuel`, `mean_length`).
   - Step 2 (Video Replay): Re-simulates Box2D physics for up to 3 successes and 3 failures
     (6 full episodes, ~1500-2000 steps) using `gym.make(..., render_mode="rgb_array")` and pipes
     raw RGB frames to ffmpeg via imageio (`libx264 -preset ultrafast`).
   - Step 3 (Best Success GIF): If `create_gif=True`, captures downscaled frames of the best success
     video and executes `imageio.mimsave(..., fps=25, loop=0)`. In pure Python/Pillow, GIF encoding
     requires color quantization (computing an optimal 256-color palette) and LZW compression on
     100-200 frames, consuming significant CPU cycles.

2. LEARNING TIMELAPSE GIF GENERATION (`crea_gif_evoluzione_apprendimento`):
   - Simulates 4 separate evaluation episodes (one for each training milestone: 25%, 50%, 75%, 100%)
     on the same seed.
   - For up to 240 steps per episode:
     - Grabs RGB frame from Gymnasium Box2D.
     - Resizes frame to 400px width using PIL `BILINEAR`.
     - Allocates PIL `ImageDraw.Draw` on each frame to render dark overlay rectangles and 2 lines of text
       (stage title, step count, outcome, rolling success rate).
     - Appends frozen trailing frames for visual readability.
   - Writes all frames into an animated GIF via `imageio.get_writer(mode="I", fps=15, loop=0)`.
   - PROFILING RESULT: Generating this GIF alone took 4.80 seconds on an 8-core host and ~12-15 seconds
     on Colab's 2 vCPUs! Color palette quantization and LZW compression across ~400 frames is heavily
     CPU-bound.
   - Applied optimization: Set default `stride = 3` (sampling every 3rd step instead of 2nd) and reduced
     frozen frames from 8 to 4. This cuts frame count by 35% and reduces host time from 3.54s to 2.41s
     (saving ~4-5s on Colab), but deeper architectural improvements are desirable.

3. REPORT GENERATION LATENCY & DUPLICATION (`genera_pdf_report` & `genera_report_esperimento`):
   - Matplotlib Vector PDF: `genera_pdf_report` uses `matplotlib.backends.backend_pdf.PdfPages` to generate
     a 2-page A4 document with flight telemetry tables and an embedded 2x2 training curve image.
     Previously hardcoded at `dpi=300`, rendering two 2480x3508 canvases in software took ~0.5s on host
     and ~1.5-2s on Colab. (We added `dpi=150` default which cuts rendering time in half while preserving
     vector text curves).
   - Inlining Multi-Megabyte Base64 Media in HTML: `genera_report_esperimento` reads `evoluzione_apprendimento.gif`
     (often 3 to 8 MB) and encodes it via `base64.b64encode().decode("ascii")` to inline directly inside
     `report_esperimento.html`. Base64-encoding multi-MB binary files in Python string memory creates
     unnecessary memory pressure.
   - Redundant Report Generation:
     - When `eval_success_rate >= 80%`, `run_lab` calls `archivia_esperimento_successo`, which runs
       `genera_report_esperimento` to generate the PDF, HTML, and Markdown reports.
     - When the student later executes notebook cell 8 (`report = risultati.genera_report()`), it was
       executing `genera_report_esperimento` a SECOND time from scratch!
     - Applied optimization: `LabResult.genera_report` now checks if `report_bundle["rep_files"]` already
       exists from automatic archiving and returns it immediately if default parameters are used.

4. JUPYTER / COLAB FRONTEND BLOAT (`display_notebook_summary`):
   - `display_notebook_summary` reads up to 4 success MP4s, 4 failure MP4s, and the animated GIF from disk,
     converts ALL of them into Base64 strings, and builds a massive HTML string passed to `IPython.display.HTML()`.
   - Transferring 15-30 MB of Base64 HTML down the Jupyter/Colab WebSocket/iframe causes browser lag,
     frontend rendering freezing, and high kernel serialization latency.

================================================================================
AREAS FOR YOUR DEEP AUDIT & REQUESTED RECOMMENDATIONS:

We seek your elite architectural, algorithmic, and systems engineering guidance to optimize
both Training Throughput and Post-Training / Reporting latency to the limit:

1. POST-TRAINING & REPORTING OPTIMIZATIONS (HIGH PRIORITY):
   a. Asynchronous / Background Generation:
      - Can video recording, milestone GIF generation, and PDF export be dispatched to a background
        `threading.Thread` or `concurrent.futures.ThreadPoolExecutor` immediately upon `model.learn()` completion,
        returning the `LabResult` object to the student with zero blocking delay?
      - How should notebook cell display (`display_notebook_summary`) handle asynchronous video availability
        gracefully (e.g., placeholder or dynamic HTML/JS polling)?
   b. Modern Web Media Formats vs Animated GIF:
      - Animated GIF uses an obsolete 1989 LZW compression algorithm with single-threaded 256-color palette
        quantization.
      - Can we switch or offer Animated WebP (`imageio.mimsave(..., format='WEBP', quality=80)`) or lightweight
        H.264 MP4 looping tags (`<video autoplay loop muted playsinline>`)?
      - How do Animated WebP and looping MP4 compare in CPU encoding speed, file size, and browser compatibility
        inside Google Colab and Jupyter Notebook?
   c. Fast-Path Frame Annotation & Downscaling:
      - PIL `Image.fromarray(f).resize(...)` + `ImageDraw.Draw` has significant per-frame overhead.
      - Can we use OpenCV (`cv2.resize` and `cv2.putText` with pre-allocated NumPy buffers) or direct NumPy
        array slicing for banner rendering? What speedup can be expected?
   d. Single-Pass Sweep & Render:
      - Currently, Phase 1 (Sweep) runs fast in `render_mode=None`, and Phase 2 re-simulates the chosen episodes
        in `render_mode="rgb_array"`.
      - Is there a way to avoid re-simulating the episodes while maintaining the fast sweep speed, or caching
        simulation seeds/states more efficiently?
   e. Lightweight PDF / HTML Generation:
      - Is there a faster, lower-overhead alternative to Matplotlib `PdfPages` for generating clean technical
        summary sheets, or can Matplotlib figure generation be streamlined further?
      - How can we eliminate massive Base64 payload bloat in `report_esperimento.html` and `display_notebook_summary`
        (e.g., using relative disk paths, blob URLs, or lightweight HTML5 video players)?

2. TRAINING THROUGHPUT & VECTORIZED ENVIRONMENT OPTIMIZATIONS:
   a. Box2D World Stepping in `DummyVecEnv`:
      - In `DummyVecEnv`, step execution runs: `[env.step(a) for env, a in zip(self.envs, actions)]`.
      - Can Box2D stepping across 8 environments be batched or accelerated?
      - Can observation NumPy array allocation in `CustomRewardWrapper` be reused with pre-allocated buffers?
   b. PyTorch JIT / TorchScript / `torch.compile` on CPU:
      - For an MLP policy of size [64, 64], Python dispatch overhead in PyTorch is significant.
      - Can the ActorCritic policy forward pass (`forward`, `predict_values`, `evaluate_actions`) be JIT-scripted
        with `@torch.jit.script` or TorchScript tracing inside Stable-Baselines3 without breaking model checkpointing
        (`model.save()` / `model.load()`) or pickling?
   c. PPO Sample-Efficiency vs Wall-Clock Trade-Off:
      - With `batch_size=128` and `n_epochs=4`, we do 64 updates per rollout.
      - Can we introduce dynamic early-stopping of epochs based on `approx_kl > target_kl`?
      - Would an adaptive batch-size schedule improve convergence on Lunar Lander and Hill Climb Racing?
   d. Fast-Path Micro-Optimizations in `CustomRewardWrapper`:
      - In `CustomRewardWrapper.step()`, inspect the fast-path (99.5% of steps).
      - Can scalar extraction or student reward dispatch be optimized further using `__slots__` or pre-allocated tuples?

================================================================================
OUTPUT FORMAT EXPECTED:

Please structure your review into the following sections:
1. EXECUTIVE SUMMARY: High-level assessment of the current optimizations and architecture.
2. TRAINING THROUGHPUT RECOMMENDATIONS: Concrete proposals for PPO, VecEnv, Box2D, and Torch JIT.
3. POST-TRAINING & REPORTING RECOMMENDATIONS: Detailed architectural plan to make evaluation,
   video/GIF creation, and report generation instantaneous or asynchronous.
4. BENCHMARKING & PROFILING PROJECTIONS: Expected wall-clock speedup on Google Colab (2 vCPUs).
5. DROP-IN CODE REPLACEMENTS: Provide ready-to-use, tested drop-in code snippets for the recommended
   changes in `play2learn_lunarlander` and `play2learn_hillclimb`.
================================================================================
"""

CODE_HEADER = """# ==============================================================================
# RL_SCHOOL - CODICE SORGENTE COMPLETO DELLA REPOSITORY
# Progetti: Lunar Lander (LunarLander-v3) & Hill Climb Racing (HillClimbRacing-v0)
# Architettura: play2learn_core + play2learn_lunarlander + play2learn_hillclimb
# ==============================================================================

"""


def bundle_repository(root_dir: Path, code_out: Path, prompt_out: Path, unified_out: Path | None = None) -> None:
    print(f"📦 Avvio bundling repository da: {root_dir}")
    
    # 1. Scrivi il prompt di review separato
    prompt_content = REVIEW_PROMPT_TEMPLATE.strip() + "\n"
    prompt_out.write_text(prompt_content, encoding="utf-8")
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

    # 3. Opzionale: scrivi bundle unificato (prompt + codice in un unico file)
    if unified_out:
        unified_content = (
            prompt_content
            + "\n"
            + "=" * 80 + "\n"
            + "EXTRA CONTEXT: COMPLETE SOURCE CODE OF THE REPOSITORY\n"
            + "=" * 80 + "\n\n"
            + full_code
        )
        unified_out.write_text(unified_content, encoding="utf-8")
        print(f"  ✅ Bundle unificato salvato in: {unified_out.name} ({unified_out.stat().st_size:,} bytes)")


def main():
    parser = argparse.ArgumentParser(description="Genera il bundle del codice e il prompt di review.")
    parser.add_argument("--root", type=str, default=".", help="Directory radice del repository (default: .)")
    parser.add_argument("--code-out", type=str, default="codice_completo_repo.txt", help="Nome file di output per il codice")
    parser.add_argument("--prompt-out", type=str, default="PROMPT_CODE_REVIEW.txt", help="Nome file di output per il prompt")
    parser.add_argument("--unified-out", type=str, default="PROMPT_OTTIMIZZAZIONE_COMPLETO.txt", help="Nome file di output unificato (prompt + codice)")
    args = parser.parse_args()

    root_path = Path(args.root).resolve()
    code_out_path = root_path / args.code_out
    prompt_out_path = root_path / args.prompt_out
    unified_out_path = (root_path / args.unified_out) if args.unified_out else None

    bundle_repository(root_path, code_out_path, prompt_out_path, unified_out_path)


if __name__ == "__main__":
    main()
