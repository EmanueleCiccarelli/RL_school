"""
Script di bundling dedicato per Hill Climb Racing e il motore RL di Play2Learn.
Estrae tutti i moduli sorgente rilevanti (escludendo Lunar Lander) e genera:
1. codice_hillclimb_engine.txt: il codice completo di engine, wrapper, lab, visualizer e test.
2. PROMPT_BUG_SPINNING_HILLCLIMB.txt: il prompt strutturato per l'analisi da parte di un modello avanzato.
"""

from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
ORIG_CLONE_DIR = Path(os.environ.get("TEMP", "/tmp")) / "hillclimbracing_orig" / "hill_racing_env" / "envs"

HILLCLIMB_FILES = [
    # Engine Box2D & Pygame
    "play2learn_hillclimb/engine/__init__.py",
    "play2learn_hillclimb/engine/hill_racing.py",
    "play2learn_hillclimb/engine/car.py",
    "play2learn_hillclimb/engine/person.py",
    "play2learn_hillclimb/engine/wheels.py",
    "play2learn_hillclimb/engine/ground.py",
    "play2learn_hillclimb/engine/perlin.py",
    "play2learn_hillclimb/engine/agent.py",

    # Play2Learn HillClimb Orchestration
    "play2learn_hillclimb/__init__.py",
    "play2learn_hillclimb/config.py",
    "play2learn_hillclimb/environment.py",
    "play2learn_hillclimb/agent.py",
    "play2learn_hillclimb/lab.py",
    "play2learn_hillclimb/evaluator.py",
    "play2learn_hillclimb/visualizer.py",
    "play2learn_hillclimb/reporting.py",
    "play2learn_hillclimb/diagnostics.py",
    "play2learn_hillclimb/favorites.py",
    "play2learn_hillclimb/reset.py",
    "play2learn_hillclimb/storage.py",

    # Play2Learn Shared Core
    "play2learn_core/__init__.py",
    "play2learn_core/agent.py",
    "play2learn_core/base_env.py",
    "play2learn_core/evaluator.py",
    "play2learn_core/favorites.py",
    "play2learn_core/reporting.py",
    "play2learn_core/storage.py",

    # Tests
    "tests/test_hillclimb.py",
]

ORIGINAL_ENGINE_FILES = [
    "car.py",
    "person.py",
    "wheels.py",
    "hill_racing.py",
    "agent.py",
]


def bundle_code() -> Path:
    out_file = ROOT_DIR / "codice_hillclimb_engine.txt"
    print(f"📦 Creazione dump codice in: {out_file.name}...")

    total_files = 0
    total_bytes = 0

    with open(out_file, "w", encoding="utf-8") as out:
        out.write("=" * 80 + "\n")
        out.write("CODEBASE DUMP: PLAY2LEARN HILL CLIMB RACING & RL ENGINE\n")
        out.write("Questo dump include tutti i file sorgente dell'ambiente, motore fisico Box2D,\n")
        out.write("rendering Pygame, wrapper Gymnasium, architettura SB3 PPO e suite di test.\n")
        out.write("=" * 80 + "\n\n")

        # 1. File del progetto corrente
        for rel_path in HILLCLIMB_FILES:
            full_path = ROOT_DIR / rel_path
            if full_path.exists():
                content = full_path.read_text(encoding="utf-8", errors="replace")
                out.write("\n" + "=" * 80 + "\n")
                out.write(f"FILE: {rel_path}\n")
                out.write("=" * 80 + "\n\n")
                out.write(content)
                out.write("\n")
                total_files += 1
                total_bytes += len(content)
            else:
                print(f"  ⚠️ File non trovato: {rel_path}")

        # 2. File di riferimento dell'implementazione originale (alexzh3/hillclimbracing)
        if ORIG_CLONE_DIR.exists():
            out.write("\n\n" + "#" * 80 + "\n")
            out.write("RIFERIMENTO: SORGENTI ORIGINALI DI ALEXZH3/HILLCLIMBRACING (PER CONFRONTO DIRETTO)\n")
            out.write("#" * 80 + "\n\n")

            for orig_name in ORIGINAL_ENGINE_FILES:
                orig_path = ORIG_CLONE_DIR / orig_name
                if orig_path.exists():
                    orig_content = orig_path.read_text(encoding="utf-8", errors="replace")
                    out.write("\n" + "-" * 80 + "\n")
                    out.write(f"ORIGINAL REPO FILE (alexzh3): {orig_name}\n")
                    out.write("-" * 80 + "\n\n")
                    out.write(orig_content)
                    out.write("\n")
                    total_files += 1
                    total_bytes += len(orig_content)

    print(f"  ✅ Dump completato: {out_file.name} ({total_bytes:,} bytes, {total_files} file inclusi)")
    return out_file


def generate_prompt() -> Path:
    out_file = ROOT_DIR / "PROMPT_BUG_SPINNING_HILLCLIMB.txt"
    print(f"📝 Generazione prompt in: {out_file.name}...")

    prompt_content = """# CONTESTO E RICHIESTA DI CODE REVIEW SPECIALISTICA: BUG SPINNING OMINO IN HILL CLIMB RACING

## 1. OBIETTIVO DEL TASK
Stiamo sviluppando `play2learn_hillclimb`, un ambiente didattico di Reinforcement Learning per il celebre gioco arcade **Hill Climb Racing**, basato sulla fisica 2D **Box2D (pybox2d)**, rendering grafico **Pygame** e algoritmi PPO di **Stable-Baselines3** su standard **Farama Gymnasium**.

Il codice dell'engine deriva dall'implementazione open-source di **alexzh3/hillclimbracing** (riadattata e ottimizzata per il training e la valutazione didattica).

---

## 2. DESCRIZIONE DEL BUG
Negli allegati troverai i video registrati durante le sessioni di verifica/valutazione dell'agente (es. `episodio_02_fallimento_seed_137.mp4`, `episodio_04_fallimento_seed_211.mp4`).

**Sintomo osservato:**
1. Durante la corsa, non appena l'auto affronta un dislivello, una salita ripida o accelera, la figura del pilota (l'omino: testa e torso) comincia a oscillare in modo anomalo e a "spinnare" (ruotare su se stesso a 360° o piegarsi completamente all'indietro fuori dall'abitacolo).
2. Questo comportamento rende il contatto tra la testa del pilota e il suolo (o il telaio) praticamente inevitabile nel giro di pochi step, causando la morte immediata dell'agente (`agent.dead = True` -> `terminated = True`) e impedendogli di scalare la collina.
3. Inoltre, dal punto di vista grafico, lo sprite del pilota sembra staccarsi dall'abitacolo e orbitare in modo vistoso intorno al telaio della macchina durante le rotazioni.

---

## 3. INDAGINI GIA' SVOLTE ED ELEMENTI IDENTIFICATI

### A. Geometria dei Giunti Box2D tra Telaio e Pilota (`car.py` e `person.py`)
Nel modello originale di `alexzh3`, il pilota è composto da due corpi dinamici (`Torso` e `Head`):
- Il `Torso` è collegato al telaio dell'auto tramite:
  1. Un **`b2RevoluteJoint`** (cerniera a rotazione libera) posizionato alla base del bacino.
  2. Un **`b2DistanceJoint`** (molla elastica con `frequencyHz=5.0`, `dampingRatio=0.1`, `length *= 1.1`) che collega la parte alta del petto (`y - height * 2/3`) alla parte anteriore del cofano dell'auto (`x + chassis_width / 2`).
- La `Head` è collegata al `Torso` tramite un altro `b2RevoluteJoint` (collo) e un `b2DistanceJoint`.

**Osservazione meccanica rilevante:**
- Il `b2DistanceJoint` funge da molla di richiamo elastica per mantenere l'omino in posizione eretta seduta.
- Tuttavia, se il busto si inclina all'indietro oltre una certa soglia angolare (~70°-90° rispetto al telaio), la linea d'azione della molla oltrepassa il centro di rotazione del giunto a cerniera (*instabilità over-center snap-through*). In quella configurazione la forza della molla inverte il braccio di leva della coppia, spingendo il busto ad accelerare ulteriormente all'indietro e a collassare dietro/sotto la vettura anziché raddrizzarlo!

### B. Collisioni Box2D (`ContactListener` in `hill_racing.py`)
- Nel `ContactListener`, il contatto tra `head` e `ground` imposta `agent.dead = True`.
- Se il busto si piega all'indietro o spinna fuori dalla sagoma dell'auto, il raggio della testa (`radius = 20 px = 0.67 m`) tocca terra anche mentre le ruote sono ancora a contatto col terreno.

### C. Rendering Grafico Pygame (`draw_person_car`, `draw_torso`, `draw_head`, `draw_wheel`)
- In Pygame, `pygame.transform.rotate(image, angle)` crea una nuova superficie il cui rettangolo di delimitazione varia di dimensioni a seconda dell'angolo $\\theta$.
- Se non si allinea il rettangolo ruotato al centro geometrico del corpo Box2D (`rotated.get_rect(center=(cx, cy))`), il disegno slitta e orbita di decine di pixel rispetto al baricentro fisico reale del corpo dinamico.

### D. Distribuzione delle Masse e Inerzie
- Nel telaio (`chassis_body`), l'originale ha 3 fixture poligonali (`mass ~ 2.59 kg`, `inertia ~ 3.71 kg·m²`).
- Il torso ha densità 0.002 (`mass ~ 0.0036 kg`) e la testa 0.001 (`mass ~ 0.0014 kg`).
- `angularDamping`: nell'originale il telaio ha `angularDamping = 0.1`.

---

## 4. COSA TI VIENE CHIESTO
Analizza attentamente il file `codice_hillclimb_engine.txt` allegato (che include sia la nostra codebase che i sorgenti originali di `alexzh3`) insieme ai video forniti.

Ti chiediamo di:
1. **Identificare con precisione la causa o le concause del bug logico dello spinning del pilota:**
   - Perché nell'ambiente originale l'omino rimaneva relativamente stabile nel veicolo mentre qui manifesta questo spinning distruttivo?
   - C'è una discrepanza tra le coordinate/scale (pixel vs metri Box2D), i punti di ancoraggio dei giunti, o i parametri fisici (frequenza, smorzamento, lunghezze a riposo)?
2. **Definire la soluzione architetturale e fisica ottimale:**
   - Come devono essere configurati i giunti (`b2RevoluteJointDef`, `b2DistanceJointDef`) affinché il pilota si muova in modo naturale come un ragdoll reattivo alle accelerazioni senza mai ribaltarsi all'indietro fuori dal sedile o compiere rotazioni a 360°?
   - Quali limiti angolari (`lowerAngle`, `upperAngle`, `enableLimit`), motori di richiamo o molle devono essere applicati?
   - Come deve essere gestita la maschera di collisione (bitmask) per evitare penetrazioni anomale o falsi contatti testa-suolo?
3. **Verificare la correttezza del rendering Pygame:**
   - Assicurarsi che le matrici di trasformazione o i pivot di disegno in Pygame riflettano esattamente la posizione fisica e l'orientamento dei corpi Box2D a qualsiasi angolo $\\theta \\in [0, 360^\\circ]$.
4. **Fornire il codice pronto e corretto (drop-in replacement):**
   - Fornisci l'implementazione esatta e pulita per i file coinvolti (principalmente `car.py`, `person.py`, ed eventualmente `hill_racing.py` / `wheels.py`).
"""

    out_file.write_text(prompt_content.strip() + "\n", encoding="utf-8")
    print(f"  ✅ Prompt completato: {out_file.name} ({len(prompt_content):,} caratteri)")
    return out_file


def main():
    dump_path = bundle_code()
    prompt_path = generate_prompt()
    print("\n🎉 Operazione completata con successo!")
    print(f"  1. Codice sorgente dump: {dump_path}")
    print(f"  2. Prompt per il modello: {prompt_path}")


if __name__ == "__main__":
    main()
