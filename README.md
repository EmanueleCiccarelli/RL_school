# 🚀 RL_school - Laboratorio di Reinforcement Learning: Lunar Lander & Hill Climb Racing

[![Open In Colab (Lunar Lander)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EmanueleCiccarelli/RL_school/blob/main/laboratorio_lunar_lander.ipynb)
[![Open In Colab (Hill Climb)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EmanueleCiccarelli/RL_school/blob/main/laboratorio_hill_climb_colab.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-Box2D-green.svg)](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
[![Stable Baselines3](https://img.shields.io/badge/Algorithm-PPO-orange.svg)](https://stable-baselines3.readthedocs.io/)
[![Unit Tests](https://img.shields.io/badge/tests-28%20passed-brightgreen.svg)](https://github.com/EmanueleCiccarelli/RL_school)

Laboratorio didattico avanzato di **Reinforcement Learning** per studenti di scuola secondaria di secondo grado (quinta superiore) e corsi introduttivi universitari. L'obiettivo formativo è padroneggiare i principi dell'apprendimento per rinforzo attraverso il **Reward Shaping** su due scenari di simulazione fisica Box2D:
1. 🚀 **Lunar Lander (`LunarLander-v3`)**: pilotaggio e atterraggio morbido tra le bandierine della piazzola lunare.
2. 🏎️ **Hill Climb Racing (`HillClimbRacing-v0`)**: controllo di trazione, velocità e stabilità di un veicolo su colline procedurali generate con Perlin Noise (ispirato al repository `alexzh3/hillclimbracing`).

Tutto il codice di orchestrazione, calcolo numerico, parallelizzazione dei processi, streaming video H.264 e rendering interattivo è strutturato con **design pattern riutilizzabili** nel core unificato `play2learn_core` e specializzato nei pacchetti `play2learn_lunarlander` e `play2learn_hillclimb`.

---

## 📂 Architettura del Progetto

```text
RL_school/
│
├── pyproject.toml                          # 📦 Metadata packaging PEP 517 / PEP 621 & tool config
├── requirements.txt                        # 📋 Dipendenze pip del laboratorio (incluso Pygame)
├── laboratorio_lunar_lander.ipynb          # 🌐 Notebook didattico Lunar Lander (Colab & Cloud)
├── laboratorio_hill_climb.ipynb            # 🏎️ Notebook didattico Hill Climb Racing
├── laboratorio_hill_climb_colab.ipynb      # ☁️ Notebook Hill Climb per Google Colab
├── README.md                               # 📖 Documentazione completa del laboratorio
├── .gitignore                              # 🙈 Esclusioni Git per artefatti pesanti e checkpoint
│
├── play2learn_core/                        # 🧠 Core comune (Design Pattern & Orchestrazione)
│   ├── base_env.py                         # BaseRewardWrapper con ispezione firma & guardie NaN/Inf
│   ├── agent.py                            # PPO orchestrator, scheduler LR & sliding window O(1)
│   ├── evaluator.py                        # Streaming video recording H.264 (<1 MB RAM) & GIF generator
│   ├── storage.py                          # Scritture atomiche, cartelle univoche & timestamp resolution
│   ├── favorites.py                        # BaseFavoritesManager per cataloghi persistenti
│   ├── reporting.py                        # Base report engine: PDF vettoriale, HTML5, Markdown
│   └── base_interactive.py                 # Sintetizzatore Web Audio & game loop fisso 50 Hz
│
├── play2learn_lunarlander/                 # 🚀 Pacchetto specializzato Lunar Lander
│   ├── config.py                           # Missioni lunari, costanti fisiche
│   ├── environment.py                      # LunarLanderWrapper & classificazione atterraggi
│   ├── diagnostics.py                      # Flight Coach: diagnostica atterraggi
│   ├── visualizer.py                       # Mappe fisiche e telemetria radar
│   ├── evaluator.py                        # Valutazione 20 seed deterministici & video
│   ├── reporting.py                        # Schede tecniche (PDF, HTML5, Markdown)
│   ├── interactive.py                      # Arcade Canvas HTML5: Single Player & Tu vs Agente
│   └── lab.py                              # run_lab() didattico
│
├── play2learn_hillclimb/                   # 🏎️ Pacchetto specializzato Hill Climb Racing
│   ├── engine/                             # Motore fisico Box2D + Pygame (human + rgb_array headless)
│   │   ├── hill_racing.py                  # Gymnasium Env HillClimbRacing-v0
│   │   ├── car.py, wheels.py, person.py    # Chassis, ruote a sospensione & ragdoll driver
│   │   ├── ground.py, perlin.py            # Terreno procedurale Perlin Noise (100% pure Python fallback)
│   │   └── pictures/                       # Sprite originali vettura, ruote, pilota ed erba
│   ├── config.py                           # Tracciati speciali: Campagna, Deserto, Montagna, Luna
│   ├── environment.py                      # HillClimbRewardWrapper & classificazione ribaltamenti/fossi
│   ├── diagnostics.py                      # Diagnostica guida: distanza, velocità, airtime, ribaltamenti
│   ├── visualizer.py                       # Mappa veicolo e dashboard telemetrica colline
│   ├── evaluator.py                        # Streaming mp4 ad alta fedeltà & GIF evoluzione
│   ├── reporting.py                        # Scheda tecnica (Distanza max, velocità, airtime, ribaltamenti)
│   ├── interactive.py                      # Arcade Canvas HTML5: Guida manuale & Gara Tu vs Agente AI
│   └── lab.py                              # run_lab() didattico per Hill Climb
│
└── tests/                                  # 🧪 Suite di test unitari automatizzati (28/28 passing)
    ├── test_hillclimb.py                   # Test suite Hill Climb Racing
    ├── test_environment.py                 # Test wrapper reward e guardie NaN
    ├── test_agent.py                       # Test callback scorrimento O(1)
    ├── test_interactive.py                 # Test Canvas interattivo & compatibilità
    ├── test_reporting.py                   # Test PDF, HTML escape e serializzazione
    ├── test_storage_favorites.py           # Test persistenza atomica
    └── test_config.py                      # Test configurazioni e iperparametri
```

---

## ⚡ Come Avviare il Laboratorio

Il laboratorio è disponibile in **due modalità sincronizzate**:

### Modalità 1: Google Colab (Consigliata per la classe)
1. Clicca sul badge in cima al README:  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EmanueleCiccarelli/RL_school/blob/main/laboratorio_lunar_lander.ipynb)
2. Il notebook clona automaticamente la repository GitHub e installa le dipendenze in circa 30 secondi.
3. Gli studenti possono subito modificare la cella di `custom_reward` ed eseguire l'addestramento sfruttando le risorse cloud gratuite.

### Modalità 2: Ambiente Locale (per docenti o sviluppo offline)

1. **Clona la repository sul tuo computer:**
   ```bash
   git clone https://github.com/EmanueleCiccarelli/RL_school.git
   cd RL_school
   ```

2. **Crea e attiva un ambiente virtuale (consigliato):**
   * **Windows (PowerShell):**
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   * **Linux / macOS:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Installa le dipendenze:**
   Puoi installare le dipendenze classiche o installare il pacchetto in modalità sviluppatore editable:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   # oppure con il nuovo standard pyproject.toml:
   pip install -e .
   ```
   > 💡 **Nota per utenti Windows:** Se `gymnasium[box2d]` segnala la mancanza di SWIG durante la compilazione, è sufficiente installare il compilatore precompilato con:
   > ```bash
   > pip install swig
   > pip install "gymnasium[box2d]"
   > ```

4. **Avvia Jupyter Notebook o VS Code:**
   ```bash
   jupyter lab
   # oppure: jupyter notebook
   ```

5. **Apri il notebook dedicato:**  
   Naviga ed apri `play2learn_lunarlander/laboratorio_lunar_lander_locale.ipynb`.

---

## 🧭 Struttura del Percorso Didattico (Le 10 Sezioni)

Il notebook guida gli studenti passo-passo attraverso 10 sezioni strutturate:

1. **Sezione 1: Setup dell'Ambiente e Importazioni**  
   Verifica delle versioni software, rilevamento automatico hardware (CPU/GPU) e configurazione thread per massimizzare gli FPS.
2. **Sezione 2: Esplorazione dell'Ambiente Box2D**  
   Ispezione visiva del vettore di stato ad 8 dimensioni (`x`, `y`, `vx`, `vy`, `angle`, `vangle`, `left_leg`, `right_leg`) e dello spazio delle 4 azioni discrete.
3. **Sezione 3: L'Obiettivo Didattico (Reward Shaping)**  
   Modellazione matematica della funzione di ricompensa con prevenzione delle trappole del *Reward Hacking* (trappola kamikaze e hovering infinito).
4. **Sezione 4: Addestramento dell'Agente PPO**  
   Esecuzione del training a 300k step con barra di avanzamento interattiva, telemetria in tempo reale (`Reward`, `Fuel`, `Success%`) e generazione opzionale di GIF animate sui milestone di apprendimento (`salva_gif_progresso=True`).
5. **Sezione 5: Valutazione e Telemetria**  
   Dashboard 2x2 ad alta risoluzione con curve di apprendimento e visualizzazione mista dei video: fino a 3 allunaggi riusciti (🟢) e fino a 3 schianti (🔴) per comprendere gli errori residui.
6. **Sezione 6: Dimostrazione Live dell'Allunaggio**  
   Esecuzione immediata di un episodio di test con rendering grafico o video HTML5.
7. **Sezione 7: Sfide Avanzate (Missioni Speciali)**  
   Metti alla prova la robustezza della policy in 6 scenari fisici estremi:
   - 🌬️ *Vento Laterale Improvviso* (`vento`)
   - 🪐 *Gravità Lunare Ridotta* (`bassa_gravita`)
   - 🪨 *Gravità Gioviana Pesante* (`alta_gravita`)
   - ⚡ *Spinta Motore Principale Depotenziata* (`motore_debole`)
   - ⛽ *Carburante Critico e Limitato* (`carburante_critico`)
   - 🌪️ *Tempesta Lunare Totale* (`tempesta_lunare`)
8. **Sezione 8: Archivio Preferiti e Successi**  
   Separazione concettuale pulita:
   - `runs/successi/`: Tutti gli esperimenti con tasso di successo $\ge 80\%$, corredati da scheda tecnica.
   - `runs/favorites/`: I video e i modelli scelti e salvati esplicitamente dallo studente con un commento personale.
9. **Sezione 9: Generatore Report da Timestamp**  
   Inserendo il timestamp di qualsiasi run storica (es. `20261115_214530`), genera istantaneamente la scheda tecnica in versione *Base* o *Avanzata* (con valutazione multi-video) e permette il confronto grafico tra esperimenti.
10. **Sezione 10: Sfida Interattiva: Single Player & Tu vs Agente**  
    Mini-gioco interattivo giocabile sia in locale (tastiera/tasti freccia) sia su Google Colab:
    - **Single Player:** Prova a pilotare manualmente il modulo spaziale per capire quanto sia difficile l'allunaggio umano!
    - **Tu vs Agente:** Sfida diretta contro il tuo modello addestrato con visualizzazione sinottica delle traiettorie fianco a fianco, confronto del consumo di carburante e calcolo del punteggio finale.

---

## 🧠 L'Obiettivo Didattico: La Funzione di Reward

Ad ogni step ($0{,}02\text{ secondi}$), l'agente riceve gli 8 sensori fisici di bordo (`obs`), il comando motore (`action`) e i flag di stato terminali:

```python
def custom_reward(obs, action, is_success, is_crashed, is_timeout):
    x, y, vx, vy, angle, vangle, left_leg, right_leg = obs
    reward = 0.0

    # =========================================================================
    # PROGETTA QUI LA TUA FORMULA DI REWARD SHAPING
    # =========================================================================

    # Esempi di termini da considerare:
    # - Posizione orizzontale: avvicinarsi alla piazzola centrale (x = 0.0)
    # - Velocità di discesa: frenare dolcemente quando vy è negativo
    # - Assetto angolare: mantenere il modulo verticale (angle = 0.0)
    # - Efficienza: premiare il contatto a terra e lo spegnimento motori

    # Eventi terminali
    if is_success:
        reward += 100.0
    if is_crashed:
        reward -= 100.0

    return float(reward)
```

---

## 🧪 Suite di Test Unitari

Il progetto include una suite completa di test unitari automatizzati, eseguibili sia tramite il modulo standard Python `unittest` (zero dipendenze esterne richieste), sia tramite `pytest`:

```bash
# Esecuzione con il modulo standard di Python:
python -m unittest discover tests

# Oppure con pytest:
pytest
```

I test verificano:
- Costanti fisiche, 6 configurazioni di missioni speciali e parametri PPO/VecNormalize.
- Flessibilità della signature di `custom_reward` (compatibilità con 1, 2, 5 argomenti o `**kwargs`).
- Calcolo in tempo reale a complessità $O(1)$ della finestra mobile del tracker di reward e telemetria.
- Persistenza e indicizzazione corretta delle cartelle di run e dei preferiti.
- Estrazione geometrica del terreno Box2D e tracciamento delle traiettorie per la sfida interattiva.
- Integrità dei file sorgente del progetto tramite `reset.py --check`.

---

## 🧹 Manutenzione e Reset dell'Ambiente

Per ripulire l'ambiente dalle cartelle di training, azzerare le celle di output dei notebook e ripristinare il laboratorio allo stato iniziale:

```bash
# Modalità interattiva con conferma:
python play2learn_lunarlander/reset.py

# Modalità forzata immediata:
python play2learn_lunarlander/reset.py --force

# Modalità verifica non-distruttiva (Dry-run):
python play2learn_lunarlander/reset.py --check
```

---

## 📜 Licenza e Crediti

Progetto didattico e scientifico sviluppato per la formazione nelle scuole secondarie di secondo grado e università.  
Repository ufficiale: [https://github.com/EmanueleCiccarelli/RL_school](https://github.com/EmanueleCiccarelli/RL_school)
