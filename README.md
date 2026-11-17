# 🚀 RL_school - Laboratorio di Reinforcement Learning: Lunar Lander & Hill Climb Racing

[![Open In Colab (Lunar Lander)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EmanueleCiccarelli/RL_school/blob/main/laboratorio_lunar_lander.ipynb)
[![Open In Colab (Hill Climb Racing)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EmanueleCiccarelli/RL_school/blob/main/laboratorio_hill_climb.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-Box2D-green.svg)](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
[![Stable Baselines3](https://img.shields.io/badge/Algorithm-PPO-orange.svg)](https://stable-baselines3.readthedocs.io/)
[![Unit Tests](https://img.shields.io/badge/tests-31%20passed-brightgreen.svg)](https://github.com/EmanueleCiccarelli/RL_school)

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
├── laboratorio_lunar_lander.ipynb          # 🚀 Notebook didattico: Lunar Lander (Colab & Locale)
├── laboratorio_hill_climb.ipynb            # 🏎️ Notebook didattico: Hill Climb Racing (Colab & Locale)
├── README.md                               # 📖 Documentazione completa del laboratorio
├── .gitignore                              # 🙈 Esclusioni Git per artefatti pesanti e checkpoint
│
├── hill_racing_env/                        # 🏎️ Motore fisico Gymnasium Box2D + Pygame (alexzh3/hillclimbracing)
│   ├── envs/                               # Ambiente HillRacing-v0, telaio, ruote, pilota e terreno
│   └── pictures/                           # Sprite originali vettura, ruote, pilota ed erba
│
├── play2learn_core/                        # 🧠 Core comune (Design Pattern & Orchestrazione)
│   ├── base_env.py                         # BaseRewardWrapper con ispezione firma & guardie NaN/Inf
│   ├── agent.py                            # PPO orchestrator, scheduler LR & sliding window O(1)
│   ├── evaluator.py                        # Streaming video recording H.264 (<1 MB RAM) & GIF generator
│   ├── storage.py                          # Scritture atomiche, cartelle univoche & timestamp resolution
│   ├── favorites.py                        # BaseFavoritesManager per cataloghi persistenti
│   └── reporting.py                        # Base report engine: PDF vettoriale, HTML5, Markdown
│
├── play2learn_lunarlander/                 # 🚀 Pacchetto specializzato Lunar Lander
│   ├── config.py                           # Missioni lunari, costanti fisiche
│   ├── environment.py                      # LunarLanderWrapper & classificazione atterraggi
│   ├── diagnostics.py                      # Flight Coach: diagnostica atterraggi
│   ├── visualizer.py                       # Mappe fisiche e telemetria radar
│   ├── evaluator.py                        # Valutazione 20 seed deterministici & video
│   ├── reporting.py                        # Schede tecniche (PDF, HTML5, Markdown)
│   └── lab.py                              # run_lab() didattico
│
├── play2learn_hillclimb/                   # 🏎️ Pacchetto specializzato Hill Climb Racing
│   ├── config.py                           # Missioni: Campagna, Deserto, Montagna, Colline, Autostrada
│   ├── environment.py                      # HillClimbRewardWrapper & adattatore Box 8D
│   ├── agent.py                            # HillClimbFeaturesExtractor & PPO
│   ├── diagnostics.py                      # Flight Coach: distanza, velocità, assetto, ribaltamenti
│   ├── visualizer.py                       # Mappa sensori e dashboard telemetrica colline
│   ├── evaluator.py                        # Streaming video H.264 & GIF evoluzione
│   ├── reporting.py                        # Scheda tecnica (PDF vettoriale, HTML5, Markdown)
│   └── lab.py                              # run_lab() didattico per Hill Climb
│
└── tests/                                  # 🧪 Suite di test unitari automatizzati
    ├── test_hillclimb.py                   # Test suite Hill Climb Racing
    ├── test_environment.py                 # Test wrapper reward e guardie NaN
    ├── test_agent.py                       # Test callback scorrimento O(1)
    ├── test_reporting.py                   # Test PDF, HTML escape e serializzazione
    ├── test_storage_favorites.py           # Test persistenza atomica
    └── test_config.py                      # Test configurazioni e iperparametri
```

---

## ⚡ Come Avviare il Laboratorio

Il laboratorio è fruibile in **due modalità sincronizzate**:

---

### 🌐 Modalità 1: Google Colab (Consigliata per la Classe - Zero Installazione)

La repository mette a disposizione **due percorsi didattici completamente separati ed indipendenti su Google Colab**, ciascuno ottimizzato per eseguire addestramento PPO, telemetria in tempo reale, streaming video H.264 headless e mini-giochi interattivi tramite browser in ambiente cloud gratuito:

#### 🚀 Percorso 1: Lunar Lander (`LunarLander-v3`)

* **Obiettivo**: Pilotare il modulo lunare verso un atterraggio morbido e controllato tra le bandierine della piazzola, coordinando spinta del motore principale e dei motori laterali e minimizzando il consumo di carburante.
* **Avvia subito su Google Colab:**  
  [![Open Lunar Lander In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EmanueleCiccarelli/RL_school/blob/main/laboratorio_lunar_lander.ipynb)
* **File notebook nel repository:** [`laboratorio_lunar_lander.ipynb`](file:///c:/Users/delop/Desktop/RL%20MATTEUCCI/laboratorio_lunar_lander.ipynb)
* **Cosa include:**
  - Setup automatico (~30 secondi): clone repository, dipendenze Gymnasium Box2D e Stable-Baselines3.
  - Simulatore fisico con radar di telemetria (quota, velocità di discesa, assetto).
  - Video streaming H.264 (fino a 3 allunaggi riusciti 🟢 e 3 schianti 🔴) e GIF evolutive dei checkpoint.
  - 6 Missioni speciali di collaudo estremo (vento laterale, gravità ridotta o gioviana, tempesta lunare).
  - Sistema di diagnostica intelligente, cataloghi successi (≥80%) e reportistica esportabile.

---

#### 🏎️ Percorso 2: Hill Climb Racing (`HillClimbRacing-v0`)

* **Obiettivo**: Progettare la funzione di reward per un veicolo fuoristrada su colline procedurali infinite generate con Perlin Noise, massimizzando la distanza percorsa e la velocità senza capovolgersi o incastrarsi nei fossi.
* **Avvia subito su Google Colab:**  
  [![Open Hill Climb Racing In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EmanueleCiccarelli/RL_school/blob/main/laboratorio_hill_climb.ipynb)
* **File notebook nel repository:** [`laboratorio_hill_climb.ipynb`](file:///c:/Users/delop/Desktop/RL%20MATTEUCCI/laboratorio_hill_climb.ipynb)
* **Cosa include:**
  - Setup automatico (~30 secondi): installazione Box2D, Pygame headless e pacchetto `play2learn_hillclimb`.
  - Motore fisico Box2D con carrozzeria, ruote a sospensione dinamica e pilota su colline Perlin Noise.
  - `HillClimbFeaturesExtractor`: architettura neurale invariante alla coordinata orizzontale e scalata per evitare la saturazione delle attivazioni.
  - 5 Tracciati speciali con profili altimetrici e difficoltà calibrate: *Campagna standard*, *Deserto dune*, *Montagna estrema*, *Colline ripide*, *Autostrada veloce*.
  - Telemetria live su 20 scenari di collaudo, streaming video H.264 e catalogazione corse.

---

### 💻 Modalità 2: Ambiente Locale (per Docenti o Sviluppo Offline)

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

4. **Avvia Jupyter Lab o VS Code:**
   ```bash
   jupyter lab
   # oppure: jupyter notebook
   ```

5. **Apri il notebook dedicato al laboratorio desiderato:**  
   * 🚀 **Per Lunar Lander:** apri `laboratorio_lunar_lander.ipynb` (oppure `play2learn_lunarlander/laboratorio_lunar_lander_locale.ipynb`).
   * 🏎️ **Per Hill Climb Racing:** apri `laboratorio_hill_climb.ipynb` (oppure `play2learn_hillclimb/laboratorio_hill_climb_locale.ipynb`).

---

## 🧭 Struttura del Percorso Didattico (Le 10 Sezioni Sincronizzate)

I notebook di **Lunar Lander** e **Hill Climb Racing** condividono la medesima struttura pedagogica in **10 sezioni progressive**, offrendo un'esperienza didattica coerente su entrambi gli ambienti:

| Sezione | 🚀 Modulo 1: Lunar Lander (`LunarLander-v3`) | 🏎️ Modulo 2: Hill Climb Racing (`HillClimbRacing-v0`) |
| :--- | :--- | :--- |
| **1. Setup & Import** | Hardware check, config thread, import `play2learn_lunarlander` | Setup Box2D/Pygame headless, import `play2learn_hillclimb` |
| **2. Esplorazione Box2D** | Stato 8D (coordinate, velocità, angolo, sensori zampette) & 4 azioni | Stato 8D (distanza, quota, velocità, assetto, contatti ruote) & 3 azioni |
| **3. Reward Shaping** | Progettazione ricompensa: allunaggio morbido vs crash/timeout | Progettazione ricompensa: avanzamento colline vs ribaltamento/fossi |
| **4. Addestramento PPO** | Training PPO (300k step), barra live, GIF evolutiva opzionale | Training PPO con `HillClimbFeaturesExtractor`, barra live e GIF |
| **5. Valutazione & Video** | Test su 20 seed fissi, dashboard 2x2, video streaming Top 3 🟢/🔴 | Test su 20 semi procedurali, streaming video alta fedeltà H.264 |
| **6. Demo Live** | Volo live immediato con telemetria e assetto | Corsa live immediata sulle colline con tracciamento quota |
| **7. Sfide Speciali** | 6 Scenari: vento, gravità variabile, tempesta lunare, motore debole | 4 Tracciati: Campagna, Deserto, Montagna ripida, Luna a bassa gravità |
| **8. Archivio Preferiti** | Gestione cataloghi `runs/successi/` (≥80%) e preferiti con note | Gestione cataloghi `runs/successi/` (≥80%) e preferiti con note |
| **9. Report da Timestamp** | Generatore istantaneo schede PDF/HTML/MD e confronto run | Generatore istantaneo schede PDF/HTML/MD con telemetria avanzata |
| **10. Esportazione & Download** | Download archivio ZIP completo esperimenti per Colab | Download archivio ZIP completo esperimenti per Colab |

---

## 🧠 L'Obiettivo Didattico: La Funzione di Reward

Ad ogni step ($0{,}02\text{ secondi}$ di simulazione fisica Box2D), l'agente riceve il vettore di stato normalizzato (`obs`), l'azione eseguita (`action`) e i flag booleani di esito terminale.

### 1. Reward Shaping per Lunar Lander
```python
def custom_reward(obs, action, is_success, is_crashed, is_timeout):
    x, y, vx, vy, angle, vangle, left_leg, right_leg = obs
    reward = 0.0

    # =========================================================================
    # FORMULA DI REWARD PER LUNAR LANDER
    # =========================================================================
    # - Posizione orizzontale: incentivare l'avvicinamento al centro (x = 0.0)
    # - Discesa controllata: frenare la velocità verticale negativa quando vicini al suolo
    # - Assetto angolare: penalizzare inclinazioni eccessive (angle != 0)
    # - Risparmio propellente: penalizzare l'accensione prolungata dei motori
    # - Condizioni terminali:
    if is_success:
        reward += 100.0
    if is_crashed:
        reward -= 100.0

    return float(reward)
```

### 2. Reward Shaping per Hill Climb Racing
```python
def custom_reward(obs, action, is_success, is_crashed, is_timeout):
    dist_x, quota_y, vx, vy, angle, ang_vel, rear_on, front_on = obs
    reward = 0.0

    # =========================================================================
    # FORMULA DI REWARD PER HILL CLIMB RACING
    # =========================================================================
    # - Avanzamento: premiare la componente di velocità in avanti (vx > 0)
    # - Stabilità: penalizzare angoli di rollio accentuati per evitare il ribaltamento
    # - Trazione: incentivare il contatto costante con il terreno (rear_on / front_on)
    # - Condizioni terminali:
    if is_success:
        reward += 100.0  # Traguardo degli 800m superato con successo
    if is_crashed:
        reward -= 100.0  # Impatto della testa del pilota (cappottamento)

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
