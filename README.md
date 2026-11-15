# 🚀 RL_school - Laboratorio di Reinforcement Learning: Lunar Lander

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EmanueleCiccarelli/RL_school/blob/main/laboratorio_lunar_lander.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-Box2D-green.svg)](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
[![Stable Baselines3](https://img.shields.io/badge/Algorithm-PPO-orange.svg)](https://stable-baselines3.readthedocs.io/)

Laboratorio didattico di **Reinforcement Learning** per studenti di scuola secondaria di secondo grado (quinta superiore). L'obiettivo formativo è comprendere il funzionamento dell'apprendimento per rinforzo attraverso il **Reward Shaping**: gli studenti progettano la logica matematica della funzione di ricompensa per insegnare a un modulo spaziale ad allunare in modo autonomo, sicuro ed efficiente.

Tutto il codice di orchestrazione, calcolo numerico, simulazione fisica e rendering è incapsulato nel motore modulare `play2learn_lunarlander`, consentendo agli studenti di concentrarsi esclusivamente sulla modellazione logico-matematica e sul debug del comportamento dell'agente.

---

## 📂 Struttura del Progetto

```text
RL_school/
│
├── laboratorio_lunar_lander.ipynb          # 🌐 Notebook ottimizzato per Google Colab
├── README.md                               # 📖 Documentazione del laboratorio
├── .gitignore                              # 🙈 Esclusioni Git
│
├── play2learn_lunarlander/                 # ⚙️ Pacchetto orchestratore modulare
│   ├── laboratorio_lunar_lander_locale.ipynb # 💻 Notebook ottimizzato per esecuzione in Locale
│   ├── __init__.py                         # API pubblica del laboratorio
│   ├── config.py                           # Costanti fisiche, 6 missioni e iperparametri PPO
│   ├── environment.py                      # Wrapper reward, telemetria e parallelizzazione
│   ├── agent.py                            # PPO agent e RewardTrackerCallback
│   ├── diagnostics.py                      # "Flight Coach": diagnostica automatica anomalie
│   ├── visualizer.py                       # Mappa fisica, dashboard 2x2 e player video HTML5
│   ├── evaluator.py                        # Valutazione mista (3 successi + 3 fallimenti)
│   ├── storage.py                          # Gestione salvataggi runs, metadati JSON e codice
│   ├── favorites.py                        # Gestione archivio modelli preferiti
│   ├── reporting.py                        # Generatore schede tecniche (PDF vettoriale, HTML, MD)
│   ├── lab.py                              # Orchestratore principale (run_lab)
│   └── reset.py                            # 🧹 Script di reset per pulire l'ambiente
│
└── runs/                                   # 📁 Cartella di archiviazione automatica delle run
    └── favorites/                          # Archivio delle run migliori salvate
        └── index.json                      # Indice persistente dei modelli preferiti
```

---

## ⚡ Come Avviare il Laboratorio

Il laboratorio è disponibile in **due modalità sincronizzate**:

### Modalità 1: Google Colab (Consigliata per la classe)
1. Clicca sul badge in cima al README:  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EmanueleCiccarelli/RL_school/blob/main/laboratorio_lunar_lander.ipynb)
2. Il notebook scaricherà automaticamente il pacchetto dalla repository GitHub ed installerà le librerie necessarie (`gymnasium[box2d]`, `stable-baselines3`, `imageio`, ecc.).
3. Gli studenti possono subito concentrarsi sulla modifica di `custom_reward` ed eseguire l'addestramento direttamente nel browser.

### Modalità 2: Ambiente Locale (per docenti o sviluppo offline)

1. **Clona la repository sul tuo computer:**
   ```bash
   git clone https://github.com/EmanueleCiccarelli/RL_school.git
   cd RL_school
   ```

2. **Crea e attiva un ambiente virtuale (consigliato):**
   * **Su Windows (PowerShell):**
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   * **Su Linux / macOS:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Installa tutte le dipendenze tramite `requirements.txt`:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
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

5. **Apri il notebook dedicato all'ambiente locale:**  
   Naviga ed apri il file:  
   📂 `play2learn_lunarlander/laboratorio_lunar_lander_locale.ipynb`

---

## 🧠 L'Obiettivo Didattico: La Funzione di Reward

Ad ogni step ($0{,}02\text{ secondi}$), l'agente riceve gli 8 sensori fisici di bordo (`obs`), il comando motore (`action`) e gli eventi di fine volo:

```python
def custom_reward(obs, action, is_success, is_crashed, is_timeout):
    x, y, vx, vy, angle, vangle, left_leg, right_leg = obs
    reward = 0.0

    # =========================================================================
    # PROGETTA QUI LA TUA FORMULA DI REWARD SHAPING
    # =========================================================================

    # Esempi di termini da considerare:
    # - Posizione orizzontale: avvicinarsi al centro (x = 0.0)
    # - Velocità di discesa: frenare quando vy è troppo negativo
    # - Assetto angolare: mantenere il modulo verticale (angle = 0.0)
    # - Efficienza: premiare lo spegnimento dei motori a terra (action == 0)

    # Eventi terminali
    if is_success:
        reward += 100.0
    if is_crashed:
        reward -= 100.0

    return float(reward)
```

### Le 2 Trappole Matematiche Classiche
Gli studenti imparano ad analizzare il fenomeno del *Reward Hacking*:
1. 💥 **La Trappola Kamikaze (Suicidio Rapido):** Se le penalità per step in volo sono troppo elevate, all'agente conviene schiantarsi subito per interrompere l'episodio ed evitare di accumulare penalità continue.
2. 🪂 **La Trappola dell'Hovering Infinito:** Se l'agente riceve premi positivi in volo o non viene penalizzato il timeout, sceglie di galleggiare a mezz'aria senza mai rischiare l'atterraggio.

---

## 📊 Telemetria, Diagnostica e Reportistica

Al termine di ogni sessione di addestramento:
* **Dashboard 2x2:** Mostra curva di ricompensa, tassi percentuali di successo/schianto/timeout nel tempo, durata del volo e consumo di carburante.
* **Valutazione Mista:** Vengono registrati e mostrati sia **esempi di allunaggi riusciti** (🟢 fino a 3 video) sia **esempi di fallimento** (🔴 fino a 3 video) per comprendere esattamente cosa correggere.
* **Diagnostica "Flight Coach":** Fornisce un feedback in linguaggio naturale spiegando la dinamica fisica del volo.
* **Scheda Tecnica di Esportazione:** Se l'allunaggio raggiunge $\ge 80\%$ di successo, viene generata una scheda riassuntiva in triplice formato:
  * **PDF Vettoriale** (`scheda_esperimento.pdf`) pronto per la stampa
  * **HTML Autonomo** (`report_esperimento.html`) visualizzabile nel browser
  * **Markdown** (`report_esperimento.md`) modificabile per aggiungere annotazioni personali
* **Confronto tra Esperimenti:** La funzione `play2learn_lunarlander.confronta_run([run_1, run_2])` permette di visualizzare i progressi tra diverse formulazioni matematiche della reward.

---

## 🧹 Reset dell'Ambiente

Per ripulire l'ambiente da tutte le run precedenti, azzerare gli output di entrambi i notebook e riportare il laboratorio allo stato iniziale:

```bash
python play2learn_lunarlander/reset.py
```
*(usa `--force` o `-y` per eseguire il reset immediato).*

---

## 📜 Licenza e Crediti

Progetto didattico sviluppato per scuole secondarie di secondo grado.  
Repository: [https://github.com/EmanueleCiccarelli/RL_school](https://github.com/EmanueleCiccarelli/RL_school)
