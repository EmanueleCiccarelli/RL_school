"""
Script di generazione pulito e deterministico per 'laboratorio_hill_climb.ipynb'.
"""

import json
from pathlib import Path


def generate_notebook() -> None:
    cells = []

    def add_md(text: str) -> None:
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().split("\n")],
        })

    def add_code(text: str) -> None:
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in text.strip().split("\n")],
        })

    # CELL 0
    add_md(
        """# 🏎️ Laboratorio di Reinforcement Learning: Hill Climb Racing
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EmanueleCiccarelli/RL_school/blob/main/laboratorio_hill_climb.ipynb)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-Box2D-green.svg)](https://gymnasium.farama.org/)
[![Stable Baselines3](https://img.shields.io/badge/Algorithm-PPO-orange.svg)](https://stable-baselines3.readthedocs.io/)

Benvenuto nel laboratorio didattico di **Reinforcement Learning** dedicato a **Hill Climb Racing**!
In questo modulo guiderai un veicolo fuoristrada attraverso colline procedurali generate con **Perlin Noise**, basato sul motore fisico 2D Box2D e rendering Pygame.

Il tuo obiettivo formativo è padroneggiare il **Reward Shaping**: progetterai la funzione matematica di ricompensa per insegnare alla rete neurale (algoritmo **PPO**) a dosare acceleratore e freno, scalare ripide salite fino al traguardo (>= 800 metri) ed evitare il ribaltamento all'indietro."""
    )

    # CELL 1
    add_md(
        """## 1. 🛠️ Installazione e Preparazione dell'Ambiente
La cella seguente verifica se l'esecuzione avviene su **Google Colab** o su un computer locale. Se rileva Colab, clona automaticamente la repository ed installa tutte le dipendenze necessarie."""
    )

    # CELL 2
    add_code(
        """# Setup automatico dell'ambiente per Google Colab e ambiente locale
import sys
import os

if 'google.colab' in sys.modules:
    print("Ambiente Google Colab rilevato. Configurazione in corso...")
    !apt-get update -qq && apt-get install -y -qq swig ffmpeg > /dev/null 2>&1
    if not os.path.exists("RL_school"):
        !git clone https://github.com/EmanueleCiccarelli/RL_school.git
    %cd RL_school
    !pip install -q "gymnasium[box2d]" stable-baselines3 pygame imageio[ffmpeg] fpdf2 matplotlib
    sys.path.insert(0, os.getcwd())

import play2learn_hillclimb
print(f"✅ Modulo play2learn_hillclimb v{play2learn_hillclimb.__version__} pronto all'uso!")"""
    )

    # CELL 3
    add_md(
        """## 2. 🔍 Esplorazione del Veicolo e dei Sensori di Bordo
Prima di scrivere il codice, visualizziamo la mappa del veicolo, lo spazio delle azioni e i dati fisici restituiti dai sensori di bordo a ogni passo temporale."""
    )

    # CELL 4
    add_code(
        """# Visualizza la mappa grafica con i sensori fisici e le azioni del veicolo
play2learn_hillclimb.mostra_mappa_ambiente()"""
    )

    # CELL 5
    add_md(
        """## 3. 🎮 Test Rapido: Come si Comporta un Pilota Casuale?
Osserviamo cosa accade se eseguiamo alcuni passi con azioni casuali. Noterai che premere a fondo il gas senza modulazione fa impennare la vettura sui dossi fino al ribaltamento."""
    )

    # CELL 6
    add_code(
        """import gymnasium as gym
import hill_racing_env
from play2learn_hillclimb.environment import HillClimbRewardWrapper

env = gym.make(play2learn_hillclimb.ENV_ID, render_mode=None)
wrapped_env = HillClimbRewardWrapper(env)
obs, info = wrapped_env.reset(seed=42)

print("Stato iniziale (vettore 8D):")
print(f" • Distanza x:     {obs[0]:.2f} m")
print(f" • Quota y:        {obs[1]:.2f} m")
print(f" • Velocità vx:    {obs[2]:.2f} m/s")
print(f" • Assetto angolo: {obs[4]:.2f} rad")
print(f" • Contatto ruote: Posteriore={obs[6]}, Anteriore={obs[7]}")

# Esegui 10 passi di prova
for step in range(10):
    obs, r, term, trunc, info = wrapped_env.step(1)
print(f"\\nDopo 10 passi di gas: distanza={obs[0]:.2f}m, inclinazione={obs[4]:.2f}rad")
wrapped_env.close()"""
    )

    # CELL 7
    add_md(
        """## 4. 🧠 Progetta la tua Funzione di Reward (Reward Shaping)
Ecco il cuore del laboratorio! Nella funzione `custom_reward` stabilisci i premi e le penalità matematiche che guideranno l'apprendimento dell'agente.

### 💡 Consigli di Guida per il Reward Shaping:
1. **Premia la velocità orizzontale positiva (`vx > 0`)**: il veicolo deve avanzare con decisione.
2. **Premia la distanza progressiva (`dist_x`)**: incentiva il superamento continuo delle creste.
3. **Penalizza l'inclinazione eccessiva (`abs(angle) > 0.4`)**: se il muso punta troppo verso l'alto, la vettura rischia di ribaltarsi all'indietro.
4. **Premia il traguardo (`is_success`)**: assegna un bonus cospicuo (+100 o +200) quando la vettura supera gli 800 metri.
5. **Penalizza lo schianto (`is_crashed`)**: scoraggia severamente il contatto della testa con il terreno."""
    )

    # CELL 8
    add_code(
        """def custom_reward(obs, action, is_success, is_crashed, is_timeout):
    \"\"\"
    Progetta qui la tua funzione matematica di Reward!

    SENSORI FISICI (obs):
      dist_x, quota_y     -> Distanza orizzontale (m, target >= 800) e quota
      vx, vy              -> Velocità orizzontale e verticale (m/s)
      angle, ang_vel      -> Inclinazione telaio (rad) e velocità angolare
      rear_on, front_on   -> Contatto ruote col suolo (1.0 = appoggiata, 0.0 = aria)

    AZIONE SCELTA (action):
      0 = Folle / Spento | 1 = Acceleratore (Gas) | 2 = Freno / Retromarcia
    \"\"\"
    dist_x = obs[0]
    quota_y = obs[1]
    vx = obs[2]
    vy = obs[3]
    angle = obs[4]
    ang_vel = obs[5]
    rear_on = obs[6]
    front_on = obs[7]

    reward = 0.0

    # 1. Trazione e avanzamento in avanti
    if vx > 0:
        reward += vx * 1.5
    else:
        reward -= 1.0

    # 2. Premio per la distanza percorsa
    reward += dist_x * 0.05

    # 3. Controllo dell'assetto per prevenire impennate e ribaltamenti
    if abs(angle) > 0.4:
        reward -= abs(angle) * 2.0

    # 4. Esiti terminali
    if is_success:
        reward += 100.0
    elif is_crashed:
        reward -= 50.0

    return float(reward)"""
    )

    # CELL 9
    add_md(
        """## 5. 🚀 Avvia l'Addestramento PPO
Avviamo l'addestramento. La barra di avanzamento mostrerà in tempo reale l'evoluzione della reward, la percentuale di successo e la distanza massima raggiunta.
Al termine, il modello verrà testato su 20 scenari di collaudo e verranno registrati i video delle migliori scalate."""
    )

    # CELL 10
    add_code(
        """# Avvia l'esperimento completo
# Suggerimento: 100.000 step per test veloci, 300.000 step per modelli completi
risultati = play2learn_hillclimb.run_lab(
    reward_fn=custom_reward,
    mission="campagna_standard",
    total_timesteps=300_000,
    crea_gif_evoluzione=True,
    seed=42,
)"""
    )

    # CELL 11
    add_md(
        """## 6. ⛰️ Sfida Estrema: Collaudo su Altre Missioni Collinari
Una volta addestrato il veicolo sulla campagna standard, verifica se il tuo pilota è in grado di affrontare pendenze più estreme o dune sabbiose.
Puoi scegliere tra:
* `"deserto_dune"`: dune irregolari con sabbia scivolosa.
* `"montagna_estrema"`: creste ripide e salite molto accentuate.
* `"colline_ripide"`: pendenze al limite della fisica del veicolo.
* `"autostrada_veloce"`: tracciato pianeggiante per test di velocità."""
    )

    # CELL 12
    add_code(
        """# Collauda la tua funzione di reward sulla missione "deserto_dune"
risultati_sfida = play2learn_hillclimb.run_lab(
    reward_fn=custom_reward,
    mission="deserto_dune",
    total_timesteps=300_000,
    crea_gif_evoluzione=True,
    seed=42,
)"""
    )

    # CELL 13
    add_md(
        """## 7. 📊 Reportistica, Timelapse e Preferiti Personali
Analizziamo ora in dettaglio le prestazioni dell'agente: timelapse dell'apprendimento, scheda tecnica e catalogo dei preferiti."""
    )

    # CELL 14
    add_code(
        """# 1. Visualizza il timelapse GIF dell'evoluzione dell'apprendimento nel tempo
risultati.mostra_evoluzione()"""
    )

    # CELL 15
    add_code(
        """# 2. Salva la run completata nei tuoi preferiti personali
risultati.salva_preferito(
    nome_personalizzato="Assetto_Sportivo_1",
    note="Buona modulazione del gas e stabilità sulle salite ripide."
)"""
    )

    # CELL 16
    add_code(
        """# 3. Visualizza la galleria dei tuoi preferiti personali
play2learn_hillclimb.mostra_preferiti()"""
    )

    # CELL 17
    add_code(
        """# 4. Genera la Scheda Tecnica dell'esperimento (PDF vettoriale, HTML5 e Markdown)
risultati.genera_report(
    titolo="Scalata Collinare Campagna",
    autore="Studente Play2Learn",
    note="Esperimento con penalità sull'angolo di beccheggio e incentivo velocità."
)"""
    )

    # CELL 18
    add_code(
        """# 5. Metti a confronto due run per analizzare l'impatto dei parametri
play2learn_hillclimb.confronta_run([risultati, risultati_sfida])"""
    )

    # CELL 19
    add_md(
        """## 8. 🕒 Cronologia e Recupero Esperimenti Passati
Puoi consultare la cronologia delle sessioni passate e rigenerare la scheda tecnica a partire dal timestamp di qualsiasi run."""
    )

    # CELL 20
    add_code(
        """# Visualizza la cronologia di tutti gli esperimenti registrati
play2learn_hillclimb.mostra_cronologia()"""
    )

    # CELL 21
    add_code(
        """# @title 📊 Generatore di Report da Timestamp { run: "auto" }
# Inserisci il timestamp (o parte del nome della cartella) della run da recuperare
timestamp_run = "run_hillclimb"  # @param {type:"string"}

try:
    report_paths = play2learn_hillclimb.genera_report_da_timestamp(timestamp_run)
    print("Scheda tecnica rigenerata con successo!")
except Exception as e:
    print(f"Avviso: {e}")"""
    )

    # CELL 22
    add_md(
        """## 9. 📦 Scarica l'Archivio degli Esperimenti (ZIP per Google Colab)
Se stai lavorando su Google Colab, la cella seguente comprime l'intera cartella `runs/` (inclusi modelli, video, GIF e report PDF) e avvia automaticamente il download sul tuo computer."""
    )

    # CELL 23
    add_code(
        """# Comprime l'intera cartella delle run e avvia il download dello ZIP
play2learn_hillclimb.comprimi_archivio()"""
    )

    nb_dict = {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    out_file = Path("laboratorio_hill_climb.ipynb")
    out_file.write_text(json.dumps(nb_dict, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ Notebook {out_file} generato con successo ({len(cells)} celle).")


if __name__ == "__main__":
    generate_notebook()
