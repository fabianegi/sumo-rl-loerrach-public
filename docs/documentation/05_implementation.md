# 5. Implementierung

## 5.1 Technologie-Stack

Die Wahl der Technologien folgt dem Prinzip minimaler Komplexitaet bei maximaler Reproduzierbarkeit. Tabelle 2 listet die Kernabhaengigkeiten mit den gepinnten Versionen.

| Komponente | Bibliothek | Version | Rolle |
|---|---|---|---|
| Sprache | Python | 3.11 | Laufzeitumgebung (ADR #3) |
| RL-Umgebung | gymnasium | 0.29.1 (pinned) | Gym-API fuer sumo-rl (ADR #2) |
| SUMO-Wrapper | sumo-rl | 1.4.5 | Gymnasium-Env um SUMO-Simulator |
| RL-Algorithmen | stable-baselines3 | 2.3.2 | DQN, PPO Implementierungen |
| Deep Learning | PyTorch | >= 2.1 | Neuronale Netze, CUDA-Support |
| Simulator | SUMO | >= 1.20 | Mikroskopische Verkehrssimulation |
| Datenanalyse | pandas | >= 2.1 | Ergebnis-Aggregation |
| Visualisierung | matplotlib, seaborn | >= 3.8 / >= 0.13 | Plots und Diagramme |
| Statistik | scipy | >= 1.11 | Mann-Whitney U, Effektstaerken |
| Monitoring | TensorBoard | >= 2.15 | Training-Monitoring (ADR #9) |
| Linting | ruff | >= 0.3 | Formatter und Linter |
| Tests | pytest | >= 8.0 | Unit- und Integrationstests |

*Tabelle 2: Technologie-Stack mit gepinnten Versionen.*

Kritische Versionsabhaengigkeiten: `gymnasium==0.29.1` ist fest gepinnt, da sumo-rl 1.4.5 mit gymnasium >= 1.0 bricht (API-Breaking-Changes in der v1.0-Release). Ebenso wird Python 3.11 statt 3.13 verwendet, da die Kompatibilitaetsmatrix SUMO + PyTorch + sumo-rl unter 3.13 nicht zuverlaessig funktioniert (ADR #2, #3). NumPy wird auf < 2.0 beschraenkt, da Version 2.x Breaking Changes in wissenschaftlichen Paketen einfuehrt.

## 5.2 Projektstruktur

Die Codebasis ist als Python-Paket mit klarer Schichtentrennung organisiert:

```
src/
  config/
    settings.py            # Frozen-Dataclass-Hierarchie, CONFIG Singleton
  environment/
    env_factory.py         # Erzeugt sumo-rl Gymnasium-Env (make_env, make_eval_env)
    rewards.py             # Custom Rewards: pressure, diff-waiting-time
  training/
    train_dqn.py           # Standalone DQN-Script mit CLI (argparse)
    train_ppo.py           # Standalone PPO-Script (gleiches Muster)
    callbacks.py           # SB3 Callbacks: Metriken, Checkpoints
  evaluation/
    evaluate.py            # Agent-Evaluation ueber N Seeds, berechnet KPIs
    baseline.py            # Festzeitsteuerung (90s Zyklus, Webster-proportional)
    metrics.py             # KPIResult Dataclass + compute_kpis
    statistical_tests.py   # Mann-Whitney U, Effektstaerken
  utils/
    seeding.py             # seed_everything() -- numpy, torch, random, CUDA
    plotting.py            # Konsistente Plot-Generierung
```

Die Architektur folgt dem Prinzip Separation of Concerns: Konfiguration (`config/`), Simulationsumgebung (`environment/`), Training (`training/`), Auswertung (`evaluation/`) und Hilfsfunktionen (`utils/`) sind strikt getrennt. Die Kommunikation zwischen den Schichten erfolgt ueber die zentrale Konfiguration und definierte Datenstrukturen (`KPIResult` Dataclass in `metrics.py`).

## 5.3 Zentrale Designentscheidungen

**Frozen-Dataclass-Konfiguration (settings.py):** Saemtliche Hyperparameter, Pfade und Simulationseinstellungen sind in einer hierarchischen Dataclass-Struktur gekapselt (`ProjectConfig` mit Unterklassen `SumoConfig`, `DQNConfig`, `PPOConfig`, `TrainingConfig`, `PathConfig`). Alle Dataclasses nutzen `frozen=True`, was versehentliche Mutation zur Laufzeit verhindert. Ein Singleton `CONFIG` wird beim Import instanziiert und projektweit genutzt. Keine Magic Numbers ausserhalb dieser Datei.

**sumo-rl als Wrapper (ADR #1):** Statt eines manuellen TraCI-Gym-Wrappers wird sumo-rl 1.4.5 als Abstraktionsschicht eingesetzt. Dies spart geschaetzt 3--4 Wochen Entwicklungszeit gegenueber einem eigenen Wrapper und liefert SOTA-nahe Defaults fuer Observation-Encoding, Action-Handling und Phase-Timing. Die Entscheidung ist zentral fuer die Machbarkeit als Solo-Projekt innerhalb von 13 Wochen.

**Standalone Training-Scripts (nicht Notebook):** Die Trainings-Scripts sind eigenstaendige Python-Dateien mit `argparse`-CLI, nicht Jupyter-Notebooks (ADR #4). Gruende: (1) Versionskontrolle von `.py`-Dateien ist zuverlaessig (im Gegensatz zu `.ipynb`-Zell-Outputs), (2) die Scripts laufen auf dem Windows-GPU-Rechner ohne Jupyter-Installation, (3) saubere Trennung von Code und Dokumentation (Markdown-Portfolio in `docs/documentation/`).

**gymnasium-Pinning (ADR #2):** Die strikte Versionsbindung `gymnasium==0.29.1` ist die wichtigste Abhaengigkeitsentscheidung. Die gymnasium-v1.0-API fuehrt inkompatible Aenderungen ein (neues `terminated`/`truncated`-Handling, geaendertes `reset()`-Interface), die sumo-rl 1.4.5 nicht unterstuetzt. Ein Upgrade wuerde einen Fork von sumo-rl erfordern -- jenseits des Projektscopes.

## 5.4 Cross-Platform-Entwicklung

Die Entwicklung erfolgt auf macOS (Apple Silicon) fuer schnelle Iteration und Debugging. GPU-intensives Training wird auf einem Windows-PC mit NVIDIA-GPU ausgefuehrt. Alle Pfade nutzen `pathlib.Path` relativ zum Projektroot; maschinenspezifische Konfiguration (z. B. `SUMO_HOME`) liegt in `.env`-Dateien, die nicht versioniert werden. Die Standalone-Training-Scripts ermoeglichen direkten Aufruf auf beiden Plattformen ohne Notebook-Abhaengigkeit.

Fuer die headless Trainingsausfuehrung wird LibSUMO (`LIBSUMO_AS_TRACI=1`) eingesetzt, was den Socket-Overhead gegenueber TraCI eliminiert und einen Speedup von ca. 8x liefert (ADR #10). Fuer Debugging mit der SUMO-GUI wird auf TraCI zurueckgeschaltet.
