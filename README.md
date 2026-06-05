# Ampelphasen-Optimierung mit Reinforcement Learning und SUMO

Studienprojekt im 2. Semester Data Science (DHBW Lörrach). Die Leitfrage: Kann ein
lernender Agent eine reale Ampel effizienter schalten als die feste Standardschaltung?

Ein Reinforcement-Learning-Agent (DQN als Hauptverfahren, PPO zum Vergleich) steuert die
Lichtsignalanlage der Kreuzung Basler Straße / Obere Riehenstraße in Lörrach-Stetten und wird
in der Verkehrssimulation SUMO (echter OSM-Import, `loerrach_osm.net.xml`, TLS-ID `1628110071`)
gegen die statische Festzeitschaltung evaluiert. Kernergebnis: Auf dem realen OSM-Netz steigert
der DQN-Agent den Durchsatz um +26,2 % (p = 2,6·10⁻⁵, r = +0,63; alle 7 KPIs signifikant
zugunsten DQN).

## Installation

```bash
git clone https://github.com/fabianegi/sumo-rl-loerrach-public.git
cd sumo-rl-loerrach-public

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run.py check                  # Umgebungs-Check (Exit 0 = bereit)
python run.py demo                   # Ergebnis in Sekunden: Tabelle und Abbildung
```

SUMO wird über `requirements.txt` als `eclipse-sumo` mitinstalliert; ein separater Download oder
das Setzen von `SUMO_HOME` ist nicht nötig. Das Setup läuft identisch unter Windows, macOS und Linux.

`python run.py demo` braucht weder SUMO noch ein trainiertes Modell: Es spielt die versionierte
Ergebnis-Belegkette (`results/csv/`) ab und erzeugt die Vorher/Nachher-Tabelle (Festzeit gegen DQN)
sowie `results/demo_comparison.png`. Eigenes Training und Live-Evaluation laufen über
`python run.py train` bzw. `evaluate` (dann mit SUMO).

## Voraussetzungen

- Python 3.11 (sumo-rl 1.4.5, `gymnasium==0.29.1`, `numpy<2.0`; Python 3.13+ wird nicht unterstützt).
- Alle übrigen Abhängigkeiten inklusive SUMO kommen aus `requirements.txt`.

## Bedienung

Ein betriebssystemunabhängiger Einstiegspunkt: `python run.py <command>`. `python run.py list`
zeigt die gültigen Werte für `--intersection`, `--scenario`, `--algo` und `--reward`.

| Befehl | Funktion | Laufzeit | Ausgabe |
|---|---|---|---|
| `check` | Umgebungs-Check: Python-Version, Imports, SUMO, Belegdaten | < 5 s | Pass/Fail-Liste |
| `list` | Gültige Kreuzungen, Szenarien, Algorithmen, Rewards | < 1 s | Auflistung |
| `demo` | Auditierte OSM-Ergebnisse abspielen (ohne SUMO/Modell) | ~5 s | KPI-Tabelle, `results/demo_comparison.png` |
| `compare` | Vergleichsabbildungen neu erzeugen (OSM und synthetisch) | ~10 s | `results/plots/compare_*.png` |
| `train` | Agent trainieren (mit SUMO) | Minuten bis Stunden | Modell unter `models/checkpoints/` |
| `evaluate` | Trainiertes Modell live evaluieren (mit SUMO) | 1 bis 5 min | KPI-CSV unter `results/csv/` |

```bash
# Eigenes Training (mit SUMO). Volle Läufe (1 bis 3 Mio. Steps) gehören auf einen
# Trainingsrechner; für einen schnellen Funktionstest kleine --timesteps wählen:
python run.py train --algo dqn --intersection osm --scenario medium --timesteps 5000 --no-gui
python run.py evaluate --intersection osm --scenario medium --n-episodes 5
```

Unter Unix spiegelt ein optionales `Makefile` diese Befehle (`make check`, `make demo`, ...).

## Methodik in Kürze

- Markov Decision Process über `sumo-rl` (Single-Agent, eine Kreuzung):
  - Zustand: Verkehrsdichte und Warteschlangen je einfahrender Spur plus aktuelle Phase.
  - Aktion: diskrete Phasenauswahl aus dem Signalplan (`Discrete(3)` beim OSM-Netz, `Discrete(4)`
    beim synthetischen 4-Arm-Netz); Entscheidung alle 5 s, `min_green = 10 s`, `yellow_time = 3 s`.
  - Reward: `diff-waiting-time` (Standard) oder `pressure`, wählbar über `--reward`.
- DQN ist das Hauptverfahren, PPO der Vergleich (beide über stable-baselines3, gleiche Env- und
  Evaluationspipeline).
- Netz und Nachfrage: echter OSM-Import der Kreuzung; die Nachfrage (`randomTrips`) ist auf die
  BASt-Spitzenstunde kalibriert (rund 680 Kfz/h, Szenario medium).
- Baseline: die aus OSM importierte Festzeitschaltung (ungetuntes 90-s-Standardprogramm),
  verglichen über 30 Seeds (Mann-Whitney-U, rank-biserial r).

## Ergebnisse

OSM-Netz (medium), DQN mit 3 Mio. Steps, Festzeit gegen DQN (je N = 30), reproduzierbar über
`python run.py demo`:

| KPI | Typ | Festzeit | DQN | Δ rel. | p-Wert | r |
|---|---|---|---|---|---|---|
| Ø Wartezeit | per-step | 289 | 93,9 | **-67,5 %** | 7,0·10⁻⁷ | -0,75 |
| Max. Rückstau [Fz] | per-step | 141 | 58,0 | **-59,0 %** | 1,7·10⁻⁵ | -0,64 |
| Ø Geschwindigkeit [m/s] | per-step | 4,26 | 6,57 | **+54,1 %** | 2,6·10⁻⁷ | +0,78 |
| **Durchsatz [Fz/Episode]** | tripinfo | 473 | 597 | **+26,2 %** | 2,6·10⁻⁵ | +0,63 |
| Ø Wartezeit/Fz [s] | tripinfo | 5,73 | 2,54 | **-55,6 %** | 4,1·10⁻⁷ | -0,76 |
| Ø Reisezeit/Fz [s] | tripinfo | 67,7 | 65,6 | -3,1 % | 2,1·10⁻³ | -0,46 |
| Ø Zeitverlust/Fz [s] | tripinfo | 16,9 | 13,8 | **-18,3 %** | 5,5·10⁻⁸ | -0,82 |

Leitindikator ist der Durchsatz (tripinfo, je abgeschlossenem Fahrzeug). Quasi-Gridlock tritt in
der Baseline bei 16 von 30 Seeds auf, beim DQN nur bei 7 von 30. Das synthetische 4-Arm-Netz dient
als Methodenbeweis (DQN und PPO, 1 Mio. Steps, -88 % Per-Step-Wartezeit, p = 0,0002; der
Durchsatz-Effekt bleibt mit +0,3 % trivial, da das idealisierte Netz unterausgelastet ist).

Hinweis zur Validität: Die Festzeit-Baseline ist das ungetunte OSM-Standardprogramm (kein
hand-optimierter Webster-Plan); evaluiert wurde nur das medium-Szenario; die Abbiegeströme sind
zufällig (`randomTrips`); die tripinfo-Mittel unterliegen einer Selektionsverzerrung (nur
abgeschlossene Fahrzeuge), weshalb der Durchsatz als Leitindikator dient. Vollständige Diskussion:
[`docs/documentation/08_discussion.md`](docs/documentation/08_discussion.md).

## Projektstruktur

```
.
├── run.py                  # Einstiegspunkt: check/list/demo/compare/train/evaluate
├── Makefile                # Optionale Unix-Variante (spiegelt run.py)
├── requirements.txt        # Gepinnte Abhängigkeiten inkl. eclipse-sumo (SUMO via pip)
├── src/
│   ├── config/             # ProjectConfig: Hyperparameter, Pfade, Konstanten
│   ├── environment/        # sumo-rl Env-Factory (make_env), eigene Rewards
│   ├── training/           # DQN-/PPO-Training, SB3-Callbacks
│   ├── evaluation/         # Festzeit-Baseline, KPI-Berechnung, Mann-Whitney-U
│   └── utils/              # Seeding, Plotting, Datenaufbereitung
├── data/                   # Rohdaten (BASt, OSM), Nachfrageprofile, SUMO-Netze
├── scripts/                # Setup-, Daten-, Analyse-, Plot- und Audit-Skripte
├── tests/                  # Unit-Tests (pytest)
├── docs/                   # Portfolio (documentation/), Abbildungen (figures/)
├── METHODOLOGY.md          # DASC-PM-Methodik
└── results/                # Berichte (*.md) und versionierte Audit-CSVs (results/csv/)
```

## Reproduzierbarkeit

- Versionen gepinnt in `requirements.txt` (u. a. `sumo-rl==1.4.5`, `stable-baselines3==2.3.2`,
  `gymnasium==0.29.1`, `numpy<2.0`, `eclipse-sumo==1.26.0`).
- Deterministisches Seeding (`src/utils/seeding.py`), Evaluation über 30 feste Seeds (0 bis 29).
- Das Training ist simulationsgebunden (SUMO via TraCI), nicht GPU-limitiert. Referenzlauf
  (3 Mio. Steps): AMD Ryzen 5 3600, Linux, rund 5,2 h.
- Die ausgelieferten Netze wurden mit Eclipse SUMO 1.18.0 erzeugt; per pip kommt die aktuelle
  1.26.0. Live-Neuläufe können daher minimal abweichen; die dokumentierten Zahlen sind über die
  CSV-Belegkette (`run.py demo`) unabhängig davon reproduzierbar.
- Trainierte `.zip`-Modelle liegen nicht im Repo (`models/checkpoints/*.zip` ist in `.gitignore`).

## Troubleshooting

- `SUMO not found`: `pip install -r requirements.txt` installiert `eclipse-sumo`; `run.py` setzt
  `SUMO_HOME` automatisch. Prüfen mit `python -c "import sumo; print(sumo.SUMO_HOME)"`. `demo`,
  `compare`, `check` und `list` laufen auch ohne SUMO.
- Schnelleres Training: `export LIBSUMO_AS_TRACI=1` (blockiert GUI und parallele Sims; auf Apple
  Silicon kann libsumo abstürzen, dann die Variable nicht setzen).
- macOS: pip-SUMO-Wheels brauchen unter Umständen einmalig `brew install xerces-c fox proj gdal`.
- `gymnasium`-Versionsfehler: es wird genau `gymnasium==0.29.1` benötigt (sumo-rl 1.4.5 ist nicht
  kompatibel mit 1.0 oder neuer).

## Datenquellen

- OpenStreetMap: Straßennetz und Geometrie der Kreuzung (Overpass-Dump in `data/raw/osm/`),
  importiert über `netconvert --osm-files`.
- BASt-Dauerzählstellen: Stundendaten B317 bei Lörrach zur Kalibrierung der Nachfrage,
  bereitgestellt von der Bundesanstalt für Straßenwesen (<https://www.bast.de>), in der Regel unter
  der Datenlizenz Deutschland Namensnennung 2.0.
- HBS-Literaturwerte: typische DTV für städtische Straßen.

## Lizenz

PolyForm Noncommercial License 1.0.0, siehe [`LICENSE`](LICENSE). Nutzung für private, schulische
und sonstige nicht-kommerzielle Zwecke ist erlaubt. Für eine kommerzielle Nutzung bitte den Autor
über <https://github.com/fabianegi> kontaktieren.
