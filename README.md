# Ampelphasen-Optimierung mit Reinforcement Learning und SUMO

> **Reproduzierbare Quelle:** <https://github.com/fabianegi/sumo-rl-loerrach-public>
> Skripte, Roh-CSV-Belegkette und Portfolio in einem Clone.

In diesem Studienprojekt (2. Semester Data Science, DHBW Lörrach) gehe ich der Frage nach, ob ein
lernender Agent eine reale Ampel effizienter schalten kann als die fest programmierte Standardschaltung.

Ein Reinforcement-Learning-Agent (DQN primär, PPO als Vergleich) steuert die Lichtsignalanlage
der Kreuzung **Basler Str. × Obere Riehenstr.** in Lörrach-Stetten adaptiv und wird in der
Verkehrssimulation **SUMO** (echter OSM-Import, `loerrach_osm.net.xml`, TLS-ID `1628110071`) gegen
die statische Festzeitschaltung evaluiert. **Kernergebnis:** auf dem realen OSM-Netz steigert der
DQN-Agent den **Durchsatz um +26,2 %** (p = 2,6·10⁻⁵, r = +0,63; alle 7 KPIs signifikant zugunsten DQN).

---

## Für Prüfer:innen - in ~2 Minuten zum Ergebnis

Vier Befehle von `git clone` bis zum sichtbaren Vorher/Nachher-Vergleich. SUMO wird per pip
mitinstalliert, eine separate SUMO-Installation ist nicht nötig. Läuft identisch auf Windows, macOS und Linux.

```bash
git clone https://github.com/fabianegi/sumo-rl-loerrach-public.git
cd sumo-rl-loerrach-public

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python run.py check                  # Umgebungs-Check (Exit 0 = bereit)
python run.py demo                   # Ergebnis in ~Sekunden: Tabelle + Figur
```

`python run.py demo` benötigt **weder SUMO noch ein trainiertes Modell**: Es spielt die im Repo
versionierte, auditierte Ergebnis-Belegkette (`results/csv/`) ab und erzeugt die Vorher/Nachher-Tabelle
(Festzeit vs. DQN) plus die Abbildung `results/demo_comparison.png`. Eigenes Training und Live-Evaluation
sind über `python run.py train` bzw. `evaluate` möglich (dann wird SUMO genutzt).

---

## Voraussetzungen

- **Python 3.11** (sumo-rl 1.4.5 + `gymnasium==0.29.1` + `numpy<2.0`; Python 3.13+ wird nicht unterstützt).
- **SUMO**: wird über `pip install -r requirements.txt` als `eclipse-sumo` automatisch mitinstalliert,
  ohne separaten Download und ohne manuelles Setzen von `SUMO_HOME`. `run.py` löst
  `SUMO_HOME` beim Start automatisch über `import sumo` auf.
- Virtuelle Umgebung aktivieren: Windows `\.venv\Scripts\activate` · macOS/Linux `source .venv/bin/activate`.

## Setup (ein Block)

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run.py check
```

## Bedienung - `python run.py <command>`

Ein einziger, betriebssystemunabhängiger Einstiegspunkt. `python run.py --list` zeigt alle gültigen
Werte für `--intersection`, `--scenario`, `--algo`, `--reward`.

| Befehl | Was er tut | Laufzeit | Erwartete Ausgabe |
|---|---|---|---|
| `check` | Umgebungs-Doktor: Python-Version, Imports, SUMO-Auflösung, Belegdaten. Exit 0/1. | < 5 s | Pass/Fail-Checkliste |
| `list` | Gültige Kreuzungen / Szenarien / Algorithmen / Rewards. | < 1 s | Auflistung |
| `demo` | Auditierte OSM-Ergebnisse abspielen (kein SUMO/Modell): Vorher/Nachher-Tabelle + Figur. | ~5 s | KPI-Tabelle + `results/demo_comparison.png` |
| `compare` | Publikationsfiguren neu erzeugen: OSM (DQN vs. Festzeit) **und** synthetisch (DQN vs. PPO vs. Festzeit). | ~10 s | `results/plots/compare_*.png` + Markdown |
| `train` | Agent trainieren (live SUMO). | Minuten-Stunden | Modell unter `models/checkpoints/*.zip` |
| `evaluate` | Trainiertes Modell live evaluieren. | ~1-5 min | KPI-CSV unter `results/csv/` |

**Beispiele:**

```bash
python run.py check                                   # Setup bestätigen
python run.py demo                                    # Headline-Ergebnis (Belegkette)
python run.py compare                                 # alle Vergleichsfiguren

# Eigenes Training (live SUMO). Volle Läufe (1-3 M Steps) gehören auf den Trainings-PC;
# für einen schnellen Funktionstest kleine --timesteps wählen:
python run.py train --algo dqn --intersection osm --scenario medium --timesteps 5000 --no-gui
python run.py train --algo ppo --intersection single --scenario medium --timesteps 5000

# Live-Evaluation des zuletzt trainierten Modells:
python run.py evaluate --intersection osm --scenario medium --n-episodes 5
```

Unter Unix spiegelt ein optionales `Makefile` diese Befehle (`make check`, `make demo`, ...);
`run.py` bleibt der maßgebliche, plattformübergreifende Weg.

## Was das Projekt macht (konzeptionell)

- **Markov Decision Process** über `sumo-rl` (Single-Agent, eine Kreuzung):
  - **Zustand** - Verkehrsdichte/Warteschlangen je einfahrender Spur plus die aktuelle Phase (Beobachtung von `sumo-rl`).
  - **Aktion** - diskrete Phasenauswahl: Der Agent wählt pro Step die nächste Grünphase aus dem Signalplan (`Discrete(N_Phasen)` - OSM-Headline-Netz = `Discrete(3)`, synthetisches 4-Arm-Netz = `Discrete(4)`). Entscheidung alle `delta_time = 5 s`, `min_green = 10 s`, `yellow_time = 3 s`; bei Phasenwechsel wird automatisch eine Gelbphase eingefügt.
  - **Reward** - `diff-waiting-time` (Standard: Reduktion der akkumulierten Wartezeit) oder `pressure`; auswählbar via `--reward`.
- **DQN** ist die primäre Methode; **PPO** dient als Vergleichsalgorithmus (beide via stable-baselines3, identische Env- und Evaluationspipeline).
- **Netz & Nachfrage** - Headline ist der echte OSM-Import der Kreuzung; die Nachfrage (`randomTrips`) ist magnitudenkalibriert auf die BASt-Spitzenstunde (≈ 680 Kfz/h, medium). Weitere Netze: synthetisches 4-Arm-Kreuz (Methodenbeweis), Korridor (3 Kreuzungen).
- **Baseline** - die aus OSM importierte **Festzeitschaltung** (un-getuntes 90-s-Default-Programm), gegen die der Agent über 30 Seeds verglichen wird (Mann-Whitney-U, rank-biserial r).

## Ergebnisse

OSM-Netz (medium), DQN 3 M Steps, Festzeit-Baseline (N = 30) vs. DQN (N = 30) - reproduzierbar via
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

Der **Durchsatz** (tripinfo, pro abgeschlossenem Fahrzeug) ist der unverzerrte Leitindikator.
Quasi-Gridlock tritt in der Baseline bei 16/30 Seeds auf, beim DQN nur bei 7/30. Das synthetische
4-Arm-Netz dient als **Methodenbeweis** (DQN + PPO, 1 M Steps, -88 % Per-Step-Wartezeit, p = 0,0002;
Durchsatz-Effekt trivial +0,3 %, da das idealisierte Netz unterausgelastet ist). Abbildungen:
`results/plots/compare_osm_*.png` (DQN vs. Festzeit) und `results/plots/compare_synth_*.png`
(DQN vs. PPO vs. Festzeit), Lernkurven unter `docs/figures/`.

> **Hinweis zur Validität (Auszug aus Kapitel 8):** Die Festzeit-Baseline ist das aus OSM importierte
> **un-getunte Default-Programm** (90 s, kein hand-optimierter Webster-Plan); evaluiert wurde **nur das
> medium-Lastszenario**; die **Abbiegeströme sind zufällig** (`randomTrips`, nicht erhoben); die
> tripinfo-Mittel unterliegen einer **Selektionsverzerrung** (nur abgeschlossene Fahrzeuge) - deshalb
> der **Durchsatz** als Leitindikator. Vollständige Diskussion:
> [`docs/documentation/08_discussion.md`](docs/documentation/08_discussion.md).

## Projektstruktur

```
.
├── run.py                  # Kanonischer Einstiegspunkt: check/list/demo/compare/train/evaluate
├── Makefile                # Optionale Unix-Bequemlichkeit (spiegelt run.py)
├── requirements.txt        # Gepinnte Abhängigkeiten inkl. eclipse-sumo (SUMO via pip)
├── src/
│   ├── config/             # ProjectConfig-Singleton: Hyperparameter, Pfade, Konstanten
│   ├── environment/        # sumo-rl Env-Factory (make_env), Custom Rewards
│   ├── training/           # DQN-/PPO-Training, SB3-Callbacks
│   ├── evaluation/         # Festzeit-Baseline, KPI-Berechnung, Mann-Whitney-U
│   └── utils/              # Seeding, Plotting, Datenaufbereitung
├── data/
│   ├── raw/                # Rohdaten (BASt-Zähldaten, OSM-Dump)
│   ├── processed/          # Aufbereitete Nachfrageprofile
│   └── sumo_config/        # SUMO-Netze (.net/.rou/.sumocfg) - synthetisch, real, OSM, Korridor
├── scripts/                # Setup-, Daten-, Analyse-, Plot- und Audit-Skripte
├── tests/                  # Unit-Tests (pytest)
├── docs/                   # Portfolio (documentation/), Abbildungen (figures/)
├── METHODOLOGY.md          # DASC-PM-Methodik
└── results/                # Berichte (*.md) + versionierte Audit-CSVs (results/csv/)
```

## Reproduzierbarkeit

- **Versionen** (gepinnt in `requirements.txt`): Python 3.11, `sumo-rl==1.4.5`,
  `stable-baselines3==2.3.2`, `gymnasium==0.29.1`, `numpy<2.0`, `eclipse-sumo==1.26.0`.
- **Seeds**: deterministisches Seeding (`src/utils/seeding.py`); Evaluation über 30 feste Seeds (0-29).
- **Hardware**: Das Training ist **simulationsgebunden** (SUMO-Mikrosimulation via TraCI), **nicht
  GPU-limitiert** - das kleine MLP-Policy-Netz lastet keine GPU aus, Engpass ist der SUMO-Schritt.
  Referenzlauf (3 Mio. Steps): AMD Ryzen 5 3600, Linux, **≈ 5,2 h** (Details:
  [`docs/documentation/06_training.md`](docs/documentation/06_training.md), Tab. 6/7). `device="auto"`
  in der Config wählt CPU/CUDA/MPS automatisch; CPU erzwingen siehe Troubleshooting.
- **SUMO-Version**: Die ausgelieferten Netze/Ergebnisse wurden mit **Eclipse SUMO 1.18.0** erzeugt; per
  pip installiert wird die aktuell verfügbare **1.26.0**. Alte `.net.xml` laden vorwärtskompatibel;
  Live-Neuläufe können daher minimal von den dokumentierten Zahlen abweichen. Die **dokumentierten
  Ergebnisse sind unabhängig davon** aus der versionierten CSV-Belegkette reproduzierbar (`run.py demo`).
- **Trainierte Modelle nicht im Repo**: Die `.zip`-Checkpoints (z. B. `dqn_..._3000000steps_seed42.zip`)
  liegen auf dem Trainings-PC und sind bewusst nicht versioniert (`models/checkpoints/*.zip` in
  `.gitignore`). Deshalb spielt `demo`/`compare` die **CSV-Belegkette** ab; für eine Live-Evaluation
  `run.py train` ausführen oder eine `.zip` unter `models/checkpoints/` ablegen.
- **Vollständiger Audit** rechnet jede berichtete Kennzahl aus einem frischen Clone neu nach:
  ```bash
  PYTHONPATH=. python scripts/audit_results.py   # 29/31 Checks aus dem Clone (2 Checks = Existenz von Modell/Log)
  ```

## Troubleshooting

- **`SUMO not found` / `SUMO_HOME`**: `pip install -r requirements.txt` installiert `eclipse-sumo`;
  `run.py` setzt `SUMO_HOME` automatisch über `import sumo`. Manuell prüfen:
  `python -c "import sumo; print(sumo.SUMO_HOME)"`. `demo`/`compare`/`check`/`list` laufen auch ohne SUMO.
- **libsumo vs. traci**: Headless wird standardmäßig traci genutzt; für ~8× schnelleres Training
  `export LIBSUMO_AS_TRACI=1` setzen (blockiert dann GUI und parallele Sims; auf Apple Silicon kann
  libsumo gelegentlich abstürzen - dann Variable nicht setzen).
- **macOS**: pip-SUMO-Wheels benötigen u. U. einmalig einige Homebrew-Bibliotheken
  (`brew install xerces-c fox proj gdal`). Tritt nur bei Fehlern beim SUMO-Import auf.
- **`sumo-gui` zeigt schwarzes Fenster**: einmal in der GUI „Play" drücken bzw. Delay erhöhen;
  für die Bewertung ist die GUI nicht nötig (`--no-gui`).
- **GPU erzwingen/abschalten**: CPU erzwingen mit `CUDA_VISIBLE_DEVICES="" python run.py train ...`.
- **`gymnasium`-Versionsfehler**: exakt `gymnasium==0.29.1` nötig (sumo-rl 1.4.5 bricht mit ≥ 1.0) -
  `pip install -r requirements.txt` in einer frischen venv stellt das sicher.

## Datenquellen

- **OpenStreetMap** - Straßennetz/Geometrie der Kreuzung (Overpass-Dump in `data/raw/osm/`),
  importiert via `netconvert --osm-files` (reproduzierbar mit `scripts/init_data_osm.py`).
- **BASt Dauerzählstellen** - Stundendaten B317 bei Lörrach (DTV-Kalibrierung der Nachfrage-Magnitude).
  Öffentlich bereitgestellt von der Bundesanstalt für Straßenwesen (<https://www.bast.de>, Bereich
  „Verkehrszählung / automatische Dauerzählstellen"); die BASt-Daten sind i. d. R. unter der
  *Datenlizenz Deutschland - Namensnennung 2.0* nutzbar. Die Original-PDF (`data/raw/bast/`) liegt zur
  Nachvollziehbarkeit der Kalibrier-Belegkette bei.
- **HBS-Literaturwerte** - typische DTV für städtische Straßen.

> **Historische Notiz:** Eine frühere, an OSM-Koordinaten *verankerte* schematische Zwischenvariante
> (`loerrach_real.*`) wurde durch den direkten OSM-Import abgelöst; ihre vorläufige Durchsatzkennzahl
> wurde als Messartefakt zurückgezogen (vgl. `docs/documentation/07_evaluation_results.md` §7.0).

## Lizenz

MIT - siehe [`LICENSE`](LICENSE).
