# Ergebnisse - echtes OSM-Geometrie-Netz (Basler Str. × Obere Riehenstr.)

**Stand:** 2026-05-21 · DASC-PM Phase 3 (Analyse), Audit-Folgearbeit **F3**
**Voraussetzung:** `results/REAL_NETWORK_RESULTS.md` (verankertes Schema-Netz) gelesen.

Dieses Dokument löst die in Audit **F3** offen gebliebene Forderung ein:
*„echtes OSM-Netz via `netconvert --osm-files` importieren, um den
Fidelitäts-Anspruch tatsächlich einzulösen."* Es nutzt außerdem die in **F1**
korrigierte Durchsatz-/Wartezeit-Messung (tripinfo statt `getArrivedNumber()`).

## Setup (ehrliche Bezeichnung)

- **Netz:** `loerrach_osm.net.xml` - **echter OSM-Import** der Kreuzung
  Basler Str. × Obere Riehenstr. (Lörrach-Stetten), via
  `netconvert --osm-files data/raw/osm/loerrach_real_osm.osm.xml`. Reale
  Straßengeometrie inkl. Nebenstraßen/Auffahrten; **eine** signalisierte
  Kreuzung (TLS OSM-ID `1628110071`). Anders als `loerrach_real.net.xml`
  (idealisiertes, an OSM-Koordinaten *verankertes* 4-Arm-Kreuz, Befund B1)
  ist hier die Topologie die tatsächliche.
- **Demand:** `loerrach_osm_medium.rou.xml`, erzeugt mit `randomTrips.py`
  (`period 2,9 s`, `--validate`, `--fringe-factor 5`, seed 42 → 681 gültige
  Trips/3600 s). Magnitude kalibriert auf die BASt-2023-Spitzenstunde
  (Zst 8570, ≈ 680 Kfz/h), identisch zur `init_data_real.py`-Kalibrierung (F2).
  OD-Relationen sind durch `randomTrips` zufällig (nicht erhobene
  Abbiegeströme) - siehe Limitierungen.
- **Agent:** DQN, **3.000.000 Steps**, Seed 42, Reward = `diff-waiting-time`
  (4.166 Episoden, ≈ 5,2 h CPU). Lernkurve konvergiert ab ≈ Episode 2.500.
- **Baseline:** Festzeit-Steuerung = **das aus OSM importierte
  Default-Signalprogramm** des Netzes (`fixed_ts=True`), **kein** hand-
  optimierter Webster-Plan. Das ist ein realistischer, aber un-tuned Vergleich
  (siehe Limitierungen).
- **Evaluation:** 30 Seeds (0-29) je Verfahren, Episode = 3.600 s, delta_time
  = 5 s → 720 RL-Schritte. Training-Seed (42) ≠ Eval-Seeds.

## KPI-Vergleich (Mittelwert ± Std über 30 Seeds)

Festzeit-Baseline (N=30) vs. DQN-Agent (N=30), Mann-Whitney U (zweiseitig,
alpha = 0,05), r = rank-biserial.

| KPI | Typ | Festzeit | DQN | Δ rel. | p-Wert | r | sig. |
|---|---|---|---|---|---|---|---|
| Ø Wartezeit [Per-Step-System] | per-step | 289 ± 262 | 93,9 ± 172 | **-67,5 %** | 7,0·10⁻⁷ | -0,75 | ja |
| Max. Rückstau [Fz] | per-step | 141 ± 109 | 58,0 ± 91 | **-59,0 %** | 1,7·10⁻⁵ | -0,64 | ja |
| Ø Geschwindigkeit [m/s] | per-step | 4,26 ± 2,54 | 6,57 ± 2,06 | **+54,1 %** | 2,6·10⁻⁷ | +0,78 | ja |
| **Durchsatz (tripinfo)** [Fz/h] | tripinfo | 473 ± 168 | **597 ± 130** | **+26,2 %** | 2,6·10⁻⁵ | +0,63 | ja |
| Ø Wartezeit/Fz [s] | tripinfo | 5,73 ± 0,32 | 2,54 ± 1,69 | **-55,6 %** | 4,1·10⁻⁷ | -0,76 | ja |
| Ø Reisezeit/Fz [s] | tripinfo | 67,7 ± 2,2 | 65,6 ± 1,3 | -3,1 % | 2,1·10⁻³ | -0,46 | ja |
| Ø Zeitverlust/Fz [s] | tripinfo | 16,9 ± 0,8 | 13,8 ± 1,5 | **-18,3 %** | 5,5·10⁻⁸ | -0,82 | ja |

Alle sieben KPI unterscheiden sich statistisch signifikant zugunsten des
DQN-Agenten. Effektgrößen |r| = 0,46-0,82 (mittel bis groß).

**Durchsatz = Abschlussquote (681 geladene Trips):** Festzeit **69,5 %**
(Spanne 36-98 %), DQN **87,7 %** (Spanne 47-99 %).

## Interpretation

**Gültige Aussagen:**

- Auf der **echten OSM-Geometrie** erhöht der DQN-Agent den **Durchsatz
  signifikant um +26 %** (≈ +18 Prozentpunkte Abschlussquote). Das ist -
  anders als am verankerten Schema-Netz - der entscheidende, praxisrelevante
  Befund: Das reale Netz ist bei „medium"-Last **kapazitätskritisch**, und der
  Agent nutzt die Knotenkapazität deutlich besser.
- Der Agent **stabilisiert** den Knoten: Die Festzeit-Steuerung läuft in
  **16 von 30 Seeds** in einen Quasi-Gridlock (Per-Step-Wartezeit > 200 s),
  der DQN-Agent nur in **7 von 30**. Sichtbar im Durchsatz-Boxplot als bimodale
  Festzeit-Verteilung vs. eng am Maximum geclusterter DQN-Verteilung
  (`results/plots/kpi_osm_throughput.png`).
- Wartezeit, Rückstau und Geschwindigkeit verbessern sich signifikant
  (per-step), ebenso die per-Fahrzeug-Wartezeit und der -Zeitverlust (tripinfo).

**Wichtiger Vorbehalt - Selektionsverzerrung der tripinfo-Mittel:**
tripinfo erfasst **nur abgeschlossene Fahrzeuge**. Da die Festzeit-Baseline
124 Fahrzeuge weniger ins Ziel bringt (die im Stau steckenden), fehlen genau
die am stärksten betroffenen Fahrzeuge in ihren tripinfo-Mitteln - erkennbar an
der **winzigen Streuung der Festzeit-Wartezeit/Fz (±0,32 s)** trotz riesiger
Per-Step-Streuung (±262 s). Folge: Die per-Fahrzeug-Reisezeit (-3,1 %) und
-Wartezeit (-55,6 %) **unterschätzen** den wahren Effekt, weil der DQN-Agent
zusätzlich die „schweren" Fahrzeuge abarbeitet, die die Baseline liegen lässt.
**Der unverzerrte Leitindikator ist hier der Durchsatz (+26 %).**

**Nicht belegbar mit diesen Daten:**

- Überlegenheit gegenüber einem **optimierten** Festzeitplan (Webster) oder
  einer **verkehrsabhängigen** Steuerung - verglichen wurde gegen das
  un-tuned OSM-Default-Programm.
- Belastbarkeit der **Abbiegeströme**: `randomTrips` erzeugt zufällige
  OD-Relationen, keine erhobenen Richtungsanteile.
- Übertragbarkeit auf andere Lastfälle (nur „medium" trainiert/evaluiert).

## Einordnung gegenüber dem verankerten Schema-Netz

| Aspekt | Verankertes 4-Arm-Kreuz | **Echtes OSM-Netz (dieses Dok.)** |
|---|---|---|
| Geometrie | idealisiert, symmetrisch | reale OSM-Topologie |
| Durchsatz-Effekt | +0,3 % (gesättigt, trivial) | **+26 % (kapazitätskritisch)** |
| Effektgröße \|r\| | 1,00 (perfekte Trennung) | 0,46-0,82 (realistische Überlappung) |
| Aussage | Methodenbeweis | **Fidelitäts-Validierung** |

Das saubere Kreuz war zur Durchsatz-Frage uninformativ (fast alle Fahrzeuge
kommen ohnehin an). Erst die reale Geometrie macht den Kapazitätsgewinn der
RL-Steuerung messbar - und liefert mit |r| < 1 zugleich glaubwürdigere,
nicht „zu perfekte" Statistik.

## Limitierungen & Folgearbeit

- **Baseline un-tuned:** Vergleich gegen optimierten Webster-Plan und/oder
  SUMO-`actuated`-Steuerung nachziehen, um den Effekt fair einzuordnen.
- **`time-to-teleport = -1` (kein Teleport):** Steckengebliebene Fahrzeuge
  werden *nicht* aus dem Stau „wegteleportiert"; ein Knoten-Deadlock bleibt
  bestehen. Das gilt **für beide Steuerungen gleich** (fairer Relativvergleich),
  beeinflusst aber die *absolute* Gridlock-Häufigkeit: Unter SUMOs Default
  (Teleport nach 300 s) würden Staus künstlich aufgelöst und die 16/30-vs-7/30-
  Zahlen fielen niedriger aus. Die Kernaussage (+26 % Durchsatz) bleibt davon
  unberührt.
- **Nur „medium"-Last:** `low`/`high` evaluieren (Last-Sensitivität).
- **Abbiegeströme:** falls erhebbar, `randomTrips` durch OD-kalibrierte
  Nachfrage ersetzen.
- **Ein Reward / ein Algorithmus:** `pressure`-Reward (durchsatzorientiert)
  und PPO als Quervergleich.

## Reproduktion

```bash
# 1) Netz + Nachfrage (idempotent)
python scripts/init_data_osm.py --force

# 2) Training (≈5 h CPU; LIBSUMO optional für Speedup)
LIBSUMO_AS_TRACI=1 PYTHONPATH=. python src/training/train_dqn.py \
  --total-timesteps 3000000 --seed 42 --reward diff-waiting-time \
  --net-file data/sumo_config/loerrach_osm.net.xml \
  --route-file data/sumo_config/loerrach_osm_medium.rou.xml

# 3) Baseline + Eval (je 30 Seeds, tripinfo-genau) + Vergleichstabelle
python src/evaluation/baseline.py  --net-file data/sumo_config/loerrach_osm.net.xml \
  --route-file data/sumo_config/loerrach_osm_medium.rou.xml --n-episodes 30 \
  --output-csv results/csv/baseline_osm_medium.csv
python src/evaluation/evaluate.py  --net-file data/sumo_config/loerrach_osm.net.xml \
  --route-file data/sumo_config/loerrach_osm_medium.rou.xml --n-episodes 30 \
  --model models/checkpoints/dqn_diff-waiting-time_3000000steps_seed42.zip \
  --output-csv results/csv/eval_dqn_osm_medium_3M.csv
python scripts/compare_baseline_agent.py \
  --baseline-csv results/csv/baseline_osm_medium.csv \
  --agent-csv results/csv/eval_dqn_osm_medium_3M.csv \
  --output-md results/csv/compare_osm_medium.md --title "OSM-Netz (medium), DQN 3M Steps"
```

**Artefakte (lokal, git-ignoriert):** `results/csv/{baseline_osm_medium,
eval_dqn_osm_medium_3M}.csv`, `results/plots/{learning_curve_osm_dqn_3M,
kpi_osm_throughput,kpi_osm_waiting_perstep,kpi_osm_maxqueue,kpi_osm_speed}.png`,
Modell `models/checkpoints/dqn_diff-waiting-time_3000000steps_seed42.zip`.
