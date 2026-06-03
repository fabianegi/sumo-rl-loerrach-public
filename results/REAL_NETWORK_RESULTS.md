# Ergebnisse - an realen OSM-Koordinaten verankertes Netzmodell

**Stand:** 2026-05-19 · Wave 3 (DASC-PM Phase 3: Analyse)
**Wichtig:** Die hier berichteten Zahlen haben methodische Grenzen - siehe
Abschnitt *Interpretation (revidiert)* weiter unten.

## Setup (ehrliche Bezeichnung)

- **Netz:** `loerrach_real.net.xml` - **kein OSM-Import**, sondern ein via
  `netconvert` gebautes idealisiertes symmetrisches 4-Arm-Kreuz (Arme 200 m,
  50 km/h), **verankert** an den echten OSM-Koordinaten der Kreuzung
  Basler Str. × Obere Riehenstr. (47.5983 N / 7.6564 E, Overpass 2026-05-13).
  Die Topologie ist schematisch, nicht die reale Straßengeometrie (Befund B1).
- **Demand:** `loerrach_real_medium.rou.xml`, parametrisch erzeugt aus
  `DTV=8037` (real-plausibel, vgl. BASt-2023 Zst 8570 = 8.291) × **HBS-2015-
  Literaturprofil** (nicht das vorhandene gemessene Profil, Befund B2) ×
  angenommene Richtungs-/Abbiegesplits (unvalidiert, Befund B3).
- **Agent:** DQN, 500.000 Steps, Seed 42, Reward = diff-waiting-time
- **Baseline:** Festzeit-Steuerung (90 s Zyklus, Webster-proportional)
- **Evaluation:** 30 Seeds je Verfahren, Episode = 3.600 s Simzeit

## KPI-Vergleich (Mittelwert ± Std über 30 Seeds)

> **Metrik-Hinweis (Befund B5):** `avg_waiting_time` ist eine sumo-rl
> **Per-Step-Systemgröße**, nicht die mittlere Wartezeit pro Fahrzeug aus
> tripinfo (reines SUMO: ~11,5 s/Fz). Die *absoluten* Werte sind daher nicht
> als „Sekunden pro Fahrzeug" lesbar. Aussagekraft hat nur der **relative**
> Vergleich (gleiche Metrik beidseitig).

| KPI | Festzeit | DQN | Δ (relativ) | p-Wert | r | signifikant |
|---|---|---|---|---|---|---|
| Ø Wartezeit (Per-Step) | 4,330 ± 0,06 | **0,360 ± 0,02** | **-91,7 %** | 3,0·10⁻¹¹ | -1,00 | ja |
| Max. Rückstau (Fz) | 8,47 ± 0,51 | **6,10 ± 0,31** | **-28,0 %** | 1,3·10⁻¹² | -1,00 | ja |
| Ø Geschwindigkeit (m/s) | 8,17 ± 0,03 | **10,30 ± 0,08** | **+26,0 %** | 3,0·10⁻¹¹ | +1,00 | ja |
| ~~Durchsatz~~ | - | - | **zurückgezogen** | - | - | ungültig |

Test: Mann-Whitney U (zweiseitig, α = 0,05, n=30), r = rank-biserial.

**Durchsatz-KPI zurückgezogen (Befund B4):** Die Zählung
(`getArrivedNumber()` je delta_time-Step akkumuliert) unterzählt systematisch
~Faktor 5 (reines SUMO: 768 von 772 Fz kommen an; Env meldet ~176). Der
früher berichtete „-31 % Durchsatz" war ein **Metrik-Artefakt, kein Befund**,
und wird nicht weiter ausgewiesen.

## Interpretation (revidiert)

**Gültige Aussage:** Der DQN-Agent senkt die env-definierte Wartezeit
gegenüber der starren 90-s-Festzeitsteuerung relativ um ~92 %, reduziert den
maximalen Rückstau und erhöht die mittlere Geschwindigkeit - alle Unterschiede
statistisch signifikant (p < 10⁻¹⁰, |r| = 1,0). Die RL-Pipeline funktioniert
auf dem verankerten Modell.

**Nicht belegbar mit diesen Daten:** absolute Wartezeit-Sekunden pro Fahrzeug,
Durchsatz-Effekte, Übertragbarkeit auf die reale Straßengeometrie.

## Einordnung Hybrid-Strategie (ADR 2026-05-13)

- **Synthetisches Netz (Methodenbeweis):** DQN+PPO, 1M Steps, -88 % Wartezeit,
  p = 0,0002 - belegt die RL-Pipeline.
- **Verankertes Modell (dieses Dokument):** Methode überträgt sich auf ein an
  realen OSM-Koordinaten verankertes Kreuz mit DTV-kalibriertem Demand. Es ist
  **keine** Fidelitäts-Validierung an realer Geometrie (vgl. Audit F3).

## Artefakte

- `results/csv/baseline_fixed_time_real_medium.csv` - 30-Seed-Baseline
- `results/csv/eval_dqn_real_medium_500k.csv` - 30-Seed-Agent-Evaluation
- `models/checkpoints/dqn_diff-waiting-time_500000steps_seed42.zip`
