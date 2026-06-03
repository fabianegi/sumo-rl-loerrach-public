# Methodischer Rahmen: DASC-PM

**Projekt:** RL-basierte Ampeloptimierung Lörrach (Basler Straße)
**Gewähltes Vorgehensmodell:** **DASC-PM** (Data Science Process Model, NORDAKADEMIE / Schulz et al., 2020)
**Entscheidung:** 2026-05-13
**Begründung:** Siehe Abschnitt "Warum DASC-PM (und nicht CRISP-DM)?" unten.

---

## Überblick: Die fünf DASC-PM-Phasen in diesem Projekt

```
   ┌───────────────────────────────────────────────────────────────┐
   │  Wissenschaftliche Methodik  (durchgängige Qualitätssicherung) │
   └───────────────────────────────────────────────────────────────┘
   ┌───────────────────────────────────────────────────────────────┐
   │  Domänenwissen Verkehrstechnik  (HBS 2015, LSA, ÖV, KFZ-Mix)   │
   └───────────────────────────────────────────────────────────────┘

   1. Projektauftrag  →  2. Datenbereitstellung  →  3. Analyse
                                                       │
                          5. Anwendung  ←  4. Nutzbarmachung
```

| # | Phase | Inhalt im Projekt | Artefakte |
|---|-------|-------------------|-----------|
| 1 | **Projektauftrag** | Forschungsfrage, Stakeholder, Scope, Risiken | `README.md`, `docs/documentation/01_introduction.md` |
| 2 | **Datenbereitstellung** | BASt-Stundenwerte, OSM-Geometrie, Felderhebung, Stadt-Absage | `data/raw/`, `data/processed/`, `docs/documentation/03_data_and_environment.md` |
| 3 | **Analyse** | SUMO-Modellierung, DQN/PPO-Training, statistische Evaluation | `src/`, `models/checkpoints/`, `results/csv/`, `results/plots/` |
| 4 | **Nutzbarmachung** | Dokumentations-Portfolio, Präsentation, Reproduzierbarkeit | `docs/documentation/`, `README.md` |
| 5 | **Anwendung** | Übergabe an Domäne (Stadt Lörrach, Folgesemester), Ausblick | `docs/documentation/09_conclusion_outlook.md`, GitHub-Repo |

---

## Phase 1 - Projektauftrag

### Forschungsfrage
> *Kann ein RL-Agent (DQN/PPO) eine Festzeitschaltung an einer Einzelkreuzung in Lörrach statistisch signifikant in den KPIs mittlere Wartezeit, maximale Rückstaulänge und Durchsatz übertreffen?*

### Stakeholder-Mapping

| Stakeholder | Rolle | Status |
|---|---|---|
| DHBW Lörrach (Prüfungsausschuss) | Bewertende Instanz | aktiv - Abgabe 28.06.2026, Präsentation 03.06.2026 |
| Stadt Lörrach (Fachbereich Tiefbau) | Datengeber (Signalplan, Zähldaten) | **Anfrage abgelehnt** - Personalmangel, schriftlich dokumentiert |
| BASt | Datengeber (Stundenwerte B317, A98) | aktiv - Daten verfügbar |
| MobiData BW | Datengeber (ergänzend) | aktiv - Plattform offen |
| Akademische Gemeinschaft | Methoden- und Literaturkontext (PressLight, SOTA) | passiv |

### Erfolgskriterien
- **MUSS:** RL-Agent zeigt klaren Lerntrend in der Reward-Kurve (Konvergenz)
- **MUSS:** Statistisch valide Evaluation (30 Seeds für die Hauptergebnisse, Mann-Whitney U, p<0.05)
- **SOLL:** RL schlägt Festzeit-Baseline in mind. 2 KPIs
- **KANN:** Reale Kreuzung Basler Str. × Obere Riehenstr. modelliert (Hybrid-Ansatz)
- **STRETCH:** Vergleich mit Actuated Control + CO₂-Auswertung

---

## Phase 2 - Datenbereitstellung

### Datenquellen-Hierarchie (Real-First-Strategie)

```
Priorität 1: Reale offizielle Daten
    - BASt-Stundenwerte B317: vorhanden (data/raw/bast/b317_zst8570_8921_2023.csv)
    - BASt-Stundenwerte A98: vorhanden (data/raw/bast/a98_zst8003_2023.csv)
    - OSM-Geometrie Basler Straße: frei verfügbar
    - Stadt-Signalplan: abgelehnt
    - Stadt-Knotenstromzählung: abgelehnt

Priorität 2: Eigenerhebung
    - Manuelle Knotenstromzählung vor Ort: geplant

Priorität 3: Dokumentierter Fallback
    - Synthetisches 4-Arm-Netz mit BASt-kalibriertem Demand: vorhanden
```

### Datenqualitäts-Dimensionen (DASC-PM-Standard)

| Dimension | Bewertung | Maßnahme |
|---|---|---|
| Vollständigkeit | Teilweise - Signalplan/TMC fehlen | Felderhebung + Annahmen dokumentiert |
| Aktualität | BASt 2023 (1 Jahr alt) | OK für Tagesganglinie |
| Richtigkeit | BASt amtlich, hohe Qualität | Stichprobenprüfung |
| Konsistenz | Format Bestandsbandformat einheitlich | Format geprüft, Kalibrierung via `scripts/init_data.py` |
| Plausibilität | DTV 8.037 Kfz/24h passt zu Lit.-Werten | HBS 2015 abgleich |

---

## Phase 3 - Analyse

### Modellierungsansatz
- **Simulator:** Eclipse SUMO 1.18.0 (mikroskopisch, Open Source, Eclipse Foundation)
- **RL-Framework:** sumo-rl 1.4.5 + stable-baselines3 2.3.2
- **Algorithmen:** DQN (primär) + PPO (Vergleichsbasis)
- **Reward-Vergleich:** `diff-waiting-time` vs. `pressure` (PressLight, Wei et al. 2019)

### MDP-Formulierung
- **State:** Phase(One-Hot) + min_green_elapsed + lane_density + lane_queue (~16-dim)
- **Action:** Diskrete Phase-Selection
- **Episode:** 3600s Simulation, 300s Warmup, Δt=5s → 720 RL-Steps
- **Training:** 1.000.000 Steps (Konvergenz bei ~400k bereits beobachtet)

### Evaluation
- 30 Seeds (OSM-Headline und reales Netz; synthetischer Methodenbeweis: 10 Seeds), KPIs: avg waiting time, max queue, throughput, avg speed
- Statistische Signifikanz: Mann-Whitney U (nicht-parametrisch)
- Effektstärke: Rank-Biserial r

### Hybrid-Strategie (entschieden 2026-05-13)
1. **Synthetisches Netz (4-Arm):** trainiert, -88% Wartezeit, -50% Queue, p=0.0002 - dient als statistisch belastbarer Methodenbeweis
2. **Reales Netz (Basler × Ob. Riehenstr.):** in Aufbau - dient als Fidelitäts-Validierung mit ehrlicher Diskussion der Annahmen

---

## Phase 4 - Nutzbarmachung

### Artefakte für Übergabe und Bewertung
- **Markdown-Portfolio** (`docs/documentation/00-10`): vollständige wissenschaftliche Narrative
- **Präsentation**: 5-10 Min, Folien + Demo-Plan + visuelle Assets (separat übergeben, nicht im Repo)
- **Code-Repository** (GitHub, öffentlich): vollständig reproduzierbar
- **README** (`README.md`): Quickstart, Datenquellen, Methodik-Verweis

### Reproduzierbarkeit
- Alle Hyperparameter in `src/config/settings.py` (frozen dataclass)
- Seeds explizit (`src/utils/seeding.py`, seed=42)
- Pinned Versions (`requirements.txt`)
- Automatisiertes Setup (`scripts/setup.sh`)

---

## Phase 5 - Anwendung

### Potenzielle Nachnutzung
1. **Stadt Lörrach:** Bei Reaktivierung der Personalsituation kann der Code-Stack mit echten Signaldaten parametrisiert werden - Übergabe-Schnittstelle: `data/sumo_config/` (austauschbare `.net.xml` / `.rou.xml`).
2. **DHBW-Folgesemester:** Erweiterung auf Multi-Agent-RL (Korridor entlang Basler Str. - Korridor-Netz bereits angelegt in `data/sumo_config/loerrach_corridor.*`).
3. **Akademische Verwertung:** Reproduzierbare Baseline für Vergleichsstudien.

### Limitationen (siehe `docs/documentation/08_discussion.md`)
- Keine offiziellen Signaldaten der Stadt (Fallback dokumentiert)
- Synthetisches Netz mit vereinfachter Geometrie
- Keine Validierung gegen reale Wartezeit-Messungen

---

## Wissenschaftliche Methodik (durchgängig)

Diese Querschnittsdimension von DASC-PM verlangt in jeder Phase:

- **Phase 1:** Klare, falsifizierbare Forschungsfrage; vorab definierte Erfolgskriterien
- **Phase 2:** Datenquellen-Triangulation (BASt + OSM + Felderhebung); Transparenz bei Fallbacks
- **Phase 3:** Hypothesentest (Mann-Whitney U), kontrolliertes Seed-Management, separater Train/Eval-Split (`make_env` vs. `make_eval_env`)
- **Phase 4:** Vollständige Methoden-Offenlegung; Versionierung aller Artefakte
- **Phase 5:** Ehrliche Limitationen, keine Übertreibung der Ergebnisse

## Domänenwissen Verkehrstechnik (durchgängig)

- **HBS 2015** (Handbuch für die Bemessung von Straßenverkehrsanlagen) - Grundlage für Festzeit-Baseline (90s Zyklus, Webster-proportional)
- **PressLight** (Wei et al. 2019) - Pressure-Reward für Throughput
- **DTV-Konventionen** (BASt) - Tagesganglinien für Demand-Generierung
- **Knotenstromzählungen** (FGSV-Standard) - Methodik der Felderhebung

---

## Warum DASC-PM (und nicht CRISP-DM)?

| Faktor | CRISP-DM (1996) | DASC-PM (2020) |
|---|---|---|
| Simulation vs. klassische ML | "Deployment" passt schlecht zu Simulationsergebnissen | "Nutzbarmachung" passt natürlich: Doku, Übergabe, Reproduzierbarkeit |
| Stakeholder-Refusal (Stadt) | Keine klare Verortung | "Projektauftrag" enthält Stakeholder-Mapping explizit |
| Domänenwissen (Verkehrstechnik) | Implizit, verteilt | Explizite Querschnittsdimension |
| Datenstrategie mit Fallback | Schwer einzubetten in "Data Understanding" | "Datenbereitstellung" deckt Beschaffung *und* Fallbacks ab |
| Deutscher akademischer Kontext | Universell, aber generisch | Speziell für DACH-Studiengänge entwickelt |
| Wissenschaftliche Rigorosität | Nicht explizit | Querschnittsdimension "Wissenschaft" |

**Fazit:** DASC-PM ist nicht nur "moderner", sondern bildet die Realität dieses Projekts (Stadt-Absage, BASt-Fallback, Felderhebung, Simulationsergebnis statt deployed Model, akademische Abgabe) strukturell sauberer ab.

---

## Referenz

Schulz, Mike; Neuhaus, Uwe et al. (2020): *DASC-PM v1.0 - Vorgehensmodell für Data-Science-Projekte.* NORDAKADEMIE Hochschule der Wirtschaft, Hamburg.
URL: https://www.nordakademie.de/dasc-pm
