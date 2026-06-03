# 4. Methodik

## 4.1 MDP-Formulierung

Das Signalsteuerungsproblem wird als Markov Decision Process (MDP) formuliert. Ein MDP ist definiert als Tupel $(S, A, R, T, \gamma)$, wobei $S$ den Zustandsraum, $A$ den Aktionsraum, $R: S \times A \rightarrow \mathbb{R}$ die Belohnungsfunktion, $T: S \times A \rightarrow S$ die Transitionsfunktion und $\gamma \in [0,1]$ den Diskontierungsfaktor beschreiben (Sutton & Barto, 2018). Der Agent -- hier die Ampelsteuerung -- beobachtet den aktuellen Zustand der Kreuzung, waehlt eine Aktion (Signalphase) und erhaelt eine Belohnung, die den Verkehrsfluss bewertet. Ziel ist die Maximierung der kumulierten diskontierten Belohnung ueber eine Episode.

Die Transitionsfunktion $T$ wird implizit durch den SUMO-Mikrosimulator bereitgestellt: Die Fahrzeugdynamik, Spurwechsel und Signalwechsel erzeugen den Folgezustand deterministisch aus dem aktuellen Zustand und der gewaehlten Aktion. Der Agent hat keinen direkten Zugriff auf $T$, sondern lernt ausschliesslich aus Interaktion -- ein model-free Ansatz, wie er in der RL-basierten Verkehrssteuerung Standard ist (Wei et al., 2020).

## 4.2 Zustandsraum (State Space)

Der Zustandsvektor kombiniert Phaseninformation mit Verkehrslage-Features. Tabelle 1 zeigt die einzelnen Komponenten.

| Feature | Typ | Dimensionen | Wertebereich | Beschreibung |
|---|---|---|---|---|
| Current phase | One-hot | $N_{phases}$ | $\{0, 1\}$ | Aktuell aktive Signalphase |
| Min green elapsed | Binary | 1 | $\{0, 1\}$ | Ob die Mindestgruenzeit erfuellt ist |
| Lane density | Float | $N_{lanes}$ | $[0, 1]$ | Fahrzeuge / Spurkapazitaet pro Zufahrt |
| Lane queue | Float | $N_{lanes}$ | $[0, 1]$ | Stehende Fahrzeuge / Spurkapazitaet |

*Tabelle 1: Zustandsvektor-Komponenten. Geschaetzte Gesamtdimension: 12--16, abhaengig von der Anzahl der Spuren und Phasen.*

Bewusst ausgeschlossen wurden: Outgoing-Lane-Density (nur relevant bei Netzwerk-Level-Optimierung), Wartezeit pro Spur (stark korreliert mit Queue, daher redundant), Geschwindigkeit pro Spur (weniger informativ als Queue fuer Signalsteuerung) sowie externe Faktoren wie Wetter oder Uhrzeit (erhoehen die State-Dimensionalitaet ohne Nutzen bei Single-Intersection). Diese Entscheidungen folgen der Empfehlung von Zheng et al. (2019), den Zustandsraum auf verkehrstechnisch relevante Minimalfeatures zu beschraenken.

## 4.3 Aktionsraum (Action Space)

Der Aktionsraum ist diskret: Jede Aktion $a \in A$ waehlt die naechste Gruenphase aus dem Signalplan. Seine Dimension ist **netzabhaengig** und entspricht der Anzahl der Gruenphasen des jeweiligen Signalplans. Fuer das synthetische 4-armige Kreuz ergeben sich **vier** Aktionen (Nord-Sued geradeaus, Nord-Sued Linksabbieger, Ost-West geradeaus, Ost-West Linksabbieger). Das aus OpenStreetMap importierte **Headline-Netz** (Quelle der in Kapitel 7 berichteten Ergebnisse) besitzt drei Gruenphasen, sein Aktionsraum ist also `Discrete(3)`.

Gewaehlt wurde Phase-Selection statt Phase-Duration-Control aus mehreren Gruenden: (1) sumo-rl implementiert Phase-Selection nativ, (2) der diskrete Aktionsraum ermoeglicht schnellere Konvergenz, und (3) SOTA-Arbeiten wie PressLight (Wei et al., 2019) und FRAP (Zheng et al., 2019) nutzen ebenfalls Phase-Selection. Duration-Control wuerde einen kontinuierlichen Aktionsraum erfordern und damit DQN als Algorithmus ausschliessen (ADR #6).

Realistische Constraints werden eingehalten: `min_green_time = 10 s` (verkehrstechnisches Minimum), `yellow_time = 3 s` (automatisch durch sumo-rl eingefuegt) und `delta_time = 5 s` (Entscheidungsfrequenz des Agents). Pro Step waehlt der Agent eine Phase; liegt ein Phasenwechsel vor und ist die Mindestgruenzeit erfuellt, wird eine Gelbphase eingeleitet. Andernfalls bleibt die aktuelle Phase bestehen.

## 4.4 Belohnungsfunktionen (Reward Functions)

Zwei Reward-Funktionen werden trainiert und verglichen (ADR #7):

**Diff-Waiting-Time** ist die primaere Belohnungsfunktion:

$$R_t = W_{t-1} - W_t$$

wobei $W_t$ die kumulative Wartezeit aller Fahrzeuge zum Zeitpunkt $t$ bezeichnet. Ein positiver Reward signalisiert, dass die Gesamtwartezeit gesunken ist. Diese Funktion ist intuitiv, stabil und in der sumo-rl-Literatur gut dokumentiert. Nachteil: Sie beruecksichtigt nicht explizit den Durchsatz.

**Pressure** (inspiriert durch PressLight, Wei et al., 2019) berechnet:

$$R_t = -\left|\sum_i d_i^{in} - \sum_i d_i^{out}\right|$$

wobei $d_i^{in}$ und $d_i^{out}$ die Fahrzeugdichten der einfahrenden bzw. ausfahrenden Spuren bezeichnen. Diese Funktion minimiert die Druckdifferenz und ist theoretisch optimal fuer Throughput-Maximierung (Max-Pressure-Theorem, Varaiya, 2013). Sie benoetigt jedoch Outgoing-Lane-Informationen und ist damit etwas komplexer.

Eine explizite Reward-Normalisierung ist nicht erforderlich: Diff-Waiting-Time ist als Differenz bereits um Null zentriert, Pressure ist durch die Fahrzeugkapazitaet begrenzt, und Stable-Baselines3 fuehrt intern eine Value-Normalization durch.

## 4.5 Episodendesign

Jede Episode simuliert eine Stunde realen Verkehr:

- **Simulationszeit:** 3600 Sekunden (deckt eine Rush-Hour-Periode ab)
- **Warmup:** 300 Sekunden (5 Minuten) zu Beginn, in denen simuliert wird, aber keine RL-Steps gezaehlt werden. Dies ermoeglicht den Aufbau eines realistischen Verkehrszustands.
- **RL-Steps pro Episode:** $3600 / \text{delta\_time} = 3600 / 5 = 720$ Steps
- **Episodenende:** Fixes Zeitlimit (kein terminales Kriterium), da Verkehr nicht "endet" -- ein zeitbasiertes Ende ist realistischer.
- **Demand-Randomisierung:** Pro Episode wird zufaellig eines von drei Demand-Szenarien (low, medium, high) gewaehlt (uniform). Innerhalb eines Szenarios sind die Routen fest. Seed-Kontrolle ueber `numpy`, `torch` und SUMO `--seed` stellt Reproduzierbarkeit sicher.

## 4.6 Baseline: Festzeitsteuerung

Als Vergleichsmaßstab dient eine Festzeitsteuerung mit 90 Sekunden Zykluszeit -- dem typischen Wert für deutsche Innenstadtkreuzungen (HBS, 2015; Webster, 1958). Bei der Auswertung kommen je nach Netz **zwei unterschiedliche Baselines** zum Einsatz; diese Unterscheidung ist für die faire Interpretation der Ergebnisse zentral (vgl. Kapitel 8, „Fairness der Baseline"):

- **Synthetisches 4-Arm-Netz (Methodenbeweis):** eine **Webster-proportionale** Festzeitsteuerung. Die Phasenverteilung erfolgt proportional zum geschätzten Verkehrsaufkommen pro Zufahrt -- trägt die B317 (Basler Straße) etwa 60 % des Verkehrs, erhält sie 60 % der Grünzeit (mindestens 15 s Grün je Phase, 3 s Gelb dazwischen). Diese Baseline ist bewusst **nicht künstlich schwach** gewählt, sondern repräsentiert eine sinnvoll ausgelegte Festzeitsteuerung (`run_fixed_time_baseline` in `baseline.py`).
- **Echtes OSM-Netz (Fidelitäts-Validierung):** das aus OpenStreetMap **importierte Default-Signalprogramm** (`fixed_ts=True`, 90 s Zyklus). Es ist realistisch -- so wäre die Kreuzung „ab Werk" geschaltet --, aber **un-tuned**, also kein hand-optimierter Webster-Plan. Die in Kapitel 7 berichteten Headline-Ergebnisse vergleichen den Agenten daher gegen dieses Default-Programm, **nicht** gegen den Stand der Praxis. Diese Einschränkung wird in Kapitel 8 (Limitation 1) offen ausgewiesen.

Der **eigentliche Praxismaßstab** -- eine verkehrsabhängige (`actuated`) Steuerung, die Grünphasen über Detektoren verlängert (in SUMO über `type="actuated"` nativ verfügbar) -- sowie ein getunter Webster-Plan auf dem OSM-Netz sind als naheliegende Folgearbeiten benannt (Kapitel 9; ADR #11).

> Alle in diesem Kapitel zitierten Quellen sind im zentralen Literaturverzeichnis (`10_references.md`) gelistet.
