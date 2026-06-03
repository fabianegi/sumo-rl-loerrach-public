# 2. Grundlagen und verwandte Arbeiten

Dieses Kapitel legt die theoretische Basis der Arbeit. Abschnitt 2.1-2.3 führen die
verwendeten Lernverfahren ein (Reinforcement Learning, wertbasiertes Lernen mit DQN,
politikbasiertes Lernen mit PPO), Abschnitt 2.4 die verkehrstechnischen Grundlagen
inklusive der Webster-Festzeitauslegung, Abschnitt 2.5 die Simulations- und
Software-Umgebung. Abschnitt 2.6 ordnet die Arbeit in den Forschungsstand zur
RL-basierten Signalsteuerung ein, Abschnitt 2.7 grenzt den eigenen Beitrag ab. Die
konkrete, problemspezifische Ausgestaltung (Zustands-, Aktions-, Belohnungsraum)
folgt in Kapitel 4.

## 2.1 Reinforcement Learning: Grundbegriffe

Reinforcement Learning (RL) beschreibt das Lernen zielgerichteten Verhaltens aus
Interaktion. Ein **Agent** beobachtet zu jedem Zeitschritt $t$ einen Zustand $s_t$,
wählt eine Aktion $a_t$, erhält eine skalare Belohnung $r_{t+1}$ und geht in den
Folgezustand $s_{t+1}$ über. Formalisiert wird dieser Ablauf als **Markov Decision
Process (MDP)** $(S, A, R, T, \gamma)$ mit Zustandsraum $S$, Aktionsraum $A$,
Belohnungsfunktion $R$, Übergangsdynamik $T$ und Diskontierungsfaktor
$\gamma \in [0,1]$ (Sutton & Barto, 2018). Die **Markov-Eigenschaft** besagt, dass der
Folgezustand nur vom aktuellen Zustand und der Aktion abhängt, nicht von der gesamten
Vorgeschichte -- eine für die Signalsteuerung vertretbare Annahme, da der aktuelle
Verkehrszustand (Warteschlangen, Dichten, aktive Phase) die entscheidungsrelevante
Information weitgehend zusammenfasst.

Ziel ist eine **Policy** $\pi(a \mid s)$, die die erwartete kumulierte diskontierte
Belohnung (den *Return* $G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$) maximiert. Zwei
zentrale Bewertungsgrößen sind die **Zustandswertfunktion** $V^\pi(s)$ und die
**Aktionswertfunktion** $Q^\pi(s,a)$, die den erwarteten Return unter $\pi$ ab Zustand
$s$ (bzw. ab dem Paar $(s,a)$) angeben. Der Diskontierungsfaktor $\gamma$ steuert die
Gewichtung zukünftiger gegenüber unmittelbaren Belohnungen. Da die Übergangsdynamik $T$
im vorliegenden Fall nicht analytisch vorliegt, sondern implizit durch den
Mikrosimulator SUMO erzeugt wird, kommen **modellfreie** Verfahren zum Einsatz, die
ausschließlich aus erlebten Transitionen lernen.

## 2.2 Wertbasiertes Lernen: Q-Learning und DQN

**Q-Learning** (Watkins & Dayan, 1992) ist ein modellfreies, *off-policy* Verfahren des
Temporal-Difference-Lernens. Es schätzt die optimale Aktionswertfunktion $Q^*(s,a)$
iterativ über die Aktualisierung

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right],$$

wobei der Term in eckigen Klammern den **TD-Fehler** bildet und $\alpha$ die Lernrate
bezeichnet. In tabellarischer Form konvergiert Q-Learning unter milden Bedingungen
gegen $Q^*$, skaliert aber nicht auf große oder kontinuierliche Zustandsräume, da für
jedes Zustands-Aktions-Paar ein eigener Eintrag nötig wäre.

**Deep Q-Networks (DQN)** (Mnih et al., 2015) überwinden diese Grenze, indem sie
$Q(s,a;\theta)$ durch ein neuronales Netz mit Parametern $\theta$ approximieren. Drei
Bausteine stabilisieren das ansonsten oszillierende Training:

1. **Experience Replay:** Erlebte Transitionen $(s_t, a_t, r_{t+1}, s_{t+1})$ werden in
   einem Puffer gespeichert; das Netz lernt aus zufällig gezogenen Minibatches. Das
   bricht die zeitliche Korrelation aufeinanderfolgender Samples und verbessert die
   Stichprobeneffizienz.
2. **Target-Network:** Ein periodisch eingefrorenes Zielnetz $\theta^-$ erzeugt die
   TD-Ziele $r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a'; \theta^-)$, wodurch das „bewegte
   Ziel"-Problem gedämpft wird.
3. **$\varepsilon$-greedy-Exploration:** Mit Wahrscheinlichkeit $\varepsilon$ wählt der
   Agent eine zufällige Aktion, sonst die aktuell beste -- $\varepsilon$ wird über das
   Training abgesenkt (Exploration → Exploitation).

Eine bekannte Schwäche von DQN ist die **Überschätzung** der Q-Werte durch den
$\max$-Operator. **Double DQN** (van Hasselt et al., 2016) entkoppelt dazu Aktionswahl
und -bewertung und reduziert diese Verzerrung. DQN eignet sich für **diskrete**
Aktionsräume und ist durch den Replay-Puffer vergleichsweise stichprobeneffizient --
beides passt zum hier gewählten Phasenauswahl-Problem.

## 2.3 Politikbasiertes Lernen: Policy Gradient und PPO

Politikbasierte Verfahren optimieren die Policy $\pi_\theta(a \mid s)$ direkt, statt den
Umweg über eine Wertfunktion zu nehmen. Der **Policy-Gradient** zeigt in Richtung
$\nabla_\theta J(\theta) = \mathbb{E}\!\left[\nabla_\theta \log \pi_\theta(a \mid s)\,
\hat{A}(s,a)\right]$, wobei der **Vorteil** $\hat{A}(s,a) = Q(s,a) - V(s)$ angibt, um
wie viel eine Aktion besser ist als der Erwartungswert im Zustand. Naive
Policy-Gradient-Updates sind jedoch instabil: zu große Schritte können die Policy
zerstören.

**Proximal Policy Optimization (PPO)** (Schulman et al., 2017) begrenzt deshalb die
Schrittweite über ein *clipped surrogate objective*

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\!\big( \rho_t(\theta)\,\hat{A}_t,\;
\text{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon)\,\hat{A}_t \big) \right],$$

mit dem Wahrscheinlichkeitsverhältnis $\rho_t(\theta) = \frac{\pi_\theta(a_t \mid
s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$. Der Clip verhindert, dass eine
einzelne Aktualisierung die Policy zu weit vom vorherigen Stand entfernt. PPO arbeitet
*on-policy* (es lernt aus Daten der aktuellen Policy), ist robust und einfach zu
tunen, dafür weniger stichprobeneffizient als off-policy-Verfahren wie DQN.

**Warum beide Verfahren vergleichen?** DQN (wertbasiert, off-policy) und PPO
(politikbasiert, on-policy) repräsentieren die zwei dominanten Familien des modernen
Deep RL. Da der Aktionsraum der Signalsteuerung diskret ist, sind beide anwendbar. Der
kontrollierte Vergleich (Kapitel 6-7) prüft, welches Paradigma für dieses Problem
stabiler konvergiert und bessere KPIs liefert -- statt vorab ein Verfahren zu setzen.

## 2.4 Verkehrstechnische Grundlagen

Eine **Lichtsignalanlage (LSA)** schaltet die widerstreitenden Verkehrsströme einer
Kreuzung zeitlich getrennt frei. Eine **Phase** ist eine Kombination gleichzeitig
freigegebener Ströme; die **Umlaufzeit (Zyklus)** $C$ ist die Dauer einer vollständigen
Phasenfolge inklusive Zwischenzeiten (Gelb/Rot). Maßgebliche Kenngrößen sind die
**Sättigungsverkehrsstärke** $s$ (maximaler Abfluss bei Dauergrün), die
Verkehrsstärke $q$ einer Zufahrt und das daraus gebildete **Auslastungsverhältnis**
$y = q/s$. Übersteigt die Summe der maßgebenden $y$-Werte die verfügbare Grünzeitanteil,
wird der Knoten übersättigt und Rückstaus wachsen über den Zyklus hinaus an (Spillback).

Das **Webster-Verfahren** (Webster, 1958) liefert eine analytisch begründete
Festzeitauslegung. Die annähernd verzögerungsoptimale Umlaufzeit ist

$$C_{\text{opt}} = \frac{1{,}5\,L + 5}{1 - Y}, \qquad Y = \sum_i y_i,$$

mit der gesamten Verlustzeit $L$ pro Umlauf. Die effektive Grünzeit wird anschließend
**proportional zu den Auslastungsverhältnissen** $y_i$ auf die Phasen verteilt. Diese
Logik bildet die Grundlage der in dieser Arbeit verwendeten Festzeit-Baseline
(Kapitel 4.6). Die typische Umlaufzeit deutscher Innenstadtknoten von rund 90 s liegt
im Normbereich der einschlägigen Bemessungsrichtlinie (HBS, 2015). Die Grenze der
Festzeitsteuerung ist ihre **Starrheit**: Ein fester Plan ist nur für die unterstellte
Nachfrage optimal und reagiert nicht auf stochastische Schwankungen -- genau hier setzt
die adaptive RL-Steuerung an.

## 2.5 Simulationsumgebung: SUMO, TraCI, sumo-rl und Gymnasium

**SUMO** (*Simulation of Urban MObility*, Lopez et al., 2018) ist ein quelloffener,
**mikroskopischer** Verkehrssimulator: Er bildet einzelne Fahrzeuge mit Folge-,
Spurwechsel- und Abbiegemodellen ab und gilt als De-facto-Standard der RL-basierten
Signalsteuerungsforschung. Über die **TraCI**-Schnittstelle (*Traffic Control
Interface*) lässt sich die laufende Simulation zur Laufzeit auslesen und steuern --
die Voraussetzung dafür, dass ein Agent zu jedem Entscheidungszeitpunkt den Zustand
beobachten und die Signalphase setzen kann.

Die Interaktion folgt der **Gymnasium**-API (Towers et al., 2023) -- dem als Nachfolger
von OpenAI Gym (Brockman et al., 2016) etablierten De-facto-Standard für RL-Umgebungen
mit dem `reset()`/`step()`-Vertrag. **sumo-rl** (Alegre, 2019) kapselt
SUMO als Gymnasium-konforme Umgebung und stellt Standard-Observationen (Phase, Queue,
Dichte) sowie Belohnungsfunktionen (u.a. `diff-waiting-time`, `pressure`) bereit. Die
Lernalgorithmen stammen aus **Stable-Baselines3** (Raffin et al., 2021), einer
geprüften PyTorch-Implementierung gängiger RL-Verfahren (DQN, PPO, A2C, SAC) mit
TensorBoard- und Callback-Unterstützung. Dieser Stack erlaubt es, den Beitrag der
Arbeit auf das *Experiment-Design* statt auf Implementierungsdetails zu konzentrieren.

## 2.6 RL für Signalsteuerung (Forschungsstand)

Die Anwendung von Reinforcement Learning auf die Verkehrssignalsteuerung hat in den
letzten Jahren stark an Bedeutung gewonnen; einen Überblick gibt der Survey von
Wei et al. (2020). Frühe Arbeiten nutzten tabellarisches Q-Learning für einzelne
Kreuzungen, mit dem Aufkommen von Deep RL verschob sich der Fokus auf größere Netze und
reichhaltigere Zustandsrepräsentationen.

**Genders & Razavi (2016)** zeigen für eine **einzelne Kreuzung**, dass ein
DQN-Agent mit einer auf Dichte und Warteschlange basierenden Zustandsrepräsentation eine
Festzeitsteuerung deutlich übertrifft -- ein direkter methodischer Vorläufer des
vorliegenden Single-Intersection-Ansatzes.

**PressLight** (Wei et al., 2019) formuliert die Signalsteuerung als
Max-Pressure-Optimierung und zeigt, dass ein druckbasierter Reward theoretisch
fundierte Vorteile gegenüber rein wartezeitbasierten Metriken bietet. Die zugrunde
liegende Max-Pressure-Theorie (Varaiya, 2013) garantiert unter bestimmten Annahmen die
Stabilisierung der Warteschlangen. Die zentrale Einsicht -- Minimierung der
Druckdifferenz zwischen ein- und ausfahrenden Spuren -- wird in dieser Arbeit als
zweite Belohnungsfunktion übernommen (Kapitel 4.4).

**LIT / FRAP** (Zheng et al., 2019) schlagen einen Phasen-Wettbewerbs-Mechanismus vor,
bei dem der Agent Phasen nach ihrer relativen Dringlichkeit auswählt. Die Arbeit
demonstriert, dass **Phase-Selection** (Auswahl der nächsten Grünphase) gegenüber
**Phase-Duration-Control** (Festlegung der Grünzeit) Vorteile bei Konvergenz und
Stabilität bietet -- ein Befund, der die Designentscheidung dieses Projekts stützt
(vgl. ADR #6).

Für **netzweite** Koordination existieren Multi-Agent-Ansätze: van der Pol & Oliehoek
(2016) koppeln mehrere Deep-RL-Steuerungen über lokale Koordinationsgraphen, und
El-Tantawy et al. (2013) integrieren ein Netz adaptiver Controller (MARLIN-ATSC). Diese
Richtung liegt außerhalb des hier gewählten Single-Intersection-Scopes, wird aber im
Ausblick (Kapitel 9) als Erweiterung aufgegriffen.

## 2.7 Abgrenzung dieser Arbeit

Die genannten Forschungsarbeiten fokussieren überwiegend auf Multi-Intersection-Szenarien
mit hohem Verkehrsaufkommen (Arterien, Gridnetze). Dieses Projekt verfolgt einen bewusst
engeren Scope:

- **Single-Intersection:** Eine einzelne Kreuzung statt eines Netzwerks. Dies erlaubt
  tiefere Analyse des Agent-Verhaltens und der Reward-Dynamik, ohne Multi-Agent-Koordination
  zu benötigen (vgl. ADR #13).
- **Realitätsnahe Kalibrierung:** Statt synthetischer oder aus anderen Städten übernommener
  Nachfrage wird die Verkehrsnachfrage aus lokalen Zähldaten (DTV 8.037, Lörrach-Stetten)
  abgeleitet.
- **Methodenvergleich statt SOTA-Jagd:** Das Ziel ist nicht ein neuer Algorithmus, sondern
  der rigorose Vergleich zweier etablierter Methoden (DQN, PPO) und Reward-Funktionen unter
  kontrollierten Bedingungen mit statistischer Absicherung.

> Alle in diesem Kapitel zitierten Quellen sind im zentralen Literaturverzeichnis (`10_references.md`) gelistet.
