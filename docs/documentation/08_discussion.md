# Diskussion

## Stärken der Methodik

Das vorliegende Projekt verfolgt einen methodisch fundierten Ansatz zur RL-basierten Lichtsignalsteuerung. Die vollständige Reproduzierbarkeit der Ergebnisse wird durch konsequente Seed-Kontrolle aller stochastischen Komponenten sichergestellt: NumPy, PyTorch, CUDA und der SUMO-Simulator werden über `seed_everything()` deterministisch initialisiert. Die Evaluation über 30 Seeds (synthetischer Methodenbeweis: 10) mit anschließendem Mann-Whitney-U-Test und Effektstärkenberechnung entspricht dem statistischen Standard der Reinforcement-Learning-Literatur und erlaubt belastbare Aussagen jenseits von Einzellauf-Anekdoten.

Die Wahl von sumo-rl als Environment-Wrapper bietet eine standardisierte Gymnasium-konforme Schnittstelle, die direkte Kompatibilität mit Stable-Baselines3 gewährleistet. Der modulare Aufbau der Codebasis -- mit getrennter Config-Hierarchie (`ProjectConfig` als frozen Dataclass), austauschbaren Reward-Funktionen und separaten Evaluationsskripten -- ermöglicht systematische Ablation einzelner Designentscheidungen. Der Vergleich zweier Reward-Funktionen (diff-waiting-time und pressure) sowie zweier Algorithmen (DQN und PPO) auf dem Methodenbeweis-Netz erhöht die Aussagekraft gegenüber einem reinen Single-Agent-Single-Reward-Setup.

Die State-Repräsentation folgt bewusst dem Minimalprinzip: Phase-Encoding, Lane Density und Lane Queue erfassen die entscheidungsrelevanten Informationen, ohne durch redundante Features (z.B. Speed, die stark mit Queue korreliert) die Dimensionalität unnötig aufzublähen.

## Interpretation der Ergebnisse

### Was bedeutet der Gridlock in mehr als der Hälfte der Festzeit-Seeds?

Der auffälligste Befund ist nicht ein Mittelwert, sondern eine **Verteilungsform**: Die Festzeit-Steuerung läuft in **16 von 30 Seeds** in einen Quasi-Gridlock (Per-Step-Wartezeit > 200 s), der DQN-Agent nur in **7 von 30**. Die Durchsatz-Verteilung der Festzeit ist dadurch **bimodal** -- entweder der Knoten arbeitet sauber ab (≈ 660 Fz/Episode) oder er kollabiert (≈ 318 Fz/Episode); ein Mittelfeld existiert kaum. Das ist verkehrstechnisch typisch: Ein signalisierter Knoten nahe der Kapazitätsgrenze kippt nicht graduell, sondern schlagartig, sobald der Zufluss die Abflussrate einer Phase übersteigt und sich Rückstaus über die Grünzeit hinaus akkumulieren (Spillback).

Praktisch heißt das: Die Festzeit-Steuerung ist an dieser Kreuzung **nicht grundsätzlich schlecht**, sondern **nicht robust**. Sie funktioniert, solange die zufällige Ankunftsfolge günstig ist, und versagt, sobald sie ungünstig ist. Der Wert der RL-Steuerung liegt damit weniger im besseren Durchschnitt als in der **Vermeidung des Worst Case** -- ein Kriterium, das für die reale Verkehrsqualität (und die subjektive Wahrnehmung von Stau) oft wichtiger ist als der Mittelwert.

### Was sagt das über die Kapazitätsgrenze des Netzes?

Dass überhaupt 16 von 30 Festzeit-Durchläufe kippen, belegt: Die gewählte **medium-Last liegt nahe der Kapazitätsgrenze** des realen Knotens. Genau das macht das Szenario aussagekräftig. Am idealisierten 4-Arm-Kreuz (Methodenbeweis) lag die Last weit unter der Kapazität -- fast alle Fahrzeuge kamen ohnehin an, der Durchsatz-Effekt der RL-Steuerung war mit +0,3 % trivial. Erst die reale OSM-Geometrie mit ihren asymmetrischen Zufahrten und kürzeren Speicherlängen erzeugt einen kapazitätskritischen Betriebspunkt, an dem die Steuerungsstrategie tatsächlich über Kollaps oder Durchsatz entscheidet. Der Kapazitätsgewinn des Agenten (+26 % Durchsatz, +18 Prozentpunkte Abschlussquote) ist daher kein Artefakt einer künstlich schwachen Baseline, sondern Ausdruck einer besseren Ausnutzung eines knappen Engpasses.

### Warum ist der Durchsatz-KPI robuster als die Wartezeit?

Die tripinfo-Mittelwerte (Wartezeit, Reisezeit, Zeitverlust **pro Fahrzeug**) sind **selektionsverzerrt**, weil SUMO `tripinfo` nur **abgeschlossene** Fahrten protokolliert. Die Festzeit-Baseline lässt ~124 Fahrzeuge mehr im Stau stehen -- gerade die am stärksten betroffenen. Diese fehlen in ihren Mittelwerten. Das Verräterindiz ist die absurd kleine Streuung der Festzeit-Wartezeit/Fz (±0,32 s) trotz einer Per-Step-Streuung von ±262 s: Die Baseline „berichtet" faktisch nur über die Fahrzeuge, die es durch den Knoten geschafft haben. Folge: Die per-Fahrzeug-Kennzahlen **unterschätzen** den wahren Vorteil des Agenten.

Der **Durchsatz** unterliegt dieser Verzerrung nicht: Er zählt schlicht, wie viele der 681 geladenen Trips abgeschlossen wurden -- die im Stau steckenden Fahrzeuge werden als nicht-abgeschlossen korrekt erfasst. Er ist damit der **unverzerrte Leitindikator**. Der Zeitverlust pro Fahrzeug (-18,3 %) ist ein sinnvoller Mittelweg: Er ist geringer verzerrt als die reine Wartezeit, weil er auch verzögerte (nicht nur völlig staufreie) Trips bewertet, bleibt aber als tripinfo-Größe mit demselben Vorbehalt behaftet.

### Was hat der Agent strukturell gelernt?

Die Analyse der gelernten Schaltentscheidungen (Abbildung 9, Kapitel 7) zeigt eine plausible Strategie: Der Agent verlagert Grünzeit auf die **B317-Hauptachse** (47 % → 61 % Grünzeitanteil), **verwirft die schwach nachgefragte Nebenphase** (`GGGGrrrrrr`) fast vollständig und schaltet die Querstraße in **kürzeren, reaktiveren** Grünphasen (Ø 37 s → 25 s). Das entspricht intuitiv dem, was ein Verkehrsingenieur tun würde -- mit dem Unterschied, dass der Agent es selbst aus der Belohnung gelernt hat und je nach Verkehrslage variieren kann.

## Limitationen

Trotz der methodischen Sorgfalt unterliegt die Arbeit wesentlichen Einschränkungen, die bei der Interpretation berücksichtigt werden müssen:

1. **Un-tuned Festzeit-Baseline.** Verglichen wurde gegen das aus OSM importierte **Default-Signalprogramm** (90 s Zyklus), **nicht** gegen einen hand-optimierten Webster-Plan oder eine verkehrsabhängige (`actuated`) Steuerung. Der Vergleich ist realistisch (so läuft die Kreuzung „ab Werk"), aber er beweist nicht die Überlegenheit gegenüber dem Stand der Praxis. Dies ist die wichtigste offene Flanke (siehe „Fairness der Baseline").

2. **Nur eine Lastfall-Stufe trainiert/evaluiert.** Trainiert und evaluiert wurde ausschließlich auf der konstanten **medium-Last**. Es gibt **keinen Tagesgang** innerhalb einer Episode; `low`/`high`-Last existieren als Routendateien, wurden aber nicht ausgewertet. Aussagen zur Lastsensitivität des gelernten Verhaltens sind daher nicht belegt.

3. **Zufällige Abbiegeströme.** Die Nachfrage wurde mit `randomTrips` erzeugt; die OD-Relationen (Richtungsanteile) sind zufällig, nicht erhoben. Die Verkehrs**menge** ist auf den realen DTV (8.037 Kfz/24h) kalibriert, die **Verteilung** auf die einzelnen Abbiegebeziehungen jedoch nicht.

4. **Single-Intersection-Optimierung.** Die Beschränkung auf einen Knoten schließt Netzwerkeffekte -- insbesondere Grüne-Welle-Koordination benachbarter Ampeln und Spillback in vorgelagerte Knoten -- aus.

5. **Ein Algorithmus / ein Reward auf dem OSM-Netz.** Auf der realen Geometrie wurde nur DQN mit `diff-waiting-time` über 3 M Steps trainiert. PPO und der durchsatzorientierte `pressure`-Reward liegen nur als Methodenbeweis auf dem synthetischen Netz vor.

6. **Eingeschränkte Hyperparameter-Suche.** Aus Ressourcengründen (Solo-Projekt) wurde kein vollständiger Grid Search durchgeführt; die meisten Parameter verblieben bei den SB3-Defaults (siehe `settings.py`).

7. **Keine Emissionsanalyse.** Die SUMO/HBEFA-Emissionsmodelle (CO₂, NOₓ, Feinstaub) wurden nicht ausgewertet.

## Fairness der Baseline

Ehrlichkeit gebietet eine differenzierte Aussage, weil zwei Netze mit **unterschiedlichen Baselines** verwendet wurden:

- **Synthetisches 4-Arm-Netz (Methodenbeweis):** Hier ist die Baseline eine **Webster-proportionale Festzeitsteuerung** mit 90 s Zykluszeit im Normbereich des HBS 2015 -- bewusst keine künstlich schwache Vergleichsgrundlage.
- **Echtes OSM-Netz (Fidelitäts-Validierung):** Hier ist die Baseline das **un-tuned Default-Programm** des importierten Netzes. Es ist realistisch (so wäre die Kreuzung ohne weitere Optimierung geschaltet), aber **nicht** der Stand der Praxis.

Der **eigentliche Maßstab** für den Mehrwert adaptiver Steuerung wäre eine **`actuated`-Steuerung** (verkehrsabhängige Verlängerung von Grünphasen über Detektoren), die in SUMO nativ verfügbar ist und in der Praxis den Standard darstellt. Ein fairer Drei-Wege-Vergleich (Festzeit vs. Actuated vs. RL) auf dem OSM-Netz würde einordnen, wie viel des +26-%-Durchsatzgewinns auf „Adaptivität an sich" entfällt und wie viel echter Vorsprung des gelernten Agenten gegenüber einer einfachen regelbasierten Adaptivität ist. Dieser Vergleich ist die naheliegendste Folgearbeit (Kapitel 9).

## Übertragbarkeit

Die allgemeine Methodik -- MDP-Formulierung mit Phase-Selection, Gymnasium-kompatible Environment-Abstraktion, standardisierte Evaluation mit multiplen Seeds und statistischen Tests, tripinfo-genaue KPIs -- ist direkt auf andere Kreuzungsgeometrien übertragbar. Die modulare Architektur erlaubt den Austausch des SUMO-Netzwerks ohne Änderungen am Trainings- oder Evaluationscode; genau das wurde beim Wechsel vom synthetischen zum OSM-Netz demonstriert.

Nicht übertragbar sind hingegen die trainierten Modellgewichte: Ein auf dieser Kreuzung trainierter Agent kann nicht ohne Transfer Learning auf eine andere Topologie, Phasenzahl oder Demand-Struktur angewendet werden. Ebenso ist die Beschränkung auf einen Knoten ein fundamentales Limit -- netzweite Koordination erfordert Multi-Agent-RL (z.B. MAPPO, QMIX) mit grundlegend anderen Kommunikationsstrukturen.
