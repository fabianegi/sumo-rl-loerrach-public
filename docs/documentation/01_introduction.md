# 1. Einleitung

## 1.1 Motivation

Verkehrsbedingte Wartezeiten an Lichtsignalanlagen (LSA) verursachen in deutschen Innenstädten erhebliche volkswirtschaftliche Kosten. Jedes stehende Fahrzeug produziert unnötige CO₂-Emissionen, erhöht die Feinstaubbelastung und senkt die Lebensqualität der Anwohner. Gleichzeitig operieren die meisten LSA in kleineren und mittelgroßen Städten mit statischen Festzeitschaltungen, die auf historischen Verkehrszählungen basieren und nicht auf Nachfrageschwankungen im Tagesverlauf reagieren.

Adaptive Signalsteuerungssysteme wie SCOOT oder SCATS existieren zwar, erfordern jedoch teure Sensorinfrastruktur und aufwändige Kalibrierung. Reinforcement Learning (RL) bietet eine Alternative: Ein Agent lernt durch Interaktion mit einer Simulationsumgebung eine Steuerungsstrategie, die den Verkehrsfluss optimiert -- ohne explizite Regelmodellierung. Aktuelle Forschungsarbeiten wie PressLight (Wei et al., 2019) und LIT (Zheng et al., 2019) zeigen, dass RL-basierte Ansätze konventionelle Steuerungen auf großen Netzwerken übertreffen können. Ob dieser Vorteil auch an einer einzelnen, realitätsnah kalibrierten Kreuzung einer deutschen Mittelstadt reproduzierbar ist, wurde bisher kaum untersucht.

## 1.2 Forschungsfrage

Die zentrale Forschungsfrage lautet: **Kann ein RL-Agent eine optimierte Festzeitschaltung an einer realen Kreuzung übertreffen?**

Konkret wird untersucht:

1. **DQN vs. Festzeit:** Erzielt ein Deep Q-Network signifikant niedrigere Wartezeiten als eine nach Webster-Verfahren optimierte Festzeitschaltung?
2. **DQN vs. PPO:** Wie unterscheiden sich DQN und PPO hinsichtlich Sample-Effizienz, Konvergenzverhalten und finaler Performance?
3. **Reward-Design:** Führt eine PressLight-inspirierte Pressure-Reward-Funktion zu besseren Ergebnissen als die Standard-Diff-Waiting-Time-Metrik?
4. **Robustheit:** Wie stabil sind die Ergebnisse über verschiedene Nachfrageszenarien und Random Seeds?

## 1.3 Beitrag

Dieses Projekt leistet drei spezifische Beiträge:

**Realitätsnahe Kalibrierung.** Die Simulationsumgebung basiert nicht auf synthetischen Standardnetzen aus der Literatur, sondern auf einer konkreten Kreuzung in Lörrach. Die Verkehrsnachfrage wird aus dem DTV-Wert der kommunalen Zählstelle Lörrach-Stetten (8.037 Kfz/24h, 2024) abgeleitet und über eine HBS-2015-Tagesganglinie auf Stundenwerte disaggregiert. Drei Nachfrageszenarien (niedrig, mittel, hoch) decken den realistischen Schwankungsbereich ab.

**Methodenvergleich.** DQN und PPO werden unter identischen Bedingungen trainiert und evaluiert. Zwei Reward-Funktionen (diff-waiting-time, pressure) erlauben einen systematischen Vergleich der Auswirkung des Reward-Designs auf das gelernte Verhalten.

**Statistische Validierung.** Die Evaluation folgt einem strengen Protokoll: N=30 Seeds für die Hauptergebnisse (OSM-Headline und reales Netz; der synthetische Methodenbeweis nutzt 10 Seeds), vier KPIs (durchschnittliche Wartezeit, maximale Warteschlangenlänge, Durchsatz, mittlere Geschwindigkeit), Mann-Whitney-U-Tests für statistische Signifikanz und Effektstärken. Das Ziel ist reproduzierbare, quantitativ belastbare Aussagen statt anekdotischer Einzelergebnisse.

## 1.4 Abgrenzung

Die Arbeit beschränkt sich bewusst auf eine einzelne Kreuzung (Single-Agent). Multi-Agent-Koordination über mehrere Kreuzungen wird in der Diskussion als Future Work eingeordnet (vgl. ADR #13). Ebenso liegt kein Deployment auf realer Hardware im Scope -- die Ergebnisse sind rein simulationsbasiert.

## 1.5 Aufbau der Arbeit

Abschnitt 2 ordnet die Arbeit in die bestehende Literatur zu RL-basierter Signalsteuerung ein. Abschnitt 3 beschreibt die Datengrundlage und Simulationsumgebung. Abschnitt 4 spezifiziert das RL-Design (State, Action, Reward, Episode). Abschnitt 5 dokumentiert das experimentelle Setup und die Trainingsprozedur. Abschnitt 6 präsentiert die Ergebnisse. Abschnitt 7 diskutiert die Befunde, Limitationen und Implikationen. Abschnitt 8 fasst die Erkenntnisse zusammen und gibt einen Ausblick auf weiterführende Arbeiten.
