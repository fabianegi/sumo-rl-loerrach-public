# Fazit und Ausblick

## Zusammenfassung

Die vorliegende Arbeit untersucht die Anwendbarkeit von Reinforcement Learning auf die Lichtsignalsteuerung einer einzelnen Kreuzung im städtischen Kontext von Lörrach. Ausgehend von der Forschungsfrage, ob ein RL-Agent eine konventionelle Festzeitsteuerung hinsichtlich Wartezeit, Rückstau und Durchsatz übertreffen kann, wurde ein vollständiger, reproduzierbarer Workflow entwickelt -- von der Datenkalibrierung über das Environment-Setup und das Training bis zur statistisch abgesicherten Evaluation.

Die belastbaren Ergebnisse stammen vom **echten, aus OpenStreetMap importierten Netz** der Kreuzung Basler Straße × Obere Riehenstraße (TLS-ID `1628110071`), kalibriert auf den realen Tagesverkehr (DTV 8.037 Kfz/24h). Ein **DQN-Agent** mit `diff-waiting-time`-Reward wurde über **3 Mio. Timesteps** (4.166 Episoden, ≈ 5,2 h) trainiert und über **30 Seeds** gegen die importierte Festzeit-Steuerung evaluiert (Mann-Whitney U, rank-biseriale Effektstärke).

**Beantwortung der Forschungsfrage:** Ja -- der DQN-Agent übertrifft die Festzeitschaltung an der realen Kreuzung statistisch signifikant. Die zentralen Kennzahlen:

- **Durchsatz +26,2 %** (p = 2,6·10⁻⁵, r = +0,63) -- unverzerrter Leitindikator; Abschlussquote 69,5 % → 87,7 %
- **Gridlock-Häufigkeit halbiert:** 16/30 → 7/30 Seeds in Quasi-Gridlock
- **Zeitverlust pro Fahrzeug -18,3 %** (p = 5,5·10⁻⁸, r = -0,82), **Rückstau -59,0 %** (p = 1,7·10⁻⁵)
- Alle sieben KPIs signifikant zugunsten des Agenten (|r| = 0,46-0,82)

Methodisch wichtig und transparent dokumentiert: Die per-Fahrzeug-Wartezeit (-55,6 %) ist **selektionsverzerrt** (tripinfo erfasst nur abgeschlossene Trips), weshalb der Durchsatz als Leitindikator berichtet wird. Ergänzend dient das synthetische 4-Arm-Netz als sauberer **Methodenbeweis** (DQN + PPO, `diff-waiting-time` + `pressure`).

Die gesamte Implementierung -- mit durchgängiger Seed-Kontrolle, strukturiertem Logging und automatisierter, tripinfo-genauer Evaluation -- ist als reproduzierbares Python-Projekt realisiert.

## Ausblick

Die identifizierten Limitationen eröffnen konkrete, priorisierte Folgearbeiten:

**1. Faire Baselines (höchste Priorität).** Der aktuelle Vergleich nutzt das un-tuned OSM-Default-Programm. Als Nächstes sollten (a) ein **Webster-optimierter Festzeitplan** und (b) eine **verkehrsabhängige `actuated`-Steuerung** (in SUMO nativ) als Baselines ergänzt werden. Erst der Drei-Wege-Vergleich (Festzeit vs. Actuated vs. RL) ordnet ein, wie viel des Durchsatzgewinns echter Vorsprung des gelernten Agenten gegenüber einfacher regelbasierter Adaptivität ist.

**2. Lastsensitivität.** Die `low`- und `high`-Last-Szenarien (bereits als Routendateien vorhanden) sollten trainiert und evaluiert werden, um zu prüfen, ob der Kapazitätsgewinn lastabhängig ist und ob der Agent unter `high`-Last weiterhin stabilisiert.

**3. Algorithmen- und Reward-Quervergleich.** **PPO** sowie der durchsatzorientierte **`pressure`-Reward** (PressLight, Wei et al. 2019) sollten auch auf dem OSM-Netz trainiert werden, um zu prüfen, ob ein explizit durchsatzorientierter Reward den +26-%-Effekt weiter steigert.

**4. Echte Zähl- und Abbiegedaten.** Die zufällig erzeugten OD-Relationen (`randomTrips`) sollten durch erhobene Richtungsanteile ersetzt werden -- etwa über **MobiData BW** oder eine eigene Verkehrserhebung an der Kreuzung. Das würde die externe Validität der Abbiegeströme erhöhen.

**5. Multi-Agent-Koordination.** Die Erweiterung von Single-Intersection auf netzweite Koordination (MAPPO, QMIX über die PettingZoo-Schnittstelle von sumo-rl) würde Grüne-Welle-Effekte und Spillback in vorgelagerte Knoten erfassen.

**6. Emissions-KPI.** Die SUMO/HBEFA-Emissionsmodelle (CO₂, NOₓ, Feinstaub) könnten als zusätzliche KPI die Arbeit an aktuelle Diskussionen zur urbanen Luftqualität anschlussfähig machen.

## Akademische Einordnung

Diese Arbeit leistet einen Beitrag zur angewandten KI-Forschung im Verkehrsbereich auf dem Niveau eines Semesterprojekts. Sie demonstriert den vollständigen Workflow von der Datenerhebung über die Modellierung und das Training bis zur statistisch abgesicherten Evaluation -- ein Prozess, der in der SOTA-Literatur (Wei et al., 2019; Zheng et al., 2019) für deutlich komplexere Szenarien beschrieben wird, hier aber auf ein einzelnes, nachvollziehbares und vollständig reproduzierbares Szenario heruntergebrochen ist. Der Wert liegt weniger in der Neuartigkeit der Methode als in der sauberen, dokumentierten und **selbstkritischen** Durchführung eines End-to-End-RL-Projekts mit realem Verkehrsbezug -- inklusive offengelegter Selektionsverzerrung und un-getunter Baseline.
