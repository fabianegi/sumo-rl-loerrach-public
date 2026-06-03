## KPI-Vergleich - OSM-Netz (medium), DQN 3M Steps

Festzeit-Baseline (N=30) vs. DQN-Agent (N=30), Mann-Whitney U (zweiseitig, alpha=0,05), r = rank-biserial.

| KPI | Typ | Festzeit | DQN | Δ rel. | p-Wert | r | sig. |
|---|---|---|---|---|---|---|---|
| Ø Wartezeit [Per-Step-System] | per-step | 289 ± 262 | 93.87 ± 172 | -67.5 % | 7.04e-07 | -0.75 | ja |
| Max. Rückstau [Fz] | per-step | 141 ± 109 | 58.03 ± 91.15 | -59.0 % | 1.68e-05 | -0.64 | ja |
| Ø Geschwindigkeit [m/s] | per-step | 4.26 ± 2.54 | 6.57 ± 2.06 | +54.1 % | 2.57e-07 | +0.78 | ja |
| Durchsatz (tripinfo) [Fz/Episode] | tripinfo | 473 ± 165 | 597 ± 128 | +26.2 % | 2.64e-05 | +0.63 | ja |
| Ø Wartezeit/Fz [s] | tripinfo | 5.73 ± 0.323 | 2.54 ± 1.66 | -55.6 % | 4.11e-07 | -0.76 | ja |
| Ø Reisezeit/Fz [s] | tripinfo | 67.74 ± 2.15 | 65.62 ± 1.30 | -3.1 % | 0.0021 | -0.46 | ja |
| Ø Zeitverlust/Fz [s] | tripinfo | 16.88 ± 0.816 | 13.78 ± 1.45 | -18.3 % | 5.53e-08 | -0.82 | ja |

> **Lesehilfe:** *Per-Step*-Größen sind sumo-rl-Systemwerte über alle Fahrzeuge je RL-Schritt - **nicht** Sekunden pro Fahrzeug. Absolut belastbar sind nur die *tripinfo*-Größen (pro abgeschlossenem Fahrzeug). ja = signifikante Verbesserung, ungünstig = signifikant, aber Richtung ungünstig, - = nicht signifikant.
