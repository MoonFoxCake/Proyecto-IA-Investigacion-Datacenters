# Decisiones técnicas — Etapa 2

| # | Decisión | Elegimos | Alternativa descartada | Por qué (evidencia) |
|---|---|---|---|---|
| 1 | Partición | `GroupShuffleSplit` por `state`, 85/15, semilla 2053. Train 7 221 plantas (41 estados), test 1 182 (8 estados: IA, LA, ME, MN, NE, OH, RI, TN). Índices en `splits.json` | Partición aleatoria por planta | Una aleatoria pone plantas del mismo estado en train y test e infla las métricas. Proporción 85/15: **completar con su razonamiento** (con 49 estados, 15 % deja 8 en test) |
| 2 | Test intacto | No se usa hasta el modelo final, una sola vez | Mirar test durante el ajuste | Evita data leakage (−15 pts) |
| 3 | Variables excluidas por fuga | `Water Consumption (m3)`, `CO2-e Emission`, `Carbon intensity` y emisiones de eGRID | Usarlas como features | **Completar:** por qué no estarían disponibles o contienen el target al predecir una planta nueva |
| 4 | Identificadores excluidos | `plant_id`, `huc8` y nombres | Usarlos como features | Memorizarían filas o subcuencas (~1 400 categorías) |
| 5 | Transformación del target | `log1p` | `log` simple | 551 ceros en `y_train`; mínimo positivo 0,0035 |
| 6 | Conjuntos de features | **A** (17, sin `state`, `ba_code`, `pca_generation_mwh`, `generation_ratio`) principal; **B** (21) y **C** (A sin `lat`/`lon`) como ablaciones | Usar las 21 directamente | El 99,3 % de las plantas de test está en regiones que ya aparecen en train |
| 7 | Regla de elección A/B/C | B o C reemplazan a A solo si mejoran el R² de CV en ≥ 0,03 **y** en 4 de 5 folds. Si no, queda A | Elegir mirando el resultado | Regla fijada **antes** de ejecutar la celda 9. Anotar fecha y hora de este commit |
| 8 | Baselines | B1 media, B2 Ridge, B3 mediana por `fuel` (respaldo `fuel_group`, luego global). CV agrupada por estado, 5 folds, solo train | — | Resultados en `baselines_cv.csv` (R² log1p: −0,010 / 0,339 / 0,070) |
| 9 | Métricas | MAE, RMSE y R² en log1p: global, subconjunto no trivial (sin solar/eólica), sin hidro y por grupo (n ≥ 50). Además MAE en m³/MWh | Solo R² global | Hidro concentra el 97,4 % del error de B3 |
| 10 | Criterio de éxito | R² ≥ 0,70 en log1p con partición espacial (definido en la Etapa 1, sin cambios) | — | Declarado antes de ver resultados |
| 11 | Hidro | Se reporta aparte y se declara como limitación | Ignorarlo | 45,2 % de ceros; correlaciones con features ≤ 0,30 |
