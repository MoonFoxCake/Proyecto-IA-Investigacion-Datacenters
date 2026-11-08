# Decisiones técnicas — Etapa 2

## A. Datos y partición

| # | Decisión | Elegimos | Alternativa descartada | Por qué (evidencia) |
|---|---|---|---|---|
| 1 | Partición | `GroupShuffleSplit` por `state`, 85/15, semilla 2053. Train: 7 221 plantas (41 estados). Test: 1 182 plantas (8 estados: IA, LA, ME, MN, NE, OH, RI, TN). Índices en `splits.json` | Partición aleatoria por planta | Una partición aleatoria pone plantas del mismo estado en train y test e infla las métricas. Proporción 85/15: **completar** (con 49 estados, 15 % deja 8 en test, y todos los grupos grandes de combustible quedan representados). Nota: el contexto inicial decía 80/20; el notebook y el artículo usan 85/15 |
| 2 | Composición del test | Se mantiene sin mover la semilla. Geotérmica queda con 0 plantas en test; nuclear (10), OTHF (11) y OFSL (3) son muy pocas y no se reportan por separado | Cambiar la semilla | La semilla se fijó antes de ver métricas de modelos. Cambiarla después sería ajustar con el test |
| 3 | Test intacto | No se usa hasta el modelo final, una sola vez | Mirar el test durante el ajuste | Evita data leakage (−15 pts) |
| 4 | Variables excluidas por fuga | `Water Consumption (m3)`, `CO2-e Emission (tons)`, `Carbon intensity` y las emisiones de eGRID | Usarlas como features | **Completar:** el consumo de agua contiene el target por construcción (intensidad = consumo / generación), y las emisiones no estarían disponibles al predecir una planta nueva |
| 5 | Identificadores excluidos | `plant_id`, `huc8` y nombres | Usarlos como features | Memorizarían filas o subcuencas (~1 400 categorías); `cf_aware` y `bws_score` los reemplazan |
| 6 | Transformación del target | `log1p` | `log` simple | 551 ceros en `y_train`, mínimo positivo 0,0035. `log(y+ε)` haría depender la distancia entre 0 y eólica de un ε arbitrario |

## B. Features

| # | Decisión | Elegimos | Alternativa descartada | Por qué (evidencia) |
|---|---|---|---|---|
| 7 | Conjuntos de features | **A** (17): sin `state`, `ba_code`, `pca_generation_mwh`, `generation_ratio`. **B** (21): A más esos cuatro. **C** (15): A sin `lat`/`lon` | Usar las 21 directamente | El 99,3 % de las plantas de test está en regiones que también aparecen en train (8 de 10 `ba_code` compartidos). Separar por estado no aísla regiones |
| 8 | Regla de elección | **B solo se reporta como ablación y nunca reemplaza a A**, porque el CV por estado no aísla regiones y su mejora no distinguiría aprendizaje de memorización regional. **C reemplaza a A solo si mejora el R² de CV en ≥ 0,03 y en 4 de 5 folds.** Si no, queda A | Elegir mirando el resultado | Regla fijada antes de ejecutar la celda 9. **Completar:** fecha y hora del commit de esta fila. Si la versión original de la regla (que permitía a B reemplazar a A) llegó a comitearse, déjenla en el historial y anótenlo en el registro de cambios al final |
| 9 | Resultado de la regla (celda 9) | **Queda A.** C vs A: ΔR² = +0,012 y mejor en 3 de 5 folds, no cumple. B vs A: mejor en 5 de 5 folds, pero solo en hidro (MAE hidro 1,624 vs 1,679; MAE sin hidro 0,116 vs 0,094). Se reporta como ablación | — | `xgb_feature_sets_cv.csv`. R² por fold, A: 0,105 / 0,354 / 0,402 / 0,144 / 0,393. El R² por fold va de 0,10 a 0,40, así que diferencias de 0,01 a 0,05 están dentro del ruido. Dato para el artículo: sin hidro, C es mejor que A (MAE 0,072 vs 0,094) y en hidro es peor (1,732 vs 1,679). Hipótesis sin verificar: las coordenadas ayudan a adivinar los ceros de hidro y meten ruido al resto. SHAP de B debería mostrar `ba_code` o `pca_generation_mwh` entre las más importantes |

## C. Baselines y métricas

| # | Decisión | Elegimos | Alternativa descartada | Por qué (evidencia) |
|---|---|---|---|---|
| 10 | Baselines | **B1** media (trivial), **B2** Ridge con features A (simple), **B3** mediana por `fuel` con respaldo a `fuel_group` y luego a la mediana global (literatura). CV agrupada por estado, 5 folds, solo train | — | `baselines_cv.csv`. R² log1p global: −0,010 / 0,339 / 0,070. Sin hidro: −0,233 / 0,744 / 0,854 |
| 11 | Métricas | MAE, RMSE y R² en log1p: global, subconjunto no trivial (sin solar ni eólica), **sin hidro** y por grupo (n ≥ 50). Además MAE en m³/MWh (con `expm1`) | Solo R² global | Hidro concentra el 97,4 % del error cuadrático de B3, así que el R² global es casi un R² de hidro |
| 12 | Criterio de éxito | R² ≥ 0,70 en log1p con partición espacial, definido en la Etapa 1 y **sin cambios**. Se reporta si se cumple o no, descompuesto en hidro y no hidro | Bajar el umbral después de ver resultados | En CV de train ningún modelo supera 0,36 global. Sin hidro, B3 ya llega a 0,854. Resultado final en test: **completar al evaluar** |
| 13 | Hidro | Se reporta aparte y se declara como limitación. Un modelo en dos partes (clasificar "cero o no" y luego regresar) queda como mejora opcional | Ignorarlo | 45,2 % de ceros en train (551 de 1 219). Correlaciones de Spearman con las features ≤ 0,30 (`lon` 0,296). El % de ceros varía por estado (UT 93 %, VT 7 %) y por región (BPAT 71 %, ISNE 15 %), lo que sugiere un origen en cómo se construyó el dato. Es hipótesis, no verificada |

## D. Arquitectura del modelo

| # | Decisión | Elegimos | Alternativa descartada | Por qué (evidencia) |
|---|---|---|---|---|
| 14 | XGBoost directo vs híbrido | Se compara XGBoost directo contra un híbrido (mediana por `fuel` + XGBoost sobre el residuo). Ambos con features A y los mismos folds. **El híbrido reemplaza al directo si baja el MAE sin hidro en 4 de 5 folds y el R² global no empeora más de 0,01.** Si no, queda el directo | Elegir mirando el resultado | Con hiperparámetros fijos, XGB A no le gana a B3 sin hidro (MAE 0,094 vs 0,027; R² 0,775 vs 0,854) y en hidro casi no aprende (MAE 1,679 contra 1,733 de la media). Hipótesis: reconstruir la tabla por combustible desde 33 columnas one-hot es difícil para el árbol. Regla fijada antes de ejecutar la celda 10. **Completar:** fecha y hora del commit |
| 15 | Optimización de hiperparámetros | Optuna con GroupKFold por estado sobre el modelo elegido en la fila 14. **Completar:** espacio de búsqueda, número de trials, costo y justificación | Ajuste manual sin registro | La rúbrica exige búsqueda sistemática con espacio y costo justificados. Se guardarán `best_params.json` y `trials.csv` |

## E. Limitaciones a declarar

1. Solo WUE_off se aprende; PUE y WUE_on son parámetros y WSF es una consulta.
2. PUE por tipo de data center, no por instalación.
3. Recomendación por región (2018); las coordenadas se asignan de forma aproximada.
4. WUE_off no descuenta importaciones entre regiones (Table 3).
5. Las subcuencas compartidas cuentan en ambas regiones.
6. Las represas con intensidad 0 sesgan el ranking.
7. Solo 56 de 69 regiones tienen todas las piezas del SCARF.
8. Hidro: casi la mitad de los valores son 0 y dependen del lugar. El modelo aprende en parte la geografía del dato de Siddik, no física independiente.
9. El 84,4 % de las plantas de train repite exactamente el valor típico de su combustible (Siddik usó promedios nacionales), así que gran parte del target es una tabla de consulta.
10. La partición por estado reduce la dependencia espacial pero no la elimina: `lat`, `lon` y las regiones compartidas dejan memorizar vecindad.

## F. Registro de cambios

| Fecha | Cambio | Motivo |
|---|---|---|
| **completar** | Fila 8 corregida: B pasa a ser solo ablación | Con B como candidato, la regla favorecía la memorización regional. Corregida **completar: antes / después** de ejecutar la celda 9 |
| **completar** | Se agregan filas 14 y 15 | El XGBoost con parámetros fijos no supera a los baselines sin hidro |
