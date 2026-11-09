# Decisiones técnicas — Etapa 2

Proyecto: Location Matters (IA, TEC, Track A). Cada decisión crítica lleva su alternativa descartada y la evidencia que la respalda.

## A. Datos y partición

| # | Decisión | Elegimos | Alternativa descartada | Por qué (evidencia) |
|---|---|---|---|---|
| 1 | Partición | `GroupShuffleSplit` por `state`, 85/15, semilla 2053. Train: 7 221 plantas (41 estados). Test: 1 182 plantas (8 estados: IA, LA, ME, MN, NE, OH, RI, TN). Índices en `splits.json` | Partición aleatoria por planta | Una partición aleatoria pone plantas del mismo estado en train y test e infla las métricas. Es espacial y no cronológica: todos los datos son de 2018 y la pregunta es predecir en estados no vistos. Proporción 85/15: **completar con su razonamiento** (con 49 estados, 15 % deja 8 en test, y todos los grupos grandes de combustible quedan representados). Nota: el contexto inicial decía 80/20; el notebook y el artículo usan 85/15 |
| 2 | Composición del test | Se mantiene sin mover la semilla. Geotérmica queda con 0 plantas en test; nuclear (10), OTHF (11) y OFSL (3) son muy pocas y no se reportan por separado. Hidro es 9-10 % del test contra 17 % del train | Cambiar la semilla | La semilla se fijó antes de ver métricas de modelos. Cambiarla después sería ajustar con el test |
| 3 | Test intacto | No se usa hasta el modelo final, una sola vez. Antes solo se contaron plantas por combustible y estado en test (sin mirar `y` ni métricas) | Mirar el test durante el ajuste | Evita data leakage (−15 pts) |
| 4 | Variables excluidas por fuga | `Water Consumption (m3)`, `CO2-e Emission (tons)`, `Carbon intensity` y las emisiones de eGRID | Usarlas como features | **Completar:** el consumo de agua contiene el target por construcción (intensidad = consumo / generación), y las emisiones no estarían disponibles al predecir una planta nueva |
| 5 | Identificadores excluidos | `plant_id`, `huc8` y nombres | Usarlos como features | Memorizarían filas o subcuencas (~1 400 categorías); `cf_aware` y `bws_score` los reemplazan |
| 6 | Transformación del target | `log1p` | `log` simple | 551 ceros en `y_train`, mínimo positivo 0,0035. `log(y+ε)` haría depender la distancia entre 0 y eólica de un ε arbitrario |

## B. Features

| # | Decisión | Elegimos | Alternativa descartada | Por qué (evidencia) |
|---|---|---|---|---|
| 7 | Conjuntos de features | **A** (17): sin `state`, `ba_code`, `pca_generation_mwh`, `generation_ratio`. **B** (21): A más esos cuatro. **C** (15): A sin `lat`/`lon` | Usar las 21 directamente | El 99,3 % de las plantas de test está en regiones que también aparecen en train (8 de 10 `ba_code` compartidos). Separar por estado no aísla regiones |
| 8 | Regla de elección | **B solo se reporta como ablación y nunca reemplaza a A**, porque el CV por estado no aísla regiones y su mejora no distinguiría aprendizaje de memorización regional. **C reemplaza a A solo si mejora el R² de CV en ≥ 0,03 y en 4 de 5 folds.** Si no, queda A | Elegir mirando el resultado | Regla fijada antes de ejecutar la celda 9. **Completar:** fecha y hora del commit de esta fila |
| 9 | Resultado de la regla (celda 9) | **Queda A.** C vs A: ΔR² = +0,012 y mejor en 3 de 5 folds, no cumple. B vs A: mejor en 5 de 5 folds, pero solo en hidro (MAE hidro 1,624 vs 1,679; MAE sin hidro 0,116 vs 0,094). Se reporta como ablación | — | `xgb_feature_sets_cv.csv`. R² por fold, A: 0,105 / 0,354 / 0,402 / 0,144 / 0,393. El R² por fold va de 0,10 a 0,40, así que diferencias de 0,01 a 0,05 están dentro del ruido. Dato para el artículo: sin hidro, C es mejor que A (MAE 0,072 vs 0,094) y en hidro es peor (1,732 vs 1,679). Hipótesis sin verificar: las coordenadas ayudan a adivinar los ceros de hidro y meten ruido al resto. SHAP de B debería mostrar `ba_code` o `pca_generation_mwh` entre las más importantes |

## C. Baselines y métricas

| # | Decisión | Elegimos | Alternativa descartada | Por qué (evidencia) |
|---|---|---|---|---|
| 10 | Baselines | **B1** media (trivial), **B2** Ridge con features A (simple), **B3** mediana por `fuel` con respaldo a `fuel_group` y luego a la mediana global (literatura). CV agrupada por estado, 5 folds, solo train | — | `baselines_cv.csv`. R² log1p global: −0,010 / 0,339 / 0,070. Sin hidro: −0,233 / 0,744 / 0,854 |
| 11 | Métricas | MAE, RMSE y R² en log1p: global, subconjunto no trivial (sin solar ni eólica), **sin hidro**, **solo hidro** y por grupo (n ≥ 50). Además MAE y mediana del error absoluto en m³/MWh (con `expm1`) | Solo R² global | Hidro concentra el 97,4 % del error cuadrático de B3, así que el R² global es casi un R² de hidro |
| 12 | Criterio de éxito | R² ≥ 0,70 en log1p con partición espacial, definido en la Etapa 1 y **sin cambios**. Se reporta si se cumple o no, descompuesto en hidro y no hidro | Bajar el umbral después de ver resultados | En CV de train el mejor modelo llega a R² = 0,408. Sin hidro, B3 ya llega a 0,854. Resultado final en test: **completar al evaluar (celda 14)** |
| 13 | Hidro | Se reporta aparte y se declara como limitación. Un modelo en dos partes (clasificar "cero o no" y luego regresar) queda como mejora opcional | Ignorarlo | 45,2 % de ceros en train (551 de 1 219). Correlaciones de Spearman con las features ≤ 0,30 (`lon` 0,296). El % de ceros varía por estado (UT 93 %, VT 7 %) y por región (BPAT 71 %, ISNE 15 %), lo que sugiere un origen en cómo se construyó el dato. Es hipótesis, no verificada |

## D. Arquitectura y optimización

| # | Decisión | Elegimos | Alternativa descartada | Por qué (evidencia) |
|---|---|---|---|---|
| 14 | XGBoost directo vs híbrido | Se compara XGBoost directo contra un híbrido (mediana por `fuel` + XGBoost sobre el residuo). Ambos con features A y los mismos folds. **El híbrido reemplaza al directo si baja el MAE sin hidro en 4 de 5 folds y el R² global no empeora más de 0,01.** Si no, queda el directo. Regla fijada antes de ejecutar la celda 10. **Completar:** fecha y hora del commit | Elegir mirando el resultado | **Resultado (celda 10):** el híbrido baja el MAE sin hidro en 2 de 5 folds; ΔR² = −0,006 → **no cumple, queda el directo.** MAE sin hidro 0,094 en ambos, contra 0,027 de B3 (`hibrido_cv.csv`). Con parámetros fijos, XGB A no le gana a B3 sin hidro (R² 0,775 vs 0,854). La hipótesis inicial (al árbol le cuesta reconstruir la tabla por combustible) no quedó respaldada: el híbrido recibe esa tabla gratis y aun así termina igual. Hipótesis nueva, sin verificar: el árbol, guiado por los residuos enormes de hidro, mete ruido en las plantas donde el combustible ya lo explica todo |
| 15 | Modelo ruteado (decisión **posterior** a ver el resultado de la fila 14) | Mediana por `fuel` para plantas no hidro y XGBoost entrenado solo con represas. **Reemplaza al directo si baja el MAE sin hidro en 5 de 5 folds, el MAE de hidro no empeora más de 0,05 respecto al directo (1,679) y el R² global no empeora más de 0,01.** Regla fijada antes de ejecutar la celda 11. **Completar:** fecha y hora del commit | Quedarse con el XGBoost directo | **Resultado (celda 11):** MAE sin hidro menor en 5 de 5 folds; ΔMAE hidro = +0,043; ΔR² = −0,004 → **se cumple, queda el ruteado** (`ruteado_cv.csv`). Aclaración: fuera de hidro el ruteado es exactamente B3 (MAE 0,027), así que ahí gana por construcción y no aprende. En grupos no hidro XGB directo pierde contra B3 (biomasa 0,093 vs 0,007; petróleo 0,096 vs 0,020; solar 0,063 vs 0,000). Esta es la tercera comparación de arquitectura sobre los mismos folds, y es exploratoria |
| 16 | Optimización de hiperparámetros (solo el modelo de hidro) | Optuna (muestreador TPE, semilla 2053), 50 trials, minimizando el RMSE medio en log1p sobre las represas de validación, con los mismos 5 folds agrupados por estado. Espacio: `n_estimators` 100–600, `learning_rate` 0,01–0,2 (log), `max_depth` 2–6, `min_child_weight` 1–20, `subsample` 0,5–1, `colsample_bytree` 0,5–1, `reg_lambda` 0,001–30 (log), `reg_alpha` 0,001–10 (log). Profundidad acotada a 6 porque hay solo 1 219 represas en train. El trial 0 parte de los parámetros anteriores. **Regla:** los ajustados reemplazan a los anteriores solo si el RMSE de hidro baja ≥ 0,03 y mejora en ≥ 3 de 5 folds. Regla fijada antes de ejecutar la celda 12. **Completar:** fecha y hora del commit | Ajuste manual sin registro; optimizar también la parte no hidro | La parte no hidro es una tabla de consulta sin nada que ajustar. **Resultado (celda 12):** RMSE de hidro: solo la media 2,227; parámetros anteriores 2,275; ajustados 2,099 (mejora +0,176; mejor en 5 de 5 folds) → **se cumple, quedan los ajustados** (`best_params.json`, `optuna_trials.csv`). Costo: 1,7 min. Parámetros: `n_estimators` 232, `learning_rate` 0,012, `max_depth` 5, `min_child_weight` 8, `subsample` 0,545, `colsample_bytree` 0,930, `reg_lambda` 12,4, `reg_alpha` 0,18. Los anteriores eran peores que predecir la media de hidro; Optuna los corrigió, pero la ganancia sobre la media es de ~6 % en RMSE. Un modelo muy regularizado (tasa baja, λ alto) es consistente con poca señal en hidro (hipótesis). El score es optimista (mínimo de 50 intentos sobre los mismos folds), y `learning_rate` y `subsample` quedaron cerca del borde del espacio |
| 17 | Modelo final y protocolo del test | Modelo final: ruteado (mediana por `fuel` fuera de hidro; XGBoost con los parámetros de la fila 16 en hidro), features A, entrenado con todo train (`models/modelo_final.joblib`). Se evalúa en test **una sola vez**, junto con los tres baselines reentrenados con todo train. Se reporta: MAE, RMSE y R² en log1p (global, sin hidro, solo hidro y por grupo con n ≥ 50), MAE y mediana del error en m³/MWh y MAE por estado de test. Criterio de éxito sin cambios (fila 12). Después de ver el test no se cambian modelo, parámetros, semilla ni partición. **Completar:** fecha y hora del commit, que debe ser anterior a ejecutar la celda 14 | Seguir ajustando mirando el test | R² de CV del modelo final = 0,408 (RMSE 0,858; MAE 0,297; MAE sin hidro 0,027; MAE hidro 1,626), contra Ridge 0,339 y mediana por combustible 0,070 (`cv_final_vs_baselines.csv`). Está algo inflado por el sesgo de selección de Optuna. Se espera que el R² ≥ 0,70 no se cumpla, y se reporta así. Resultado en test: **completar al evaluar** |

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
11. Fuera de hidro el modelo final es el baseline de literatura. Todo el aporte del aprendizaje automático está en hidro, y ahí es modesto (~6 % menos de RMSE que predecir la media).
12. El R² de CV del modelo final (0,408) es optimista: los parámetros se eligieron entre 50 intentos sobre los mismos folds, y las decisiones de features y arquitectura también usaron esos folds. El test es la única medida independiente.
13. El test tiene 8 estados y ~10 % de hidro contra 17 % en train; el resultado depende de qué estados cayeron en test.
14. El espacio de Optuna no se amplió después de ver que `learning_rate` y `subsample` quedaron cerca del borde.
15. `net_generation_mwh` es el denominador del target y `capacity_factor` se deriva de ella. El agente las conoce al consultar una planta, pero están ligadas al target por construcción.

## F. Registro de cambios

| Fecha | Cambio | Motivo |
|---|---|---|
| 7-11-26 | Fila 8 corregida: B pasa a ser solo ablación | Con B como candidato, la regla favorecía la memorización regional. Corregida **completar: antes / después** de ejecutar la celda 9 |
| 8-11-26 | Se agrega la fila 14 (directo vs híbrido) | El XGBoost con parámetros fijos no supera a los baselines sin hidro |
| 9-11-26 | Se agrega la fila 15 (ruteado), decisión posterior a ver la fila 14 | El híbrido no cumplió la regla y el MAE sin hidro quedaba en 0,094 |
| 9-11-26 | La antigua fila 15 (Optuna) pasa a ser la 16, con espacio, costo y resultado; se agrega la fila 17 | Se definió Optuna solo sobre el modelo de hidro, y el protocolo del test se fija antes de evaluar |
