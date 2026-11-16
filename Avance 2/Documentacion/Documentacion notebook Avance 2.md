
## Que es el notebook

La idea del notebook es poder calcular la intensidad hidrica de las plantas de estados unidos junto a las plantas de energia de estados unidos, y lo que se hace es calcular el AWI.


El AWI se calcula como:

```
AWI = (WUE_on + PUE × WUE_off) × WSF      (por MWh de TI)
```

Solo **WUE_off** se aprende con machine learning. PUE y WUE_on son parámetros fijos, y WSF es una consulta a una tabla.

## Librerías principales

- Pandas
- Numpy
- scikit learn
- xgboost
- optuna
- shap
- geopandas
- pyogrio
- openpyxl
- requests

## Estructura

| Contenido                                                         |
| ----------------------------------------------------------------- |
| Construcción del dataset de entrenamiento (una fila = una planta) |
| Tabla SCARF por región (una fila = una región PCA)                |
| Partición, pipeline, baselines y optimización                     |
| Métricas, análisis de errores, sobreajuste y SHAP                 |
| Contribución de cada componente del modelo                        |

## Datos de entrada

Todos los archivos se guardan en `data/raw/` y corresponden al año 2018.

| Archivo                         | Fuente                             |
| ------------------------------- | ---------------------------------- |
| `Input_data.xlsx`               | Siddik et al. (2021)               |
| `6_2_EnviroEquip_Y2018.xlsx`    | EIA-860                            |
| `egrid2018_data_v2.xlsx`        | EPA eGRID                          |
| `Global_Analysis_SI_Final.xlsx` | Masanet et al. (2020), para el PUE |
| Carpeta `aqueduct/`             | WRI Aqueduct 4.0                   |

## Montar el Dataset de entrenamiento

1. **Preparar carpetas y archivos:** descarga los datos y define las rutas del proyecto.
2. **Table 2 (base):** carga 8 403 plantas, renombra columnas y marca con `LEAK_` las columnas que contienen el target o dependen de él.
3. **Table 5:** une `cf_aware` (factor de escasez hídrica) por `huc8_id`.
4. **EIA-860:** agrega el tipo de enfriamiento como banderas 0/1 (`cool_*`).
5. **eGRID:** agrega `capacity_mw`, `capacity_factor` y `n_generadores`.
6. **Aqueduct (opcional):** agrega `bws_score` con un join espacial por coordenadas.
7. **Quitar fuga:** elimina las columnas `LEAK_*` (consumo de agua, CO₂, intensidad de carbono) y define el rol de cada columna.
8. **Revisión:** muestra faltantes y el target por combustible.
9. **Guardar:** escribe los archivos en `data/processed/`.

Todos los joins usan `how="left"` para no perder plantas.

### Archivos generados

| Archivo | Uso |
|---|---|
| `plants_enriched.parquet` | Dataset final |
| `columnas.json` | Roles de las columnas (ID, grupos, features, target) |
| `diccionario_datos.csv` | Diccionario de datos |
| `reporte_cobertura.csv` | Cobertura de cada join |

## Parte B · Tabla SCARF por región

10. **PUE:** se toma de Masanet et al. (2020) según el tipo de data center (interno, colocation o hyperscale).
11. **WUE_off:** promedio de las plantas de la región, ponderado por generación.
12. **WUE_on:** constante de 1,8 m³/MWh (Siddik et al.).
13. **WSF:** promedio de las subcuencas de cada región, ponderado por la energía de data centers.
14. **AWI y ranking:** se calcula el AWI por tipo de data center. Solo 56 de las 69 regiones tienen todas las piezas.
15. **Escenario:** ejemplo de una planta nueva en una región.
16. **Coordenadas a región:** asigna una región a unas coordenadas (por ejemplo Phoenix, Ashburn y Des Moines).

## Modelado

### Partición

- Método: `GroupShuffleSplit`, agrupado por estado (`state`).
- Proporción: 85 % entrenamiento y 15 % prueba, siempre se puede cambiar la semilla
- Se agrupa por estado para evitar fuga espacial, ya que plantas vecinas comparten cuenca, clima y regulación.

### Pipeline y prevención de fuga

- La imputación, el escalado y el one-hot van dentro de un `Pipeline` que se reajusta en cada fold con su train.
- Se excluyen el consumo de agua, las emisiones y los identificadores.
- Se incluyen verificaciones automáticas de que no hay columnas de fuga ni estados compartidos entre train y test.

### Transformación del target

Se usa `log1p`. El target va de 0 a 22 346 m³/MWh y tiene una cola dominada por las hidroeléctricas.

### Baselines

| Baseline | Descripción |
|---|---|
| B1 | Media del target |
| B2 | Regresión Ridge |
| B3 | Mediana por combustible, con respaldo a `fuel_group` y luego a la mediana global |

### Modelo final (ruteado)

- **Plantas no hidroeléctricas:** mediana por combustible (B3).
- **Plantas hidroeléctricas:** XGBoost entrenado solo con hidro.
- Los hiperparámetros se optimizan con **Optuna**, usando validación cruzada agrupada de 5 folds y minimizando el RMSE.

## Resultados en test

Las métricas están en escala `log1p`.

| Modelo | R² | RMSE | MAE | MAE no-hidro | MAE hidro |
|---|---|---|---|---|---|
| B1 media | −0.001 | 1.017 | 0.650 | 0.478 | 2.170 |
| B2 Ridge | 0.490 | 0.726 | 0.276 | 0.093 | 1.900 |
| B3 mediana por fuel | 0.196 | 0.912 | 0.241 | 0.020 | 2.198 |
| **Ruteado final** | **0.523** | **0.702** | **0.203** | **0.020** | **1.816** |


## Estudio de ablación

Se quita o agrega un componente a la vez, con 20 particiones agrupadas por `state` y por `ba_code`, y con comparación pareada.

| Componente | Efecto |
|---|---|
| Ruteo | Aporta |
| XGBoost en hidro frente a una constante | Aporta poco |
| Optuna | Aporta poco |
| `log1p` | Aporta |
| Ubicación (`lat`, `lon`) | Aporta |
| Escalado, imputación, enfriamiento y `cf_aware` | Sin efecto |
| Granularidad `fuel` frente a `fuel_group` | Sin diferencia |

La tabla de consulta por combustible explica cerca del 90 % al 97 % de la mejora total de MAE sobre la media.

## Limitaciones

1. No se cumplió el objetivo de R² ≥ 0.70 (el resultado es 0.523).
2. Aproximadamente el 45 % de las hidroeléctricas de train tienen intensidad exactamente 0. Es un artefacto del método de Siddik et al., y condiciona el modelo en hidro.
3. Las represas con valor 0 favorecen artificialmente a las regiones con más hidro en el ranking del AWI.
4. El modelo subestima fuertemente las hidroeléctricas extremas (entre 60 y 461 veces).
5. La validación cruzada puede ser algo optimista, porque los hiperparámetros se ajustaron con los mismos folds.
6. Solo 56 de las 69 regiones del SCARF tienen todas las piezas.
7. PUE y WUE_on son parámetros fijos, y WUE_off no descuenta importaciones entre regiones.

## Cómo ejecutarlo

Para ejecutarlo se sube a google collab, o si no localmente, en 
