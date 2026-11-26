# SUS Analytics & Clinical Intelligence (Brasil 2024–2026 • V6 Continuo)

> 🌐 **Idioma / Language:** [🇺🇸 English](README.md) · [🇧🇷 Português](README.pt.md) · [🇪🇸 Español](README.es.md)

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://contecte-sus-lucasleodata.streamlit.app/)
[![DuckDB](https://img.shields.io/badge/DuckDB-OLAP-FFF000.svg)](https://duckdb.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-V6%20Continuo-2ea44f.svg)](https://lightgbm.readthedocs.io/)
[![MLOps Cloud](https://img.shields.io/badge/GitHub%20Actions-Cron%20Quincenal-2088FF.svg)](.github/workflows/pipeline_continuo_mlops.yml)

**Aplicación Online (Live Demo):** [contecte-sus-lucasleodata.streamlit.app](https://contecte-sus-lucasleodata.streamlit.app/)

Plataforma continua integral de **Ingeniería de Datos, Epidemiología e Inteligencia Artificial Hospitalaria** construida sobre todos los microdatos públicos del **SIH/SUS (Sistema de Información Hospitalaria del SUS)** de Brasil de **2024 a 2026** (**38.306.556 ingresos deduplicados / ~38,31M** en **888 particiones** de los **27 Estados**), con ingesta incremental automatizada en la nube, interfaz multilingüe (**PT / EN / ES**) y gobernanza **Champion vs. Challenger**.

---

## Problema de Negocio e Impacto en Salud Pública

Los hospitales públicos enfrentan diariamente dos desafíos críticos al momento del ingreso hospitalario:

1. **Saturación y rotación de camas:** Identificar tempranamente pacientes con alto riesgo de **Larga Estancia (15 días o más)** para activar Trabajo Social, Fisioterapia y planificación de alta desde el primer día.
2. **Asignación de cuidados intensivos:** Estimar el riesgo real de **Mortalidad Intrahospitalaria** al ingreso usando solo variables disponibles en la admisión, priorizando monitoreo continuo y camas de UCI.

---

## Arquitectura Continua en la Nube (V6 Continuo)

```text
[ DATASUS FTP Oficial (888+ particiones • 2024/01–2026/09 • 27 Estados) ]
                             │
                             ▼ (GitHub Actions cron 0 6 5,20 * * + GitHub Releases)
[ Lakehouse Parquet Particionado Hive ZSTD (~38,31M / 38.306.556 filas) ]
                             │
          ┌──────────────────┴──────────────────┐
          ▼                                     ▼
[ Capa Analítica DuckDB ]            [ Modelos V6 Continuo (Nube • Champion/Challenger) ]
 - Cubo Multianual (11,81 MB exacto)   - 4 Algoritmos + Bayes CIE3 + Calibración Isotónica
 - Muestra Panel Clínico (30k casos)   - Reentreno Temporal (2,4M) vs. Prueba Ciega (300k)
 - Historial de Cargas (Pestaña 2)     - Monitor Drift (60k) y Figuras 300 DPI Dinámicas
          │                                     │
          └──────────────────┬──────────────────┘
                             ▼
            [ Dashboard Interactivo Streamlit (6 Pestañas • PT/EN/ES) ]
             1. Epidemiología y Equidad en Salud (Multianual 2024–2026)
             2. Calidad de Datos, Cargas Incrementales e Integridad
             3. Cómo Funciona la IA (Modelos V6 Continuo y Champion/Challenger)
             4. Gestión Clínica y Camas (Simulador CIE3 + Alertas)
             5. Diccionario Completo SIH/SUS (113 Columnas + 14 Derivadas) y Glosario
             6. Trayectoria, Decisiones y Arquitectura (V1 → V6 Continuo)
```

---

## Resumen de las 6 Pestañas del Dashboard (`PT / EN / ES`)

1. **Epidemiología y Equidad en Salud:** Filtros dinámicos por año (`2024–2026`), región, estado, raza/color, especialidad, sexo y mes; gráficos interactivos (edad × sexo, equidad racial, estacionalidad mensual, capítulos CIE-10), comparativo interanual y tabla estatal con **DuckDB**.
2. **Calidad de Datos, Cargas Incrementales y Contrato:** Resumen de la última carga incremental, **historial de cargas por lote** (`brutos`, `aceptados`, `filtrados`, `deduplicados` e indicadores de subregistro), contrato columna por columna de los 10 bloques clave (`Perfil + Entreno IA`) y completitud por Estado.
3. **Cómo Funciona la IA (Modelos y MLOps):** Comparativo temporal entre **Regresión Logística**, **Random Forest**, **XGBoost** y **LightGBM V6 Continuo**, **3 figuras diagnósticas en 300 DPI** sincronizadas automáticamente con el Champion activo (regeneración automática al promover un ganador e invalidación de caché por `mtime`), agenda en nube, política de reentreno, **monitor de drift clínico** e **historial Champion vs. Challenger**.
4. **Gestión Clínica y Camas:** Deslizadores calibrados desde el **Índice de Youden V6 Continuo** (`3,9%` Mortalidad y `6,1%` Larga Estancia), **simulador en tiempo real** con selección de diagnóstico `cid3` específico + Matriz de Decisión 2×2 y **Panel Clínico (`30k` casos estratificados)** con auditoría de aciertos (`VP`, `VN`, `Alarma Preventiva` vs. `Falso Negativo`).
5. **Diccionario Completo SIH/SUS (`127` Variables) y Glosario:** Auditoría buscable de las **113 columnas oficiales** más **14 derivadas**, indicando explícitamente su rol (`IA + Epidemiología`, `Solo Epidemiología`, `Objetivo`, `Data Leakage` o `Admin`), junto con un **glosario de 15 términos** técnicos.
6. **Trayectoria, Decisiones y Arquitectura (`V1 → V6 Continuo`):** Registro transparente de ingeniería (`V1` a `V6 Continuo`), tabla comparativa de las 6 generaciones y **stack de 10 tecnologías**.

---

## Resultados del Modelo Activo — LightGBM V6 Continuo (Prueba Ciega Temporal `2026/07–09`: `300.000` ingresos)

Entrenado temporalmente con **2.400.000** ingresos (`2024/01 → 2026/06`) mediante codificación híbrida de diagnóstico (`cid3` 3 dígitos Top-200 OHE + Target Encoding Bayesiano *Out-of-Fold* `m=50` con respaldo en el capítulo), calibrado vía **Calibración Isotónica** en **300.000** casos y evaluado en **300.000** ingresos futuros inéditos (`2026/07 → 2026/09`):

| Desenlace | Modelo Activo | ROC-AUC | PR-AUC | Brier Score (Calibrado) | Sensibilidad (Corte Youden) | Especificidad |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Mortalidad Intrahospitalaria** | **LightGBM V6 Continuo** | `0,9107` | `0,3089` | `0,0300` | `89,55%` (`3,87%`) | `77,49%` |
| **Larga Estancia (>= 15d)** | **LightGBM V6 Continuo** | `0,8447` | `0,3493` | `0,0469` | `83,74%` (`6,14%`) | `67,78%` |

> **Evolución V5 → V6 Continuo:** El Champion inaugural **V5 Nacional (Base 2024, 13,58M)** obtuvo `ROC 0,9082 / 0,8481` (`PR 0,3035 / 0,3663`, `Brier 0,0306 / 0,0471`) y fue sustituido en la nube por **V6 Continuo** tras ganar sin regresiones en 2/2 desenlaces, con respaldo en `modelos/archive/` y registro auditable en `modelos/registro_modelos_v5.json`.

### Gobernanza MLOps Automática, Monitor de Drift y Compromiso Ético
- **Desacoplamiento Ético y Cero Leakage:** `raca_cor` y `mes_internacao` se conservan en la Pestaña 1 (equidad y estacionalidad), pero se **excluyen** de la IA; solo **9 variables conocidas al ingreso** alimentan la predicción (`idade_anos`, `sexo`, `capitulo_cid`, `cid3`, `especialidade_leito`, `carater_internacao`, `complexidade`, `transferido_intermun`, `internacao_fim_de_semana`).
- **Disparador Híbrido en Nube (`modelos/politica_mlops.json`):** Ejecuta **Champion vs. Challenger** al acumular **`>= 3` meses nuevos** (o **`>= 3.000.000` registros**) tras **`30` días** de espera mínima, o inmediatamente si el **monitor de drift** (`60.000` casos recientes) detecta cambio de prevalencia (`>= 0,50` p.p. Mortalidad / `>= 0,80` p.p. Larga Estancia), caída de `ROC-AUC` (`< 0,8950` / `< 0,8300`) o aumento de `Brier` (`> 0,0330` / `> 0,0500`).
- **Promoción Segura y Figuras Actualizadas:** Solo sustituye producción si mejora sin regresiones; archiva respaldos en `modelos/archive/`, actualiza `.joblib`, `metricas_v5_nacional.csv`, `config_limiares_v5.json`, `top_cid3_por_capitulo_v5.json` y regenera automáticamente las **3 figuras 300 DPI** (incluso en modo `recalibrar`).

---

## Cómo Ejecutar Localmente y Orquestar el Pipeline

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Ejecutar el Dashboard Streamlit
streamlit run app.py
```

Para ejecutar el flujo continuo completo (descarga lotes nuevos del FTP DATASUS, regenera el cubo deploy, evalúa drift clínico y lanza Champion vs. Challenger si se activa el gatillo):
```bash
python scripts/orquestrador_continuo.py --workers 6
python scripts/orquestrador_continuo.py --forcar-retreino
python scripts/retreinar_e_promover.py --regenerar-figuras-champion
```

En GitHub, `.github/workflows/pipeline_continuo_mlops.yml` ejecuta este ciclo automáticamente los días **5 y 20** de cada mes (`06:00 UTC`), sincroniza el Parquet particionado mediante `GitHub Releases (lakehouse-data)` con `scripts/sync_lakehouse_release.py` y publica actualizaciones en **Streamlit Cloud**.

---

## Estructura del Proyecto
- `app.py` — Aplicación web Streamlit (6 pestañas interactivas en PT/EN/ES con caché dinámica por `mtime`).
- `.github/workflows/pipeline_continuo_mlops.yml` — Automatización quincenal en nube (ingesta FTP + MLOps + auto-deploy).
- `requirements.txt` & `requirements-pipeline.txt` — Dependencias de Streamlit y dependencias ligeras del runner MLOps.
- `scripts/orquestrador_continuo.py` — Orquestador incremental, monitor de drift clínico (`60k`) y gatillo híbrido Champion vs. Challenger.
- `scripts/retreinar_e_promover.py` — Evaluación temporal justa Champion vs. Challenger, respaldo, promoción segura y generación de las 3 figuras 300 DPI.
- `scripts/sync_lakehouse_release.py` — Sincronización de `parquet_*.tar` y `historico_cargas.json` vía GitHub Releases (`lakehouse-data`).
- `scripts/pipeline_nacional_sih.py` — Ingesta paralela e idempotente DATASUS a Parquet particionado Hive (`ano/uf/mes`) con deduplicación natural.
- `scripts/gerar_camada_deploy.py` — Generador del cubo multianual (`11,81 MB`) y muestra estratificada (`30k` casos).
- `scripts/duckdb_service.py` — Motor SQL híbrido (Parquet bruto local o cubo compacto en nube) + lectura de historiales de cargas y modelos.
- `scripts/features_v5.py` & `scripts/treinar_modelos_v5_nacional.py` — Preprocesador clínico híbrido `cid3` (`AdmissionV5Preprocessor` compartido V5/V6 Continuo) y entrenamiento base 2024.
- `scripts/catalogo_sih.py` & `scripts/i18n.py` — Auditoría de las 113 columnas SIH/SUS (+14 derivadas) e internacionalización PT/EN/ES (textos, filtros, tablas y glosario).
- `dados/deploy/` — Cubo epidemiológico (`11,81 MB`), muestra estratificada (`0,34 MB`) y manifiesto `resumo_ultima_carga.json`.
- `modelos/` — Artefactos LightGBM V6 Continuo calibrados (`0,62 MB` cada uno), `metricas_v5_nacional.csv`, `config_limiares_v5.json`, `top_cid3_por_capitulo_v5.json`, `politica_mlops.json`, `registro_modelos_v5.json` y respaldos en `modelos/archive/`.
- `figuras/` — Evidencias visuales en alta resolución (300 DPI) generadas para el Champion activo en producción.
