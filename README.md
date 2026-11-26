# SUS Analytics & Clinical Intelligence (Brazil 2024–2026 • Continuous V6)

> 🌐 **Language / Idioma:** [🇺🇸 English](README.md) · [🇧🇷 Português](README.pt.md) · [🇪🇸 Español](README.es.md)

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://contecte-sus-lucasleodata.streamlit.app/)
[![DuckDB](https://img.shields.io/badge/DuckDB-OLAP-FFF000.svg)](https://duckdb.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Continuous%20V6-2ea44f.svg)](https://lightgbm.readthedocs.io/)
[![MLOps Cloud](https://img.shields.io/badge/GitHub%20Actions-Biweekly%20Cron-2088FF.svg)](.github/workflows/pipeline_continuo_mlops.yml)

**Live Demo:** [contecte-sus-lucasleodata.streamlit.app](https://contecte-sus-lucasleodata.streamlit.app/)

End-to-end continuous **Data Engineering, Epidemiology, and Hospital Clinical AI** platform built on all official public microdata from Brazil's Unified Health System (**SIH/SUS — Hospital Information System**) covering **2024–2026** (**38,306,556 deduplicated hospitalizations / ~38.31M** across **888 partitions** and all **27 Brazilian States**), featuring automated incremental cloud ingestion, trilingual UI (**PT / EN / ES**), and **Champion vs. Challenger** MLOps governance.

---

## Business Problem & Public Health Impact

Public hospitals and health networks face two critical operational challenges every day right at inpatient admission:

1. **Bed overcrowding & discharge planning:** Flag patients at high risk of **Long Length of Stay (15+ days)** right at admission so Social Work, Physical Therapy, and bed management teams can act from day one.
2. **Intensive care allocation:** Estimate true **In-Hospital Mortality** risk using only intake variables known at arrival to prioritize continuous monitoring and ICU beds.

---

## Continuous Cloud Architecture (Continuous V6)

```text
[ Official DataSUS FTP (888+ partitions • 2024/01–2026/09 • 27 States) ]
                             │
                             ▼ (GitHub Actions cron 0 6 5,20 * * + GitHub Releases)
[ Hive ZSTD Partitioned Parquet Lakehouse (~38.31M / 38,306,556 rows) ]
                             │
          ┌──────────────────┴──────────────────┐
          ▼                                     ▼
[ DuckDB Analytics Layer ]           [ Continuous V6 Models (Cloud • Champion/Challenger) ]
 - Multi-Year Cube (11.81 MB exact)    - 4 Algorithms + Bayesian CID3 + Isotonic Cal.
 - Ward Cockpit Sample (30k cases)     - Temporal Retrain (2.4M) vs. Blind Holdout (300k)
 - Batch Load History (Tab 2)          - Drift Monitor (60k) & Auto 300-DPI Plots
          │                                     │
          └──────────────────┬──────────────────┘
                             ▼
            [ Interactive Streamlit Dashboard (6 Tabs • PT/EN/ES) ]
             1. Epidemiology & Health Equity (Multi-Year 2024–2026)
             2. Data Quality, Incremental Loads & Integrity Contract
             3. How the AI Works (Continuous V6 & Champion/Challenger)
             4. Clinical Management & Ward Cockpit (CID3 Simulator + Flags)
             5. Complete SIH/SUS Dictionary (113 Columns + 14 Derived) & Glossary
             6. Project Journey, Decisions & Architecture (V1 → Continuous V6)
```

---

## Dashboard Overview — 6 Interactive Tabs (`PT / EN / ES`)

1. **Epidemiology & Health Equity:** Dynamic filters across year (`2024–2026`), region, state, race/ethnicity, specialty, sex, and month; interactive Plotly visualizations (age × sex, racial equity, monthly seasonality, top ICD-10 chapters), year-over-year trends, and state breakdown powered by **DuckDB**.
2. **Data Quality, Incremental Loads & Integrity Contract:** Latest incremental load summary, **per-batch ingestion history** (`raw`, `accepted`, `filtered`, `deduplicated`, and underreporting metrics), column-by-column audit across 10 core blocks (`Profile + AI Training`), and state completeness table.
3. **How the AI Works (Models & MLOps):** Temporal benchmark comparing **Logistic Regression**, **Random Forest**, **XGBoost**, and **Continuous LightGBM V6**, **three 300-DPI diagnostic figures** synced automatically to the active Champion (re-rendered on winner promotion with `mtime`-based app cache invalidation), cloud schedule, retraining policy, **clinical drift monitor**, and **Champion vs. Challenger history**.
4. **Clinical Management & Ward Cockpit:** Interactive threshold sliders anchored to the **Continuous V6 Youden Index** (`3.9%` Mortality and `6.1%` Long Stay), **real-time intake simulator** with specific 3-character ICD-10 (`cid3`) support + 2×2 Decision Matrix, and **hospital ward cockpit (`30k` stratified cases)** auditing True Positives/Negatives and Preventive Alerts vs. False Negatives.
5. **Complete SIH/SUS Dictionary (`127` Fields) & Glossary:** Searchable audit of all **113 official columns** plus **14 derived features**, tagging every field's exact role (`AI + Epidemiology`, `Epidemiology Only`, `Target`, `Data Leakage`, or `Admin`), alongside a **15-term** plain-language clinical/AI glossary.
6. **Project Journey, Decisions & Architecture (`V1 → Continuous V6`):** Full engineering log from single-state pilot (`V1`) to automated multi-year cloud (`Continuous V6`), version evolution table, and the **10-technology stack** table.

---

## Active Production Results — Continuous LightGBM V6 (Blind Temporal Holdout `2026/07–09`: `300,000` admissions)

Trained temporally on **2,400,000** admissions (`2024/01 → 2026/06`) with hybrid diagnosis encoding (`cid3` 3-char Top-200 OHE + Out-of-Fold Bayesian Target Encoding `m=50` with chapter fallback), calibrated via **Isotonic Calibration** on **300,000** cases, and evaluated on **300,000** unseen future hospitalizations (`2026/07 → 2026/09`):

| Outcome | Active Model | ROC-AUC | PR-AUC | Brier Score (Calibrated) | Recall (Youden Cutoff) | Specificity |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **In-Hospital Mortality** | **Continuous LightGBM V6** | `0.9107` | `0.3089` | `0.0300` | `89.55%` (`3.87%`) | `77.49%` |
| **Long Stay (>= 15d)** | **Continuous LightGBM V6** | `0.8447` | `0.3493` | `0.0469` | `83.74%` (`6.14%`) | `67.78%` |

> **V5 → Continuous V6 Progression:** The inaugural **National V5 (2024 Base, 13.58M)** Champion achieved `ROC 0.9082 / 0.8481` (`PR 0.3035 / 0.3663`, `Brier 0.0306 / 0.0471`) and was replaced in production by **Continuous V6** after winning with zero regressions across 2/2 outcomes, with prior weights archived in `modelos/archive/` and audited in `modelos/registro_modelos_v5.json`.

### Automated MLOps Governance, Drift Monitoring & Ethics
- **Ethical Decoupling & Zero Data Leakage:** `raca_cor` and `mes_internacao` remain in Tab 1 for equity and seasonality auditing, but are **excluded** from predictive modeling; only **9 intake features** known at admission enter inference (`idade_anos`, `sexo`, `capitulo_cid`, `cid3`, `especialidade_leito`, `carater_internacao`, `complexidade`, `transferido_intermun`, `internacao_fim_de_semana`).
- **Cloud Hybrid Trigger (`modelos/politica_mlops.json`):** Fires **Champion vs. Challenger** once **`>= 3` unseen months** (or **`>= 3,000,000` rows**) accumulate after a **`30`-day cooldown**, or triggers early if the **drift monitor** (`60,000` recent cases) flags prevalence shift (`>= 0.50` pp Mortality / `>= 0.80` pp Long Stay), `ROC-AUC` drop (`< 0.8950` / `< 0.8300`), or `Brier` degradation (`> 0.0330` / `> 0.0500`).
- **Safe Artifact & Plot Promotion:** Only replaces production upon non-regressive temporal gains; saves backups under `modelos/archive/`, updates `.joblib` bundles, `metricas_v5_nacional.csv`, `config_limiares_v5.json`, `top_cid3_por_capitulo_v5.json`, and regenerates all **three 300-DPI figures** (including in `recalibrar` mode).

---

## Local Setup & Continuous Orchestration

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the Streamlit Dashboard
streamlit run app.py
```

To run the continuous pipeline locally (fetches unseen DataSUS FTP partitions, rebuilds deploy cubes, evaluates clinical drift, and triggers Champion vs. Challenger when conditions are met):
```bash
python scripts/orquestrador_continuo.py --workers 6
python scripts/orquestrador_continuo.py --forcar-retreino
python scripts/retreinar_e_promover.py --regenerar-figuras-champion
```

On GitHub, `.github/workflows/pipeline_continuo_mlops.yml` runs automatically on the **5th and 20th** of every month (`06:00 UTC`), syncs partitioned Parquet archives via `GitHub Releases (lakehouse-data)` using `scripts/sync_lakehouse_release.py`, and triggers auto-deploy to **Streamlit Cloud**.

---

## Repository Structure
- `app.py` — Interactive Streamlit web app (6 tabs in PT/EN/ES with `mtime`-based cache invalidation).
- `.github/workflows/pipeline_continuo_mlops.yml` — Biweekly cloud MLOps workflow (FTP ingestion + training + auto-deploy).
- `requirements.txt` & `requirements-pipeline.txt` — Streamlit runtime dependencies and lean cloud pipeline dependencies.
- `scripts/orquestrador_continuo.py` — Incremental orchestrator, clinical drift monitor (`60k`), and hybrid retraining gate.
- `scripts/retreinar_e_promover.py` — Temporal Champion vs. Challenger evaluation, archiving, promotion, and 300-DPI plot generation.
- `scripts/sync_lakehouse_release.py` — `parquet_*.tar` and `historico_cargas.json` sync via GitHub Releases (`lakehouse-data`).
- `scripts/pipeline_nacional_sih.py` — Parallel idempotent DataSUS FTP ingestion into Hive-partitioned Parquet (`ano/uf/mes`) with natural-key deduplication.
- `scripts/gerar_camada_deploy.py` — Generates the compact multi-year cube (`11.81 MB`) and stratified ward sample (`30k` rows).
- `scripts/duckdb_service.py` — Hybrid analytical SQL engine (local raw Parquet or cloud compact cube) + batch/model manifest readers.
- `scripts/features_v5.py` & `scripts/treinar_modelos_v5_nacional.py` — Hybrid `cid3` clinical preprocessor (`AdmissionV5Preprocessor` shared by V5/Continuous V6) and 2024 baseline trainer.
- `scripts/catalogo_sih.py` & `scripts/i18n.py` — Audited 113-column (+14 derived) catalog and full PT/EN/ES localization (UI, filters, tables, and glossary).
- `dados/deploy/` — Multi-year epidemiological cube (`11.81 MB`), stratified sample (`0.34 MB`), and `resumo_ultima_carga.json`.
- `modelos/` — Calibrated Continuous LightGBM V6 bundles (`0.62 MB` each), `metricas_v5_nacional.csv`, `config_limiares_v5.json`, `top_cid3_por_capitulo_v5.json`, `politica_mlops.json`, `registro_modelos_v5.json`, and `modelos/archive/`.
- `figuras/` — High-resolution (300 DPI) model diagnostics synced with the active Champion.
