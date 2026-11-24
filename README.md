# 🏥 SUS Analytics & Clinical Intelligence (Brasil 2024)

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![DuckDB](https://img.shields.io/badge/DuckDB-OLAP-FFF000.svg)](https://duckdb.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-V4%20Calibrado-2ea44f.svg)](https://lightgbm.readthedocs.io/)

Plataforma ponta a ponta de **Engenharia de Dados, Epidemiologia e Inteligência Artificial Hospitalar** construída sobre todos os microdados públicos do **SIH/SUS (Sistema de Informações Hospitalares do SUS)** do Brasil em **2024** (**13.584.534 internações** nas **27 Unidades da Federação**).

---

## 🎯 Problema de Negócio & Impacto na Saúde Pública
Hospitais públicos e operadoras enfrentam diariamente dois desafios críticos na porta de entrada:
1. **Superlotação e giro de leitos:** Identificar precocemente pacientes com alto risco de **Longa Permanência (≥ 15 dias)** para acionar Serviço Social, Fisioterapia e desospitalização desde o dia 1.
2. **Alocação de cuidado intensivo:** Estimar o risco real de **Óbito Intra-hospitalar** na admissão usando apenas informações conhecidas no acolhimento, priorizando monitoramento e leitos de UTI.

---

## 🏗️ Arquitetura da Solução

```
[ DataSUS FTP Oficial (324 arquivos .dbc • 27 UFs x 12 meses) ]
                             │
                             ▼ (Pipeline ETL paralelo + Arrow)
[ Lakehouse Local Parquet Particionado (186 MB • 13,58M linhas) ]
                             │
          ┌──────────────────┴──────────────────┐
          ▼                                     ▼
[ Camada Analítica DuckDB ]           [ Modelos V4 Nacional (1,5M coorte) ]
 - Cubo Deploy (3,99 MB exato)         - 4 Algoritmos + Calibração Isotônica
 - Amostra Cockpit (30k casos)         - Ética: Sem raça/cor ou mês na predição
          │                                     │
          └──────────────────┬──────────────────┘
                             ▼
            [ Dashboard Interativo Streamlit (4 Abas) ]
             1. Perfil Epidemiológico & Equidade
             2. Como a IA Funciona (Modelos & Métricas)
             3. Painel de Gestão Clínica & Leitos (Flags)
             4. Dicionário de Dados & Glossário
```

---

## 🏆 Resultados dos Modelos V4 Nacional *(Teste Cego: 300.000 internações das 27 UFs)*

Treinados com **900.000** internações estratificadas, calibrados via **Calibração Isotônica** em **300.000** registros dedicados (eliminando distorções de probabilidade) e avaliados em **300.000** internações inéditas:

| Desfecho | Modelo Escolhido | ROC-AUC | PR-AUC | Brier Score (Calibrado) | Sensibilidade (Limiar Youden) | Especificidade |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Óbito Intra-hospitalar** | **LightGBM V4** | `0,8793` | `0,2101` | `0,0329` | `87,58%` (`3,89%`) | `73,49%` |
| **Longa Permanência (≥15d)** | **LightGBM V4** | `0,8160` | `0,3401` | `0,0480` | `83,92%` (`6,37%`) | `61,41%` |

### Compromisso Ético e Governança (*No Data Leakage*)
- **Desacoplamento Ético:** `raca_cor` é mantida na Aba 1 para visibilidade de equidade em saúde pública, mas **excluída** do modelo preditivo para evitar viés algorítmico na regulação de leitos.
- **Zero Vazamento Temporal:** Apenas 8 variáveis disponíveis no acolhimento entram na predição (`idade_anos`, `sexo`, `capitulo_cid`, `especialidade_leito`, `carater_internacao`, `complexidade`, `transferido_intermun`, `internacao_fim_de_semana`).

---

## 🚀 Como Executar Localmente

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Executar o Dashboard Streamlit
streamlit run app.py
```

Para reprocessar toda a base nacional do zero direto do FTP do DataSUS:
```bash
python scripts/pipeline_nacional_sih.py --ano 2024 --workers 4
python scripts/gerar_camada_deploy.py
python scripts/treinar_modelos_v4_nacional.py
```

---

## 📂 Estrutura do Projeto
- `app.py` — Aplicação web Streamlit (4 abas interativas).
- `scripts/pipeline_nacional_sih.py` — Ingestão paralela DataSUS → Parquet particionado Hive.
- `scripts/duckdb_service.py` — Motor SQL híbrido (local bruto ou cubo compacto na nuvem).
- `scripts/treinar_modelos_v4_nacional.py` — Treino, calibração isotônica e gráficos 300 DPI.
- `dados/deploy/` — Cubo epidemiológico (`3,99 MB`) + amostra estratificada (`0,34 MB`) prontos para nuvem.
- `modelos/` — Artefatos LightGBM V4 calibrados e métricas auditadas.
- `figuras/` — Evidências visuais em alta resolução (300 DPI).
