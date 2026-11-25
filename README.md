# SUS Analytics & Clinical Intelligence (Brasil 2024)

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://contecte-sus-lucasleodata.streamlit.app/)
[![DuckDB](https://img.shields.io/badge/DuckDB-OLAP-FFF000.svg)](https://duckdb.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-V5%20Calibrado-2ea44f.svg)](https://lightgbm.readthedocs.io/)

**Aplicação Online (Live Demo):** [contecte-sus-lucasleodata.streamlit.app](https://contecte-sus-lucasleodata.streamlit.app/)

Plataforma ponta a ponta de **Engenharia de Dados, Epidemiologia e Inteligência Artificial Hospitalar** construída sobre todos os microdados públicos do **SIH/SUS (Sistema de Informações Hospitalares do SUS)** do Brasil em **2024** (**13.584.534 internações** nas **27 Unidades da Federação**).

---

## Problema de Negócio e Impacto na Saúde Pública

Hospitais públicos e operadoras enfrentam diariamente dois desafios críticos no momento da admissão hospitalar:

1. **Superlotação e giro de leitos:** Identificar precocemente pacientes com alto risco de **Longa Permanência (15 dias ou mais)** para acionar Serviço Social, Fisioterapia e planejamento de desospitalização desde o primeiro dia.
2. **Alocação de cuidado intensivo:** Estimar o risco real de **Óbito Intra-hospitalar** na admissão usando apenas informações disponíveis no acolhimento, priorizando monitoramento contínuo e leitos de UTI.

---

## Arquitetura da Solução

```text
[ DataSUS FTP Oficial (324 arquivos .dbc • 27 UFs x 12 meses) ]
                             │
                             ▼ (Pipeline ETL paralelo + Arrow)
[ Lakehouse Local Parquet Particionado (186 MB • 13,58M linhas) ]
                             │
          ┌──────────────────┴──────────────────┐
          ▼                                     ▼
[ Camada Analítica DuckDB ]           [ Modelos V5 Nacional (2,4M treino / 3M amostra) ]
 - Cubo Deploy (3,99 MB exato)         - 4 Algoritmos + Bayes CID3 + Calibração Isotônica
 - Amostra Cockpit (30k casos)         - Ética: Sem raça/cor ou mês na predição
          │                                     │
          └──────────────────┬──────────────────┘
                             ▼
            [ Dashboard Interativo Streamlit (6 Abas • PT/EN/ES) ]
             1. Perfil Epidemiológico & Equidade
             2. Qualidade dos Dados & Integridade
             3. Como a IA Funciona (Modelos V5 & Métricas)
             4. Painel de Gestão Clínica & Leitos (Flags)
             5. Dicionário Completo SIH/SUS (113 Colunas)
             6. Jornada, Decisões & Arquitetura (V1 a V5)
```

---

## Resultados dos Modelos V5 Nacional (Teste Cego: 300.000 internações das 27 UFs)

Treinados com **2.400.000** internações estratificadas (split total de **3.000.000**), usando codificação híbrida de diagnóstico (`cid3` 3 dígitos Top-200 OHE + Target Encoding Bayesiano *Out-of-Fold* `m=50` com fallback no capítulo), calibrados via **Calibração Isotônica** em **300.000** registros dedicados e avaliados em **300.000** internações inéditas:

| Desfecho | Modelo Escolhido | ROC-AUC | PR-AUC | Brier Score (Calibrado) | Sensibilidade (Limiar Youden) | Especificidade |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Óbito Intra-hospitalar** | **LightGBM V5** | `0,9082` | `0,3035` | `0,0306` | `89,58%` (`4,09%`) | `77,14%` |
| **Longa Permanência (>= 15d)** | **LightGBM V5** | `0,8481` | `0,3663` | `0,0471` | `85,32%` (`6,27%`) | `66,60%` |

### Compromisso Ético e Governança (No Data Leakage)
- **Desacoplamento Ético:** A variável `raca_cor` é preservada na Aba 1 para análise de equidade em saúde pública, mas **excluída** do modelo preditivo para evitar viés algorítmico na regulação de leitos.
- **Zero Vazamento Temporal:** Apenas 9 variáveis conhecidas no acolhimento entram na predição (`idade_anos`, `sexo`, `capitulo_cid`, `cid3`, `especialidade_leito`, `carater_internacao`, `complexidade`, `transferido_intermun`, `internacao_fim_de_semana`).

---

## Como Executar Localmente

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
python scripts/treinar_modelos_v5_nacional.py
```

---

## Estrutura do Projeto
- `app.py` — Aplicação web Streamlit (6 abas interativas em PT/EN/ES).
- `scripts/pipeline_nacional_sih.py` — Ingestão paralela DataSUS para Parquet particionado Hive.
- `scripts/gerar_camada_deploy.py` — Gerador do cubo exato (`3,99 MB`) e amostra cockpit (`0,34 MB`).
- `scripts/duckdb_service.py` — Motor SQL híbrido (base bruta local ou cubo compacto na nuvem).
- `scripts/features_v5.py` — Pré-processador clínico híbrido `cid3` + Empirical Bayes Shrinkage.
- `scripts/treinar_modelos_v5_nacional.py` — Treino V5 (`2,4M`), calibração isotônica e gráficos em 300 DPI.
- `scripts/catalogo_sih.py` e `scripts/i18n.py` — Auditoria das 113 colunas SIH/SUS e internacionalização PT/EN/ES.
- `dados/deploy/` — Cubo epidemiológico (`3,99 MB`) e amostra estratificada (`0,34 MB`) para publicação online.
- `modelos/` — Artefatos LightGBM V5 calibrados (`0,62 MB` cada), limiares e métricas auditadas.
- `figuras/` — Evidências visuais V5 em alta resolução (300 DPI).
