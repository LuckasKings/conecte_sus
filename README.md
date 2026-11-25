# SUS Analytics & Clinical Intelligence (Brasil 2024–2026 • V6 Contínuo)

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://contecte-sus-lucasleodata.streamlit.app/)
[![DuckDB](https://img.shields.io/badge/DuckDB-OLAP-FFF000.svg)](https://duckdb.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-V6%20Cont%C3%ADnuo-2ea44f.svg)](https://lightgbm.readthedocs.io/)
[![MLOps Cloud](https://img.shields.io/badge/GitHub%20Actions-Cron%20Quinzenal-2088FF.svg)](.github/workflows/pipeline_continuo_mlops.yml)

**Aplicação Online (Live Demo):** [contecte-sus-lucasleodata.streamlit.app](https://contecte-sus-lucasleodata.streamlit.app/)

Plataforma contínua ponta a ponta de **Engenharia de Dados, Epidemiologia e Inteligência Artificial Hospitalar** construída sobre os microdados públicos do **SIH/SUS (Sistema de Informações Hospitalares do SUS)** do Brasil de **2024 a 2026** (**38.306.556 internações deduplicadas / ~38,31M** em **888 partições** nas **27 Unidades da Federação**), com ingestão incremental automatizada em nuvem, interface multilíngue (**PT / EN / ES**) e governança **Champion vs. Challenger**.

---

## Problema de Negócio e Impacto na Saúde Pública

Hospitais públicos e operadoras enfrentam diariamente dois desafios críticos no momento da admissão hospitalar:

1. **Superlotação e giro de leitos:** Identificar precocemente pacientes com alto risco de **Longa Permanência (15 dias ou mais)** para acionar Serviço Social, Fisioterapia e planejamento de desospitalização desde o primeiro dia.
2. **Alocação de cuidado intensivo:** Estimar o risco real de **Óbito Intra-hospitalar** na admissão usando apenas informações disponíveis no acolhimento, priorizando monitoramento contínuo e leitos de UTI.

---

## Arquitetura Contínua em Nuvem (V6 Contínuo)

```text
[ DataSUS FTP Oficial (888+ partições • 2024/01–2026/09 • 27 UFs) ]
                             │
                             ▼ (GitHub Actions cron 0 6 5,20 * * + GitHub Releases)
[ Lakehouse Parquet Particionado Hive ZSTD (~38,31M / 38.306.556 linhas) ]
                             │
          ┌──────────────────┴──────────────────┐
          ▼                                     ▼
[ Camada Analítica DuckDB ]           [ Modelos V6 Contínuo (Nuvem • Champion/Challenger) ]
 - Cubo Multi-Ano (11,81 MB exato)     - 4 Algoritmos + Bayes CID3 + Calibração Isotônica
 - Amostra Cockpit (30k casos)         - Retreino Temporal (2,4M) vs. Teste Cego (300k)
 - Histórico de Cargas (Aba 2)         - Monitor de Drift (60k) & Figuras 300 DPI Dinâmicas
          │                                     │
          └──────────────────┬──────────────────┘
                             ▼
            [ Dashboard Interativo Streamlit (6 Abas • PT/EN/ES) ]
             1. Perfil Epidemiológico & Equidade (Multi-Ano 2024–2026)
             2. Qualidade dos Dados, Lotes Incrementais & Integridade
             3. Como a IA Funciona (Modelos V6 Contínuo & Champion/Challenger)
             4. Painel de Gestão Clínica & Leitos (Simulador CID3 + Flags)
             5. Dicionário Completo SIH/SUS (113 Colunas + 14 Derivadas) & Glossário
             6. Jornada, Decisões & Arquitetura (V1 → V6 Contínuo)
```

---

## Visão Geral das 6 Abas do Dashboard (`PT / EN / ES`)

1. **Perfil Epidemiológico & Equidade em Saúde:** Filtros dinâmicos por ano (`2024–2026`), região, UF, raça/cor, especialidade, sexo e mês; gráficos interativos (faixa etária × sexo, equidade racial, sazonalidade mensal, Top capítulos CID-10), comparativo interanual e tabela estadual via **DuckDB**.
2. **Qualidade dos Dados, Lotes Incrementais & Contrato:** Resumo da última carga incremental, **histórico de cargas por lote** (`brutos`, `aceitos`, `filtrados`, `deduplicados` e indicadores de subnotificação), contrato coluna a coluna dos 10 blocos críticos (`Perfil + Treino IA`) e completude por UF.
3. **Como a IA Funciona (Modelos & MLOps):** Benchmark temporal entre **Regressão Logística**, **Random Forest**, **XGBoost** e **LightGBM V6 Contínuo**, **3 figuras diagnósticas em 300 DPI** sempre sincronizadas com o Champion ativo (regeneração automática quando um Challenger vence e cache dinâmico por `mtime`), além de painel com cron, política automática de retreino, **monitor de drift clínico** e **histórico Champion vs. Challenger**.
4. **Painel de Gestão Clínica & Leitos:** Sliders interativos partindo do **Limiar de Youden V6 Contínuo** (`3,9%` Óbito e `6,1%` Longa Permanência), **simulador em tempo real** com escolha de `cid3` específico + Matriz de Decisão 2×2 e **Cockpit hospitalar (`30k` casos estratificados)** com auditoria de acertos (`VP`, `VN`, `Alerta Preventivo` vs. `Falso Negativo`).
5. **Dicionário Completo SIH/SUS (`127` Variáveis) & Glossário:** Auditoria pesquisável das **113 colunas oficiais** da AIH Reduzida + **14 derivadas**, destacando o papel exato (`IA + Epidemio`, `Apenas Epidemio`, `Alvo`, `Data Leakage` ou `Admin`), acompanhada de **15 verbetes** técnicos explicados em linguagem clínica.
6. **Jornada, Decisões & Arquitetura (`V1 → V6 Contínuo`):** Registro completo de engenharia (`V1` a `V6 Contínuo`), tabela comparativa das 6 gerações e **stack de 10 tecnologias** utilizadas.

---

## Resultados do Modelo Ativo — LightGBM V6 Contínuo (Teste Cego Temporal `2026/07–09`: `300.000` internações)

Treinado temporalmente em **2.400.000** internações (`2024/01 → 2026/06`) com codificação híbrida de diagnóstico (`cid3` 3 dígitos Top-200 OHE + Target Encoding Bayesiano *Out-of-Fold* `m=50` com fallback no capítulo), calibrado via **Calibração Isotônica** em **300.000** casos e avaliado em **300.000** internações futuras inéditas (`2026/07 → 2026/09`):

| Desfecho | Modelo Ativo | ROC-AUC | PR-AUC | Brier Score (Calibrado) | Sensibilidade (Limiar Youden) | Especificidade |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Óbito Intra-hospitalar** | **LightGBM V6 Contínuo** | `0,9107` | `0,3089` | `0,0300` | `89,55%` (`3,87%`) | `77,49%` |
| **Longa Permanência (>= 15d)** | **LightGBM V6 Contínuo** | `0,8447` | `0,3493` | `0,0469` | `83,74%` (`6,14%`) | `67,78%` |

> **Evolução V5 → V6 Contínuo:** O Champion inaugural **V5 Nacional (Base 2024, 13,58M)** atingiu `ROC 0,9082 / 0,8481` (`PR 0,3035 / 0,3663`, `Brier 0,0306 / 0,0471`) e foi substituído em nuvem pelo **V6 Contínuo** após vitória sem regressões em 2/2 desfechos, com backup em `modelos/archive/` e trilha auditável em `modelos/registro_modelos_v5.json`.

### Governança MLOps Automática, Monitor de Drift & Compromisso Ético
- **Desacoplamento Ético & Zero Leakage:** `raca_cor` e `mes_internacao` permanecem na Aba 1 (equidade e sazonalidade), mas são **excluídos** da IA; apenas **9 variáveis disponíveis na admissão** entram na predição (`idade_anos`, `sexo`, `capitulo_cid`, `cid3`, `especialidade_leito`, `carater_internacao`, `complexidade`, `transferido_intermun`, `internacao_fim_de_semana`).
- **Gatilho Híbrido em Nuvem (`modelos/politica_mlops.json`):** Dispara competição **Champion vs. Challenger** ao acumular **`>= 3` competências inéditas** (ou **`>= 3.000.000` registros**) após **`30` dias** de *cooldown*, ou imediatamente se o **monitor de drift** (`60.000` casos recentes) detectar desvio de prevalência (`>= 0,50` p.p. Óbito / `>= 0,80` p.p. Longa), queda de `ROC-AUC` (`< 0,8950` / `< 0,8300`) ou alta de `Brier` (`> 0,0330` / `> 0,0500`).
- **Atualização Segura de Artefatos & Figuras:** Só promove o Challenger se superar o Champion sem regressões; grava backup em `modelos/archive/`, atualiza os `.joblib`, `metricas_v5_nacional.csv`, `config_limiares_v5.json`, `top_cid3_por_capitulo_v5.json` e regenera automaticamente as **3 figuras 300 DPI** (inclusive no modo `recalibrar`).

---

## Como Executar Localmente e Orquestrar o Pipeline

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Executar o Dashboard Streamlit
streamlit run app.py
```

Para executar o pipeline contínuo completo (busca lotes inéditos no DataSUS FTP, atualiza o cubo deploy, monitora drift e avalia Champion vs. Challenger se o gatilho híbrido disparar):
```bash
python scripts/orquestrador_continuo.py --workers 6
python scripts/orquestrador_continuo.py --forcar-retreino
python scripts/retreinar_e_promover.py --regenerar-figuras-champion
```

No GitHub, o workflow `.github/workflows/pipeline_continuo_mlops.yml` executa esse ciclo automaticamente nos dias **5 e 20** de cada mês (`06:00 UTC`), restaura/atualiza o Parquet particionado via `GitHub Releases (lakehouse-data)` com `scripts/sync_lakehouse_release.py` e publica atualizações no **Streamlit Cloud**.

---

## Estrutura do Projeto
- `app.py` — Aplicação web Streamlit (6 abas interativas em PT/EN/ES com cache dinâmico por `mtime`).
- `.github/workflows/pipeline_continuo_mlops.yml` — Automação quinzenal em nuvem (ingestão FTP + MLOps + auto-deploy).
- `requirements.txt` & `requirements-pipeline.txt` — Dependências do app Streamlit e dependências enxutas do runner MLOps.
- `scripts/orquestrador_continuo.py` — Orquestrador incremental, monitor de drift clínico (`60k`) e gatilho híbrido Champion vs. Challenger.
- `scripts/retreinar_e_promover.py` — Avaliação temporal justa entre Champion e Challenger, backup, promoção segura e geração das 3 figuras 300 DPI.
- `scripts/sync_lakehouse_release.py` — Sincronização dos arquivos `parquet_*.tar` e `historico_cargas.json` via GitHub Releases (`lakehouse-data`).
- `scripts/pipeline_nacional_sih.py` — Ingestão idempotente e paralela DataSUS para Parquet particionado Hive (`ano/uf/mes`) com deduplicação natural.
- `scripts/gerar_camada_deploy.py` — Gerador do cubo multi-ano (`11,81 MB`) e amostra cockpit (`30k` casos).
- `scripts/duckdb_service.py` — Motor SQL híbrido (Parquet bruto local ou cubo compacto na nuvem) + leitura dos manifestos de cargas e modelos.
- `scripts/features_v5.py` & `scripts/treinar_modelos_v5_nacional.py` — Pré-processador clínico híbrido `cid3` (`AdmissionV5Preprocessor` compartilhado V5/V6 Contínuo) e treino baseline 2024.
- `scripts/catalogo_sih.py` & `scripts/i18n.py` — Auditoria das 113 colunas SIH/SUS (+14 derivadas) e internacionalização PT/EN/ES (textos, filtros, tabelas e glossário).
- `dados/deploy/` — Cubo epidemiológico (`11,81 MB`), amostra estratificada (`0,34 MB`) e manifesto `resumo_ultima_carga.json`.
- `modelos/` — Artefatos LightGBM V6 Contínuo calibrados (`0,62 MB` cada), `metricas_v5_nacional.csv`, `config_limiares_v5.json`, `top_cid3_por_capitulo_v5.json`, `politica_mlops.json`, `registro_modelos_v5.json` e backups em `modelos/archive/`.
- `figuras/` — Evidências visuais em alta resolução (300 DPI) geradas para o Champion ativo em produção.
