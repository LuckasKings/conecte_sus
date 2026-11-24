# 📋 Diretrizes e Roadmap do Projeto SUS Brasil 2024
## Sistema Integrado de Inteligência de Dados, Gestão Clínica e Triagem Preditiva

> **Objetivo do Projeto:** Construir uma solução ponta a ponta de Inteligência Hospitalar e Epidemiológica utilizando microdados do SIH/SUS 2024 de todo o território nacional. O projeto servirá como portfólio executivo e técnico sênior para galgar posições de destaque em instituições de saúde, healthtechs e operadoras.

---

## 🏗️ 1. Visão Geral da Arquitetura Alvo

```
[ DataSUS FTP / PySUS (Online) ]
               │
               ▼
   [ Pipeline de Ingestão ETL/ELT ] ─── (Particionamento UF/Mês)
               │
               ▼
[ Armazenamento Nuvem / Colunar (Parquet / GCS / BigQuery / DuckDB) ]
               │
        ┌──────┴───────────────────────────┐
        ▼                                  ▼
[ Camada Analítica & Modelagem ]   [ Dashboard Online Multi-Abas ]
 - Modelos Dual-Triage (Óbito/Longa)    ├─ Aba 1: Perfil Epidemiológico
 - Métricas & Explicabilidade           ├─ Aba 2: Engenharia & Modelos
                                       └─ Aba 3: Cockpit de Gestão Clínica
```

---

## 🖥️ 2. Especificação do Dashboard Interativo (3 Abas)

A aplicação será desenvolvida em **Streamlit** (ou Dash/FastAPI) com interface responsiva, profissional e voltada a tomada de decisão em saúde.

### 📌 Aba 1: Inteligência Epidemiológica & Equidade em Saúde
* **Objetivo:** Revelar o perfil de morbimortalidade e internações no SUS, permitindo comparações estaduais e regionais.
* **Componentes Principais:**
  1. **Filtros Globais:** Seleção de Região (Norte, Nordeste, etc.), UF (27 estados), Faixa Etária, Sexo e Período (meses de 2024).
  2. **Pirâmide Etária e Sexo:** Dinâmica populacional das internações.
  3. **Vulnerabilidade e Equidade:** Análise por Raça/Cor e Populações Indígenas (tempo de permanência e taxa de mortalidade).
  4. **Sazonalidade Mensal:** Variação mensal das internações e taxa de mortalidade ao longo do ano por estado/região (ex.: síndromes respiratórias, surtos infecciosos e demanda hospitalar sazonal).
  5. **Carga de Doença (CID-10):** Top causas de admissão e top causas de mortalidade por capítulo e grupo diagnóstico.
  6. **Fluxo de Regulação Intermunicipal:** Taxa de pacientes transferidos fora do domicílio e sobrecarga de capitais/polos.

### 📌 Aba 2: Inteligência de Dados & Performance dos Modelos (MLOps)
* **Objetivo:** Transparência técnica e governança dos modelos preditivos para profissionais de dados e auditoria clínica.
* **Componentes Principais:**
  1. **Painel de Dados de Entrada:** Dicionário das variáveis de admissão, taxas de nulos e tratamento de *leakage* (sem variáveis pós-admissão como cobrança).
  2. **Benchmarking de Modelos:** Comparativo de 4 algoritmos (Regressão Logística, Random Forest, XGBoost e LightGBM).
  3. **Métricas Validadas (10-Fold Stratified CV):** ROC-AUC, PR-AUC, F1-Score, Sensibilidade (Recall), Especificidade, Brier Score.
  4. **Curvas Interativas:** Curvas ROC e Precision-Recall com controle dinâmico do limiar de corte (*threshold slider*).
  5. **Explicabilidade Global (XAI):** Importância de variáveis e impacto clínico (SHAP Values ou Gain Importance).

### 📌 Aba 3: Painel de Gestão Clínica & Cockpit de Leitos
* **Objetivo:** Simular um prontuário/painel em tempo real de beira de leito e central de regulação hospitalar.
* **Componentes Principais:**
  1. **Tabela Dinâmica de Pacientes:**
     - **ID do Paciente:** Hash anonimizado único (ex.: `#PAC-2024-8841`).
     - **Dados Básicos:** Idade, Sexo, Município de Origem, Tipo de Admissão (Urgência vs. Eletiva).
     - **Especialidade do Leito:** Clínica Médica, Cirúrgica, Obstetrícia, Pediatria, UTI, etc.
     - **Motivo da Internação:** Diagnóstico principal (código e descrição legível do CID-10).
     - **Dias Internado:** Tempo decorrido de internação atual.
  2. **Flags Preditivas de Decisão Rápida:**
     - 🚩 **Flag de Longa Permanência (>= 15 dias):** 
       - *Score Preditivo de Risco:* (ex.: 78% - Alto Risco).
       - *Ação Recomendada:* Planejamento precoce de desospitalização, avaliação de retaguarda, fisioterapia motora precoce.
     - ⚠️ **Flag de Risco de Óbito:**
       - *Score Preditivo de Risco:* (ex.: 84% - Crítico).
       - *Ação Recomendada:* Alocação prioritária em leito monitorado/UTI, acionamento do time de resposta rápida (TRR).
  3. **Filtros e Alertas em Tempo Real:** Filtrar pacientes críticos por enfermaria/unidade, buscar paciente por ID e ordenar por prioridade clínica.

### 📌 Aba 4: Dicionário de Dados & Glossário (Planejada para Fase 4)
* **Objetivo:** Tornar o projeto 100% compreensível tanto para recrutadores técnicos/cientistas de dados e profissionais de saúde quanto para gestores e público leigo.
* **Componentes Principais:**
  1. **Dicionário das Variáveis do SIH/SUS:** Explicação clara de cada coluna utilizada (ex.: `n_aih`, `diag_princ`, `qt_diarias`, `carater_internacao`, `transferido_intermun`, `uti_mes_to`).
  2. **Glossário de Métricas e Termos de IA:** Tradução simples e exemplos práticos de termos como `ROC-AUC`, `PR-AUC`, `Brier Score`, `Sensibilidade`, `Especificidade`, `Data Leakage`, `Calibração Isotônica` e `Limiar de Youden`.
  3. **Regra de Comunicação do Dashboard:** Manter termos técnicos padrão da indústria acompanhados de explicações breves e acessíveis em todas as abas.

---

## 🇧🇷 3. Expansão para Escala Brasil (27 UFs - SIH/SUS 2024)

### 📊 Desafio de Volume de Dados:
- **Abrangência:** 27 Unidades da Federação x 12 meses de 2024 = **324 arquivos `.dbc`**.
- **Volume Estimado:** ~12 a 14 milhões de registros de internação/ano.
- **Tamanho Bruto:** ~15 a 25 GB de dados tabulares.

### ⚙️ Estratégia de Processamento de Alto Desempenho:
1. **Conversão Colunar Otimizada:**
   - Descompactação de `.dbc` para arquivos `.parquet` com compressão `snappy` ou `zstd`.
   - **Particionamento:** `dados_sih/ano=2024/uf={UF}/mes={MM}.parquet`.
   - Redução drástica de memória: leitura seletiva apenas das colunas estritamente necessárias (*projection pushdown*).
2. **Motor de Processamento Eficiente:**
   - Utilização de **DuckDB** ou **Polars** em vez de Pandas tradicional para processamento local ou em instâncias econômicas.
   - O DuckDB permite consultas analíticas SQL em segundos diretamente sobre múltiplos arquivos Parquet sem estourar a memória RAM.

---

## ☁️ 4. Arquitetura de Dados em Nuvem & Ingestão Online

### Opção A: Cloud-Native Serverless (Google Cloud Platform - Recomendada)
* **Ingestão Automatizada:** Script agendado (Cloud Functions / Cloud Run Job) utilizando `pysus` ou FTP oficial do DataSUS (`ftp.datasus.gov.br`).
* **Data Lake:** Google Cloud Storage (GCS) em bucket seguro contendo a base bruta e refinada em Parquet.
* **Data Warehouse:** **Google BigQuery** (Serverless).
  - Permite consultas analíticas instantâneas sobre os 13 milhões de registros.
  - Custo baixíssimo (coberto quase integralmente pela camada gratuita mensal de 1 TB de processamento do BigQuery).
* **Camada de Aplicação:** O dashboard consome views otimizadas ou tabelas agregadas do BigQuery via Service Account.

### Opção B: Arquitetura Híbrida / Low-Cost (DuckDB + Hugging Face / GitHub / GCS)
* Pipeline de extração roda em lote (batch ETL) e gera:
  1. Base agregada nacional para a Aba 1 (Epidemiologia) - < 50 MB.
  2. Artefatos de modelos treinados (`.joblib` ou `ONNX`) para a Aba 2.
  3. Amostra estratificada representativa (~100 mil pacientes) para o simulador de gestão da Aba 3.
* O dashboard roda localmente ou na nuvem carregando dados diretamente em cache.

---

## 🌐 5. Deploy e Disponibilização Online Pública

Para transformar o projeto em uma vitrine de alto impacto para recrutadores e executivos de saúde:

### 1. Plataforma de Hospedagem:
* **Streamlit Community Cloud (Gratuito):** Deploy contínuo integrado diretamente ao repositório GitHub do projeto.
* **Hugging Face Spaces (Docker / Streamlit):** Excelente estabilidade, suporte a até 16 GB de RAM gratuita e alta visibilidade na comunidade de IA/ML.
* **Google Cloud Run (Produção Corporativa):** Container Docker escalável até zero (paga-se frações de centavos por acesso).

### 2. Estratégia de Storytelling & Portfólio Executivo:
* **README.md do GitHub no padrão internacional:**
  - Badges de status, arquitetura visual (Mermaid), GIFs/vídeos curtos demonstrando a usabilidade do dashboard.
  - Resumo executivo voltado ao negócio: *“Como a predição precoce de permanência reduz custos hospitalares e melhora a alocação de leitos no SUS”*.
* **Artigo de Divulgação (LinkedIn / Medium / Substack):**
  - Publicação detalhando o desafio técnico de processar 13M de registros públicos e o impacto clínico do modelo dual-triage.
  - Link direto clicável para o dashboard online interativo.

---

## 🗓️ 6. Fases de Execução

| Fase | Etapa | Entregável Principal |
| :---: | :--- | :--- |
| **Fase 1** | **Construção do Dashboard Inicial (Acre - Piloto)** | Dashboard Streamlit e HTML com as 3 abas funcionais (Epidemiologia, Modelos, Cockpit Clínico com flags). *(Concluído - Layout será retomado e refinado futuramente)* |
| **Fase 2** | **Pipeline de Ingestão Online & Cloud** | Script automatizado para download dos 324 `.dbc` do Brasil 2024 via FTP/PySUS e conversão para Parquet particionado. |
| **Fase 3** | **Escalabilidade & Treinamento Nacional** | Integração com DuckDB/BigQuery e calibração dos modelos preditivos com dados nacionais. |
| **Fase 4** | **Aba 4 (Dicionário), Layout (UI/UX) & Deploy Público** | Implementação da Aba 4 (Dicionário de Dados & Glossário), revisão visual/ergonômica do dashboard, publicação na nuvem, documentação do repositório e material de portfólio profissional. |

> [!NOTE]
> **Entregáveis Registrados para Encerramento (Fase 4 / Final):**
> 1. **Aba 4 — Dicionário de Dados & Glossário:** *(Implementado no dashboard com busca em tempo real e quebra de linha).*
> 2. **Layout do Dashboard (UI/UX):** *(Refinamento inicial aplicado; ajustes visuais finais podem ser retomados após feedback da publicação online).*
> 3. **Documento Final Consolidado (Storytelling & Metodologia):** Ao final de todas as etapas e publicação, gerar documento completo reunindo todo o racional técnico e clínico — justificativas de amostragem (`1,5M` e `30k` vs. `13,58M`), evolução do V1 ao V4 Nacional, remoção ética de variáveis, calibração isotônica e passo a passo integral até o modelo em produção.
