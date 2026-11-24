# Relatório de Análise Preditiva e Storytelling: Internações de Longa Permanência no SUS (Acre, 2024)

**Autoria:** Ciência de Dados & Inteligência em Saúde Pública  
**Data:** Novembro de 2026  
**Base Analisada:** SIH/SUS Acre (2024) — Arquivos `sih_acre_2024_completo.csv` e `sih_modelagem.csv`  

---

## 1. Resumo Executivo

### O Contexto e o Desafio Operacional
No Sistema Único de Saúde (SUS), a gestão eficiente dos leitos hospitalares é um dos maiores desafios de saúde coletiva, especialmente em estados da Região Norte como o Acre, onde os recursos assistenciais e de terapia intensiva são geograficamente concentrados e escassos. A identificação precoce — já no ato da admissão — dos pacientes com alto risco de **longa permanência** ($\ge 15$ dias) permite planejamento antecipado de desospitalização, alocação racional de equipes multidisciplinares e prevenção de eventos adversos (como infecções hospitalares).

### Principais Conclusões
1. **Desbalanceamento Extremo (17,88 : 1):** Das 56.459 internações válidas analisadas em 2024, apenas **2.991 (5,30%)** evoluíram para longa permanência, enquanto **53.468 (94,70%)** foram internações curtas (< 15 dias).
2. **Fatores Determinantes de Risco:** 
   - A idade média dos pacientes com longa permanência é de **49,40 anos** (mediana de **52,0 anos**), comparada a **35,72 anos** (mediana de **32,0 anos**) nas internações curtas — uma diferença de **20 anos na mediana**.
   - Transtornos Mentais e Comportamentais (Capítulo V do CID-10) apresentam a maior proporção clínica de longa permanência (**18,84%** dos casos), seguidos por Neoplasias (**12,27%**). Em contraste, o grupo de Gravidez, Parto e Puerpério (Capítulo XV), apesar de representar o maior volume absoluto de internações (14.404 casos), evolui para longa internação em apenas **0,56%** dos eventos.
3. **Desempenho dos Modelos e Escolha Operacional:**
   - Para o SUS, o **Falso Negativo custa mais do que o Falso Positivo**: deixar de identificar um paciente crônico/grave impede a intervenção preventiva.
   - A **Regressão Logística** demonstrou a melhor eficiência para triagem de admissão, atingindo um **Recall de 76,53%** para a classe "Longa" (capturando 2.289 das 2.991 internações longas) com **72,07% de acurácia** e **71,82% de especificidade**.
   - O **LightGBM** apresentou o melhor equilíbrio global com o maior F1-Score da classe Longa (**27,06%**), acurácia de **83,71%**, especificidade de **85,20%** e recall de **57,04%**.
   - O **Random Forest** sofreu com viés conservador devido ao desbalanceamento (Recall de apenas **31,13%**), enquanto o **XGBoost** sofreu superpenalização pelo uso combinado de SMOTE e `scale_pos_weight=18` (Recall artificial de **97,93%** à custa de 45.237 alarmes falsos e acurácia de apenas **19,77%**).

---

## 2. Metodologia e Desenho Experimental

### 2.1 Origem dos Dados e Critérios de Coorte
Os microdados brutos foram extraídos dos arquivos mensais de Autorização de Internação Hospitalar (AIH Reduzida) do estado do Acre referentes a todo o ano de 2024 (`sih_acre_2024_completo.csv`), consolidando 12 meses de processamento.

Para viabilizar a modelagem sem vazamento de dados (*data leakage*):
- Foram filtradas as internações com diárias válidas (`qt_diarias > 0`).
- O target binário `Longa_Permanencia` foi fixado conforme o marco clínico e de regulação hospitalar:
  $$\text{Longa\_Permanencia} = \begin{cases} 1, & \text{se } \text{qt\_diarias} \ge 15 \\ 0, & \text{se } \text{qt\_diarias} < 15 \end{cases}$$
- Foram mantidas estritamente **variáveis disponíveis no momento da internação**:
  - Numéricas: `idade_anos` (idade calculada em anos contínuos a partir de `idade` e `cod_idade`).
  - Categóricas: `sexo`, `capitulo_cid` (derivado da primeira letra do `diag_princ`), `complex` (nível de complexidade) e `cobranca` (motivo/tipo de cobrança na autorização).
- Variáveis apuradas apenas na alta ou fechamento da AIH (como `val_tot`, `val_sh`, `val_sp`, `val_uti` e `morte`) foram excluídas da matriz de modelagem para garantir que o preditor seja aplicável em tempo real na recepção do paciente.

### 2.2 Tratamento do Desbalanceamento e Validação Cruzada
- **Pipeline:** Imputação/Pré-processamento com `ColumnTransformer` (StandardScaler para idade; OneHotEncoder para variáveis categóricas com tratamento de categorias desconhecidas).
- **Balanceamento:** Aplicação de **SMOTE (Synthetic Minority Over-sampling Technique)** dentro de cada dobra de validação (via `imblearn.pipeline.Pipeline`), assegurando isolamento contra vazamento entre treino e teste.
- **Validação Cruzada:** Esquema estratificado de 10 dobras (`StratifiedKFold`, $k=10$, `random_state=42`), reproduzindo exatamente as proporções populacionais em cada partição de teste.

---

## 3. Análise Exploratória de Dados (EDA)

### 3.1 Perfil Dimensional e Qualidade de Preenchimento
- **Base Bruta (`sih_acre_2024_completo.csv`):** 57.272 linhas e 113 colunas.
- **Percentual de Missing nas Colunas:**
  - Diagnósticos secundários: `diagsec6` a `diagsec9` apresentam **100,00%** de nulos; `diagsec4` tem **99,998%** (57.271 faltantes); `diagsec3` tem **99,988%**; `diagsec2` tem **99,925%**; e `diagsec1` possui **89,65%** de dados ausentes (51.344 faltantes).
  - Campos administrativos como `num_proc`, `cpf_aut`, `gestor_dt`, `infehosp` e `sis_just` registraram **100,00%** de nulos.
  - Campos de habilitação e custeio: `faec_tp` com **81,10%** de ausência; `cnpj_mant` com **14,12%**; e `cgc_hosp` com **0,45%**.
- **Base Filtrada de Modelagem (`sih_modelagem.csv`):** 56.459 linhas e 7 colunas, sem valores ausentes remanescentes.

### 3.2 Distribuição da Variável Alvo
Conforme demonstrado visualmente no gráfico [fig_distribuicao_target.png](file:///c:/Projetos/conecte_sus/figuras/fig_distribuicao_target.png), a base apresenta um desbalanceamento acentuado:
- **Curta Permanência (< 15 dias):** 53.468 registros (**94,70%**).
- **Longa Permanência ($\ge 15$ dias):** 2.991 registros (**5,30%**).
- **Razão de Desbalanceamento:** **17,88 : 1**.

### 3.3 Estatísticas Descritivas das Variáveis Numéricas por Classe
A tabela abaixo sintetiza o comportamento das variáveis quantitativas obtidas diretamente dos microdados:

| Variável | Classe | Contagem | Média | Desvio Padrão | Mínimo | Q25 (1º Q) | Mediana | Q75 (3º Q) | Máximo |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Idade (anos)** | Curta (0) | 53.468 | 35,72 | 21,92 | 0,0 | 21,0 | 32,0 | 50,0 | 99,0 |
| **Idade (anos)** | Longa (1) | 2.991 | 49,40 | 23,46 | 0,0 | 33,0 | 52,0 | 68,0 | 98,0 |
| **Diárias (dias)** | Curta (0) | 53.468 | 3,32 | 2,79 | 1,0 | 1,0 | 2,0 | 4,0 | 14,0 |
| **Diárias (dias)** | Longa (1) | 2.991 | 25,82 | 13,90 | 15,0 | 17,0 | 22,0 | 30,0 | 337,0 |

As distribuições comparativas e assimetrias de idade e diárias estão ilustradas em [fig_boxplots_classe.png](file:///c:/Projetos/conecte_sus/figuras/fig_boxplots_classe.png).

### 3.4 Principais Variáveis Categóricas e a Relação com a Classe Alvo

#### Sexo do Paciente
- **Feminino (`sexo = 3`):** 35.185 internações (62,32% da amostra total), das quais apenas **1.307 (3,71%)** foram de longa permanência.
- **Masculino (`sexo = 1`):** 21.274 internações (37,68% da amostra total), com **1.684 (7,92%)** de longa permanência.
- *Insight:* Pacientes do sexo masculino apresentam taxa de longa permanência **2,13 vezes superior** às mulheres, reflexo da menor busca por cuidados preventivos e maior exposição a traumas e agravos cardiovasculares.

#### Complexidade da Internação (`complex`)
- **Média Complexidade (`complex = 2`):** 54.376 internações (96,31% do total), com **2.762 (5,08%)** de longa permanência.
- **Alta Complexidade (`complex = 3`):** 2.083 internações (3,69% do total), com **229 (10,99%)** de longa permanência.
- *Insight:* A alta complexidade dobra o risco proporcional de estadia prolongada (**10,99% vs 5,08%**).

#### Diagnóstico Principal por Capítulo do CID-10
A distribuição das condições clínicas revela disparidades marcantes na probabilidade de internação prolongada:
- **Capítulo V - Transtornos Mentais:** 1.619 casos, sendo **305 de longa permanência (18,84%)** — a mais alta proporção entre todos os capítulos.
- **Capítulo II - Neoplasias:** 2.030 casos, sendo **249 de longa permanência (12,27%)**.
- **Capítulo IX - Aparelho Circulatório:** 3.583 casos, com **342 longas (9,55%)**.
- **Capítulo I - Doenças Infecciosas:** 2.807 casos, com **251 longas (8,94%)**.
- **Capítulo X - Aparelho Respiratório:** 4.979 casos, com **425 longas (8,54%)**.
- **Capítulo XV - Gravidez e Parto:** 14.404 casos (maior contingente isolado), com apenas **81 longas (0,56%)**.

#### Motivo de Cobrança (`cobranca`)
- Códigos de permanência contínua ou prorrogação (`cobranca = 22` e `21`) apresentam taxas de longa permanência de **85,29%** (145 de 170) e **62,26%** (33 de 53).
- O código `61` (alta da parturiente) concentra 11.305 eventos com taxa ínfima de longa permanência (**0,44%**).

### 3.5 Matriz de Correlação das Variáveis Numéricas
Conforme evidenciado em [fig_correlacao_heatmap.png](file:///c:/Projetos/conecte_sus/figuras/fig_correlacao_heatmap.png), as diárias hospitalares apresentam forte correlação positiva com:
- **Valor Total (`val_tot`):** $r = 0,55$, comprovando o impacto orçamentário direto do prolongamento da estadia.
- **Serviços Hospitalares (`val_sh`):** $r = 0,52$.
- **Diárias de Acompanhante (`diar_acom`):** $r = 0,38$.
- **Valor de UTI (`val_uti`):** $r = 0,26$.
- **Óbito (`morte`):** $r = 0,08$, demonstrando que internações longas não têm necessariamente correlação direta linear com desfecho fatal, mas sim com cronicidade e reabilitação lenta.

### 3.6 Os 4 Achados Mais Relevantes do EDA
1. **O Efeito "Obstetrícia":** As internações obstétricas (Capítulo XV) inflam o denominador hospitalar (mais de 25% de todo o volume), mas quase nunca viram longa permanência (apenas 0,56%). Isso mascara a taxa real de ocupação de leitos de clínica médica e cirúrgica.
2. **Carga Psiquiátrica Crônica:** Transtornos mentais representam o maior ponto crítico de retenção de leitos no estado (quase 1 em cada 5 internações psiquiátricas ultrapassa 15 dias).
3. **Assimetria Etária Crítica:** Pacientes idosos (Q75 de 68 anos) dominam as longas permanências, exigindo linhas de cuidado integradas de geriatria e retaguarda.
4. **Subnotificação Quase Total de Comorbidades Secundárias:** A ausência de dados em mais de 89% dos campos de diagnósticos secundários (`diagsec1`-`9`) impede que o algoritmo saiba o grau de multimorbidade prévia do paciente na admissão.

---

## 4. Avaliação Comparativa dos Modelos Preditivos (Baseline)

Todos os modelos foram avaliados via validação cruzada estratificada em 10 dobras sobre as 56.459 observações.

### 4.1 Tabela Geral de Desempenho

| Modelo | TN | FP | FN | TP | Acurácia | Precisão (Longa) | Recall (Longa) | Especificidade (Curta) | F1-Score (Longa) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **38.403** | 15.065 | **702** | **2.289** | **72,07%** | 13,19% | **76,53%** | **71,82%** | 22,50% |
| **Random Forest** | 49.403 | **4.065** | 2.060 | 931 | 89,15% | **18,63%** | 31,13% | **92,40%** | 23,31% |
| **XGBoost** | 8.231 | 45.237 | **62** | 2.929 | 19,77% | 6,08% | **97,93%** | 15,39% | 11,45% |
| **LightGBM** | 45.557 | 7.911 | 1.285 | 1.706 | **83,71%** | 17,74% | 57,04% | 85,20% | **27,06%** |

*As matrizes de confusão individuais e o painel comparativo estão disponíveis em [fig_matrizes_confusao_todos_sus.png](file:///c:/Projetos/conecte_sus/figuras/fig_matrizes_confusao_todos_sus.png).*

---

### 4.2 Interpretação Detalhada por Modelo e Custos de Decisão

No contexto da gestão hospitalar do SUS:
- **Custo do Falso Negativo (FN - Internação longa classificada como curta):** Altíssimo. O hospital não antecipa a reserva de leito de retaguarda, não programa suporte nutricional/fisioterápico intensivo precoce e enfrenta bloqueio involuntário de leito que sobrecarrega a emergência.
- **Custo do Falso Positivo (FP - Internação curta classificada como longa):** Moderado. O paciente recebe monitoramento preventivo ou visita da comissão de desospitalização que confirma a alta precoce, gerando apenas pequeno esforço administrativo extra.

#### 1. Regressão Logística (Modelo Recomendado para Triagem Operacional)
- **Desempenho:** Atingiu **Recall de 76,53%**, recuperando 2.289 dos 2.991 casos de longa permanência, enquanto manteve uma especificidade aceitável de **71,82%** (38.403 verdadeiros negativos) e acurácia global de **72,07%**.
- **Vantagem:** Apresenta fronteira de decisão suave e linear calibrada pelo SMOTE, sem hiperajustar ruídos. É interpretável, permitindo expressar o risco em termos de *odds ratios* compreensíveis para médicos reguladores.

#### 2. LightGBM (Modelo Recomendado para Alocação Balanceada de Recursos)
- **Desempenho:** Campeão no **F1-Score (27,06%)**, combinando sólida acurácia (**83,71%**), boa especificidade (**85,20%**) e um Recall moderado-alto (**57,04%**, identificando 1.706 casos longos).
- **Vantagem:** Produz muito menos alarmes falsos (apenas 7.911 FP contra 15.065 da Regressão Logística). Ideal quando a equipe de gestão de leitos possui capacidade operacional restrita para auditar pacientes sinalizados.

#### 3. Random Forest (Inadequado para Triagem de Risco)
- **Desempenho:** Atingiu alta acurácia aparente (**89,15%**) e alta especificidade (**92,40%**), mas falhou no objetivo central: seu Recall foi de apenas **31,13%**, gerando **2.060 Falsos Negativos**. O algoritmo convergiu para privilegiar a classe majoritária, ignorando mais de dois terços dos pacientes crônicos.

#### 4. XGBoost (Diagnóstico de Efeito Colateral na Ponderação)
- **Desempenho:** Exibiu Recall de **97,93%**, mas com acurácia desastrosa de **19,77%** e apenas **15,39%** de especificidade, classificando 48.166 pacientes como "Longa".
- **Causa Técnica:** A combinação cumulativa do reamostragem sintética (SMOTE) com o hiperparâmetro de penalização `scale_pos_weight=18` gerou uma supercompensação de ordem de grandeza duplicada ($18 \times \text{SMOTE}$), empurrando quase todas as predições para a classe positiva.

---

## 5. Interpretação das Variáveis Mais Importantes

O ranking e a magnitude de importância de variáveis (disponíveis em [fig_importancias_todos_sus.png](file:///c:/Projetos/conecte_sus/figuras/fig_importancias_todos_sus.png)) revelam os principais vetores determinantes:

### Top Variáveis por Modelo
1. **Regressão Logística (Magnitude dos Coeficientes $|\beta|$):**
   - `cobranca_22` ($|\beta| = 4,6178$): Código de prorrogação continuada.
   - `cobranca_64` ($|\beta| = 3,1181$) e `cobranca_21` ($|\beta| = 2,8270$).
   - `capitulo_cid_V - Mentais` ($|\beta| = 1,8796$): Forte coeficiente positivo elevando o risco de longa permanência.
   - `capitulo_cid_XXI - Fatores de saude` ($|\beta| = 1,8500$).
2. **Random Forest (Gini Importance):**
   - `idade_anos` (**0,5856**): Detém mais de 58% de todo o ganho de pureza das árvores.
   - `capitulo_cid_XV - Gravidez/Parto` (**0,0495**): Principal divisor negativo de águas.
   - `cobranca_12` (**0,0449**) e `capitulo_cid_V - Mentais` (**0,0359**).
3. **LightGBM (Feature Splits / Ganho):**
   - `idade_anos` (**1.690 splits**): Principal variável contínua para ramificações profundas.
   - `sexo_1` (Masculino, **81 splits**).
   - `complex_2` (Média complexidade, **70 splits**).
   - `cobranca_12` (**69 splits**) e `cobranca_15` (**61 splits**).

### Significado Clínico e Administrativo
- **Idade como proxy de reserva fisiológica:** Pacientes idosos internam com menor capacidade funcional e maior incidência de delirium, descondicionamento e complicações infecciosas, prolongando o tempo necessário para desmame medicamentoso e reabilitação.
- **Saúde Mental e a Fragilidade da Rede de Atenção Psicossocial (RAPS):** O longo tempo de permanência em Transtornos Mentais reflete um gargalo crônico do SUS: a dificuldade de transferir o paciente em crise estabilizada para residências terapêuticas, CAPS III ou suporte familiar acolhedor, transformando leitos de internação em leitos de acolhimento social involuntário.
- **A Obstetrícia como Fator Protetor Operacional:** A parturição e procedimentos ginecológicos possuem protocolos clínicos estritos de alta em 24 a 48 horas (código `61` e Capítulo XV), servindo aos modelos como fortes âncoras de decisão para curta permanência.

---

## 6. Limitações Metodológicas e Viés dos Dados

1. **Natureza Faturadora do SIH/SUS (Viés de Registro):** O banco do SIH baseia-se na AIH, instrumento originalmente desenhado para ressarcimento financeiro de procedimentos, e não prontuário médico. Procedimentos mais complexos ou que remuneram melhor podem induzir escolhas de codificação específicas pelo faturista hospitalar.
2. **Ausência de Comorbidades Secundárias na Admissão:** Conforme detectado no EDA, campos essenciais como `diagsec1` a `diagsec9` estão ausentes em quase **90% a 100%** dos prontuários. Isso impossibilita a criação de índices consagrados de comorbidade (como o Índice de Comorbidade de Charlson ou Elixhauser).
3. **Ausência de Dados Clínicos Granulares:** A base não contempla sinais vitais à admissão (pressão arterial, saturação de $O_2$, Glasgow), exames laboratoriais (creatinina, leucócitos) ou uso prévio de medicamentos, fatores que influenciam criticamente o tempo de recuperação.
4. **Generalização Geográfica Restrita ao Acre:** Os resultados refletem a realidade socioassistencial do Acre em 2024 (população com forte contingente florestal/indígena, transporte fluvial dependente de regimes de cheias para transferências e alta concentração tecnológica exclusiva em Rio Branco). Aplicar este modelo diretamente a estados com maior malha hospitalar privada ou redes de retaguarda robustas (como SP ou RS) exigirá calibração local prévia.

---

## 7. Recomendações Práticas e Próximos Passos

1. **Implementação de Gatilho de Alerta na Admissão:**
   - Adotar a **Regressão Logística** como triagem padrão para identificar pacientes em risco de longa permanência (sensibilidade de **76,53%**).
   - Todo paciente admitido com pontuação positiva deve acionar automaticamente a equipe multiprofissional (assistência social, fisioterapia e nutrição) nas primeiras 48 horas.
2. **Protocolo Dedicado para Saúde Mental:**
   - Estabelecer fluxo direto entre a regulação de leitos hospitalares e os CAPS/redes de acolhimento psicossocial para evitar que internações psiquiátricas superem 14 diárias por falta de suporte sociofamiliar.
3. **Refinamento Técnico do Algoritmo:**
   - Remover a duplicação no XGBoost (eliminar `scale_pos_weight=18` quando o pipeline já utilizar SMOTE, ou usar apenas ponderação de perda sem gerar instâncias sintéticas).
   - Implementar calibração de probabilidade (*Isotonic Regression* ou *Platt Scaling*) e curvas de custo-benefício para definir limiares decisórios customizados de acordo com a taxa de ocupação da UTI e enfermaria.

---

## 8. Arquitetura Definitiva (V3): IA Clínica de Admissão & Base Epidemiológica

Para garantir conformidade bioética estrita e eliminar sobreajustes temporais, o projeto consolidou-se em dois pilares independentes:
1. **Sistema Preditivo Clínico de Admissão (8 Features Estritas):** Idade, sexo, diagnóstico primário (CID-10), especialidade do leito, caráter da internação, complexidade tecnológica, transferência intermunicipal e fim de semana. Foram expurgadas `raca_cor` (para evitar viés algorítmico discriminatório na alocação de leitos) e `mes_internacao` (para evitar sobreajuste temporal ao calendário de 2024 e *split bias* das árvores).
2. **Base de Vigilância Epidemiológica ([sih_perfil_epidemiologico_acre.csv](file:///c:/Projetos/conecte_sus/dados/sih_perfil_epidemiologico_acre.csv)):** Mantém todas as 13 variáveis (incluindo raça/cor para monitoramento de equidade de saúde indígena/ribeirinha, sazonalidade e geografia) para a futura construção do **Dashboard de Vigilância e Gestão do SUS**.

O documento interativo detalhando a matriz de evolução completa está disponível em **[resumo_mudanca_variaveis_v3.html](file:///c:/Projetos/conecte_sus/relatorios/resumo_mudanca_variaveis_v3.html)**.

### 8.1 Desempenho Consolidado dos 4 Modelos Clínicos V3 (10-Fold Stratified CV)

Todos os 4 modelos foram avaliados com ponderação de custo de classe (`class_weight='balanced'` e `scale_pos_weight = N_neg / N_pos`), eliminando distorções de SMOTE e gerando probabilidades calibradas:

| Desfecho | Modelo | ROC-AUC | PR-AUC (Baseline) | Sensibilidade (Recall) | Especificidade | Acurácia | Brier Score | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Longa Permanência** | **Logistic Regression** | 0,8095 | 0,1941 (0,053) | **78,23%** (2.340/2.991) | 71,10% | 71,48% | 0,1841 | 22,52% |
| **Longa Permanência** | **Random Forest** | 0,8186 | 0,2189 (0,053) | 74,36% (2.224/2.991) | **74,26%** | **74,27%** | 0,1640 | 23,44% |
| **Longa Permanência** | **XGBoost** | 0,8258 | 0,2272 (0,053) | **78,00%** (2.333/2.991) | 73,17% | 73,43% | 0,1675 | 23,72% |
| **Longa Permanência** | **LightGBM** | **0,8263** | **0,2294** (0,053) | **77,93%** (2.331/2.991) | **73,49%** | **73,73%** | **0,1630** | **23,91%** |
| **Óbito Intra-hospitalar** | **Logistic Regression** | 0,8882 | 0,1804 (0,027) | **83,01%** (1.265/1.524) | 79,54% | 79,63% | 0,1454 | 18,03% |
| **Óbito Intra-hospitalar** | **Random Forest** | 0,8896 | 0,1854 (0,027) | 79,86% (1.217/1.524) | **82,82%** | **82,74%** | 0,1263 | **19,99%** |
| **Óbito Intra-hospitalar** | **XGBoost** | **0,8958** | **0,1903** (0,027) | **82,55%** (1.258/1.524) | 81,12% | 81,16% | 0,1309 | 19,13% |
| **Óbito Intra-hospitalar** | **LightGBM** | **0,8949** | **0,1889** (0,027) | **82,22%** (1.253/1.524) | **81,47%** | **81,49%** | **0,1231** | 19,35% |

*Gráficos diagnósticos definitivos gerados:*
- Painel de Matrizes de Confusão - Longa Permanência: [fig_matrizes_confusao_longa_v3.png](file:///c:/Projetos/conecte_sus/figuras/fig_matrizes_confusao_longa_v3.png)
- Painel de Matrizes de Confusão - Óbito Hospitalar: [fig_matrizes_confusao_obito_v3.png](file:///c:/Projetos/conecte_sus/figuras/fig_matrizes_confusao_obito_v3.png)
- Curvas ROC e Precision-Recall (4 Modelos x 2 Desfechos): [fig_roc_pr_curvas_v3.png](file:///c:/Projetos/conecte_sus/figuras/fig_roc_pr_curvas_v3.png)
- Curvas de Calibração de Probabilidade: [fig_calibracao_probabilidade_v3.png](file:///c:/Projetos/conecte_sus/figuras/fig_calibracao_probabilidade_v3.png)
- Importâncias Clínicas Consolidadas: [fig_importancias_admissao_v3.png](file:///c:/Projetos/conecte_sus/figuras/fig_importancias_admissao_v3.png)

### 8.2 Principais Conclusões do Modelo V3
1. **Desempenho Notável de XGBoost e LightGBM (ROC-AUC ~0,896 e ~0,895):** O ajuste fino do `scale_pos_weight` corrigiu a falha anterior do XGBoost, elevando-o ao topo do ranking juntamente com o LightGBM, capturando mais de **82% de todos os óbitos hospitalares**.
2. **Máxima Sensibilidade na Regressão Logística (83,01% em Óbito e 78,23% em Longa):** Para triagem médica que prioriza minimizar Falsos Negativos, a Regressão Logística continua sendo a ferramenta mais sensível e clinicamente transparente.
3. **Simulador Operacional em Produção:** O módulo [simulador_admissao_sus.py](file:///c:/Projetos/conecte_sus/scripts/simulador_admissao_sus.py) opera com as 8 variáveis limpas, entregando escores instantâneos na admissão médica.

### 8.3 Roadmap do Futuro Dashboard Epidemiológico do Acre (Planejado)
A base [sih_perfil_epidemiologico_acre.csv](file:///c:/Projetos/conecte_sus/dados/sih_perfil_epidemiologico_acre.csv) servirá como fonte para o futuro dashboard estruturado em 5 eixos:
1. **Equidade & Saúde Indígena:** Proporção de hospitalizações e letalidade por raça/cor nas regiões do Juruá e Tarauacá.
2. **Sazonalidade Amazônica:** Monitoramento temporal das cheias (Jan-Abr) vs estiagem/fumaça (Ago-Out).
3. **Fluxos Intermunicipais:** Matriz origem-destino dos 22 municípios acreanos em direção a Rio Branco.
4. **Carga de Doença:** Curvas de morbimortalidade por capítulos do CID-10 e faixas etárias.
5. **Capacidade Hospitalar:** Tempo Médio de Permanência (TMP) e giro de leitos por especialidade.

---
*Relatório concluído e validado com base nos microdados reais do SIH/SUS Acre 2024.*
