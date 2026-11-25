# -*- coding: utf-8 -*-
"""
Módulo de Internacionalização (i18n) — Português (PT) & English (EN)
Centraliza textos da interface, mapeamentos de categorias clínicas do SIH/SUS
e tabelas bilíngues do Dicionário de Dados e Glossário.
"""

import pandas as pd

MAPAS_CAT = {
    "pt": {
        "Todas": "Todas",
        "Todos": "Todos",
        "Masculino": "Masculino",
        "Feminino": "Feminino",
        "Outro": "Outro",
        "Nao_Declarada": "Não Declarada",
        "Indigena": "Indígena",
        "Clinica_Medica": "Clínica Médica",
        "Cirurgica": "Cirúrgica",
        "Obstetrica": "Obstétrica",
        "Pediatrica": "Pediatria",
        "Hospital_Dia": "Hospital Dia",
        "Psiquiatrica": "Psiquiatria",
        "Outras": "Outras",
        "Urgencia_Emergencia": "Urgência / Emergência",
        "Eletiva": "Eletiva",
        "Media_Complexidade": "Média Complexidade",
        "Alta_Complexidade": "Alta Complexidade",
    },
    "en": {
        "Todas": "All",
        "Todos": "All",
        "Norte": "North",
        "Nordeste": "Northeast",
        "Centro-Oeste": "Central-West",
        "Sudeste": "Southeast",
        "Sul": "South",
        "Masculino": "Male",
        "Feminino": "Female",
        "Outro": "Other",
        "Branca": "White",
        "Preta": "Black",
        "Parda": "Mixed / Pardo",
        "Amarela": "Asian",
        "Indigena": "Indigenous",
        "Nao_Declarada": "Not Declared",
        "Clinica_Medica": "Internal Medicine",
        "Cirurgica": "Surgical",
        "Obstetrica": "Obstetrics",
        "Pediatrica": "Pediatrics",
        "Hospital_Dia": "Day Hospital",
        "Psiquiatrica": "Psychiatry",
        "Outras": "Other",
        "Urgencia_Emergencia": "Urgency / Emergency",
        "Eletiva": "Elective",
        "Media_Complexidade": "Medium Complexity",
        "Alta_Complexidade": "High Complexity",
        "<1 ano": "<1 yr",
        "1-11 anos": "1-11 yrs",
        "12-18 anos": "12-18 yrs",
        "19-39 anos": "19-39 yrs",
        "40-59 anos": "40-59 yrs",
        "60-79 anos": "60-79 yrs",
        "80+ anos": "80+ yrs",
        "I - Infecciosas": "I - Infectious Diseases",
        "II - Neoplasias": "II - Neoplasms",
        "III - Sangue": "III - Blood Disorders",
        "IV - Endocrinas": "IV - Endocrine / Metabolic",
        "V - Mentais": "V - Mental Disorders",
        "VI - Nervoso": "VI - Nervous System",
        "VII/VIII - Olho/Ouvido": "VII/VIII - Eye & Ear",
        "IX - Circulatorio": "IX - Circulatory System",
        "X - Respiratorio": "X - Respiratory System",
        "XI - Digestivo": "XI - Digestive System",
        "XII - Pele": "XII - Skin & Subcutaneous",
        "XIII - Osteomuscular": "XIII - Musculoskeletal",
        "XIV - Geniturinario": "XIV - Genitourinary",
        "XV - Gravidez/Parto": "XV - Pregnancy / Childbirth",
        "XVI - Perinatal": "XVI - Perinatal Conditions",
        "XVII - Congenitas": "XVII - Congenital Malformations",
        "XVIII - Sintomas": "XVIII - Symptoms & Signs",
        "XIX - Lesoes": "XIX - Injury & Poisoning",
        "XX - Causas externas": "XX - External Causes",
        "XXI - Fatores de saude": "XXI - Health Status Factors",
        "Ignorado": "Unknown",
        "Logistic Regression": "Logistic Regression",
        "Random Forest": "Random Forest",
        "XGBoost": "XGBoost",
        "LightGBM": "LightGBM",
    },
    "es": {
        "Todas": "Todas",
        "Todos": "Todos",
        "Norte": "Norte",
        "Nordeste": "Nordeste",
        "Centro-Oeste": "Centro-Oeste",
        "Sudeste": "Sudeste",
        "Sul": "Sur",
        "Masculino": "Masculino",
        "Feminino": "Femenino",
        "Outro": "Otro",
        "Branca": "Blanca",
        "Preta": "Negra",
        "Parda": "Parda / Mestiza",
        "Amarela": "Asiática",
        "Indigena": "Indígena",
        "Nao_Declarada": "No Declarada",
        "Clinica_Medica": "Medicina Interna",
        "Cirurgica": "Quirúrgica",
        "Obstetrica": "Obstetricia",
        "Pediatrica": "Pediatría",
        "Hospital_Dia": "Hospital de Día",
        "Psiquiatrica": "Psiquiatría",
        "Outras": "Otras",
        "Urgencia_Emergencia": "Urgencia / Emergencia",
        "Eletiva": "Electiva",
        "Media_Complexidade": "Complejidad Media",
        "Alta_Complexidade": "Alta Complejidad",
        "<1 ano": "<1 año",
        "1-11 anos": "1-11 años",
        "12-18 anos": "12-18 años",
        "19-39 anos": "19-39 años",
        "40-59 anos": "40-59 años",
        "60-79 anos": "60-79 años",
        "80+ anos": "80+ años",
        "I - Infecciosas": "I - Infecciosas",
        "II - Neoplasias": "II - Neoplasias",
        "III - Sangue": "III - Sangre",
        "IV - Endocrinas": "IV - Endocrinas / Metabólicas",
        "V - Mentais": "V - Trastornos Mentales",
        "VI - Nervoso": "VI - Sistema Nervioso",
        "VII/VIII - Olho/Ouvido": "VII/VIII - Ojo y Oído",
        "IX - Circulatorio": "IX - Aparato Circulatorio",
        "X - Respiratorio": "X - Aparato Respiratorio",
        "XI - Digestivo": "XI - Aparato Digestivo",
        "XII - Pele": "XII - Piel y Subcutáneo",
        "XIII - Osteomuscular": "XIII - Osteomuscular",
        "XIV - Geniturinario": "XIV - Genitourinario",
        "XV - Gravidez/Parto": "XV - Embarazo y Parto",
        "XVI - Perinatal": "XVI - Afecciones Perinatales",
        "XVII - Congenitas": "XVII - Malformaciones Congénitas",
        "XVIII - Sintomas": "XVIII - Síntomas y Signos",
        "XIX - Lesoes": "XIX - Traumatismos / Lesiones",
        "XX - Causas externas": "XX - Causas Externas",
        "XXI - Fatores de saude": "XXI - Factores de Salud",
        "Ignorado": "Ignorado",
        "Logistic Regression": "Regresión Logística",
        "Random Forest": "Random Forest",
        "XGBoost": "XGBoost",
        "LightGBM": "LightGBM",
    },
}


def tr_val(valor: str, lang: str) -> str:
    s = str(valor)
    return MAPAS_CAT.get(lang, {}).get(s, s)


def tr_series(series: pd.Series, lang: str) -> pd.Series:
    mapa = MAPAS_CAT.get(lang, {})
    if isinstance(series.dtype, pd.CategoricalDtype):
        novas = [mapa.get(str(c), str(c)) for c in series.cat.categories]
        return series.cat.rename_categories(novas)
    return series.astype(str).map(lambda x: mapa.get(x, x))


TEXTOS = {
    "pt": {
        "page_title": "SUS Analytics & Clinical Intelligence",
        "sidebar_title": "## 🏥 SUS Analytics 2024",
        "sidebar_caption": "Projeto de Portfólio em Ciência de Dados & Saúde Pública",
        "sidebar_resumo_hdr": "**📌 Resumo das Abas:**",
        "sidebar_resumo_body": """
- **📊 Aba 1 (Epidemiologia):** Perfil das internações no Brasil por idade, sexo, raça/cor, meses e causas CID-10.
- **🛡️ Aba 2 (Qualidade):** Auditoria de completude (27 UFs × 12 meses), subnotificação e regras de integridade clínica.
- **🤖 Aba 3 (Modelos IA):** Desempenho dos algoritmos, prós/contras, calibração e escolha do LightGBM.
- **🏥 Aba 4 (Gestão Clínica):** Simulador de admissão e painel de leitos com alertas preditivos ajustáveis.
- **📖 Aba 5 (Dicionário):** Catálogo auditado das **113 colunas oficiais do SIH/SUS** + 13 derivadas e glossário.
- **🧭 Aba 6 (Jornada):** Histórico criativo, decisões técnicas, stack usada e evolução V1 → V5.
""",
        "sidebar_motor": "**⚙️ Motor Analítico:**",
        "sidebar_prod": "**🧠 Produção:**\n`LightGBM V5 (Bayes CID3 + Isotônica)`",
        "sidebar_etica": "⚖️ **IA Responsável:** Raça/cor e mês são mantidos na epidemiologia para equidade, mas excluídos dos modelos clínicos preditivos.",
        "modo_local": "Local Completo (13,58M Partições + Cubo)",
        "modo_cloud": "Cloud Otimizado (Cubo 13,58M + Amostra 30k)",
        "main_title": "🏥 SUS Analytics & Clinical Intelligence",
        "sub_title": "Plataforma de Inteligência em Saúde Pública (SIH/SUS Brasil 2024): perfil epidemiológico, previsão de riscos na admissão hospitalar e painel clínico de leitos.",
        "kpi1_lbl": "Internações Analisadas",
        "kpi1_sub": "Microdados SIH/SUS 2024",
        "kpi2_lbl": "Estados Cobertos",
        "kpi2_sub": "Cobertura Nacional",
        "kpi3_lbl": "Média de Óbitos",
        "kpi3_sub": "Mortalidade Hospitalar",
        "kpi4_lbl": "Internações ≥ 15 dias",
        "kpi4_sub": "Longa Permanência",
        "kpi5_lbl": "Poder Preditivo (ROC-AUC)",
        "kpi5_sub": "Óbito / Longa Perm. (V5)",
        "tabs": [
            "📊 1. Perfil Epidemiológico & Equidade",
            "🛡️ 2. Qualidade dos Dados & Integridade",
            "🤖 3. Como a IA Funciona (Modelos & Métricas)",
            "🏥 4. Painel de Gestão Clínica & Leitos",
            "📖 5. Dicionário SIH/SUS (113 Colunas)",
            "🧭 6. Jornada, Decisões & Arquitetura",
        ],
        "epi_hdr": "### 📊 Panorama Epidemiológico das Internações Hospitalares",
        "epi_sub": "Explore quem utiliza os hospitais do SUS no Brasil em 2024, as principais causas de internação e diferenças entre regiões e perfis demográficos.",
        "f_regiao": "Região:",
        "f_uf": "Estado (UF):",
        "f_raca": "Raça / Cor:",
        "f_espec": "Especialidade:",
        "f_sexo": "Sexo:",
        "f_mes": "Mês (2024):",
        "epi_info": "🔎 Exibindo **{n}** internações para os filtros selecionados ({pct:.1f}% do total nacional). *(Consultas processadas diretamente via motor DuckDB)*",
        "g1_hdr": "#### Distribuição por Faixa Etária e Sexo",
        "g1_cap": "Mostra o volume de internações em cada idade, separando homens e mulheres.",
        "g1_x": "Faixa Etária",
        "g1_y": "Internações",
        "g1_c": "Sexo",
        "g2_hdr": "#### Equidade: Mortalidade e Permanência por Raça/Cor",
        "g2_cap": "Compara o tempo médio no hospital e o risco de óbito entre grupos raciais (tamanho da bolha = volume).",
        "g2_x": "Taxa de Óbito (%)",
        "g2_y": "Média de Diárias",
        "g3_hdr": "#### Sazonalidade Mensal de Internações e Óbitos",
        "g3_cap": "Acompanha como o volume de entradas (barras) e a mortalidade (linha) variam ao longo dos 12 meses.",
        "g3_b": "Internações",
        "g3_l": "Taxa Óbito (%)",
        "g3_x": "Mês de 2024",
        "g3_y1": "Total de Internações",
        "g3_y2": "Mortalidade (%)",
        "g4_hdr": "#### Top Causas por Capítulo da CID-10",
        "g4_cap": "Principais motivos médicos das internações; passe o mouse para ver a taxa exata de óbito e longa permanência.",
        "g4_y": "Capítulo CID-10",
        "g4_x": "Internações",
        "g4_c": "Óbito (%)",
        "g4_hover": "<b>%{y}</b><br>Internações: %{x:,}<br>Taxa de Óbito: <b>%{customdata[0]:.2f}%</b><br>Longa Permanência: %{customdata[1]:.2f}%<extra></extra>",
        "ufs_hdr": "#### 🗺️ Comparativo por Estado (UFs Carregadas)",
        "col_uf": "UF",
        "col_reg": "Região",
        "col_int": "Internações",
        "col_ob_pct": "Taxa Óbito (%)",
        "col_lg_pct": "Longa Perm. (%)",
        "col_med_d": "Média Diárias",
        "dq_hdr": "### 🛡️ Qualidade dos Dados: Perfil Epidemiológico & Treinamento da IA",
        "dq_sub": "Auditoria focada exclusivamente nas variáveis que alimentam o **Perfil Epidemiológico (Aba 1)** e o **Treinamento Preditivo (Aba 3 — 9 features + 2 alvos)**.",
        "dq_banner_local": "🔎 **Escopo Local (100% da Base):** Cobertura e completude medidas sobre **todas as {n} internações** no Cubo; nulos e validade das variáveis de treino auditados linha a linha nos Parquets locais (`{na}` linhas).",
        "dq_banner_cloud": "🔎 **Escopo Híbrido (Nuvem):** Cobertura e subnotificação do Perfil medidas sobre **todas as {n} internações** no Cubo; nulos e limites das features de treino auditados na amostra estratificada (`{na}` casos das 27 UFs).",
        "dq_kpi1_lbl": "Cobertura UF × Mês",
        "dq_kpi1_sub": "Partições Ativas (Perfil)",
        "dq_kpi2_lbl": "Completude Treino (9 Features)",
        "dq_kpi2_sub": "Zero Nulos na Admissão",
        "dq_kpi3_lbl": "Violações Domínio / Alvo",
        "dq_kpi3_sub": "Idade / Diárias / CID Inválidos",
        "dq_kpi4_lbl": "Subnotificação / Inespecíficos",
        "dq_kpi4_sub": "Raça N.Decl. / Espec. Outras",
        "dq_contract_hdr": "#### 📋 Auditoria Coluna a Coluna (Apenas Variáveis do Perfil & Treinamento)",
        "dq_contract_sub": "Completude, faixa válida e papel exato de cada campo que entra nos painéis epidemiológicos e nos pipelines do LightGBM V5:",
        "dq_status_col": "Status",
        "dq_uf_tbl_hdr": "#### 🗺️ Completude Estadual das Variáveis de Perfil",
        "dq_uf_cols": {
            "uf": "UF",
            "regiao": "Região",
            "internacoes": "Internações",
            "meses_ativos": "Meses Ativos",
            "pct_raca_nd": "Raça Não Decl. (%)",
            "pct_cid_ign": "CID Ignorado (%)",
            "pct_esp_outras": "Espec. Outras (%)",
        },
        "dq_other_note": """⚠️ **Sobre as demais colunas do SIH/SUS (não auditadas em detalhe aqui):**
Por enquanto esta aba foca apenas no que entra no **Perfil (Aba 1)** e no **Treinamento da IA (Aba 3)**. Outras colunas do arquivo oficial que poderiam ser clinicamente ricas ficaram de fora após auditoria prévia por apresentarem **extrema falta de dados** (ex.: diagnósticos secundários/comorbidades `diagsec1`–`diagsec9`, ausentes em `89,65%` a `100%` dos registros), **graves inconsistências de faturamento/preenchimento** na origem ou **vazamento temporal pós-alta (*data leakage*)**. Todas as `113` colunas seguem documentadas na **Aba 5 (Dicionário)**.""",
        "ml_hdr": "### 🤖 Como a Inteligência Artificial Funciona & Escolha do Modelo (V5 Nacional)",
        "ml_intro": """No momento em que o paciente entra no hospital, dois modelos preditivos operam juntos usando apenas **9 informações da admissão** (idade, sexo, capítulo CID-10 + diagnóstico específico `cid3`, especialidade do leito, caráter urgência/eletivo, complexidade, se veio transferido de outro município e se entrou no fim de semana):
1. **Modelo de Longa Permanência (≥ 15 dias):** estima a chance de o paciente precisar de internação prolongada, apoiando o giro de leitos e o serviço social.
2. **Modelo de Óbito Intra-hospitalar:** estima a gravidade de risco fatal para priorizar monitoramento intensivo ou UTI.""",
        "ml_why": """🏆 **Por que escolhemos o LightGBM V5 como modelo oficial?**
- **Líder isolado nos dois desfechos:** Com a codificação híbrida de `cid3` (`Top-200 OHE` + *Target Encoding Bayesiano* com *shrinkage* no capítulo) treinada sobre `2,4M` internações, venceu todos os concorrentes em **Óbito (`ROC-AUC 0,9082` | `PR-AUC 0,3035` | `Brier 0,0306`)** e **Longa Permanência (`ROC-AUC 0,8481` | `PR-AUC 0,3663` | `Brier 0,0471`)**.
- **Salto expressivo em precisão de eventos raros:** Subiu o `PR-AUC` de óbito em **+44,5%** frente à V4 (`0,2101 → 0,3035`), mantendo **`89,58%`** de detecção em óbito e **`85,32%`** em longa permanência.
- **Leveza extrema na nuvem:** Cada pipeline calibrado completo ocupa apenas **`0,62 MB`** e pontua em milissegundos.""",
        "ml_guide": """📖 **Mini-Guia das Métricas (Para todos os públicos):**
- **ROC-AUC (0 a 1):** Mede o poder de distinção entre pacientes graves e estáveis (`0,5` = chute aleatório; acima de `0,80` = excelente; `> 0,90` = excepcional).
- **PR-AUC:** Precisão focada em eventos raros (óbito e longa permanência são minoria dos casos).
- **Brier Score (quanto menor, melhor):** Mede se a probabilidade prevista reflete a realidade graças à **Calibração Isotônica** (ex.: se diz 20%, ~20 em 100 pacientes com aquele perfil realmente têm o desfecho).
- **Ética & Sem *Data Leakage*:** `raca_cor` e `mes_internacao` não entram no modelo para evitar preconceito algorítmico; dados pós-alta foram removidos para não "trapacear".""",
        "ml_bench": "#### 📊 Comparativo Nacional V5 (Teste Cego em `300.000` internações das 27 UFs)",
        "ml_ob_sub": "##### 1. Óbito Intra-hospitalar *(Média na Amostra: 3,83%)*",
        "ml_lg_sub": "##### 2. Longa Permanência ≥ 15d *(Média na Amostra: 6,19%)*",
        "ml_cols": {"Modelo": "Modelo", "ROC_AUC": "ROC-AUC", "PR_AUC": "PR-AUC", "Brier_Calibrado": "Brier Score", "Limiar_Otimo_Pct": "Corte Alerta (%)", "Sensibilidade_Pct": "Detecção (%)", "Especificidade_Pct": "Especif. (%)"},
        "ml_models_hdr": "#### 🧠 Como Funciona Cada Modelo — Prós & Contras",
        "ml_models_sub": "Resumo didático de como cada algoritmo avaliado no benchmark nacional V5 decide o risco na admissão hospitalar:",
        "ml_models_cards": [
            {
                "title": "1. Regressão Logística *(Baseline Linear)*",
                "how": "**Como funciona:** Modelo estatístico clássico que soma pesos fixos para cada informação da entrada (idade, CID, leito) e aplica uma curva sigmoide para estimar a chance do desfecho.",
                "pros": "**✅ Prós:** Muito rápido, transparente (mostra o peso exato de cada variável), consome quase zero memória e já nasce bem calibrado (`Brier 0,0312 / 0,0483` na V5).",
                "cons": "**⚠️ Contras:** Assume relação linear simples — não combina interações não lineares (ex.: idade avançada × CID3 específico × urgência), ficando atrás em `ROC-AUC` (`0,8265` em Longa e `0,8956` em Óbito)."
            },
            {
                "title": "2. Random Forest *(Ensemble Bagging)*",
                "how": "**Como funciona:** Constrói centenas de árvores de decisão independentes em sorteios aleatórios dos dados e decide a probabilidade final pela média dos votos de toda a floresta.",
                "pros": "**✅ Prós:** Captura relações não lineares naturalmente e resiste bem a outliers, mantendo bom equilíbrio com `ROC-AUC 0,8942` (Óbito) e `0,8286` (Longa).",
                "cons": "**⚠️ Contras:** Gera arquivos pesados na memória RAM/nuvem, exige subamostragem (`250k`) para treinar rápido e ficou atrás do boosting em `PR-AUC` (`0,2684` em Óbito vs `0,3035` do LightGBM)."
            },
            {
                "title": "3. XGBoost *(Gradient Boosting — Histograma)*",
                "how": "**Como funciona:** Treina árvores em sequência em que cada nova árvore corrige os erros residuais das anteriores, usando histogramas numéricos e regularização L1/L2 integrada.",
                "pros": "**✅ Prós:** Excelente desempenho geral (`ROC-AUC 0,9075` em Óbito e `0,8467` em Longa), maior sensibilidade bruta em óbito (`90,12%`) e `PR-AUC` muito próximo ao topo (`0,3024 / 0,3654`).",
                "cons": "**⚠️ Contras:** Ligeiramente mais lento no treino sobre `2,4M` linhas e ficou milésimos atrás do LightGBM tanto em `ROC-AUC` quanto em `PR-AUC` e `Brier Score` nos dois alvos."
            },
            {
                "title": "4. LightGBM *(Gradient Boosting Leaf-Wise — Modelo Oficial V5)*",
                "how": "**Como funciona:** Evolução de gradient boosting baseada em histogramas que expande as folhas de maior ganho de erro (*leaf-wise*), reduzindo cálculos e acelerando o aprendizado sobre milhões de registros.",
                "pros": "**✅ Prós:** 1º lugar isolado em todos os indicadores principais (`ROC-AUC 0,9082 / 0,8481`; `PR-AUC 0,3035 / 0,3663`; `Brier 0,0306 / 0,0471`), ótima especificidade (`77,14% / 66,60%`), arquivo final de apenas `0,62 MB` e resposta em milissegundos.",
                "cons": "**⚠️ Contras:** Crescimento *leaf-wise* pode superajustar subgrupos raros sem regularização — por isso na V5 combinamos `min_child_samples=100`, suavização Bayesiana em `cid3` (`m=50`) e calibração isotônica."
            },
        ],
        "ml_evid": "#### 📈 Evidências Visuais e Explicabilidade (V5 Nacional)",
        "ml_img1": "Curvas ROC e PR: quanto mais perto do topo/canto, maior a precisão do modelo.",
        "ml_img2": "Calibração Isotônica: curvas coladas na linha pontilhada provam que o % previsto é realista.",
        "ml_img3": "Fatores que mais pesam na decisão da IA (ex.: Idade, Clínica Médica e Urgência).",
        "cli_hdr": "### 🏥 Painel de Gestão Clínica e Risco de Pacientes",
        "cli_sub": "Simulação de um painel hospitalar real de acompanhamento de leitos: mostra o identificador anônimo da internação (`#AIH`), diagnóstico (`CID-10`), tempo já internado e duas **Flags Preditivas** calculadas na admissão.",
        "cli_exp": "⚙️ **Calibrar Cortes das Flags & Entender as Porcentagens**",
        "cli_exp_txt": "💡 **Por que os cortes padrão não são 50%?** No Brasil (2024), a média real de óbito hospitalar é de **3,83%** e a de longa permanência (≥15 dias) é de **6,19%** na base modelada. Se o hospital esperasse 50% de risco na porta de entrada, quase todos os casos graves passariam sem alerta. Por isso, o corte **Amarelo/Laranja** nasce do **Limiar de Youden V5** (`4,1%` para Óbito e `6,3%` para Longa Permanência — ponto estatístico que maximiza a detecção precoce) e o corte **Vermelho** marca perfis com **3x a 4x mais risco que a média nacional** (`15,0%` e `20,0%`). Ajuste abaixo conforme a lotação do seu hospital:",
        "sl_ob": "⚠️ Cortes Risco de Óbito — Atenção / Crítico (%):",
        "sl_lg": "🚩 Cortes Longa Permanência — Médio / Alto Risco (%):",
        "leg_lg": "🚩 **Longa Permanência (≥ 15 dias):** `🟢 Baixo < {a:.1f}%` | `🟡 Médio {a:.1f}%–{b:.1f}%` | `🔴 Alto Risco ≥ {c:.1f}%`",
        "leg_ob": "⚠️ **Risco de Óbito:** `🟢 Estável < {a:.1f}%` | `🟠 Atenção {a:.1f}%–{b:.1f}%` | `🔴 Crítico ≥ {c:.1f}%`",
        "cli_guide_hdr1": "**🚦 Faixas de Risco Ativas** *(o `%` é chance calibrada na admissão)*",
        "cli_guide_txt1": "Um risco de **25%** não significa certeza — quer dizer: *em 100 pacientes semelhantes na entrada (idade, CID3 e leito), ~25 agravam e ~75 evoluem bem*. Como a média nacional de óbito é **3,83%** (e longa perm. **6,19%**), 25% já representa **~6,5× mais risco** e justifica ligar o alerta **🟡🟠🔴 (`≥ corte`)** para priorizar UTI/cuidado, mesmo quando o paciente se recupera e sai de alta. **🟢 Verde (`< corte`)** indica baixo risco.",
        "cli_guide_hdr2": "**🎯 Como Ler `Alarme Preventivo vs. Não Detectado`**",
        "cli_guide_txt2": "- **`✅ Verdadeiro Positivo / Negativo`:** decisão nos dois alvos bateu com `🏥 Desfecho Real` e `Dias Internado`.\n- **`✅ Alerta Bem-Sucedido (Falso Positivo em 1 ou Ambos)`:** a equipe recebeu alerta preventivo (`🟡🟠🔴`), mas o evento não ocorreu *(**Falso Positivo** = alarme preventivo com cuidado extra e recuperação — na saúde é muito menos problemático que um **Falso Negativo**)*. *(Veja mais na Aba 5 → Glossário)*\n- **`🔕 Falso Negativo` / `❌ Ambos FN`:** ficou no `🟢 Verde`, porém houve internação ≥15d ou óbito *(**Falso Negativo** = caso grave não sinalizado na entrada)*.",
        "sim_exp": "➕ **Simular Nova Admissão de Paciente (Scorecard em Tempo Real)**",
        "sim_idade": "Idade (anos):",
        "sim_sexo": "Sexo Biológico:",
        "sim_espec": "Especialidade do Leito:",
        "sim_car": "Caráter de Internação:",
        "sim_cid": "Diagnóstico Principal (Capítulo CID-10):",
        "sim_cid3": "CID-10 Específico (3 dígitos — Opcional):",
        "sim_cid3_default": "Geral do Capítulo (Prior Hierárquico)",
        "sim_comp": "Complexidade:",
        "sim_transf": "Paciente Transferido de Outro Município?",
        "sim_fds": "Admissão em Fim de Semana?",
        "sim_nao": "Não",
        "sim_sim": "Sim",
        "sim_btn": "🚨 Calcular Score de Admissão",
        "sim_res": "#### Resultado da Triagem (Probabilidades Calibradas Brasil V5):",
        "sim_ob_tit": "Risco de Óbito",
        "sim_lg_tit": "Longa Permanência",
        "badge_ob": ("CRÍTICO", "ATENÇÃO", "ESTÁVEL"),
        "badge_lg": ("ALTO RISCO", "MÉDIO", "BAIXO"),
        "quads": {
            "red": ("ZONA VERMELHA (CRÍTICA)", "🚨 Acionar UTI/Semi-Intensiva imediatamente. Suporte avançado de vida."),
            "acute": ("ZONA DE EMERGÊNCIA AGUDA", "⚠️ Sala Vermelha: Monitoramento intensivo nas primeiras 48h."),
            "chronic": ("ZONA DE CRONICIDADE", "📋 Acionar Serviço Social e Fisioterapia para planejamento de alta precoce."),
            "fast": ("ZONA DE FLUXO RÁPIDO", "✅ Linha de cuidado convencional. Previsão de alta em até 48h."),
        },
        "panel_hdr": "#### 📋 Painel de Pacientes Internados *(Amostra Dinâmica do Brasil)*",
        "p_uf": "Estado (UF):",
        "p_esp": "Especialidade:",
        "p_grav": "Status de Gravidade:",
        "p_grav_opts": ["Todos", "Crítico (Óbito Alto)", "Longa Permanência Elevada"],
        "p_dias": "Dias Internado (Mínimo):",
        "p_busca": "Buscar ID (AIH) ou CID:",
        "p_empty": "Nenhum paciente encontrado para os filtros informados.",
        "p_count": "Exibindo **{n}** pacientes amostrados",
        "p_match_summary": "acerto nos cortes atuais — **Ambos (2/2): {both:.1f}%** | Longa Perm.: **{lg:.1f}%** | Óbito: **{ob:.1f}%**",
        "match_tokens": {
            "obito": "Óbito",
            "alta": "Alta",
            "vn": "Verdadeiro Negativo",
            "vp": "Verdadeiro Positivo",
            "alert_ok": "Alerta Bem-Sucedido",
            "fn": "Falso Negativo",
            "both_fp": "Ambos FP",
            "both_fn": "Ambos FN",
            "fp_short": "FP",
            "fn_short": "FN",
            "short_lg": "Perm.",
            "short_ob": "Óbito",
        },
        "p_cols": {
            "id_anonimo": "ID (AIH)",
            "uf": "UF",
            "idade_anos": "Idade",
            "sexo_desc": "Sexo",
            "especialidade_leito": "Especialidade",
            "motivo_cid": "Motivo (CID-10)",
            "qt_diarias": "Dias Internado",
            "uti_mes_to": "Diárias UTI",
            "desfecho_real": "🏥 Desfecho Real",
            "flag_longa": "🚩 Longa Perm.",
            "flag_obito": "⚠️ Risco Óbito",
            "desfecho_match": "🎯 Alarme Preventivo vs. Não Detectado",
        },
        "fmt_idade": "%.0f anos",
        "dic_hdr": "### 📖 Dicionário Completo do SIH/SUS (113 Colunas Oficiais + Derivadas) & Glossário",
        "dic_sub": "Auditoria integral de cada campo da AIH Reduzida do DataSUS, deixando explícito o que alimenta a **IA**, o que fica apenas no **Perfil Epidemiológico** e o que foi **Descartado**.",
        "dic_search": "🔍 Pesquisar coluna, bloco ou significado:",
        "dic_ph": "Ex.: cobranca, diagsec, raca_cor, ROC-AUC, Brier...",
        "dic_filter_lbl": "Filtrar por Uso no Projeto:",
        "dic_filter_opts": [
            ("all", "Todas as Colunas (126)"),
            ("ia_epi", "✅ Usadas no Modelo IA + Epidemio"),
            ("epi", "📊 Apenas Perfil Epidemiológico"),
            ("alvo_painel", "🎯 Alvo / Painel Clínico"),
            ("nao_leakage", "⛔ Não Usadas (Data Leakage / Pós-alta)"),
            ("nao_admin", "⛔ Não Usadas (Admin / Nula / Redundante)"),
        ],
        "dic_s1": "#### 🗂️ 1. Catálogo Auditado de Todas as Variáveis (SIH/SUS + Derivadas)",
        "dic_s2": "#### 🧠 2. Glossário de Termos Técnicos (Ciência de Dados & Saúde)",
        "dic_empty1": "Nenhuma variável encontrada para esse filtro.",
        "dic_empty2": "Nenhum termo técnico encontrado para esse filtro.",
        "dic_count": "Exibindo **{n}** variáveis do catálogo completo (`113` originais do SIH/SUS + `13` derivadas).",
        "j_hdr": "### 🧭 Jornada do Projeto: Processo Criativo, Decisões & Arquitetura (V1 → V5)",
        "j_sub": "Registro transparente de como um estudo exploratório no Acre evoluiu para uma plataforma nacional calibrada sobre **13.584.534 internações** — mostrando erros corrigidos, dilemas éticos e escolhas de engenharia.",
        "j_v1_tit": "1️⃣ Fase V1 — Baseline no Acre & Descoberta de Armadilhas (56.459 casos)",
        "j_v1_txt": """
- **Ponto de partida:** Baixamos os 12 arquivos mensais `.dbc` do Acre (2024) e construímos o primeiro classificador para prever Longa Permanência (`>= 15 dias`, apenas `5,30%` dos casos — desbalanceamento de `17,88:1`).
- **Erro crítico descoberto (#1 — Data Leakage):** A variável `cobranca` apareceu dominando o modelo (`85,29%` de acerto aparente nos códigos `21/22`). Ao investigar o significado no DataSUS, descobrimos que pedidos de prorrogação só são faturados **após** o paciente já ter ficado longo tempo internado — informação inexistente na entrada!
- **Erro crítico descoberto (#2 — Dupla penalização no XGBoost):** Combinar `SMOTE` com `scale_pos_weight=18` no pipeline empurrou `85%` das previsões para a classe positiva (`45.237` alarmes falsos e apenas `19,77%` de acurácia). **Decisão:** eliminar o `SMOTE` e passar a usar ponderação nativa isolada.
- **Auditoria de dados faltantes:** Descobrimos que comorbidades secundárias (`diagsec1` a `diagsec9`) têm **`89,65%` a `100%` de valores nulos**, inviabilizando índices como Charlson e motivando o catálogo auditado de todas as 113 colunas na Aba 5.
""",
        "j_v2_tit": "2️⃣ Fase V2 — Virada para Admissão Clínica & Matriz Dual-Triage",
        "j_v2_txt": """
- **Mudança de paradigma:** Decidimos prever exclusivamente com variáveis preenchidas **no ato da recepção/admissão**, removendo `cobranca` e outras 24 variáveis financeiras/pós-alta (`val_tot`, `dt_saida`, `dias_perm`, `proc_rea`).
- **Segundo desfecho simultâneo (`Obito`):** Adicionamos a predição de mortalidade intra-hospitalar (`2,70%` no Acre), criando a **Matriz de Decisão 2x2** (*Zona Vermelha Crítica, Emergência Aguda, Cronicidade e Fluxo Rápido*).
- **Engenharia de features clínicas e geográficas:**
  - `especialidade_leito`: revelou que a Clínica Médica concentra **`78,9%` de todos os óbitos** (`7,40%` de mortalidade vs. `0,009%` na Obstetrícia).
  - `transferido_intermun` (`munic_res != munic_mov`): mostrou que pacientes vindos do interior sofrem risco de óbito **`28%` maior** (`3,72%` vs. `2,91%`) por atrasos logísticos.
  - `internacao_fim_de_semana`: captura fragilidade diagnóstica em sábados e domingos.
""",
        "j_v3_tit": "3️⃣ Fase V3 — Bioética em Saúde & Desacoplamento Sazonal",
        "j_v3_txt": """
- **Questionamento ético e crítico:** Ao analisar o gráfico de importância de variáveis do V2, notamos destaque para `raca_cor` e `mes_internacao`. Questionamos se fazia sentido usá-las na triagem médica.
- **Decisão #1 — Remoção de `raca_cor` do modelo preditivo:** Cor da pele não é mecanismo biológico de gravidade, e mantê-la na IA poderia perpetuar viés discriminatório na prioridade de leitos de UTI. **Solução:** removemos da IA clínica e criamos uma camada dedicada de **Equidade em Saúde Pública** (Aba 1) para dar visibilidade às populações vulneráveis sem enviesar o atendimento.
- **Decisão #2 — Remoção de `mes_internacao`:** Árvores de decisão superajustavam padrões específicos de meses de 2024 (*split bias* e risco de *data drift* em anos futuros). Mantivemos apenas na análise temporal da Aba 1.
- **Prova de robustez (10-Fold CV Acre):** Sem essas duas variáveis, os modelos mantiveram excelente poder de separação (**LightGBM V3: `ROC-AUC 0,8949` em Óbito e `0,8263` em Longa**).
""",
        "j_v4_tit": "4️⃣ Fase V4 — Escala Brasil (13,58M), Amostragem, Calibração Isotônica & Nuvem",
        "j_v4_txt": """
- **Salto nacional (27 UFs / 324 arquivos `.dbc`):** Construímos um pipeline paralelo via FTP DataSUS + Arrow (`datasuspy`), comprimindo **~20 GB** brutos de `13.584.534` internações em apenas **`186,06 MB`** de Parquet particionado Hive (`ano/uf/mes`) consultados via **DuckDB**.
- **Cubo Exato vs. Amostragem:** Aba 1 opera sobre o **Cubo Agregado de `3,99 MB`** (`100%` das `13.584.534` internações), o Cockpit usa `30.000` casos (`0,34 MB`) na nuvem, e o V4 validou o treino nacional (`900k`) com **Calibração Isotônica** (`Brier 0,0329 / 0,0480`).
- **Produto & Auditoria:** Interface PT/EN/ES, Aba 2 de Qualidade dos Dados e Dicionário auditado de todas as 113 colunas SIH/SUS.
""",
        "j_v5_tit": "5️⃣ Fase V5 — Re-auditoria das 113 Colunas, CID-10 Específico (`cid3`) & Treino em `2,4M`",
        "j_v5_txt": """
- **Segunda auditoria empírica:** Reavaliamos as 113 colunas para garantir que nada relevante havia ficado de fora. Descartamos armadilhas de *leakage* (`ident`, `marca_uti`), flags quebradas (`gestrisco=1` em `99,75%` da base) e nulos extremos (`diagsec1`–`9`), mas confirmamos ganho massivo ao abrir **`diag_princ` em 3 dígitos (`cid3`, `1.665` doenças)** para separar patologias leves das graves dentro do mesmo capítulo (ex.: Resfriado `J00` vs. Pneumonia `J18`, Infarto `I21` vs. Varizes `I83`).
- **Controle de Ruído na Cauda Rara (Por que e Efeito):**
  - *Motivo:* Dos `1.665` códigos `cid3`, `698` doenças raras aparecem pouco (`0,73%` das internações). Sem proteção, poucos casos isolados (ex.: 1 óbito em 2 entradas) fariam a IA superestimar o risco e memorizar acaso estatístico (*overfitting*).
  - *Decisão e efeito no modelo:* Isolamos as `200` doenças mais frequentes (`Top-200 OHE`) e, para os demais códigos, aplicamos **Target Encoding Bayesiano *Out-of-Fold* (`m=50`)**: quanto menos casos a doença tiver, mais a estimativa recua para a média segura do capítulo (`capitulo_cid`), sem vazar respostas no treino. Isso evita alarmes erráticos sem ignorar casos atípicos — nas doenças menos comuns, a separação saltou de `ROC-AUC 0,8526 → 0,8930` (Óbito) e `0,8047 → 0,8448` (Longa), reduzindo também o erro das probabilidades (*Brier* `0,0306` / `0,0471`).
- **Escala V5 (`2,4M` treino / `300k` calibração / `300k` teste = `3,0M`):** **LightGBM V5** liderou todos os concorrentes atingindo **`ROC-AUC 0,9082` (`PR-AUC 0,3035`, `Brier 0,0306`, detecção `89,58%`)** em Óbito e **`ROC-AUC 0,8481` (`PR-AUC 0,3663`, `Brier 0,0471`, detecção `85,32%`)** em Longa Permanência, ocupando apenas `0,62 MB` cada arquivo na nuvem.
""",
        "j_evol_hdr": "#### 📊 Linha do Tempo Comparativa das Versões (V1 → V5)",
        "j_tech_hdr": "#### 🛠️ Tecnologias Utilizadas & Por Que Cada Escolha Foi Feita",
        "j_tech_sub": "Arquitetura pensada para processar mais de 13,5 milhões de registros em computador comum e rodar gratuitamente em nuvem com resposta em milissegundos:",
        "footer": "SUS Analytics Brasil 2024 • DuckDB + Parquet + LightGBM V5 Calibrado • Portfólio de Inteligência em Saúde",
    },
    "en": {
        "page_title": "SUS Analytics & Clinical Intelligence",
        "sidebar_title": "## 🏥 SUS Analytics 2024",
        "sidebar_caption": "Data Science & Public Health Portfolio Project",
        "sidebar_resumo_hdr": "**📌 Tabs Overview:**",
        "sidebar_resumo_body": """
- **📊 Tab 1 (Epidemiology):** Hospitalization profile across Brazil by age, sex, race, months, and ICD-10 chapters.
- **🛡️ Tab 2 (Data Quality):** Completeness audit (27 States × 12 months), underreporting, and clinical integrity rules.
- **🤖 Tab 3 (AI Models):** Algorithm benchmarks, pros/cons, calibration, and why LightGBM was chosen.
- **🏥 Tab 4 (Clinical Panel):** Admission simulator and ward cockpit with adjustable predictive risk alerts.
- **📖 Tab 5 (Dictionary):** Audited catalog of all **113 official SIH/SUS columns** + 13 derived fields & glossary.
- **🧭 Tab 6 (Journey):** Creative timeline, technical decisions, tech stack, and V1 → V5 evolution.
""",
        "sidebar_motor": "**⚙️ Analytics Engine:**",
        "sidebar_prod": "**🧠 Production Model:**\n`LightGBM V5 (Bayes CID3 + Isotonic)`",
        "sidebar_etica": "⚖️ **Responsible AI:** Race and calendar month are kept for health equity audits in Tab 1, but excluded from clinical prediction models.",
        "modo_local": "Full Local (13.58M Partitions + Cube)",
        "modo_cloud": "Cloud Optimized (13.58M Cube + 30k Sample)",
        "main_title": "🏥 SUS Analytics & Clinical Intelligence",
        "sub_title": "Public Health Intelligence Platform (Brazil SIH/SUS 2024): national epidemiology, admission risk forecasting, and clinical ward management.",
        "kpi1_lbl": "Hospitalizations Analyzed",
        "kpi1_sub": "SIH/SUS 2024 Microdata",
        "kpi2_lbl": "States Covered",
        "kpi2_sub": "National Coverage",
        "kpi3_lbl": "Mortality Rate",
        "kpi3_sub": "In-Hospital Death",
        "kpi4_lbl": "Stays ≥ 15 Days",
        "kpi4_sub": "Long Length of Stay",
        "kpi5_lbl": "Predictive Power (ROC-AUC)",
        "kpi5_sub": "Mortality / Long Stay (V5)",
        "tabs": [
            "📊 1. Epidemiology & Health Equity",
            "🛡️ 2. Data Quality & Integrity",
            "🤖 3. How the AI Works (Models & Metrics)",
            "🏥 4. Clinical Management & Ward Cockpit",
            "📖 5. SIH/SUS Dictionary (113 Columns)",
            "🧭 6. Journey, Decisions & Architecture",
        ],
        "epi_hdr": "### 📊 National Epidemiological Overview",
        "epi_sub": "Explore who uses Brazil's public hospital network (SUS) in 2024, leading causes of admission, and regional or demographic disparities.",
        "f_regiao": "Region:",
        "f_uf": "State (UF):",
        "f_raca": "Race / Ethnicity:",
        "f_espec": "Ward Specialty:",
        "f_sexo": "Sex:",
        "f_mes": "Month (2024):",
        "epi_info": "🔎 Showing **{n}** hospitalizations for the selected filters ({pct:.1f}% of national total). *(Queries powered directly by DuckDB)*",
        "g1_hdr": "#### Distribution by Age Group and Sex",
        "g1_cap": "Displays admission volumes across age brackets split by biological sex.",
        "g1_x": "Age Group",
        "g1_y": "Hospitalizations",
        "g1_c": "Sex",
        "g2_hdr": "#### Equity: Mortality and Length of Stay by Race",
        "g2_cap": "Compares average hospital days and mortality rate across racial groups (bubble size = volume).",
        "g2_x": "Mortality Rate (%)",
        "g2_y": "Average Stay (Days)",
        "g3_hdr": "#### Monthly Seasonality of Admissions & Mortality",
        "g3_cap": "Tracks how admission volumes (bars) and mortality rates (line) fluctuate across the 12 months of 2024.",
        "g3_b": "Hospitalizations",
        "g3_l": "Mortality Rate (%)",
        "g3_x": "Month (2024)",
        "g3_y1": "Total Admissions",
        "g3_y2": "Mortality (%)",
        "g4_hdr": "#### Top Causes by ICD-10 Chapter",
        "g4_cap": "Leading diagnostic chapters; hover over bars to inspect exact mortality and long-stay rates.",
        "g4_y": "ICD-10 Chapter",
        "g4_x": "Hospitalizations",
        "g4_c": "Mortality (%)",
        "g4_hover": "<b>%{y}</b><br>Admissions: %{x:,}<br>Mortality Rate: <b>%{customdata[0]:.2f}%</b><br>Long Stay (≥15d): %{customdata[1]:.2f}%<extra></extra>",
        "ufs_hdr": "#### 🗺️ Breakdown by State (Loaded UFs)",
        "col_uf": "State",
        "col_reg": "Region",
        "col_int": "Admissions",
        "col_ob_pct": "Mortality (%)",
        "col_lg_pct": "Long Stay (%)",
        "col_med_d": "Avg. Days",
        "dq_hdr": "### 🛡️ Data Quality Audit: Epidemiology Profile & AI Training Inputs",
        "dq_sub": "Focused audit covering strictly the fields feeding the **Epidemiology Profile (Tab 1)** and **Predictive AI Training (Tab 3 — 9 features + 2 targets)**.",
        "dq_banner_local": "🔎 **Local Full Scope (100% of Records):** Coverage and completeness measured across **all {n} admissions** in the Cube; training feature nulls and ranges audited row-by-row on local Parquet partitions (`{na}` rows).",
        "dq_banner_cloud": "🔎 **Hybrid Cloud Scope:** Profile coverage and underreporting measured across **all {n} admissions** via Cube; training feature nulls and ranges audited on the national stratified sample (`{na}` cases across 27 States).",
        "dq_kpi1_lbl": "State × Month Coverage",
        "dq_kpi1_sub": "Active Partitions (Profile)",
        "dq_kpi2_lbl": "AI Training Completeness",
        "dq_kpi2_sub": "Zero Intake Nulls (9 Features)",
        "dq_kpi3_lbl": "Domain / Target Violations",
        "dq_kpi3_sub": "Invalid Age / Stay / ICD",
        "dq_kpi4_lbl": "Underreporting / Generic",
        "dq_kpi4_sub": "Race Not Decl. / Other Spec.",
        "dq_contract_hdr": "#### 📋 Column-by-Column Audit (Profile & Model Training Variables Only)",
        "dq_contract_sub": "Completeness, valid value bounds, and exact role of every column entering the epidemiological charts and LightGBM V5 training:",
        "dq_status_col": "Status",
        "dq_uf_tbl_hdr": "#### 🗺️ State-Level Completeness for Profile Variables",
        "dq_uf_cols": {
            "uf": "State",
            "regiao": "Region",
            "internacoes": "Admissions",
            "meses_ativos": "Active Months",
            "pct_raca_nd": "Race Not Decl. (%)",
            "pct_cid_ign": "Unknown ICD (%)",
            "pct_esp_outras": "Other Spec. (%)",
        },
        "dq_other_note": """⚠️ **About remaining SIH/SUS columns (not audited in detail here):**
For now, this tab focuses strictly on variables entering the **Epidemiology Profile (Tab 1)** and **AI Model Training (Tab 3)**. Other potentially clinically valuable raw columns were excluded after prior auditing due to **severe missingness** (e.g., secondary diagnoses/comorbidities `diagsec1`–`diagsec9`, missing in `89.65%`–`100%` of records), **serious source recording/billing inconsistencies**, or **post-discharge data leakage**. All `113` official fields remain documented in **Tab 5 (Dictionary)**.""",
        "ml_hdr": "### 🤖 How the AI Works & Model Selection (V5 National)",
        "ml_intro": """At the exact moment of hospital admission, two predictive models run simultaneously using only **9 intake variables** (age, sex, ICD-10 chapter + specific 3-character `cid3` code, ward specialty, admission urgency, complexity level, inter-municipal transfer, and weekend admission):
1. **Long Length of Stay Model (≥ 15 days):** estimates the probability of prolonged hospitalization to guide bed turnover and social work early on.
2. **In-Hospital Mortality Model:** estimates fatal deterioration risk to prioritize intensive monitoring or ICU allocation.""",
        "ml_why": """🏆 **Why did we choose LightGBM V5 as our production model?**
- **Outright #1 across both outcomes:** Powered by hybrid `cid3` encoding (`Top-200 OHE` + Hierarchical Bayesian Target Encoding) trained on `2.4M` admissions, it beat every baseline on **Mortality (`ROC-AUC 0.9082` | `PR-AUC 0.3035` | `Brier 0.0306`)** and **Long Stay (`ROC-AUC 0.8481` | `PR-AUC 0.3663` | `Brier 0.0471`)**.
- **Major precision gain on rare events:** Boosted Mortality `PR-AUC` by **+44.5%** over V4 (`0.2101 → 0.3035`) while capturing **`89.58%`** of fatal cases and **`85.32%`** of prolonged stays right at intake.
- **Tiny cloud footprint:** Each calibrated pipeline takes just **`0.62 MB`** on disk and scores patients in milliseconds.""",
        "ml_guide": """📖 **Metrics Mini-Guide (Plain Language):**
- **ROC-AUC (0 to 1):** Measures how well the model separates high-risk from stable patients (`0.5` = random guess; `> 0.80` = strong; `> 0.90` = exceptional).
- **PR-AUC:** Precision focus on rare events (mortality and long stays represent a small minority of admissions).
- **Brier Score (lower is better):** Measures how close predicted probabilities are to actual outcomes after **Isotonic Calibration** (e.g., 20% predicted means ~20 in 100 patients experience the event).
- **Ethics & No *Data Leakage*:** Race and month are excluded from model training to prevent algorithmic bias; post-discharge billing fields were removed to prevent leakage.""",
        "ml_bench": "#### 📊 National Benchmark V5 (Blind Test on `300,000` admissions across 27 States)",
        "ml_ob_sub": "##### 1. In-Hospital Mortality *(Sample Baseline: 3.83%)*",
        "ml_lg_sub": "##### 2. Long Stay ≥ 15 Days *(Sample Baseline: 6.19%)*",
        "ml_cols": {"Modelo": "Model", "ROC_AUC": "ROC-AUC", "PR_AUC": "PR-AUC", "Brier_Calibrado": "Brier Score", "Limiar_Otimo_Pct": "Alert Cut (%)", "Sensibilidade_Pct": "Recall (%)", "Especificidade_Pct": "Specif. (%)"},
        "ml_models_hdr": "#### 🧠 How Each Model Works — Pros & Cons",
        "ml_models_sub": "Concise breakdown of how each algorithm evaluated in the V5 national benchmark scores admission risk:",
        "ml_models_cards": [
            {
                "title": "1. Logistic Regression *(Linear Baseline)*",
                "how": "**How it works:** Classical statistical model that sums fixed weights across intake features and passes the total through a sigmoid curve to estimate outcome odds.",
                "pros": "**✅ Pros:** Very fast, transparent feature coefficients, near-zero RAM footprint, and well calibrated out of the box (`Brier 0.0312 / 0.0483` in V5).",
                "cons": "**⚠️ Cons:** Assumes linear additivity — misses complex non-linear interactions (e.g., age × specific `cid3` × urgency), trailing on `ROC-AUC` (`0.8265` Long Stay, `0.8956` Mortality)."
            },
            {
                "title": "2. Random Forest *(Bagging Ensemble)*",
                "how": "**How it works:** Builds hundreds of independent decision trees on random data and feature subsamples, averaging predictions across the entire forest.",
                "pros": "**✅ Pros:** Captures non-linear relationships naturally and resists outliers well, reaching `ROC-AUC 0.8942` (Mortality) and `0.8286` (Long Stay).",
                "cons": "**⚠️ Cons:** Produces large serialized files in memory/cloud, requires subsampling (`250k`) for training speed, and trailed boosting on rare-event precision (`PR-AUC 0.2684` Mortality)."
            },
            {
                "title": "3. XGBoost *(Histogram Gradient Boosting)*",
                "how": "**How it works:** Trains trees sequentially so each new tree corrects residual errors from prior trees, using histogram binning and built-in L1/L2 regularization.",
                "pros": "**✅ Pros:** Very strong runner-up (`ROC-AUC 0.9075` Mortality, `0.8467` Long Stay), highest raw Mortality recall (`90.12%`), and tight `PR-AUC` (`0.3024 / 0.3654`).",
                "cons": "**⚠️ Cons:** Slightly slower than LightGBM across `2.4M` training rows and edged out narrowly across `ROC-AUC`, `PR-AUC`, and `Brier Score` on both outcomes."
            },
            {
                "title": "4. LightGBM *(Leaf-Wise Gradient Boosting — Production V5)*",
                "how": "**How it works:** Fast histogram-based gradient boosting expanding leaves with highest loss reduction (*leaf-wise*), scaling smoothly to millions of rows.",
                "pros": "**✅ Pros:** Overall #1 on every headline metric (`ROC-AUC 0.9082 / 0.8481`; `PR-AUC 0.3035 / 0.3663`; `Brier 0.0306 / 0.0471`), strong specificity (`77.14% / 66.60%`), `0.62 MB` bundle size, and millisecond scoring.",
                "cons": "**⚠️ Cons:** Leaf-wise splitting can overfit rare subgroups if unregularized — mitigated in V5 via `min_child_samples=100`, Bayesian `cid3` shrinkage (`m=50`), and isotonic calibration."
            },
        ],
        "ml_evid": "#### 📈 Visual Diagnostics & Explainability (V5 National)",
        "ml_img1": "ROC & PR Curves: curves closer to the top-left (ROC) or top-right (PR) indicate stronger accuracy.",
        "ml_img2": "Isotonic Calibration: curves hugging the diagonal dashed line confirm true clinical probabilities.",
        "ml_img3": "Top features driving AI decisions (Age and Bayesian `cid3` severity lead both outcomes).",
        "cli_hdr": "### 🏥 Clinical Management & Patient Risk Cockpit",
        "cli_sub": "Real-time hospital ward simulation displaying anonymized admission IDs (`#AIH`), primary diagnosis (`ICD-10`), current stay length, and two **Predictive Risk Flags** computed at admission.",
        "cli_exp": "⚙️ **Calibrate Alert Thresholds & Understand the Percentages**",
        "cli_exp_txt": "💡 **Why aren't default thresholds set to 50%?** Across Brazil (2024), baseline in-hospital mortality is **3.83%** and long stays (≥15 days) occur in **6.19%** of modeled admissions. Waiting for 50% risk at intake would miss nearly all high-risk patients. Thus, **Yellow/Orange** cuts start at the **V5 Youden Index** (`4.1%` Mortality, `6.3%` Long Stay — the statistical optimum maximizing early detection), while **Red** alerts mark **3x to 4x higher risk than baseline** (`15.0%` and `20.0%`). Adjust below based on hospital capacity:",
        "sl_ob": "⚠️ Mortality Risk Cutoffs — Watch / Critical (%):",
        "sl_lg": "🚩 Long Stay Cutoffs — Medium / High Risk (%):",
        "leg_lg": "🚩 **Long Stay (≥ 15 days):** `🟢 Low < {a:.1f}%` | `🟡 Medium {a:.1f}%–{b:.1f}%` | `🔴 High Risk ≥ {c:.1f}%`",
        "leg_ob": "⚠️ **Mortality Risk:** `🟢 Stable < {a:.1f}%` | `🟠 Watch {a:.1f}%–{b:.1f}%` | `🔴 Critical ≥ {c:.1f}%`",
        "cli_guide_hdr1": "**🚦 Active Risk Bands** *(`%` is calibrated admission probability)*",
        "cli_guide_txt1": "A **25%** risk is not certainty — it means: *out of 100 similar admissions (same age, ICD3, and ward), ~25 worsen and ~75 recover*. Against Brazil's baseline (**3.83%** mortality; **6.19%** long stay), 25% is **~6.5× higher risk**, so firing **🟡🟠🔴 (`≥ cutoff`)** makes full clinical sense to prioritize ICU care even when the patient recovers and goes home. **🟢 Green (`< cutoff`)** means low risk.",
        "cli_guide_hdr2": "**🎯 How to Read `Preventive Alert vs. Missed`**",
        "cli_guide_txt2": "- **`✅ True Positive / Negative`:** alerts matched both `🏥 Actual Outcome` and `Stay Days`.\n- **`✅ Successful Alert (False Positive on 1 or Both)`:** staff received a preventive alert (`🟡🟠🔴`), yet the event did not occur *(**False Positive** = preventive alarm with extra caution and recovery — in healthcare, far less harmful than a **False Negative**)*. *(Details in Tab 5 → Glossary)*\n- **`🔕 False Negative` / `❌ Both FN`:** stayed in `🟢 Green`, yet ≥15d stay or death occurred *(**False Negative** = severe outcome not flagged at admission)*.",
        "sim_exp": "➕ **Simulate New Patient Admission (Real-Time Triage Scorecard)**",
        "sim_idade": "Age (years):",
        "sim_sexo": "Biological Sex:",
        "sim_espec": "Ward Specialty:",
        "sim_car": "Admission Type:",
        "sim_cid": "Primary Diagnosis (ICD-10 Chapter):",
        "sim_cid3": "Specific 3-Char ICD-10 (Optional):",
        "sim_cid3_default": "Chapter Average (Hierarchical Prior)",
        "sim_comp": "Complexity Level:",
        "sim_transf": "Transferred from Another Municipality?",
        "sim_fds": "Weekend Admission?",
        "sim_nao": "No",
        "sim_sim": "Yes",
        "sim_btn": "🚨 Compute Admission Risk Score",
        "sim_res": "#### Triage Result (Calibrated Brazil V5 Probabilities):",
        "sim_ob_tit": "Mortality Risk",
        "sim_lg_tit": "Long Stay",
        "badge_ob": ("CRITICAL", "WATCH", "STABLE"),
        "badge_lg": ("HIGH RISK", "MEDIUM", "LOW"),
        "quads": {
            "red": ("RED ZONE (CRITICAL)", "🚨 Immediate Action: Assign ICU/Step-Down bed and advanced life support."),
            "acute": ("ACUTE EMERGENCY ZONE", "⚠️ Resuscitation Bay: High acute deterioration risk. Continuous 48h monitoring."),
            "chronic": ("CHRONICITY / RETENTION ZONE", "📋 Bed Planning: Engage Social Work and Rehab early for discharge planning."),
            "fast": ("FAST-TRACK FLOW ZONE", "✅ Standard Care Pathway: Expected short stay and early discharge."),
        },
        "panel_hdr": "#### 📋 Inpatient Ward Panel *(Dynamic Brazil Sample)*",
        "p_uf": "State (UF):",
        "p_esp": "Specialty:",
        "p_grav": "Risk Filter:",
        "p_grav_opts": ["All", "Critical (High Mortality)", "Elevated Long Stay"],
        "p_dias": "Days Admitted (Min):",
        "p_busca": "Search ID (AIH) or ICD:",
        "p_empty": "No patients match the selected filters.",
        "p_count": "Showing **{n}** sampled inpatients",
        "p_match_summary": "match at current cutoffs — **Both (2/2): {both:.1f}%** | Long Stay: **{lg:.1f}%** | Mortality: **{ob:.1f}%**",
        "match_tokens": {
            "obito": "Death",
            "alta": "Discharge",
            "vn": "True Negative",
            "vp": "True Positive",
            "alert_ok": "Successful Alert",
            "fn": "False Negative",
            "both_fp": "Both FP",
            "both_fn": "Both FN",
            "fp_short": "FP",
            "fn_short": "FN",
            "short_lg": "Stay",
            "short_ob": "Death",
        },
        "p_cols": {
            "id_anonimo": "Patient ID",
            "uf": "State",
            "idade_anos": "Age",
            "sexo_desc": "Sex",
            "especialidade_leito": "Specialty",
            "motivo_cid": "Reason (ICD-10)",
            "qt_diarias": "Stay Days",
            "uti_mes_to": "ICU Days",
            "desfecho_real": "🏥 Actual Outcome",
            "flag_longa": "🚩 Long Stay",
            "flag_obito": "⚠️ Mortality Risk",
            "desfecho_match": "🎯 Preventive Alert vs. Missed",
        },
        "fmt_idade": "%.0f yrs",
        "dic_hdr": "### 📖 Complete SIH/SUS Dictionary (113 Official Columns + Derived) & Glossary",
        "dic_sub": "Full audit of every DATASUS hospital admission column, explicitly stating what powers the **AI Model**, what stays in the **Epidemiology Profile**, and what was **Excluded**.",
        "dic_search": "🔍 Search column, domain or meaning:",
        "dic_ph": "E.g.: cobranca, diagsec, raca_cor, ROC-AUC, Brier...",
        "dic_filter_lbl": "Filter by Project Usage:",
        "dic_filter_opts": [
            ("all", "All Columns (126)"),
            ("ia_epi", "✅ Used in AI Model + Epidemiology"),
            ("epi", "📊 Epidemiology Profile Only"),
            ("alvo_painel", "🎯 Target / Clinical Ward Panel"),
            ("nao_leakage", "⛔ Not Used (Data Leakage / Post-Stay)"),
            ("nao_admin", "⛔ Not Used (Admin / Sparse / Redundant)"),
        ],
        "dic_s1": "#### 🗂️ 1. Audited Catalog of All Variables (SIH/SUS + Derived)",
        "dic_s2": "#### 🧠 2. Technical Glossary (Data Science & Healthcare)",
        "dic_empty1": "No variables match your filter.",
        "dic_empty2": "No technical terms match your filter.",
        "dic_count": "Showing **{n}** cataloged variables (`113` original SIH/SUS fields + `13` derived fields).",
        "j_hdr": "### 🧭 Project Journey: Creative Process, Decisions & Architecture (V1 → V5)",
        "j_sub": "Transparent engineering log showing how a single-state pilot evolved into a calibrated national platform across **13,584,534 hospitalizations** — including bugs fixed, ethical choices, and architectural trade-offs.",
        "j_v1_tit": "1️⃣ Phase V1 — Acre Pilot & Uncovering Traps (56,459 admissions)",
        "j_v1_txt": """
- **Starting point:** Ingested 12 monthly `.dbc` files from Acre (2024) to predict Long Length of Stay (`>= 15 days`, only `5.30%` of cases — `17.88:1` class imbalance).
- **Critical finding #1 (Data Leakage):** The billing column `cobranca` dominated initial models (`85.29%` apparent long-stay rate on codes `21/22`). Auditing DATASUS rules revealed those extension codes are only billed **after** a patient has already stayed long — impossible to know at admission!
- **Critical finding #2 (XGBoost double-penalty collapse):** Combining `SMOTE` + `scale_pos_weight=18` pushed `85%` of predictions into the positive class (`45,237` false alarms, `19.77%` accuracy). **Decision:** drop SMOTE entirely and rely solely on native loss weighting.
- **Missingness audit:** Secondary diagnoses (`diagsec1`–`diagsec9`) were missing in **`89.65%` to `100%`** of records, motivating a full 113-column audit (now searchable in Tab 5).
""",
        "j_v2_tit": "2️⃣ Phase V2 — Shift to Intake Triage & Dual-Outcome Matrix",
        "j_v2_txt": """
- **Paradigm shift:** Scoped strictly to variables known **at admission intake**, purging `cobranca` and 24 post-discharge/financial fields (`val_tot`, `dt_saida`, `dias_perm`, `proc_rea`).
- **Dual-Triage Matrix:** Added in-hospital mortality (`Obito`, `2.70%` in Acre) alongside long stay to build the **2x2 Decision Matrix** (*Red Critical Zone, Acute Emergency, Chronicity/Retention, and Fast-Track Flow*).
- **Clinical & geographic feature engineering:**
  - `especialidade_leito`: revealed Internal Medicine concentrates **`78.9%` of all hospital deaths** (`7.40%` vs. `0.009%` in Obstetrics).
  - `transferido_intermun` (`munic_res != munic_mov`): showed inter-municipal transfers carry **`+28%` higher mortality** (`3.72%` vs. `2.91%`) due to transit delays.
  - `internacao_fim_de_semana`: captures weekend diagnostic/staffing constraints.
""",
        "j_v3_tit": "3️⃣ Phase V3 — Healthcare Bioethics & Temporal Decoupling",
        "j_v3_txt": """
- **Ethical & methodological question:** Feature importance plots highlighted `raca_cor` (race) and `mes_internacao` (month). Should these drive bedside triage decisions?
- **Decision #1 — Remove `raca_cor` from AI:** Skin color is not a biological mechanism of acute severity; leaving it in bedside scoring risks encoding systemic inequities into ICU prioritization. **Solution:** decoupled race from predictive AI while preserving it in Tab 1 for **Health Equity Auditing**.
- **Decision #2 — Remove `mes_internacao`:** Decision trees overfit calendar-specific patterns (`split bias` and future `data drift`). Kept exclusively for seasonality visualization in Tab 1.
- **Validation (10-Fold Stratified CV):** Removing both improved ethical robustness while preserving strong discrimination (**LightGBM V3: `ROC-AUC 0.8949` Mortality, `0.8263` Long Stay**).
""",
        "j_v4_tit": "4️⃣ Phase V4 — National Scale (13.58M), Sampling, Isotonic Calibration & Cloud",
        "j_v4_txt": """
- **National ingestion (27 States / 324 `.dbc` files):** Built a parallel DataSUS FTP + Arrow pipeline (`datasuspy`), compressing **~20 GB** of raw `13,584,534` admissions down to **`186.06 MB`** of Hive-partitioned ZSTD Parquet queried via **DuckDB**.
- **Exact Cube vs. Sampling:** Tab 1 runs on a **`3.99 MB` Cube** (`100%` of all `13,584,534` hospitalizations), the Ward Cockpit filters `30,000` real cases (`0.34 MB`) in Cloud RAM, and V4 validated national `900k` training with **Isotonic Calibration** (`Brier 0.0329 / 0.0480`).
- **Product & Audit:** Added PT/EN/ES switcher, Tab 2 Data Quality audits, and full 113-column audit table.
""",
        "j_v5_tit": "5️⃣ Phase V5 — Re-Auditing All 113 Columns, Specific `cid3` Encoding & `2.4M` Training Scale",
        "j_v5_txt": """
- **Second empirical audit:** Re-evaluated all 113 raw fields so no strong intake signal was missed. Rejected leakage traps (`ident`, `marca_uti`), broken flags (`gestrisco=1` in `99.75%` of rows), and severe nulls (`diagsec1`–`9`), but uncovered huge untapped signal by expanding **`diag_princ` into 3-character ICD-10 (`cid3`, `1,665` categories)** to separate minor conditions from life-threatening diagnoses within the same chapter (e.g., Cold `J00` vs. Pneumonia `J18`, Heart Attack `I21` vs. Varicose Veins `I83`).
- **Taming Rare-Tail Noise (Why & Model Effect):**
  - *Why:* Out of `1,665` `cid3` codes, `698` rare conditions make up just `0.73%` of admissions. Without smoothing, a handful of isolated cases (e.g., 1 death out of 2 admissions) would trick the model into mistaking random noise for extreme severity (*overfitting*).
  - *Decision & effect:* We gave dedicated columns to the `200` most common diagnoses (`Top-200 OHE`) and applied **Out-of-Fold Bayesian Target Encoding (`m=50`)** to the rest: the fewer records a diagnosis has, the closer its score shrinks toward the stable `capitulo_cid` average without leaking labels during training. This prevents erratic alerts while preserving rare-disease signal — among less common diagnoses, `ROC-AUC` jumped `0.8526 → 0.8930` (Mortality) and `0.8047 → 0.8448` (Long Stay), while sharpening probability accuracy (*Brier* down to `0.0306` / `0.0471`).
- **V5 Scale (`2.4M` train / `300k` calibration / `300k` test = `3.0M`):** **LightGBM V5** topped all 4 algorithms with **`ROC-AUC 0.9082` (`PR-AUC 0.3035`, `Brier 0.0306`, recall `89.58%`)** on Mortality and **`ROC-AUC 0.8481` (`PR-AUC 0.3663`, `Brier 0.0471`, recall `85.32%`)** on Long Stay, while keeping each compressed artifact at `0.62 MB`.
""",
        "j_evol_hdr": "#### 📊 Version Evolution Summary Table (V1 → V5)",
        "j_tech_hdr": "#### 🛠️ Tech Stack Used & Architectural Rationale",
        "j_tech_sub": "Every library was chosen to process 13.58M federal admissions on standard hardware and serve sub-second cloud triage for free:",
        "footer": "SUS Analytics Brazil 2024 • DuckDB + Parquet + Calibrated LightGBM V5 • Healthcare Intelligence Portfolio",
    },
    "es": {
        "page_title": "SUS Analytics & Clinical Intelligence",
        "sidebar_title": "## 🏥 SUS Analytics 2024",
        "sidebar_caption": "Proyecto de Portafolio en Ciencia de Datos y Salud Pública",
        "sidebar_resumo_hdr": "**📌 Resumen de Pestañas:**",
        "sidebar_resumo_body": """
- **📊 Pestaña 1 (Epidemiología):** Perfil hospitalario en Brasil por edad, sexo, raza/color, meses y causas CIE-10.
- **🛡️ Pestaña 2 (Calidad):** Auditoría de completitud (27 Estados × 12 meses), subregistro y reglas de integridad clínica.
- **🤖 Pestaña 3 (Modelos IA):** Desempeño de algoritmos, pros/contras, calibración y elección de LightGBM.
- **🏥 Pestaña 4 (Gestión Clínica):** Simulador de ingreso y panel de camas con alertas predictivas ajustables.
- **📖 Pestaña 5 (Diccionario):** Catálogo auditado de las **113 columnas oficiales del SIH/SUS** + 13 derivadas y glosario.
- **🧭 Pestaña 6 (Trayectoria):** Historial creativo, decisiones técnicas, stack usada y evolución V1 → V5.
""",
        "sidebar_motor": "**⚙️ Motor Analítico:**",
        "sidebar_prod": "**🧠 Producción:**\n`LightGBM V5 (Bayes CIE3 + Isotónica)`",
        "sidebar_etica": "⚖️ **IA Responsable:** Raza/color y mes se mantienen en epidemiología por equidad, pero se excluyen de los modelos clínicos predictivos.",
        "modo_local": "Local Completo (13,58M Particiones + Cubo)",
        "modo_cloud": "Cloud Optimizado (Cubo 13,58M + Muestra 30k)",
        "main_title": "🏥 SUS Analytics & Clinical Intelligence",
        "sub_title": "Plataforma de Inteligencia en Salud Pública (SIH/SUS Brasil 2024): perfil epidemiológico, predicción de riesgos al ingreso y panel clínico.",
        "kpi1_lbl": "Ingresos Analizados",
        "kpi1_sub": "Microdatos SIH/SUS 2024",
        "kpi2_lbl": "Estados Cubiertos",
        "kpi2_sub": "Cobertura Nacional",
        "kpi3_lbl": "Tasa de Mortalidad",
        "kpi3_sub": "Óbito Intrahospitalario",
        "kpi4_lbl": "Estancias ≥ 15 Días",
        "kpi4_sub": "Larga Estancia",
        "kpi5_lbl": "Poder Predictivo (ROC-AUC)",
        "kpi5_sub": "Óbito / Larga Est. (V5)",
        "tabs": [
            "📊 1. Epidemiología y Equidad",
            "🛡️ 2. Calidad de Datos e Integridad",
            "🤖 3. Cómo Funciona la IA (Modelos)",
            "🏥 4. Gestión Clínica y Camas",
            "📖 5. Diccionario SIH/SUS (113 Cols)",
            "🧭 6. Trayectoria, Decisiones y Diseño",
        ],
        "epi_hdr": "### 📊 Panorama Epidemiológico de los Ingresos Hospitalarios",
        "epi_sub": "Explore quién utiliza los hospitales públicos de Brasil (SUS) en 2024, las principales causas de ingreso y las brechas regionales o demográficas.",
        "f_regiao": "Región:",
        "f_uf": "Estado (UF):",
        "f_raca": "Raza / Color:",
        "f_espec": "Especialidad:",
        "f_sexo": "Sexo:",
        "f_mes": "Mes (2024):",
        "epi_info": "🔎 Mostrando **{n}** ingresos para los filtros seleccionados ({pct:.1f}% del total nacional). *(Consultas procesadas con DuckDB)*",
        "g1_hdr": "#### Distribución por Grupo de Edad y Sexo",
        "g1_cap": "Muestra el volumen de ingresos por grupo etario separado entre hombres y mujeres.",
        "g1_x": "Grupo de Edad",
        "g1_y": "Ingresos",
        "g1_c": "Sexo",
        "g2_hdr": "#### Equidad: Mortalidad y Estancia por Raza/Color",
        "g2_cap": "Compara el promedio de días en el hospital y el riesgo de fallecimiento entre grupos raciales (tamaño = volumen).",
        "g2_x": "Tasa de Mortalidad (%)",
        "g2_y": "Promedio de Días",
        "g3_hdr": "#### Estacionalidad Mensual de Ingresos y Mortalidad",
        "g3_cap": "Sigue cómo varían el volumen mensual (barras) y la tasa de mortalidad (línea) a lo largo de los 12 meses.",
        "g3_b": "Ingresos",
        "g3_l": "Mortalidad (%)",
        "g3_x": "Mes de 2024",
        "g3_y1": "Total de Ingresos",
        "g3_y2": "Mortalidad (%)",
        "g4_hdr": "#### Principales Causas por Capítulo CIE-10",
        "g4_cap": "Principales diagnósticos de ingreso; pase el cursor para ver la tasa exacta de mortalidad y larga estancia.",
        "g4_y": "Capítulo CIE-10",
        "g4_x": "Ingresos",
        "g4_c": "Mortalidad (%)",
        "g4_hover": "<b>%{y}</b><br>Ingresos: %{x:,}<br>Tasa de Mortalidad: <b>%{customdata[0]:.2f}%</b><br>Larga Estancia (≥15d): %{customdata[1]:.2f}%<extra></extra>",
        "ufs_hdr": "#### 🗺️ Comparativo por Estado (UFs Cargadas)",
        "col_uf": "Estado",
        "col_reg": "Región",
        "col_int": "Ingresos",
        "col_ob_pct": "Mortalidad (%)",
        "col_lg_pct": "Larga Est. (%)",
        "col_med_d": "Prom. Días",
        "dq_hdr": "### 🛡️ Calidad de Datos: Perfil Epidemiológico y Entrenamiento de la IA",
        "dq_sub": "Auditoría enfocada estrictamente en las variables que alimentan el **Perfil Epidemiológico (Pestaña 1)** y el **Entrenamiento Predictivo (Pestaña 3 — 9 features + 2 objetivos)**.",
        "dq_banner_local": "🔎 **Alcance Local (100% de la Base):** Cobertura y completitud medidas sobre **todos los {n} ingresos** en el Cubo; nulos y validez de variables de entrenamiento auditados fila por fila en Parquets locales (`{na}` filas).",
        "dq_banner_cloud": "🔎 **Alcance Híbrido (Nube):** Cobertura y subregistro del Perfil medidos sobre **todos los {n} ingresos** vía Cubo; nulos y límites de entrenamiento auditados en la muestra estratificada (`{na}` casos de 27 Estados).",
        "dq_kpi1_lbl": "Cobertura Estado × Mes",
        "dq_kpi1_sub": "Particiones Activas (Perfil)",
        "dq_kpi2_lbl": "Completitud Entreno (9 Features)",
        "dq_kpi2_sub": "Cero Nulos al Ingreso",
        "dq_kpi3_lbl": "Violaciones Dominio / Objetivo",
        "dq_kpi3_sub": "Edad / Días / CIE Inválidos",
        "dq_kpi4_lbl": "Subregistro / Inespecíficos",
        "dq_kpi4_sub": "Raza No Decl. / Espec. Otras",
        "dq_contract_hdr": "#### 📋 Auditoría Columna por Columna (Solo Variables de Perfil y Entrenamiento)",
        "dq_contract_sub": "Completitud, rango válido y rol exacto de cada campo utilizado en los gráficos epidemiológicos y en el entrenamiento de LightGBM V5:",
        "dq_status_col": "Estado",
        "dq_uf_tbl_hdr": "#### 🗺️ Completitud Estatal de las Variables de Perfil",
        "dq_uf_cols": {
            "uf": "Estado",
            "regiao": "Región",
            "internacoes": "Ingresos",
            "meses_ativos": "Meses Activos",
            "pct_raca_nd": "Raza No Decl. (%)",
            "pct_cid_ign": "CIE Ignorado (%)",
            "pct_esp_outras": "Espec. Otras (%)",
        },
        "dq_other_note": """⚠️ **Sobre las demás columnas del SIH/SUS (no detalladas aquí):**
Por ahora esta pestaña se enfoca solo en los datos usados en el **Perfil (Pestaña 1)** y el **Entrenamiento de la IA (Pestaña 3)**. Otras columnas oficiales potencialmente informativas quedaron fuera tras auditoría previa por presentar **falta severa de datos** (ej.: diagnósticos secundarios `diagsec1`–`diagsec9`, ausentes en `89,65%`–`100%` de los registros), **graves inconsistencias de registro/facturación** o **fuga temporal post-alta (*data leakage*)**. Todas las `113` columnas siguen documentadas en la **Pestaña 5 (Diccionario)**.""",
        "ml_hdr": "### 🤖 Cómo Funciona la IA y Elección del Modelo (V5 Nacional)",
        "ml_intro": """Al momento del ingreso hospitalario, dos modelos predictivos operan en paralelo usando solo **9 datos de admisión** (edad, sexo, capítulo CIE-10 + diagnóstico específico `cid3`, especialidad de cama, carácter urgencia/electivo, complejidad, traslado intermunicipal y entrada en fin de semana):
1. **Modelo de Larga Estancia (≥ 15 días):** estima la probabilidad de hospitalización prolongada para apoyar el giro de camas y trabajo social.
2. **Modelo de Mortalidad Intrahospitalaria:** estima el riesgo vital agudo para priorizar monitoreo intensivo o cama de UCI.""",
        "ml_why": """🏆 **¿Por qué elegimos LightGBM V5 como modelo oficial?**
- **Líder aislado en ambos desenlaces:** Con la codificación híbrida de `cid3` (`Top-200 OHE` + *Target Encoding Bayesiano* con *shrinkage* al capítulo) entrenada sobre `2,4M` ingresos, superó a todos los competidores en **Mortalidad (`ROC-AUC 0,9082` | `PR-AUC 0,3035` | `Brier 0,0306`)** y **Larga Estancia (`ROC-AUC 0,8481` | `PR-AUC 0,3663` | `Brier 0,0471`)**.
- **Salto expresivo en precisión de eventos raros:** Elevó el `PR-AUC` de mortalidad en **+44,5%** frente a V4 (`0,2101 → 0,3035`), manteniendo **`89,58%`** de detección en mortalidad y **`85,32%`** en larga estancia.
- **Ligereza extrema en la nube:** Cada pipeline calibrado completo ocupa apenas **`0,62 MB`** y responde en milisegundos.""",
        "ml_guide": """📖 **Mini-Guía de Métricas (Lenguaje Claro):**
- **ROC-AUC (0 a 1):** Mide cuánto distingue el modelo entre pacientes graves y estables (`0,5` = azar; `> 0,80` = excelente; `> 0,90` = excepcional).
- **PR-AUC:** Precisión enfocada en eventos poco frecuentes (fallecimiento y estancias largas).
- **Brier Score (menor es mejor):** Mide si la probabilidad prevista coincide con la realidad tras la **Calibración Isotónica** (ej.: 20% previsto equivale a ~20 casos reales cada 100 pacientes).
- **Ética y Cero *Data Leakage*:** Raza y mes no entran al modelo para evitar sesgos; variables post-alta fueron eliminadas.""",
        "ml_bench": "#### 📊 Comparativo Nacional V5 (Prueba Ciega en `300.000` ingresos de los 27 Estados)",
        "ml_ob_sub": "##### 1. Mortalidad Intrahospitalaria *(Promedio en la Muestra: 3,83%)*",
        "ml_lg_sub": "##### 2. Larga Estancia ≥ 15 Días *(Promedio en la Muestra: 6,19%)*",
        "ml_cols": {"Modelo": "Modelo", "ROC_AUC": "ROC-AUC", "PR_AUC": "PR-AUC", "Brier_Calibrado": "Brier Score", "Limiar_Otimo_Pct": "Corte Alerta (%)", "Sensibilidade_Pct": "Detección (%)", "Especificidade_Pct": "Especif. (%)"},
        "ml_models_hdr": "#### 🧠 Cómo Funciona Cada Modelo — Pros y Contras",
        "ml_models_sub": "Resumen claro de cómo cada algoritmo evaluado en el comparativo nacional V5 calcula el riesgo al ingreso:",
        "ml_models_cards": [
            {
                "title": "1. Regresión Logística *(Baseline Lineal)*",
                "how": "**Cómo funciona:** Modelo estadístico clásico que suma pesos fijos por cada variable de ingreso y aplica una función sigmoide para estimar la probabilidad del evento.",
                "pros": "**✅ Pros:** Muy rápida, coeficientes transparentes, consumo casi nulo de RAM y naturalmente bien calibrada (`Brier 0,0312 / 0,0483` en V5).",
                "cons": "**⚠️ Contras:** Asume linealidad simple — no captura interacciones clínicas complejas (ej.: edad avanzada × CIE3 específico × urgencia), quedando atrás en `ROC-AUC` (`0,8265` Larga Estancia y `0,8956` Mortalidad)."
            },
            {
                "title": "2. Random Forest *(Ensemble Bagging)*",
                "how": "**Cómo funciona:** Entrena cientos de árboles de decisión independientes sobre submuestras aleatorias de datos y variables, promediando el voto de todo el bosque.",
                "pros": "**✅ Pros:** Captura relaciones no lineales de forma robusta frente a valores atípicos, manteniendo buen equilibrio con `ROC-AUC 0,8942` (Mortalidad) y `0,8286` (Larga Estancia).",
                "cons": "**⚠️ Contras:** Genera archivos pesados en RAM/nube, exige submuestreo (`250k`) para entrenar rápido y quedó por detrás del boosting en `PR-AUC` (`0,2684` en Mortalidad vs. `0,3035` de LightGBM)."
            },
            {
                "title": "3. XGBoost *(Gradient Boosting — Histograma)*",
                "how": "**Cómo funciona:** Entrena árboles de forma secuencial donde cada árbol corrige los errores residuales anteriores, usando histogramas numéricos y regularización L1/L2.",
                "pros": "**✅ Pros:** Excelente rendimiento general (`ROC-AUC 0,9075` Mortalidad y `0,8467` Larga Estancia), mayor sensibilidad bruta en mortalidad (`90,12%`) y `PR-AUC` muy cercano a la cima (`0,3024 / 0,3654`).",
                "cons": "**⚠️ Contras:** Ligeramente más lento al entrenar `2,4M` filas y quedó milésimas detrás de LightGBM tanto en `ROC-AUC` como en `PR-AUC` y `Brier Score` en ambos desenlaces."
            },
            {
                "title": "4. LightGBM *(Gradient Boosting Leaf-Wise — Modelo Oficial V5)*",
                "how": "**Cómo funciona:** Algoritmo rápido por histogramas que expande primero las hojas con mayor ganancia de error (*leaf-wise*), reduciendo cómputo y acelerando el aprendizaje sobre millones de registros.",
                "pros": "**✅ Pros:** 1.er lugar aislado en todos los indicadores principales (`ROC-AUC 0,9082 / 0,8481`; `PR-AUC 0,3035 / 0,3663`; `Brier 0,0306 / 0,0471`), gran especificidad (`77,14% / 66,60%`), archivo final de solo `0,62 MB` y respuesta en milisegundos.",
                "cons": "**⚠️ Contras:** El crecimiento *leaf-wise* puede sobreajustar subgrupos raros sin regularización — por eso en V5 combinamos `min_child_samples=100`, suavizado Bayesiano en `cid3` (`m=50`) y calibración isotónica."
            },
        ],
        "ml_evid": "#### 📈 Evidencias Visuales y Explicabilidad (V5 Nacional)",
        "ml_img1": "Curvas ROC y PR: cuanto más cerca de la esquina superior, mayor precisión discriminativa.",
        "ml_img2": "Calibración Isotónica: curvas pegadas a la diagonal punteada prueban probabilidades realistas.",
        "ml_img3": "Factores con mayor peso en la IA (ej.: Edad, Medicina Interna e Ingreso de Urgencia).",
        "cli_hdr": "### 🏥 Panel de Gestión Clínica y Riesgo de Pacientes",
        "cli_sub": "Simulación hospitalaria de seguimiento de camas: muestra identificador anonimizado (`#AIH`), diagnóstico (`CIE-10`), días ingresados y dos **Alertas Predictivas** calculadas al ingreso.",
        "cli_exp": "⚙️ **Calibrar Umbrales de Alerta y Entender los Porcentajes**",
        "cli_exp_txt": "💡 **¿Por qué los cortes por defecto no son 50%?** En Brasil (2024), la tasa real de mortalidad es **3,83%** y la de larga estancia (≥15 días) es **6,19%** en la base modelada. Esperar un 50% al ingreso dejaría pasar casi todos los casos graves. Por eso, el corte **Amarillo/Naranja** parte del **Umbral de Youden V5** (`4,1%` Mortalidad y `6,3%` Larga Estancia), mientras la zona **Roja** señala perfiles con **3x a 4x más riesgo que el promedio nacional** (`15,0%` y `20,0%`). Ajuste abajo según la ocupación del hospital:",
        "sl_ob": "⚠️ Cortes Riesgo de Mortalidad — Atención / Crítico (%):",
        "sl_lg": "🚩 Cortes Larga Estancia — Medio / Alto Riesgo (%):",
        "leg_lg": "🚩 **Larga Estancia (≥ 15 días):** `🟢 Bajo < {a:.1f}%` | `🟡 Medio {a:.1f}%–{b:.1f}%` | `🔴 Alto Riesgo ≥ {c:.1f}%`",
        "leg_ob": "⚠️ **Riesgo de Mortalidad:** `🟢 Estable < {a:.1f}%` | `🟠 Atención {a:.1f}%–{b:.1f}%` | `🔴 Crítico ≥ {c:.1f}%`",
        "cli_guide_hdr1": "**🚦 Rangos de Riesgo Activos** *(el `%` es probabilidad calibrada al ingreso)*",
        "cli_guide_txt1": "Un riesgo del **25%** no es certeza — significa: *de 100 ingresos iguales (misma edad, CIE3 y cama), ~25 se agravan y ~75 evolucionan bien*. Frente al promedio nacional (**3,83%** muerte; **6,19%** estancia larga), 25% equivale a **~6,5× más riesgo**, justificando activar **🟡🟠🔴 (`≥ corte`)** para priorizar UCI aunque el paciente se recupere y salga de alta. **🟢 Verde (`< corte`)** indica bajo riesgo.",
        "cli_guide_hdr2": "**🎯 Cómo Leer `Alarma Preventiva vs. No Detectado`**",
        "cli_guide_txt2": "- **`✅ Verdadero Positivo / Negativo`:** las alertas coincidieron con `🏥 Desenlace Real` y `Días Estancia`.\n- **`✅ Alerta Exitosa (Falso Positivo en 1 o Ambos)`:** el equipo recibió alerta preventiva (`🟡🟠🔴`), pero no ocurrió el evento *(**Falso Positivo** = alarma preventiva con cuidado extra y alta — en salud es mucho menos problemático que un **Falso Negativo**)*. *(Ver más en Pestaña 5 → Glosario)*\n- **`🔕 Falso Negativo` / `❌ Ambos FN`:** quedó en `🟢 Verde`, pero ocurrió estancia ≥15d u óbito *(**Falso Negativo** = caso grave no detectado al ingreso)*.",
        "sim_exp": "➕ **Simular Nuevo Ingreso de Paciente (Scorecard en Tiempo Real)**",
        "sim_idade": "Edad (años):",
        "sim_sexo": "Sexo Biológico:",
        "sim_espec": "Especialidad de Cama:",
        "sim_car": "Tipo de Ingreso:",
        "sim_cid": "Diagnóstico Principal (Capítulo CIE-10):",
        "sim_cid3": "CIE-10 Específico (3 dígitos — Opcional):",
        "sim_cid3_default": "General del Capítulo (Prior Jerárquico)",
        "sim_comp": "Complejidad:",
        "sim_transf": "¿Trasladado desde Otro Municipio?",
        "sim_fds": "¿Ingreso en Fin de Semana?",
        "sim_nao": "No",
        "sim_sim": "Sí",
        "sim_btn": "🚨 Calcular Score de Ingreso",
        "sim_res": "#### Resultado del Triaje (Probabilidades Calibradas Brasil V5):",
        "sim_ob_tit": "Riesgo de Mortalidad",
        "sim_lg_tit": "Larga Estancia",
        "badge_ob": ("CRÍTICO", "ATENCIÓN", "ESTABLE"),
        "badge_lg": ("ALTO RIESGO", "MEDIO", "BAJO"),
        "quads": {
            "red": ("ZONA ROJA (CRÍTICA)", "🚨 Acción Inmediata: Asignar cama de UCI/Intermedios y soporte vital avanzado."),
            "acute": ("ZONA DE EMERGENCIA AGUDA", "⚠️ Sala Crítica: Alto riesgo agudo. Monitoreo continuo en las primeras 48h."),
            "chronic": ("ZONA DE CRONICIDAD", "📋 Planificación de Cama: Activar Trabajo Social y Rehabilitación para alta temprana."),
            "fast": ("ZONA DE FLUJO RÁPIDO", "✅ Vía Estándar: Protocolo convencional con previsión de alta rápida."),
        },
        "panel_hdr": "#### 📋 Panel de Pacientes Ingresados *(Muestra Dinámica de Brasil)*",
        "p_uf": "Estado (UF):",
        "p_esp": "Especialidad:",
        "p_grav": "Filtro de Gravedad:",
        "p_grav_opts": ["Todos", "Crítico (Alta Mortalidad)", "Larga Estancia Elevada"],
        "p_dias": "Días Ingresado (Mín):",
        "p_busca": "Buscar ID (AIH) o CIE:",
        "p_empty": "Ningún paciente coincide con los filtros.",
        "p_count": "Mostrando **{n}** pacientes muestreados",
        "p_match_summary": "acierto en los cortes actuales — **Ambos (2/2): {both:.1f}%** | Larga Est.: **{lg:.1f}%** | Óbito: **{ob:.1f}%**",
        "match_tokens": {
            "obito": "Óbito",
            "alta": "Alta",
            "vn": "Verdadero Negativo",
            "vp": "Verdadero Positivo",
            "alert_ok": "Alerta Exitosa",
            "fn": "Falso Negativo",
            "both_fp": "Ambos FP",
            "both_fn": "Ambos FN",
            "fp_short": "FP",
            "fn_short": "FN",
            "short_lg": "Est.",
            "short_ob": "Óbito",
        },
        "p_cols": {
            "id_anonimo": "ID Paciente",
            "uf": "Estado",
            "idade_anos": "Edad",
            "sexo_desc": "Sexo",
            "especialidade_leito": "Especialidad",
            "motivo_cid": "Motivo (CIE-10)",
            "qt_diarias": "Días Estancia",
            "uti_mes_to": "Días UCI",
            "desfecho_real": "🏥 Desenlace Real",
            "flag_longa": "🚩 Larga Estancia",
            "flag_obito": "⚠️ Riesgo Óbito",
            "desfecho_match": "🎯 Alarma Preventiva vs. No Detectado",
        },
        "fmt_idade": "%.0f años",
        "dic_hdr": "### 📖 Diccionario Completo SIH/SUS (113 Columnas Oficiales + Derivadas) y Glosario",
        "dic_sub": "Auditoría integral de cada campo del DATASUS, indicando explícitamente qué alimenta a la **IA**, qué queda solo en el **Perfil Epidemiológico** y qué fue **Descartado**.",
        "dic_search": "🔍 Buscar columna, bloque o significado:",
        "dic_ph": "Ej.: cobranca, diagsec, raca_cor, ROC-AUC, Brier...",
        "dic_filter_lbl": "Filtrar por Uso en el Proyecto:",
        "dic_filter_opts": [
            ("all", "Todas las Columnas (126)"),
            ("ia_epi", "✅ Usadas en IA + Epidemiología"),
            ("epi", "📊 Solo Perfil Epidemiológico"),
            ("alvo_painel", "🎯 Objetivo / Panel Clínico"),
            ("nao_leakage", "⛔ No Usadas (Data Leakage / Post-Alta)"),
            ("nao_admin", "⛔ No Usadas (Admin / Nula / Redundante)"),
        ],
        "dic_s1": "#### 🗂️ 1. Catálogo Auditado de Todas las Variables (SIH/SUS + Derivadas)",
        "dic_s2": "#### 🧠 2. Glosario de Términos Técnicos (Ciencia de Datos y Salud)",
        "dic_empty1": "No se encontraron variables con ese filtro.",
        "dic_empty2": "No se encontraron términos técnicos con ese filtro.",
        "dic_count": "Mostrando **{n}** variables del catálogo completo (`113` originales SIH/SUS + `13` derivadas).",
        "j_hdr": "### 🧭 Trayectoria del Proyecto: Proceso Creativo, Decisiones y Arquitectura (V1 → V5)",
        "j_sub": "Registro transparente de cómo un piloto estatal en Acre evolucionó hacia una plataforma nacional calibrada sobre **13.584.534 ingresos** — mostrando errores corregidos, dilemas éticos y decisiones técnicas.",
        "j_v1_tit": "1️⃣ Fase V1 — Piloto en Acre y Descubrimiento de Trampas (56.459 casos)",
        "j_v1_txt": """
- **Punto de partida:** Descargamos los 12 archivos mensuales `.dbc` de Acre (2024) para predecir Larga Estancia (`>= 15 días`, solo `5,30%` de los casos — desbalance de `17,88:1`).
- **Hallazgo crítico #1 (*Data Leakage*):** La variable `cobranca` dominaba los modelos (`85,29%` aparente en códigos `21/22`). Al auditar las reglas de DATASUS descubrimos que esas prórrogas se facturan **después** de que el paciente ya tuvo estancia larga — ¡imposible saberlo al ingreso!
- **Hallazgo crítico #2 (Colapso por doble penalización en XGBoost):** Combinar `SMOTE` + `scale_pos_weight=18` empujó el `85%` de predicciones a positivo (`45.237` falsas alarmas, `19,77%` de exactitud). **Decisión:** eliminar `SMOTE` y usar solo ponderación nativa.
- **Auditoría de nulos:** Los diagnósticos secundarios (`diagsec1`–`9`) tenían **`89,65%` a `100%`** de ausentes, motivando la auditoría de las 113 columnas en la Pestaña 5.
""",
        "j_v2_tit": "2️⃣ Fase V2 — Enfoque al Ingreso Clínico y Matriz Dual-Triage",
        "j_v2_txt": """
- **Cambio de paradigma:** Restringimos la predicción estrictamente a variables disponibles **en la admisión**, purgando `cobranca` y 24 variables financieras/post-alta (`val_tot`, `dt_saida`, `dias_perm`, `proc_rea`).
- **Segundo objetivo simultáneo (`Obito`):** Incorporamos mortalidad intrahospitalaria (`2,70%` en Acre), creando la **Matriz de Decisión 2x2** (*Zona Roja Crítica, Emergencia Aguda, Cronicidad y Flujo Rápido*).
- **Ingeniería de variables clínicas y geográficas:**
  - `especialidade_leito`: mostró que Medicina Interna concentra el **`78,9%` de los fallecimientos** (`7,40%` vs. `0,009%` en Obstetricia).
  - `transferido_intermun` (`munic_res != munic_mov`): reveló **`+28%` mayor mortalidad** en pacientes trasladados desde el interior (`3,72%` vs. `2,91%`).
  - `internacao_fim_de_semana`: captura el impacto de los ingresos en sábado y domingo.
""",
        "j_v3_tit": "3️⃣ Fase V3 — Bioética en Salud y Desacoplamiento Estacional",
        "j_v3_txt": """
- **Cuestionamiento ético:** En los gráficos de importancia aparecieron `raca_cor` (raza) y `mes_internacao` (mes). ¿Debían usarse para decidir camas al ingreso?
- **Decisión #1 — Excluir `raca_cor` de la IA:** El color de piel no es un mecanismo biológico de gravedad clínica, y usarlo en triaje podría perpetuar desigualdades en asignación de UCI. **Solución:** se excluyó de la IA y se conservó en la Pestaña 1 para auditoría de **Equidad en Salud**.
- **Decisión #2 — Excluir `mes_internacao`:** Los árboles sobreajustaban patrones del calendario 2024 (*split bias* y riesgo de *data drift*). Se mantuvo solo para análisis temporal en la Pestaña 1.
- **Validación (10-Fold Stratified CV):** Sin estas dos variables, el modelo mantuvo gran capacidad predictiva (**LightGBM V3: `ROC-AUC 0,8949` Mortalidad y `0,8263` Larga Estancia**).
""",
        "j_v4_tit": "4️⃣ Fase V4 — Escala Brasil (13,58M), Muestreo, Calibración Isotónica y Nube",
        "j_v4_txt": """
- **Salto nacional (27 Estados / 324 archivos `.dbc`):** Pipeline paralelo vía FTP DATASUS + Arrow (`datasuspy`), comprimiendo **~20 GB** brutos de `13.584.534` ingresos a **`186,06 MB`** en Parquet particionado Hive (`ano/uf/mes`) consultado con **DuckDB**.
- **Cubo Exacto vs. Muestreo:** La Pestaña 1 opera sobre el **Cubo Agregado de `3,99 MB`** (`100%` de los `13.584.534` ingresos), el Panel Clínico usa `30.000` casos (`0,34 MB`) en la nube y V4 validó el entrenamiento nacional (`900k`) con **Calibración Isotónica** (`Brier 0,0329 / 0,0480`).
- **Producto y Auditoría:** Interfaz PT/EN/ES, Pestaña 2 de Calidad de Datos y Diccionario auditado de todas las 113 columnas SIH/SUS.
""",
        "j_v5_tit": "5️⃣ Fase V5 — Re-auditoría de 113 Columnas, CIE-10 Específico (`cid3`) y Entreno en `2,4M`",
        "j_v5_txt": """
- **Segunda auditoría empírica:** Reevaluamos las 113 columnas para asegurar que nada relevante quedara fuera. Descartamos trampas de *leakage* (`ident`, `marca_uti`), indicadores rotos (`gestrisco=1` en `99,75%` de la base) y nulos extremos (`diagsec1`–`9`), pero confirmamos una mejora masiva al abrir **`diag_princ` en 3 dígitos (`cid3`, `1.665` códigos)** para separar patologías leves de graves dentro del mismo capítulo (ej.: Resfriado `J00` vs. Neumonía `J18`, Infarto `I21` vs. Várices `I83`).
- **Control de Ruido en la Cola Rara (Por qué y Efecto):**
  - *Motivo:* De `1.665` códigos `cid3`, `698` enfermedades raras suman apenas el `0,73%` de los ingresos. Sin protección, pocos casos aislados (ej.: 1 fallecimiento en 2 ingresos) harían que la IA confundiera azar estadístico con riesgo extremo (*overfitting*).
  - *Decisión y efecto en el modelo:* Separamos las `200` patologías más frecuentes (`Top-200 OHE`) y aplicamos **Target Encoding Bayesiano *Out-of-Fold* (`m=50`)** al resto: cuantos menos casos tenga un código, más se acerca su estimación al promedio estable del `capitulo_cid`, sin fuga de datos en el entrenamiento. Así evitamos alarmas erráticas sin perder señal clínica — en diagnósticos poco comunes, el `ROC-AUC` subió de `0,8526 → 0,8930` (Mortalidad) y `0,8047 → 0,8448` (Larga Estancia), afinando también las probabilidades (*Brier* `0,0306` / `0,0471`).
- **Escala V5 (`2,4M` entreno / `300k` calibración / `300k` prueba = `3,0M`):** **LightGBM V5** lideró a todos los competidores alcanzando **`ROC-AUC 0,9082` (`PR-AUC 0,3035`, `Brier 0,0306`, detección `89,58%`)** en Mortalidad y **`ROC-AUC 0,8481` (`PR-AUC 0,3663`, `Brier 0,0471`, detección `85,32%`)** en Larga Estancia, ocupando apenas `0,62 MB` cada archivo en la nube.
""",
        "j_evol_hdr": "#### 📊 Tabla Resumen de Evolución de Versiones (V1 → V5)",
        "j_tech_hdr": "#### 🛠️ Stack Tecnológico Utilizado y Por Qué Se Eligió",
        "j_tech_sub": "Arquitectura diseñada para procesar más de 13,5 millones de registros en hardware estándar y operar gratis en la nube con latencia de milisegundos:",
        "footer": "SUS Analytics Brasil 2024 • DuckDB + Parquet + LightGBM V5 Calibrado • Portafolio de Inteligencia en Salud",
    },
}


try:
    from catalogo_sih import get_catalogo_completo_df
except ImportError:
    from scripts.catalogo_sih import get_catalogo_completo_df


def get_dicionario_df(lang: str) -> pd.DataFrame:
    return get_catalogo_completo_df(lang)


def get_evolucao_df(lang: str) -> pd.DataFrame:
    if lang == "en":
        return pd.DataFrame([
            {"Version": "V1 (Baseline)", "Scope": "Acre (56,459)", "Targets": "Long Stay Only", "Features": "7 (incl. cobranca leakage)", "Key Milestone / Decision": "Detected `cobranca` post-intake leakage & fixed XGBoost SMOTE + weight collapse."},
            {"Version": "V2 (Dual-Triage)", "Scope": "Acre (56,459)", "Targets": "Long Stay + Mortality", "Features": "10 (strict intake)", "Key Milestone / Decision": "Removed 24 financial/post-stay cols; created 2x2 Decision Matrix & transfer/ward features."},
            {"Version": "V3 (Ethical AI)", "Scope": "Acre (56,459)", "Targets": "Long Stay + Mortality", "Features": "8 (ethical clinical)", "Key Milestone / Decision": "Removed `raca_cor` & `mes_internacao` from AI to prevent bias/drift; kept for equity tab."},
            {"Version": "V4 (National Cloud)", "Scope": "Brazil 27 UFs (13.58M)", "Targets": "Long Stay + Mortality", "Features": "8 (calibrated LightGBM)", "Key Milestone / Decision": "DuckDB/Parquet lakehouse, 1.5M stratified sample, Isotonic Calibration (Brier 0.033/0.048), PT/EN/ES Cloud App."},
            {"Version": "V5 (CID3 + 2.4M)", "Scope": "Brazil 27 UFs (3.0M)", "Targets": "Long Stay + Mortality", "Features": "9 (+ cid3 Hybrid Bayes TE)", "Key Milestone / Decision": "113-col audit + Top-200 OHE & OOF Bayes TE (m=50); ROC-AUC 0.9082/0.8481 & PR-AUC +44.5% (Death) at 0.62 MB."},
        ])
    if lang == "es":
        return pd.DataFrame([
            {"Versión": "V1 (Baseline)", "Alcance": "Acre (56.459)", "Objetivos": "Solo Larga Est.", "Features": "7 (con fuga cobranca)", "Decisión / Hito Principal": "Detección de fuga en `cobranca` y corrección del colapso SMOTE + peso en XGBoost."},
            {"Versión": "V2 (Dual-Triage)", "Alcance": "Acre (56.459)", "Objetivos": "Larga Est. + Óbito", "Features": "10 (estrictas ingreso)", "Decisión / Hito Principal": "Eliminación de 24 cols financieras/post-alta; Matriz 2x2 y variables de traslado/cama."},
            {"Versión": "V3 (IA Ética)", "Alcance": "Acre (56.459)", "Objetivos": "Larga Est. + Óbito", "Features": "8 (clínicas y éticas)", "Decisión / Hito Principal": "Exclusión de `raca_cor` y `mes_internacao` de la IA (mantenidas solo en Equidad)."},
            {"Versión": "V4 (Nacional Cloud)", "Alcance": "Brasil 27 UFs (13,58M)", "Objetivos": "Larga Est. + Óbito", "Features": "8 (LightGBM calibrado)", "Decisión / Hito Principal": "Lakehouse DuckDB/Parquet, muestra 1,5M, Calibración Isotónica (Brier 0,033/0,048), app PT/EN/ES."},
            {"Versión": "V5 (CIE3 + 2,4M)", "Alcance": "Brasil 27 UFs (3,0M)", "Objetivos": "Larga Est. + Óbito", "Features": "9 (+ cid3 Híbrido Bayes)", "Decisión / Hito Principal": "Auditoría 113 cols + Top-200 OHE & TE Bayesiano (m=50); ROC 0,9082/0,8481 y PR-AUC +44,5% en 0,62 MB."},
        ])
    return pd.DataFrame([
        {"Versão": "V1 (Baseline)", "Escopo": "Acre (56.459)", "Alvos": "Apenas Longa Perm.", "Features": "7 (com leakage cobranca)", "Decisão / Marco Principal": "Descoberta do vazamento em `cobranca` e correção do colapso SMOTE + peso no XGBoost."},
        {"Versão": "V2 (Dual-Triage)", "Escopo": "Acre (56.459)", "Alvos": "Longa Perm. + Óbito", "Features": "10 (estritas de entrada)", "Decisão / Marco Principal": "Expurgo de 24 colunas financeiras/pós-alta; criação da Matriz 2x2 e flags geográficas/leito."},
        {"Versão": "V3 (IA Ética)", "Escopo": "Acre (56.459)", "Alvos": "Longa Perm. + Óbito", "Features": "8 (clínicas e éticas)", "Decisão / Marco Principal": "Remoção de `raca_cor` e `mes_internacao` da IA clínica (mantidas só na Aba 1 de Equidade)."},
        {"Versão": "V4 (Nacional Cloud)", "Escopo": "Brasil 27 UFs (13,58M)", "Alvos": "Longa Perm. + Óbito", "Features": "8 (LightGBM calibrado)", "Decisão / Marco Principal": "Lakehouse DuckDB/Parquet, amostra de 1,5M, Calibração Isotônica (Brier 0,033/0,048), app PT/EN/ES na nuvem."},
        {"Versão": "V5 (CID3 + 2,4M)", "Escopo": "Brasil 27 UFs (3,0M)", "Alvos": "Longa Perm. + Óbito", "Features": "9 (+ cid3 Bayes Híbrido)", "Decisão / Marco Principal": "Re-auditoria 113 cols + Top-200 OHE & TE Bayesiano (m=50); ROC 0,9082/0,8481 e PR-AUC +44,5% em 0,62 MB."},
    ])


_TECH_STACK = [
    (
        "Python 3.11+ & Joblib",
        {"pt": "Linguagem Base & Serialização", "en": "Core Language & Serialization", "es": "Lenguaje Base y Serialización"},
        {"pt": "Orquestra ETL, modelagem, calibração e app web; Joblib empacota pipelines V5 (0,62 MB) em .joblib com cache instantâneo.", "en": "Unifies ETL, modeling, calibration, and web UI; Joblib packages V5 pipelines (0.62 MB) into .joblib for instant caching.", "es": "Unifica ETL, modelado, calibración y app web; Joblib empaqueta los pipelines V5 (0,62 MB) en .joblib con caché instantánea."},
        {"pt": "Evita fragmentação entre pesquisa e produção e garante reprodutibilidade determinística na nuvem.", "en": "Avoids research-to-production friction and ensures deterministic model loading in the cloud.", "es": "Evita fricción entre investigación y producción y garantiza carga determinística en la nube."},
    ),
    (
        "datasuspy + FTP DataSUS",
        {"pt": "Ingestão de Dados Públicos", "en": "Public Health Ingestion", "es": "Ingesta de Salud Pública"},
        {"pt": "Baixa em paralelo (ThreadPoolExecutor) e descompacta os 324 arquivos .dbc oficiais das 27 UFs direto do Ministério da Saúde.", "en": "Downloads in parallel and decompresses all 324 official .dbc state files directly from the Ministry of Health FTP.", "es": "Descarga en paralelo y descomprime los 324 archivos .dbc oficiales de los 27 estados desde el FTP oficial."},
        {"pt": "Elimina cliques manuais no TabNet/DataSUS e torna a coleta 100% automatizável e auditável.", "en": "Eliminates manual portal downloads and makes national data acquisition 100% reproducible.", "es": "Elimina descargas manuales y vuelve la recolección nacional 100% automatizable y reproducible."},
    ),
    (
        "PyArrow & Parquet (Hive)",
        {"pt": "Armazenamento Lakehouse", "en": "Lakehouse Storage", "es": "Almacenamiento Lakehouse"},
        {"pt": "Converte e comprime ~20 GB brutos de 13,58M internações em apenas 186,06 MB particionados por ano/uf/mes (ZSTD).", "en": "Converts and compresses ~20 GB of 13.58M admissions into 186.06 MB partitioned by year/state/month (ZSTD).", "es": "Convierte y comprime ~20 GB de 13,58M ingresos en solo 186,06 MB particionados por año/estado/mes (ZSTD)."},
        {"pt": "Leitura colunar até 100x menor que CSV, lendo só as colunas e estados filtrados sem estourar disco ou RAM.", "en": "Columnar compression reduces size ~100x vs CSV and prunes unread columns/partitions from RAM.", "es": "Compresión columnar ~100x menor que CSV; lee solo columnas y estados necesarios sin saturar RAM."},
    ),
    (
        "DuckDB",
        {"pt": "Motor Analítico OLAP", "en": "Embedded OLAP SQL Engine", "es": "Motor Analítico OLAP"},
        {"pt": "Executa consultas SQL sobre partições Parquet locais ou sobre o Cubo de 3,99 MB com 100% de exatidão nos 13,58M registros.", "en": "Runs SQL queries over local Parquet partitions or the 3.99 MB Cube with 100% exact parity over 13.58M rows.", "es": "Ejecuta consultas SQL sobre Parquet local o el Cubo de 3,99 MB con 100% de exactitud sobre 13,58M filas."},
        {"pt": "Dispensa servidor Postgres/BigQuery pago, roda embutido no Python e responde filtros em milissegundos.", "en": "Requires no paid external warehouse server, runs inside Python, and answers aggregations in milliseconds.", "es": "No requiere servidor externo pago, corre embebido en Python y responde agregaciones en milisegundos."},
    ),
    (
        "Pandas & NumPy",
        {"pt": "Manipulação & Auditoria", "en": "Data Wrangling & Audits", "es": "Transformación y Auditoría"},
        {"pt": "Padroniza faixas etárias, capítulos CID-10, flags geográficas (transferido_intermun) e valida nulidade das 113 colunas.", "en": "Standardizes age brackets, ICD-10 chapters, geographic flags (transferido_intermun), and nulls across all 113 columns.", "es": "Estandariza rangos etarios, capítulos CIE-10, variables geográficas y auditoría de nulos de las 113 columnas."},
        {"pt": "Integração nativa com PyArrow, DuckDB e Scikit-Learn para pré-processamento tabular expresso.", "en": "Seamless interoperability with PyArrow, DuckDB, and Scikit-Learn for clean tabular preprocessing.", "es": "Interoperabilidad directa con PyArrow, DuckDB y Scikit-Learn para preprocesamiento tabular limpio."},
    ),
    (
        "Scikit-Learn",
        {"pt": "Pipelines, Baselines & Calibração", "en": "Pipelines, Baselines & Calibration", "es": "Pipelines, Baselines y Calibración"},
        {"pt": "Encapsula AdmissionV5Preprocessor (OOF Target Encoding Bayesiano m=50 + Top-200 OHE + StandardScaler) e CalibratedClassifierCV isotônica.", "en": "Wraps AdmissionV5Preprocessor (OOF Bayesian Target Encoding m=50 + Top-200 OHE + StandardScaler) and isotonic calibration.", "es": "Encapsula AdmissionV5Preprocessor (OOF Target Encoding Bayesiano m=50 + Top-200 OHE + StandardScaler) y Calibración Isotónica."},
        {"pt": "Impede data leakage no pré-processamento e derrubou o Brier Score para 0,0306 / 0,0471 na V5.", "en": "Prevents preprocessing leakage and cut probability error (Brier Score) down to 0.0306 / 0.0471 in V5.", "es": "Evita fugas en preprocesamiento y bajó el Brier Score a 0,0306 / 0,0471 en V5."},
    ),
    (
        "Plotly & Matplotlib",
        {"pt": "Visualização & Evidências", "en": "Interactive & Scientific Charts", "es": "Gráficos Interactivos y Científicos"},
        {"pt": "Plotly gera gráficos interativos filtráveis na Aba 1; Matplotlib gera figuras estáticas 300 DPI (ROC/PR, calibração e pesos).", "en": "Plotly powers interactive Tab 1 exploration; Matplotlib exports 300-DPI ROC/PR, calibration, and feature-importance plots.", "es": "Plotly crea gráficos interactivos en Pestaña 1; Matplotlib exporta curvas ROC/PR, calibración e importancias a 300 DPI."},
        {"pt": "Une exploração clínica interativa no navegador com transparência científica auditável no repositório.", "en": "Combines interactive browser exploration for clinicians with publication-grade model diagnostics.", "es": "Combina exploración clínica interactiva en la web con diagnóstico científico auditable del modelo."},
    ),
    (
        "Streamlit (+ Cloud)",
        {"pt": "Aplicação Web & Deploy", "en": "Web App & Cloud Hosting", "es": "Aplicación Web y Nube"},
        {"pt": "Interface web em Python puro com cache (st.cache_data), seletores PT/EN/ES, simulador de admissão e cockpit de leitos.", "en": "Pure-Python web app with caching (st.cache_data), PT/EN/ES switcher, admission simulator, and ward cockpit.", "es": "Interfaz web en Python con caché (st.cache_data), selector PT/EN/ES, simulador de ingreso y panel de camas."},
        {"pt": "Permite deploy gratuito contínuo via GitHub e prototipagem clínica ágil sem necessidade de frontend separado.", "en": "Enables free continuous GitHub deployment and rapid clinical prototyping without a separate frontend stack.", "es": "Permite despliegue gratuito continuo desde GitHub y prototipado clínico rápido sin otro frontend."},
    ),
]


def get_tecnologias_df(lang: str) -> pd.DataFrame:
    cols_map = {
        "pt": ("Tecnologia", "Papel na Arquitetura", "O Que Faz no Projeto", "Por Que Foi Escolhida"),
        "en": ("Technology", "Architecture Role", "What It Does Here", "Why We Chose It"),
        "es": ("Tecnología", "Rol en la Arquitectura", "Qué Hace en el Proyecto", "Por Qué Se Eligió"),
    }
    cols = cols_map.get(lang, cols_map["pt"])
    key = lang if lang in ("pt", "en", "es") else "pt"
    rows = [{cols[0]: name, cols[1]: role[key], cols[2]: what[key], cols[3]: why[key]} for name, role, what, why in _TECH_STACK]
    return pd.DataFrame(rows)


def _fmt_int_lang(v: int, lang: str) -> str:
    return f"{v:,}" if lang == "en" else f"{v:,}".replace(",", ".")


def _fmt_pct_lang(v: float, casas: int, lang: str) -> str:
    s = f"{v:.{casas}f}%"
    return s if lang == "en" else s.replace(".", ",")


def get_qualidade_variaveis_df(res_dq: dict, esperado_ufs: int, lang: str) -> tuple[pd.DataFrame, list[str]]:
    """Gera auditoria de qualidade estritamente das variáveis usadas no Perfil Epidemiológico e Treino da IA."""
    k = lang if lang in ("pt", "en", "es") else "pt"
    cols_map = {
        "pt": ("Variável / Bloco", "Uso no Projeto", "Completude / Validade", "Status", "Diagnóstico de Qualidade"),
        "en": ("Variable / Block", "Project Usage", "Completeness / Validity", "Status", "Quality Assessment"),
        "es": ("Variable / Bloque", "Uso en el Proyecto", "Completitud / Validez", "Estado", "Diagnóstico de Calidad"),
    }[k]

    n_am = res_dq["amostra_n"]
    esperado_parts = esperado_ufs * 12
    cov_ok = res_dq["particoes_uf_mes"] >= esperado_parts and res_dq["particoes_uf_mes"] > 0
    idade_ok = n_am > 0 and res_dq["bad_idade"] == 0
    sexo_ok = n_am > 0 and res_dq["bad_sexo"] == 0 and res_dq["pct_sexo_outro"] == 0.0
    raca_ok = res_dq["pct_raca_nd"] < 1.0
    cid_ok = n_am > 0 and res_dq["bad_cid"] == 0 and res_dq["pct_cid_ign"] < 1.0
    esp_ok = res_dq["pct_esp_outras"] < 1.0
    cat_ok = n_am > 0 and (res_dq["null_carater"] + res_dq["null_complex"]) == 0
    flags_ok = n_am > 0 and (res_dq["null_transf"] + res_dq["null_fds"]) == 0
    alvo_ok = n_am > 0 and res_dq["bad_diarias"] == 0

    lbl_ok = {"pt": "APROVADO", "en": "PASS", "es": "APROBADO"}[k]
    lbl_warn = {"pt": "MONITORAR", "en": "WATCH", "es": "MONITOREAR"}[k]
    lbl_alert = {"pt": "ATENÇÃO", "en": "ALERT", "es": "ALERTA"}[k]

    u_epi = {"pt": "Perfil (Aba 1)", "en": "Epidemiology (Tab 1)", "es": "Perfil (Pestaña 1)"}[k]
    u_ia = {"pt": "Treino IA (Aba 3)", "en": "AI Training (Tab 3)", "es": "Entreno IA (Pestaña 3)"}[k]
    u_both = {"pt": "Perfil + Treino IA", "en": "Profile + AI Train", "es": "Perfil + Entreno IA"}[k]
    u_tgt = {"pt": "Perfil + Alvo IA", "en": "Profile + AI Target", "es": "Perfil + Objetivo IA"}[k]

    def _pct_ok(erros: int) -> str:
        if n_am <= 0:
            return "-"
        pct = max(0.0, (n_am - erros) * 100.0 / n_am)
        return _fmt_pct_lang(pct, 2, k)

    linhas = [
        (
            "regiao / uf",
            u_epi,
            f"{res_dq['ufs_cobertas']}/{esperado_ufs} UFs ({_fmt_pct_lang(100.0 if res_dq['ufs_cobertas'] == esperado_ufs else (res_dq['ufs_cobertas']*100.0/max(esperado_ufs,1)), 0, k)})",
            lbl_ok if res_dq["ufs_cobertas"] >= esperado_ufs else lbl_alert,
            "dq_ok" if res_dq["ufs_cobertas"] >= esperado_ufs else "dq_alert",
            {"pt": "Cobertura nacional integral das Unidades da Federação sem lacunas regionais.", "en": "Full national state coverage with zero missing states.", "es": "Cobertura nacional íntegra de los estados sin brechas regionales."}[k],
        ),
        (
            "mes_internacao",
            u_epi,
            f"{res_dq['particoes_uf_mes']} / {esperado_parts} ({res_dq['meses_cobertos']}/12m)",
            lbl_ok if cov_ok else lbl_alert,
            "dq_ok" if cov_ok else "dq_alert",
            {"pt": "Todos os 12 meses ativos por UF; excluída da IA na V3 para evitar data drift.", "en": "All 12 months active per state; excluded from AI in V3 to avoid drift.", "es": "Los 12 meses activos por estado; excluida de la IA en V3 para evitar drift."}[k],
        ),
        (
            "raca_cor_desc",
            u_epi,
            f"{_fmt_pct_lang(max(0.0, 100.0 - res_dq['pct_raca_nd']), 2, k)} ({_fmt_pct_lang(res_dq['pct_raca_nd'], 2, k)} ND)",
            lbl_ok if raca_ok else lbl_warn,
            "dq_ok" if raca_ok else "dq_warn",
            {"pt": "Monitora subnotificação estadual; usada só para equidade (nunca no treino da IA).", "en": "Tracks state-level underreporting; kept for equity only (never in AI training).", "es": "Monitorea subregistro estatal; usada solo en equidad (nunca en entreno IA)."}[k],
        ),
        (
            "idade_anos / faixa_etaria",
            u_both,
            f"{_pct_ok(res_dq['bad_idade'])} ({res_dq['min_idade']:.0f}–{res_dq['max_idade']:.0f})",
            lbl_ok if idade_ok else lbl_alert,
            "dq_ok" if idade_ok else "dq_alert",
            {"pt": "Convertida de idade + cod_idade para anos contínuos; zero valores fora de [0, 120].", "en": "Normalized from idade + cod_idade to continuous years; zero values outside [0, 120].", "es": "Normalizada de idade + cod_idade a años continuos; cero valores fuera de [0, 120]."}[k],
        ),
        (
            "sexo",
            u_both,
            f"{_pct_ok(res_dq['bad_sexo'])} (0% outro/inv.)",
            lbl_ok if sexo_ok else lbl_alert,
            "dq_ok" if sexo_ok else "dq_alert",
            {"pt": "100% aderente aos códigos válidos 1 (Masculino) e 3 (Feminino) no cubo e no treino.", "en": "100% compliant with valid codes 1 (Male) and 3 (Female) across cube and training.", "es": "100% conforme a los códigos válidos 1 (Masculino) y 3 (Femenino)."}[k],
        ),
        (
            "capitulo_cid & cid3 (diag_princ)",
            u_both,
            f"{_pct_ok(res_dq['bad_cid'])} ({_fmt_pct_lang(res_dq['pct_cid_ign'], 3, k)} Ign.)",
            lbl_ok if cid_ok else lbl_warn,
            "dq_ok" if cid_ok else "dq_warn",
            {"pt": "Mapeia diag_princ em 20 capítulos + 1.665 códigos cid3 (encoder Bayesiano V5) equilibrando detalhe e ruído.", "en": "Maps diag_princ into 20 chapters + 1,665 cid3 codes (V5 Bayesian encoder) balancing granularity and noise.", "es": "Mapea diag_princ en 20 capítulos + 1.665 códigos cid3 (encoder Bayesiano V5) equilibrando detalle y ruido."}[k],
        ),
        (
            "especialidade_leito",
            u_both,
            f"{_fmt_pct_lang(max(0.0, 100.0 - res_dq['pct_esp_outras']), 2, k)} ({_fmt_pct_lang(res_dq['pct_esp_outras'], 2, k)} Outras)",
            lbl_ok if esp_ok else lbl_warn,
            "dq_ok" if esp_ok else "dq_warn",
            {"pt": "Variável crítica na admissão; categoria 'Outras' monitorada por variações de cadastro.", "en": "Key intake predictor; 'Other' category monitored for hospital coding gaps.", "es": "Predictor clave al ingreso; categoría 'Otras' monitoreada por variaciones de registro."}[k],
        ),
        (
            "carater_internacao & complexidade",
            u_ia,
            f"{_pct_ok(res_dq['null_carater'] + res_dq['null_complex'])} (0 nulos)",
            lbl_ok if cat_ok else lbl_alert,
            "dq_ok" if cat_ok else "dq_alert",
            {"pt": "Disponíveis no acolhimento (Urgência/Eletiva e Média/Alta), sem vazamento pós-alta.", "en": "Available right at intake (Urgency/Elective & Complexity) with zero leakage.", "es": "Disponibles al ingreso (Urgencia/Electiva y Complejidad), sin fuga de datos."}[k],
        ),
        (
            "transferido_intermun & fds",
            u_ia,
            f"{_pct_ok(res_dq['null_transf'] + res_dq['null_fds'])} (0 nulos)",
            lbl_ok if flags_ok else lbl_alert,
            "dq_ok" if flags_ok else "dq_alert",
            {"pt": "Flags binárias derivadas de munic_res!=munic_mov e dt_inter na entrada hospitalar.", "en": "Binary flags derived from munic_res!=munic_mov and intake weekday.", "es": "Variables binarias derivadas de munic_res!=munic_mov y día de ingreso."}[k],
        ),
        (
            "qt_diarias & Obito (Alvos)",
            u_tgt,
            f"{_pct_ok(res_dq['bad_diarias'])} ({res_dq['min_diarias']}–{res_dq['max_diarias']}d)",
            lbl_ok if alvo_ok else lbl_alert,
            "dq_ok" if alvo_ok else "dq_alert",
            {"pt": "ETL expurga registros com qt_diarias <= 0 ou morte nulo antes de definir os alvos.", "en": "ETL removes qt_diarias <= 0 or missing death flags before defining model targets.", "es": "ETL filtra registros con qt_diarias <= 0 o desenlace nulo antes de generar objetivos."}[k],
        ),
    ]

    rows = [
        {cols_map[0]: var, cols_map[1]: uso, cols_map[2]: med, cols_map[3]: st_lbl, cols_map[4]: obs}
        for var, uso, med, st_lbl, _, obs in linhas
    ]
    status_keys = [r[4] for r in linhas]
    return pd.DataFrame(rows), status_keys


def get_glossario_df(lang: str) -> pd.DataFrame:
    if lang == "en":
        return pd.DataFrame([
            {"Term": "SIH/SUS", "Domain": "Public Health", "Plain Explanation": "Brazil's Unified Health System Hospital Information System. Official public Ministry of Health database covering all publicly funded hospital admissions."},
            {"Term": "LightGBM", "Domain": "Machine Learning", "Plain Explanation": "Fast gradient-boosted decision tree algorithm. Selected for production due to top accuracy, strong recall, and millisecond scoring latency."},
            {"Term": "ROC-AUC", "Domain": "AI Metric", "Plain Explanation": "Ranges from 0.5 (random coin flip) to 1.0 (perfect). Measures how well the model assigns higher risk scores to severe cases than stable ones."},
            {"Term": "PR-AUC", "Domain": "AI Metric", "Plain Explanation": "Precision-Recall area under curve focused on rare events (mortality and long stays), evaluating detection quality without excessive false alarms."},
            {"Term": "Brier Score", "Domain": "Calibration", "Plain Explanation": "Measures probability calibration error (closer to 0 is better). Improved from ~0.13–0.16 (V3) down to 0.031–0.047 (V5)."},
            {"Term": "Isotonic Calibration", "Domain": "Machine Learning", "Plain Explanation": "Non-parametric calibration fitted on a dedicated holdout set ensuring that a 20% prediction corresponds to ~20 real cases out of 100."},
            {"Term": "Youden Index", "Domain": "Clinical Statistics", "Plain Explanation": "Optimal threshold maximizing Sensitivity + Specificity - 1, used as the baseline Yellow/Orange alert cutoff (4.1% and 6.3% in V5)."},
            {"Term": "Sensitivity (Recall)", "Domain": "Epidemiology / AI", "Plain Explanation": "Share of true severe cases (deaths or 15+ day stays) successfully flagged right at admission (89.6% and 85.3%)."},
            {"Term": "Specificity", "Domain": "Epidemiology / AI", "Plain Explanation": "Share of standard non-severe hospitalizations accurately kept in the green zone without unnecessary alarms."},
            {"Term": "False Positive (Preventive Alert)", "Domain": "Clinical Triage / AI", "Plain Explanation": "Occurs when the AI triggers an alert above the cutoff (e.g., ≥ 4.1% mortality or ≥ 6.3% long stay), yet the patient recovers or goes home before 15 days. In clinical triage it translates into extra vigilance and protective care — far safer than missing an unstable patient, though overly low cutoffs would cause alert fatigue."},
            {"Term": "False Negative (Missed Case)", "Domain": "Clinical Triage / AI", "Plain Explanation": "Occurs when an admission stays in the green zone below the cutoff, yet the patient later dies or stays ≥ 15 days. In healthcare, False Negatives carry the highest clinical risk by delaying ICU or discharge care — which is why our Youden cutoffs prioritize 85%–90% recall over late thresholds like 50%."},
            {"Term": "Data Leakage", "Domain": "Data Engineering", "Plain Explanation": "Using future post-admission information (such as final stay length or billing extension codes) to predict intake risk. Eliminated 100%."},
            {"Term": "DuckDB + Parquet", "Domain": "Data Engineering", "Plain Explanation": "Embedded analytical SQL engine and compressed columnar file format enabling sub-second queries over 13.58M rows without heavy servers."},
        ])
    if lang == "es":
        return pd.DataFrame([
            {"Término": "SIH/SUS", "Área": "Salud Pública", "Explicación Clara": "Sistema de Información Hospitalaria del SUS. Base pública oficial del Ministerio de Salud de Brasil con todos los ingresos hospitalarios financiados por el sistema público."},
            {"Término": "LightGBM", "Área": "Machine Learning", "Explicación Clara": "Algoritmo moderno basado en árboles de decisión rápidos. Elegido para producción por su alta precisión, gran detección y respuesta en milisegundos."},
            {"Término": "ROC-AUC", "Área": "Métrica de IA", "Explicación Clara": "Va de 0,5 (azar) a 1,0 (perfecto). Mide la capacidad del modelo para asignar mayor riesgo a los casos graves que a los estables."},
            {"Término": "PR-AUC", "Área": "Métrica de IA", "Explicación Clara": "Precisión enfocada en eventos poco frecuentes (mortalidad y estancias largas), evaluando aciertos sin exceso de falsas alarmas."},
            {"Término": "Brier Score", "Área": "Calibración", "Explicación Clara": "Mide el error entre la probabilidad prevista y la realidad (más cerca de 0 es mejor). Bajó de ~0,13–0,16 (V3) a 0,031–0,047 (V5)."},
            {"Término": "Calibración Isotónica", "Área": "Machine Learning", "Explicación Clara": "Técnica ajustada en un conjunto independiente que garantiza que una predicción del 20% corresponda a ~20 casos reales de cada 100 pacientes."},
            {"Término": "Índice de Youden", "Área": "Estadística Clínica", "Explicación Clara": "Punto matemático óptimo que equilibra Sensibilidad y Especificidad — usado como corte inicial de alerta amarilla/naranja (4,1% y 6,3% en V5)."},
            {"Término": "Sensibilidad (Detección)", "Área": "Epidemiología / IA", "Explicación Clara": "Porcentaje de los casos graves reales (óbitos o estancias ≥15d) que el sistema logra detectar al ingreso (89,6% y 85,3%)."},
            {"Término": "Especificidad", "Área": "Epidemiología / IA", "Explicación Clara": "Porcentaje de ingresos normales que el sistema mantiene correctamente en zona verde sin alarmas innecesarias."},
            {"Término": "Falso Positivo (Alarma Preventiva)", "Área": "Triaje Clínico / IA", "Explicación Clara": "Ocurre cuando la IA activa la alerta por encima del corte (ej.: ≥ 4,1% mortalidad o ≥ 6,3% estancia larga), pero el paciente se recupera o sale antes de 15 días. En triaje hospitalario significa vigilancia preventiva y cuidado extra: mucho menos riesgoso que omitir un caso grave, aunque demasiados FP generan fatiga de alertas."},
            {"Término": "Falso Negativo (Caso No Detectado)", "Área": "Triaje Clínico / IA", "Explicación Clara": "Ocurre cuando el ingreso queda en zona verde por debajo del corte, pero luego fallece o permanece ≥ 15 días. En salud es el fallo más crítico por retrasar cama de UCI o planificación del alta; por eso los umbrales de Youden priorizan ~85%–90% de detección frente a cortes tardíos como 50%."},
            {"Término": "Data Leakage", "Área": "Ingeniería de Datos", "Explicación Clara": "Fuga de información del futuro (ej.: usar facturación o días finales para predecir el riesgo de entrada). Eliminado al 100% en este proyecto."},
            {"Término": "DuckDB + Parquet", "Área": "Ingeniería de Datos", "Explicación Clara": "Motor analítico SQL embebido y formato columnar comprimido que permiten consultar 13,58 millones de filas en segundos sin servidores pesados."},
        ])
    return pd.DataFrame([
        {"Termo": "SIH/SUS", "Área": "Saúde Pública", "Explicação Clara": "Sistema de Informações Hospitalares do SUS. Base oficial pública do Ministério da Saúde com todas as internações financiadas pelo SUS no Brasil."},
        {"Termo": "LightGBM", "Área": "Machine Learning", "Explicação Clara": "Algoritmo moderno baseado em árvores de decisão rápidas. Escolhido para produção pela alta precisão, ótima detecção e resposta em milissegundos."},
        {"Termo": "ROC-AUC", "Área": "Métrica de IA", "Explicação Clara": "Vai de 0,5 (chute aleatório) a 1,0 (perfeito). Mede a capacidade do modelo de dar um escore maior a quem realmente vai agravar do que a quem está estável."},
        {"Termo": "PR-AUC", "Área": "Métrica de IA", "Explicação Clara": "Precisão focada em eventos raros (como óbito e longa permanência), onde acertar o grupo minoritário sem gerar excesso de falsos alarmes é desafiador."},
        {"Termo": "Brier Score", "Área": "Calibração", "Explicação Clara": "Mede o erro entre a probabilidade prevista e a realidade (quanto mais perto de 0, melhor). Caiu de ~0,13–0,16 (V3) para 0,031–0,047 (V5)."},
        {"Termo": "Calibração Isotônica", "Área": "Machine Learning", "Explicação Clara": "Técnica aplicada em dados separados que garante que uma previsão de 20% realmente corresponda a ~20 casos reais a cada 100 pacientes."},
        {"Termo": "Limiar de Youden", "Área": "Estatística Clínica", "Explicação Clara": "Ponto matemático ideal que equilibra Sensibilidade e Especificidade — usado como linha de corte inicial para o alerta amarelo/laranja (4,1% e 6,3% na V5)."},
        {"Termo": "Sensibilidade (Detecção)", "Área": "Epidemiologia / IA", "Explicação Clara": "De todos os pacientes que realmente morreram ou ficaram 15+ dias, quantos por cento o sistema conseguiu sinalizar na entrada (89,6% e 85,3%)."},
        {"Termo": "Especificidade", "Área": "Epidemiologia / IA", "Explicação Clara": "De todos os pacientes que tiveram alta normal, quantos por cento o sistema corretamente manteve no sinal verde."},
        {"Termo": "Falso Positivo (Alarme Preventivo)", "Área": "Triagem Clínica / IA", "Explicação Clara": "Acontece quando a IA acende o alerta acima do corte (ex.: ≥ 4,1% óbito ou ≥ 6,3% longa perm.), mas o paciente se recupera ou sai antes de 15 dias. Na saúde significa vigilância preventiva e cuidado extra — muito menos problemático do que deixar um paciente grave sem alerta, embora cortes baixos demais gerem fadiga de alertas."},
        {"Termo": "Falso Negativo (Caso Não Detectado)", "Área": "Triagem Clínica / IA", "Explicação Clara": "Acontece quando a admissão fica na faixa verde abaixo do corte, mas o paciente evolui para ≥ 15 dias de internação ou óbito. Na saúde é a falha mais crítica por atrasar UTI ou serviço social; por isso usamos o Limiar de Youden para priorizar ~85%–90% de sensibilidade em vez de esperar 50%."},
        {"Termo": "Data Leakage", "Área": "Engenharia de Dados", "Explicação Clara": "Vazamento de informação do futuro (ex.: usar cobrança ou número de dias finais para prever o risco de entrada). Eliminado 100% neste projeto."},
        {"Termo": "DuckDB + Parquet", "Área": "Engenharia de Dados", "Explicação Clara": "Banco analítico embutido e formato de armazenamento colunar compactado que permitem consultar 13,58 milhões de registros em segundos sem servidor pesado."},
    ])
