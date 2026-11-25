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
- **📖 Aba 5 (Dicionário):** Catálogo auditado das **113 colunas oficiais do SIH/SUS** + derivadas e glossário.
- **🧭 Aba 6 (Jornada):** Histórico criativo, decisões técnicas, stack usada e evolução V1 → V4.
""",
        "sidebar_motor": "**⚙️ Motor Analítico:**",
        "sidebar_prod": "**🧠 Produção:**\n`LightGBM V4 (Calibração Isotônica)`",
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
        "kpi5_sub": "Óbito / Longa Perm. (V4)",
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
        "dq_hdr": "### 🛡️ Auditoria de Qualidade dos Dados & Contrato de Integridade (SIH/SUS)",
        "dq_sub": "Monitoramento automático de **cobertura espaço-temporal, completude, validade clínica e consistência de faturamento** executado via SQL/DuckDB antes da modelagem preditiva.",
        "dq_banner_local": "🔎 **Auditoria Integral Local:** Cobertura e preenchimento demográfico calculados sobre **100% das {n} internações** no Cubo; regras de domínio e consistência clínica verificadas registro a registro nos Parquets locais (`{na}` linhas).",
        "dq_banner_cloud": "🔎 **Auditoria Híbrida (Nuvem):** Cobertura, volumes e subnotificação demográfica medidos sobre **100% das {n} internações** via Cubo; regras linha a linha (`idade`, `qt_diarias`, `uti_mes_to`, nulos da IA) auditadas sobre a amostra estratificada nacional (`{na}` casos das 27 UFs).",
        "dq_kpi1_lbl": "Cobertura UF × Mês",
        "dq_kpi1_sub": "Partições Ativas (2024)",
        "dq_kpi2_lbl": "Completude IA (8 Features)",
        "dq_kpi2_sub": "Zero Nulos na Admissão",
        "dq_kpi3_lbl": "Subnotificação Raça / CID",
        "dq_kpi3_sub": "Nao Declarada / Ignorado",
        "dq_kpi4_lbl": "UTI > Total de Diárias",
        "dq_kpi4_sub": "Anomalia Faturada (Amostra)",
        "dq_contract_hdr": "#### 📋 Checklist Automatizado de Contrato de Dados (Data Quality Framework)",
        "dq_contract_sub": "Cada regra abaixo testa ao vivo os dados carregados no DuckDB segundo 4 dimensões de qualidade (Cobertura, Completude, Validade e Consistência):",
        "dq_status_col": "Status",
        "dq_c1_hdr": "#### Estabilidade Mensal de Envio (Volume & UFs Ativas)",
        "dq_c1_cap": "Verifica se algum mês de 2024 sofreu quebra ou falha de envio nos arquivos `.dbc` do DataSUS.",
        "dq_c1_x": "Mês (2024)",
        "dq_c1_y": "Internações Recebidas",
        "dq_c2_hdr": "#### Subnotificação & Categorias Inespecíficas por UF",
        "dq_c2_cap": "Compara os estados com maior taxa de especialidade `Outras` ou `Raça/Cor Não Declarada` (%), indicando diferenças de preenchimento hospitalar.",
        "dq_c2_x": "Percentual (%)",
        "dq_c2_y": "UF",
        "dq_c2_leg": "Indicador",
        "dq_c2_opt1": "Espec. Outras (%)",
        "dq_c2_opt2": "Raça Não Declarada (%)",
        "dq_uf_tbl_hdr": "#### 🗺️ Diagnóstico de Completude por Unidade da Federação",
        "dq_uf_cols": {
            "uf": "UF",
            "regiao": "Região",
            "internacoes": "Internações",
            "meses_ativos": "Meses Ativos",
            "pct_raca_nd": "Raça Não Decl. (%)",
            "pct_cid_ign": "CID Ignorado (%)",
            "pct_esp_outras": "Espec. Outras (%)",
        },
        "dq_exp_uti_tit": "🔍 **Descoberta Real do SIH/SUS: Casos onde `uti_mes_to > qt_diarias` ({n} ocorrências / {pct:.2f}%)**",
        "dq_exp_uti_txt": "No SIH/SUS, é possível encontrar AIHs onde o número faturado de diárias de UTI (`uti_mes_to`) supera `qt_diarias` (por reapresentações, continuações ou inconsistências contábeis na origem). Identificar essa anomalia na Auditoria de Qualidade comprova por que campos de faturamento pós-entrada **não devem** alimentar modelos de admissão nem substituir `qt_diarias` como alvo.",
        "dq_uti_cols": {
            "n_aih": "ID Anonimizado (AIH)",
            "uf": "UF",
            "especialidade_leito": "Especialidade",
            "diag_princ": "CID-10",
            "qt_diarias": "Diárias Totais",
            "uti_mes_to": "Diárias UTI",
            "diferenca_dias": "Excesso UTI (d)",
        },
        "dq_exp_null_tit": "🧹 **Auditoria Histórica de Nulidade & Anti-Leakage do Schema Original (113 Colunas)**",
        "dq_exp_null_txt": """
- **Diagnósticos Secundários (`diagsec1` a `diagsec9` e `tpdisec1`–`9`):** apresentaram **`89,65%` a `100%` de nulos** na auditoria do piloto completo, inviabilizando escores automáticos de comorbidade (Charlson/Elixhauser) em produção e motivando a transparência da Aba 5.
- **Campos Constantes ou Redundantes (`ano_cmpt`, `uf_zi`, `dt_inter` bruta):** removidos ou convertidos em atributos estruturais (`mes_internacao`, `internacao_fim_de_semana`) e chaves de particionamento Hive.
- **Expurgo Anti-Leakage (`34` colunas pós-alta):** campos como `cobranca`, `val_tot`, `dt_saida`, `proc_rea` e `uti_mes_*` existem nos arquivos brutos do DataSUS, mas só surgem após a admissão — bloqueados contratualmente antes de treinar a IA (Aba 3).
""",
        "ml_hdr": "### 🤖 Como a Inteligência Artificial Funciona & Escolha do Modelo (V4 Nacional)",
        "ml_intro": """No momento em que o paciente entra no hospital, dois modelos preditivos operam juntos usando apenas **8 informações da admissão** (idade, sexo, motivo CID-10, especialidade do leito, caráter urgência/eletivo, complexidade, se veio transferido de outro município e se entrou no fim de semana):
1. **Modelo de Longa Permanência (≥ 15 dias):** estima a chance de o paciente precisar de internação prolongada, apoiando o giro de leitos e o serviço social.
2. **Modelo de Óbito Intra-hospitalar:** estima a gravidade de risco fatal para priorizar monitoramento intensivo ou UTI.""",
        "ml_why": """🏆 **Por que escolhemos o LightGBM como modelo oficial?**
- **Empate técnico no topo com o XGBoost:** Alcançou excelente poder de separação (**ROC-AUC de `0,8793`** para óbito e **`0,8160`** para longa permanência) com a melhor calibração empírica (**Brier Score de `0,0329` e `0,0480`**).
- **Maior sensibilidade em longa permanência:** Detecta **`83,92%`** dos casos de internação prolongada e **`87,58%`** dos casos de alto risco de óbito.
- **Agilidade em produção:** Responde em milissegundos consumindo pouca memória RAM — ideal para triagem em tempo real e publicação online, sendo mais rápido e leve que *Random Forest* e muito mais preciso que *Regressão Logística*.""",
        "ml_guide": """📖 **Mini-Guia das Métricas (Para todos os públicos):**
- **ROC-AUC (0 a 1):** Mede o poder de distinção entre pacientes graves e estáveis (`0,5` = chute aleatório; acima de `0,80` = excelente).
- **PR-AUC:** Precisão focada em eventos raros (óbito e longa permanência são minoria dos casos).
- **Brier Score (quanto menor, melhor):** Mede se a probabilidade prevista reflete a realidade graças à **Calibração Isotônica** (ex.: se diz 20%, ~20 em 100 pacientes com aquele perfil realmente têm o desfecho).
- **Ética & Sem *Data Leakage*:** `raca_cor` e `mes_internacao` não entram no modelo para evitar preconceito algorítmico; dados pós-alta foram removidos para não "trapacear".""",
        "ml_bench": "#### 📊 Comparativo Nacional (Teste Cego em `300.000` internações das 27 UFs)",
        "ml_ob_sub": "##### 1. Óbito Intra-hospitalar *(Média Nacional: 3,85%)*",
        "ml_lg_sub": "##### 2. Longa Permanência ≥ 15d *(Média Nacional: 6,17%)*",
        "ml_cols": {"Modelo": "Modelo", "ROC_AUC": "ROC-AUC", "PR_AUC": "PR-AUC", "Brier_Calibrado": "Brier Score", "Limiar_Otimo_Pct": "Corte Alerta (%)", "Sensibilidade_Pct": "Detecção (%)", "Especificidade_Pct": "Especif. (%)"},
        "ml_models_hdr": "#### 🧠 Como Funciona Cada Modelo — Prós & Contras",
        "ml_models_sub": "Resumo didático de como cada algoritmo avaliado no benchmark nacional decide o risco na admissão hospitalar:",
        "ml_models_cards": [
            {
                "title": "1. Regressão Logística *(Baseline Linear)*",
                "how": "**Como funciona:** Modelo estatístico clássico que soma pesos fixos para cada informação da entrada (idade, CID, leito) e aplica uma curva sigmoide para estimar a chance do desfecho.",
                "pros": "**✅ Prós:** Muito rápido, transparente (mostra o peso exato de cada variável), consome quase zero memória e já nasce bem calibrado (`Brier 0,0334 / 0,0493`).",
                "cons": "**⚠️ Contras:** Assume relação linear simples — não combina interações complexas (ex.: idade avançada + capítulo CID + urgência), ficando atrás em discriminação (`ROC-AUC 0,7805` e `PR-AUC 0,2916` em Longa Permanência)."
            },
            {
                "title": "2. Random Forest *(Ensemble Bagging)*",
                "how": "**Como funciona:** Constrói centenas de árvores de decisão independentes em sorteios aleatórios dos dados e decide a probabilidade final pela média dos votos de toda a floresta.",
                "pros": "**✅ Prós:** Captura relações não lineares naturalmente, é robusto contra ruídos/outliers e atingiu a maior especificidade no teste nacional (`74,94%` em Óbito e `62,32%` em Longa).",
                "cons": "**⚠️ Contras:** Gera arquivos pesados na memória RAM/nuvem, tem latência maior por consultar muitas árvores profundas e obteve menor precisão em eventos raros que o boosting (`PR-AUC 0,1994` em Óbito)."
            },
            {
                "title": "3. XGBoost *(Gradient Boosting — Histograma)*",
                "how": "**Como funciona:** Treina árvores em sequência em que cada nova árvore corrige os erros residuais das anteriores, usando histogramas numéricos e regularização L1/L2 integrada.",
                "pros": "**✅ Prós:** Empatou na liderança absoluta em separação (`ROC-AUC 0,8795` em Óbito e `0,8161` em Longa), obteve maior detecção de óbitos (`88,47%`) e maior `PR-AUC` numérico (`0,2114 / 0,3427`).",
                "cons": "**⚠️ Contras:** Ligeiramente mais lento e pesado no treino de 900 mil linhas do que o LightGBM, e sensível à dupla penalização em classes desbalanceadas (como visto ao combinar SMOTE + peso na V1)."
            },
            {
                "title": "4. LightGBM *(Gradient Boosting Leaf-Wise — Modelo Oficial V4)*",
                "how": "**Como funciona:** Evolução de gradient boosting baseada em histogramas que expande as folhas de maior ganho de erro (*leaf-wise*), reduzindo cálculos e acelerando drasticamente o aprendizado.",
                "pros": "**✅ Prós:** Empate técnico no topo com o XGBoost (`ROC-AUC 0,8793 / 0,8160`; `Brier 0,0329 / 0,0480` após calibração), maior sensibilidade em Longa Permanência (`83,92%`), baixíssimo uso de RAM e resposta em milissegundos na nuvem.",
                "cons": "**⚠️ Contras:** Crescimento *leaf-wise* pode superajustar bases pequenas sem controle de profundidade/folhas e exige calibração isotônica (feita na V4) mais gráficos de importância para explicar cada decisão clínica."
            },
        ],
        "ml_evid": "#### 📈 Evidências Visuais e Explicabilidade (V4 Nacional)",
        "ml_img1": "Curvas ROC e PR: quanto mais perto do topo/canto, maior a precisão do modelo.",
        "ml_img2": "Calibração Isotônica: curvas coladas na linha pontilhada provam que o % previsto é realista.",
        "ml_img3": "Fatores que mais pesam na decisão da IA (ex.: Idade, Clínica Médica e Urgência).",
        "cli_hdr": "### 🏥 Painel de Gestão Clínica e Risco de Pacientes",
        "cli_sub": "Simulação de um painel hospitalar real de acompanhamento de leitos: mostra o identificador anônimo da internação (`#AIH`), diagnóstico (`CID-10`), tempo já internado e duas **Flags Preditivas** calculadas na admissão.",
        "cli_exp": "⚙️ **Calibrar Cortes das Flags & Entender as Porcentagens**",
        "cli_exp_txt": "💡 **Por que os cortes padrão não são 50%?** No Brasil (2024), a média real de óbito hospitalar é de **3,85%** e a de longa permanência (≥15 dias) é de **6,17%**. Se o hospital esperasse chegar a 50% de risco na porta de entrada, quase todos os casos graves passariam sem alerta. Por isso, o corte **Amarelo/Laranja** nasce do **Limiar de Youden** (`4,0%` para Óbito e `6,4%` para Longa Permanência — ponto estatístico que maximiza a detecção precoce) e o corte **Vermelho** marca perfis com **3x a 4x mais risco que a média nacional** (`15,0%` e `20,0%`). Ajuste abaixo conforme a lotação do seu hospital:",
        "sl_ob": "⚠️ Cortes Risco de Óbito — Atenção / Crítico (%):",
        "sl_lg": "🚩 Cortes Longa Permanência — Médio / Alto Risco (%):",
        "leg_lg": "🚩 **Longa Permanência (≥ 15 dias):** `🟢 Baixo < {a:.1f}%` | `🟡 Médio {a:.1f}%–{b:.1f}%` | `🔴 Alto Risco ≥ {c:.1f}%`",
        "leg_ob": "⚠️ **Risco de Óbito:** `🟢 Estável < {a:.1f}%` | `🟠 Atenção {a:.1f}%–{b:.1f}%` | `🔴 Crítico ≥ {c:.1f}%`",
        "sim_exp": "➕ **Simular Nova Admissão de Paciente (Scorecard em Tempo Real)**",
        "sim_idade": "Idade (anos):",
        "sim_sexo": "Sexo Biológico:",
        "sim_espec": "Especialidade do Leito:",
        "sim_car": "Caráter de Internação:",
        "sim_cid": "Diagnóstico Principal (Capítulo CID-10):",
        "sim_comp": "Complexidade:",
        "sim_transf": "Paciente Transferido de Outro Município?",
        "sim_fds": "Admissão em Fim de Semana?",
        "sim_nao": "Não",
        "sim_sim": "Sim",
        "sim_btn": "🚨 Calcular Score de Admissão",
        "sim_res": "#### Resultado da Triagem (Probabilidades Calibradas Brasil V4):",
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
        "p_count": "Exibindo **{n}** pacientes amostrados:",
        "p_cols": {
            "id_anonimo": "ID (AIH)",
            "uf": "UF",
            "idade_anos": "Idade",
            "sexo_desc": "Sexo",
            "especialidade_leito": "Especialidade",
            "motivo_cid": "Motivo (CID-10)",
            "qt_diarias": "Dias Internado",
            "uti_mes_to": "Diárias UTI",
            "flag_longa": "🚩 Longa Perm.",
            "flag_obito": "⚠️ Risco Óbito",
        },
        "fmt_idade": "%.0f anos",
        "dic_hdr": "### 📖 Dicionário Completo do SIH/SUS (113 Colunas Oficiais + Derivadas) & Glossário",
        "dic_sub": "Auditoria integral de cada campo da AIH Reduzida do DataSUS, deixando explícito o que alimenta a **IA**, o que fica apenas no **Perfil Epidemiológico** e o que foi **Descartado**.",
        "dic_search": "🔍 Pesquisar coluna, bloco ou significado:",
        "dic_ph": "Ex.: cobranca, diagsec, raca_cor, ROC-AUC, Brier...",
        "dic_filter_lbl": "Filtrar por Uso no Projeto:",
        "dic_filter_opts": [
            ("all", "Todas as Colunas (125)"),
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
        "dic_count": "Exibindo **{n}** variáveis do catálogo completo (`113` originais do SIH/SUS + `12` derivadas).",
        "j_hdr": "### 🧭 Jornada do Projeto: Processo Criativo, Decisões & Arquitetura (V1 → V4)",
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
- **Por que amostrar `1.500.000` internações no treino e `30.000` no painel da nuvem?**
  - *Aba 1 (100% sem amostragem):* Geramos um **Cubo Agregado de `3,99 MB`** (`419.801` linhas pré-somadas) com **100% de paridade exata** das `13.584.534` internações, `520.884` óbitos e `839.972` longas permanências.
  - *Treino V4 (`1,5M` = 900k treino / 300k calibração / 300k teste cego):* Com apenas 8 variáveis (~40 colunas codificadas), `900 mil` casos já saturam o aprendizado proporcional de todos os 27 estados sem estourar memória RAM.
  - *Cockpit Cloud (`30.000` casos = `0,34 MB`):* Permite que o painel online gratuito no Streamlit Cloud exiba pacientes reais de todas as 27 UFs em milissegundos dentro do limite de RAM da nuvem.
- **Calibração Isotônica & Limiar de Youden:** No V3, `scale_pos_weight` inflava escores (`Brier Score 0,12–0,16`). No V4, treinamos em *log-loss* nativo e aplicamos **Calibração Isotônica** nos `300k` de calibração, derrubando o **Brier Score para `0,0329` (Óbito) e `0,0480` (Longa)** — garantindo probabilidades reais e definindo alertas pelo **Limiar de Youden (`4,0%` / `6,4%`)** com barras deslizantes ajustáveis pelo hospital.
- **Produto Final & Portfólio:** Adicionamos tradução instantânea PT/EN, filtro cascata Região → UF, Dicionário completo das 113 colunas SIH/SUS e publicação gratuita no Streamlit Cloud.
""",
        "j_evol_hdr": "#### 📊 Linha do Tempo Comparativa das Versões (V1 → V4)",
        "j_tech_hdr": "#### 🛠️ Tecnologias Utilizadas & Por Que Cada Escolha Foi Feita",
        "j_tech_sub": "Arquitetura pensada para processar mais de 13,5 milhões de registros em computador comum e rodar gratuitamente em nuvem com resposta em milissegundos:",
        "footer": "SUS Analytics Brasil 2024 • DuckDB + Parquet + LightGBM V4 Calibrado • Portfólio de Inteligência em Saúde",
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
- **📖 Tab 5 (Dictionary):** Audited catalog of all **113 official SIH/SUS columns** + derived fields & glossary.
- **🧭 Tab 6 (Journey):** Creative timeline, technical decisions, tech stack, and V1 → V4 evolution.
""",
        "sidebar_motor": "**⚙️ Analytics Engine:**",
        "sidebar_prod": "**🧠 Production Model:**\n`LightGBM V4 (Isotonic Calibration)`",
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
        "kpi5_sub": "Mortality / Long Stay (V4)",
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
        "dq_hdr": "### 🛡️ Data Quality Audit & Integrity Contract (SIH/SUS)",
        "dq_sub": "Automated monitoring of **spatio-temporal coverage, completeness, clinical validity, and billing consistency** executed via SQL/DuckDB prior to predictive modeling.",
        "dq_banner_local": "🔎 **Full Local Audit:** Coverage and demographic completeness computed across **100% of {n} admissions** in the Cube; domain validity and clinical consistency audited row-by-row across local Parquet partitions (`{na}` rows).",
        "dq_banner_cloud": "🔎 **Hybrid Audit (Cloud):** Coverage, volumes, and demographic underreporting measured over **100% of {n} admissions** via Cube; row-level domain checks (`idade`, `qt_diarias`, `uti_mes_to`, AI nulls) audited on the national stratified sample (`{na}` cases across 27 States).",
        "dq_kpi1_lbl": "State × Month Coverage",
        "dq_kpi1_sub": "Active Partitions (2024)",
        "dq_kpi2_lbl": "AI Completeness (8 Features)",
        "dq_kpi2_sub": "Zero Intake Nulls",
        "dq_kpi3_lbl": "Race / ICD Underreporting",
        "dq_kpi3_sub": "Not Declared / Unknown",
        "dq_kpi4_lbl": "ICU > Total Stay Days",
        "dq_kpi4_sub": "Billed Anomaly (Sample)",
        "dq_contract_hdr": "#### 📋 Automated Data Contract Checklist (Data Quality Framework)",
        "dq_contract_sub": "Each expectation below validates live DuckDB tables across 4 data quality dimensions (Coverage, Completeness, Validity, Consistency):",
        "dq_status_col": "Status",
        "dq_c1_hdr": "#### Monthly Ingestion Stability (Volume & Active States)",
        "dq_c1_cap": "Checks whether any 2024 month suffered submission drops or missing `.dbc` state files from DATASUS.",
        "dq_c1_x": "Month (2024)",
        "dq_c1_y": "Admissions Ingested",
        "dq_c2_hdr": "#### Underreporting & Unspecified Labels by State",
        "dq_c2_cap": "Compares states with the highest share of `Other` ward specialty or `Not Declared` race/ethnicity (%), exposing hospital reporting gaps.",
        "dq_c2_x": "Share (%)",
        "dq_c2_y": "State (UF)",
        "dq_c2_leg": "Metric",
        "dq_c2_opt1": "Other Specialty (%)",
        "dq_c2_opt2": "Race Not Declared (%)",
        "dq_uf_tbl_hdr": "#### 🗺️ Completeness Breakdown by State (UF)",
        "dq_uf_cols": {
            "uf": "State",
            "regiao": "Region",
            "internacoes": "Admissions",
            "meses_ativos": "Active Months",
            "pct_raca_nd": "Race Not Decl. (%)",
            "pct_cid_ign": "Unknown ICD (%)",
            "pct_esp_outras": "Other Spec. (%)",
        },
        "dq_exp_uti_tit": "🔍 **Real SIH/SUS Discovery: Cases where `uti_mes_to > qt_diarias` ({n} cases / {pct:.2f}%)**",
        "dq_exp_uti_txt": "In official SIH/SUS records, billed ICU days (`uti_mes_to`) can exceed total stay days (`qt_diarias`) due to continuation AIHs, multi-month rollovers, or source billing anomalies. Surfacing this early explains why post-admission billing fields **must not** enter intake models or replace `qt_diarias` as the stay target.",
        "dq_uti_cols": {
            "n_aih": "Anonymized ID (AIH)",
            "uf": "State",
            "especialidade_leito": "Specialty",
            "diag_princ": "ICD-10",
            "qt_diarias": "Total Stay Days",
            "uti_mes_to": "Billed ICU Days",
            "diferenca_dias": "ICU Excess (d)",
        },
        "dq_exp_null_tit": "🧹 **Historical Raw Schema Missingness & Anti-Leakage Audit (113 Columns)**",
        "dq_exp_null_txt": """
- **Secondary Diagnoses (`diagsec1`–`diagsec9` & `tpdisec1`–`9`):** exhibited **`89.65%` to `100%` missing values** during the full pilot audit, making automated comorbidity indices (Charlson/Elixhauser) unreliable and documented transparently in Tab 5.
- **Constant or Redundant Fields (`ano_cmpt`, `uf_zi`, raw `dt_inter`):** pruned or transformed into structural features (`mes_internacao`, `internacao_fim_de_semana`) and Hive partition keys.
- **Anti-Leakage Guardrail (`34` post-stay columns):** billing/outcome fields (`cobranca`, `val_tot`, `dt_saida`, `proc_rea`, `uti_mes_*`) exist in raw DATASUS dumps only after stay closure — blocked before AI model training (Tab 3).
""",
        "ml_hdr": "### 🤖 How the AI Works & Model Selection (V4 National)",
        "ml_intro": """At the exact moment of hospital admission, two predictive models run simultaneously using only **8 intake variables** (age, sex, ICD-10 chapter, ward specialty, admission urgency, complexity level, inter-municipal transfer, and weekend admission):
1. **Long Length of Stay Model (≥ 15 days):** estimates the probability of prolonged hospitalization to guide bed turnover and social work early on.
2. **In-Hospital Mortality Model:** estimates fatal deterioration risk to prioritize intensive monitoring or ICU allocation.""",
        "ml_why": """🏆 **Why did we choose LightGBM as our production model?**
- **Statistical tie at the top with XGBoost:** Achieved strong discrimination (**ROC-AUC `0.8793`** for mortality and **`0.8160`** for long stay) alongside optimal empirical calibration (**Brier Score `0.0329` and `0.0480`**).
- **Highest long-stay detection:** Captures **`83.92%`** of prolonged stays and **`87.58%`** of fatal cases on admission.
- **Production efficiency:** Scores patients in milliseconds with a tiny memory footprint — ideal for real-time triage and cloud hosting, outperforming *Random Forest* in speed and *Logistic Regression* in accuracy.""",
        "ml_guide": """📖 **Metrics Mini-Guide (Plain Language):**
- **ROC-AUC (0 to 1):** Measures how well the model separates high-risk from stable patients (`0.5` = random guess; `> 0.80` = strong).
- **PR-AUC:** Precision focus on rare events (mortality and long stays represent a small minority of admissions).
- **Brier Score (lower is better):** Measures how close predicted probabilities are to actual outcomes after **Isotonic Calibration** (e.g., 20% predicted means ~20 in 100 patients experience the event).
- **Ethics & No *Data Leakage*:** Race and month are excluded from model training to prevent algorithmic bias; post-discharge billing fields were removed to prevent leakage.""",
        "ml_bench": "#### 📊 National Benchmark (Blind Test on `300,000` admissions across 27 States)",
        "ml_ob_sub": "##### 1. In-Hospital Mortality *(National Avg: 3.85%)*",
        "ml_lg_sub": "##### 2. Long Stay ≥ 15 Days *(National Avg: 6.17%)*",
        "ml_cols": {"Modelo": "Model", "ROC_AUC": "ROC-AUC", "PR_AUC": "PR-AUC", "Brier_Calibrado": "Brier Score", "Limiar_Otimo_Pct": "Alert Cut (%)", "Sensibilidade_Pct": "Recall (%)", "Especificidade_Pct": "Specif. (%)"},
        "ml_models_hdr": "#### 🧠 How Each Model Works — Pros & Cons",
        "ml_models_sub": "Concise breakdown of how each algorithm evaluated in the national benchmark scores admission risk:",
        "ml_models_cards": [
            {
                "title": "1. Logistic Regression *(Linear Baseline)*",
                "how": "**How it works:** Classical statistical model that sums fixed weights across intake features and passes the total through a sigmoid curve to estimate outcome odds.",
                "pros": "**✅ Pros:** Very fast, transparent feature coefficients, near-zero RAM footprint, and naturally well calibrated out of the box (`Brier 0.0334 / 0.0493`).",
                "cons": "**⚠️ Cons:** Assumes strictly linear additivity — misses complex clinical interactions (e.g., advanced age × ICD-10 chapter × urgency), yielding lower Long Stay discrimination (`ROC-AUC 0.7805`, `PR-AUC 0.2916`)."
            },
            {
                "title": "2. Random Forest *(Bagging Ensemble)*",
                "how": "**How it works:** Builds hundreds of independent decision trees on random data and feature subsamples, averaging predictions across the entire forest.",
                "pros": "**✅ Pros:** Captures non-linear relationships naturally, resists outliers/noise well, and reached the highest test specificity (`74.94%` Mortality, `62.32%` Long Stay).",
                "cons": "**⚠️ Cons:** Produces large serialized files in memory/cloud, has slower inference across deep trees, and trailed gradient boosting on rare-event precision (`PR-AUC 0.1994` Mortality)."
            },
            {
                "title": "3. XGBoost *(Histogram Gradient Boosting)*",
                "how": "**How it works:** Trains trees sequentially so each new tree corrects residual errors from prior trees, using histogram binning and built-in L1/L2 regularization.",
                "pros": "**✅ Pros:** Co-leader in overall ranking accuracy (`ROC-AUC 0.8795` Mortality, `0.8161` Long Stay), highest Mortality recall (`88.47%`), and highest raw `PR-AUC` (`0.2114 / 0.3427`).",
                "cons": "**⚠️ Cons:** Slightly heavier and slower to train on 900k rows than LightGBM, and sensitive to stacked imbalance penalties (seen in the V1 SMOTE + class-weight collapse)."
            },
            {
                "title": "4. LightGBM *(Leaf-Wise Gradient Boosting — Production V4)*",
                "how": "**How it works:** Fast histogram-based gradient boosting that expands tree leaves with the highest error reduction (*leaf-wise*), cutting compute overhead substantially.",
                "pros": "**✅ Pros:** Statistical tie at the top with XGBoost (`ROC-AUC 0.8793 / 0.8160`; calibrated `Brier 0.0329 / 0.0480`), top Long Stay recall (`83.92%`), low RAM, and millisecond cloud latency.",
                "cons": "**⚠️ Cons:** Leaf-wise growth can overfit small/noisy datasets without leaf constraints and relies on post-hoc isotonic calibration plus importance charts for clinical explainability."
            },
        ],
        "ml_evid": "#### 📈 Visual Diagnostics & Explainability (V4 National)",
        "ml_img1": "ROC & PR Curves: curves closer to the top-left (ROC) or top-right (PR) indicate stronger accuracy.",
        "ml_img2": "Isotonic Calibration: curves hugging the diagonal dashed line confirm true clinical probabilities.",
        "ml_img3": "Top features driving AI decisions (e.g., Age, Internal Medicine ward, and Emergency intake).",
        "cli_hdr": "### 🏥 Clinical Management & Patient Risk Cockpit",
        "cli_sub": "Real-time hospital ward simulation displaying anonymized admission IDs (`#AIH`), primary diagnosis (`ICD-10`), current stay length, and two **Predictive Risk Flags** computed at admission.",
        "cli_exp": "⚙️ **Calibrate Alert Thresholds & Understand the Percentages**",
        "cli_exp_txt": "💡 **Why aren't default thresholds set to 50%?** Across Brazil (2024), baseline in-hospital mortality is **3.85%** and long stays (≥15 days) occur in **6.17%** of admissions. Waiting for 50% risk at intake would miss nearly all high-risk patients. Thus, **Yellow/Orange** cuts start at the **Youden Index** (`4.0%` Mortality, `6.4%` Long Stay — the statistical optimum maximizing early detection), while **Red** alerts mark **3x to 4x higher risk than the national baseline** (`15.0%` and `20.0%`). Adjust below based on hospital capacity:",
        "sl_ob": "⚠️ Mortality Risk Cutoffs — Watch / Critical (%):",
        "sl_lg": "🚩 Long Stay Cutoffs — Medium / High Risk (%):",
        "leg_lg": "🚩 **Long Stay (≥ 15 days):** `🟢 Low < {a:.1f}%` | `🟡 Medium {a:.1f}%–{b:.1f}%` | `🔴 High Risk ≥ {c:.1f}%`",
        "leg_ob": "⚠️ **Mortality Risk:** `🟢 Stable < {a:.1f}%` | `🟠 Watch {a:.1f}%–{b:.1f}%` | `🔴 Critical ≥ {c:.1f}%`",
        "sim_exp": "➕ **Simulate New Patient Admission (Real-Time Triage Scorecard)**",
        "sim_idade": "Age (years):",
        "sim_sexo": "Biological Sex:",
        "sim_espec": "Ward Specialty:",
        "sim_car": "Admission Type:",
        "sim_cid": "Primary Diagnosis (ICD-10 Chapter):",
        "sim_comp": "Complexity Level:",
        "sim_transf": "Transferred from Another Municipality?",
        "sim_fds": "Weekend Admission?",
        "sim_nao": "No",
        "sim_sim": "Yes",
        "sim_btn": "🚨 Compute Admission Risk Score",
        "sim_res": "#### Triage Result (Calibrated Brazil V4 Probabilities):",
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
        "p_count": "Showing **{n}** sampled inpatients:",
        "p_cols": {
            "id_anonimo": "Patient ID",
            "uf": "State",
            "idade_anos": "Age",
            "sexo_desc": "Sex",
            "especialidade_leito": "Specialty",
            "motivo_cid": "Reason (ICD-10)",
            "qt_diarias": "Stay Days",
            "uti_mes_to": "ICU Days",
            "flag_longa": "🚩 Long Stay",
            "flag_obito": "⚠️ Mortality Risk",
        },
        "fmt_idade": "%.0f yrs",
        "dic_hdr": "### 📖 Complete SIH/SUS Dictionary (113 Official Columns + Derived) & Glossary",
        "dic_sub": "Full audit of every DATASUS hospital admission column, explicitly stating what powers the **AI Model**, what stays in the **Epidemiology Profile**, and what was **Excluded**.",
        "dic_search": "🔍 Search column, domain or meaning:",
        "dic_ph": "E.g.: cobranca, diagsec, raca_cor, ROC-AUC, Brier...",
        "dic_filter_lbl": "Filter by Project Usage:",
        "dic_filter_opts": [
            ("all", "All Columns (125)"),
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
        "dic_count": "Showing **{n}** cataloged variables (`113` original SIH/SUS fields + `12` derived fields).",
        "j_hdr": "### 🧭 Project Journey: Creative Process, Decisions & Architecture (V1 → V4)",
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
- **Why sample `1,500,000` for training and `30,000` for the cloud cockpit while keeping `100%` in Tab 1?**
  - *Tab 1 (0% sampling):* Built a **`3.99 MB` pre-aggregated Cube** (`419,801` grouped rows) preserving **100% exact parity** across all `13,584,534` admissions, `520,884` deaths, and `839,972` long stays.
  - *V4 Training (`1.5M` = 900k train / 300k calibration / 300k blind test):* With 8 intake features (~40 encoded columns), `900k` stratified rows saturate learning curves across all 27 states cleanly within workstation RAM.
  - *Cloud Cockpit (`30,000` cases = `0.34 MB`):* Enables instant multi-state ward filtering inside free Streamlit Cloud memory limits.
- **Isotonic Calibration & Youden Cutoffs:** Replacing extreme `scale_pos_weight` with native log-loss + **Isotonic Calibration** on `300k` holdout cases cut **Brier Score from `0.12–0.16` down to `0.0329` (Mortality) and `0.0480` (Long Stay)**, producing true clinical probabilities and **Youden Index cutoffs (`4.0%` / `6.4%`)** with hospital-adjustable sliders.
- **Portfolio Product:** Added PT/EN bilingual switcher, cascading Region → State filters, full 113-column audit, and Streamlit Cloud deployment (`13.2 MiB` Git repo).
""",
        "j_evol_hdr": "#### 📊 Version Evolution Summary Table (V1 → V4)",
        "j_tech_hdr": "#### 🛠️ Tech Stack Used & Architectural Rationale",
        "j_tech_sub": "Every library was chosen to process 13.58M federal admissions on standard hardware and serve sub-second cloud triage for free:",
        "footer": "SUS Analytics Brazil 2024 • DuckDB + Parquet + Calibrated LightGBM V4 • Healthcare Intelligence Portfolio",
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
- **📖 Pestaña 5 (Diccionario):** Catálogo auditado de las **113 columnas oficiales del SIH/SUS** + derivadas y glosario.
- **🧭 Pestaña 6 (Trayectoria):** Historial creativo, decisiones técnicas, stack usada y evolución V1 → V4.
""",
        "sidebar_motor": "**⚙️ Motor Analítico:**",
        "sidebar_prod": "**🧠 Producción:**\n`LightGBM V4 (Calibración Isotónica)`",
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
        "kpi5_sub": "Óbito / Larga Est. (V4)",
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
        "dq_hdr": "### 🛡️ Auditoría de Calidad de Datos y Contrato de Integridad (SIH/SUS)",
        "dq_sub": "Monitoreo automatizado de **cobertura espacio-temporal, completitud, validez clínica y consistencia de facturación** ejecutado mediante SQL/DuckDB antes del modelado predictivo.",
        "dq_banner_local": "🔎 **Auditoría Integral Local:** Cobertura y completitud demográfica calculadas sobre el **100% de {n} ingresos** en el Cubo; reglas de dominio e integridad verificadas fila por fila en los Parquets locales (`{na}` filas).",
        "dq_banner_cloud": "🔎 **Auditoría Híbrida (Nube):** Cobertura, volúmenes y subregistro demográfico medidos sobre el **100% de {n} ingresos** mediante Cubo; reglas fila a fila (`idade`, `qt_diarias`, `uti_mes_to`, nulos de la IA) auditadas en la muestra estratificada (`{na}` casos de 27 Estados).",
        "dq_kpi1_lbl": "Cobertura Estado × Mes",
        "dq_kpi1_sub": "Particiones Activas (2024)",
        "dq_kpi2_lbl": "Completitud IA (8 Features)",
        "dq_kpi2_sub": "Cero Nulos al Ingreso",
        "dq_kpi3_lbl": "Subregistro Raza / CIE",
        "dq_kpi3_sub": "No Declarada / Ignorado",
        "dq_kpi4_lbl": "UCI > Días Totales",
        "dq_kpi4_sub": "Anomalía Facturada (Muestra)",
        "dq_contract_hdr": "#### 📋 Lista de Verificación del Contrato de Datos (Data Quality Framework)",
        "dq_contract_sub": "Cada regla valida en vivo las tablas cargadas en DuckDB en 4 dimensiones (Cobertura, Completitud, Validez y Consistencia):",
        "dq_status_col": "Estado",
        "dq_c1_hdr": "#### Estabilidad Mensual de Envío (Volumen y Estados Activos)",
        "dq_c1_cap": "Verifica si algún mes de 2024 sufrió caídas anómalas o pérdida de archivos `.dbc` en DATASUS.",
        "dq_c1_x": "Mes (2024)",
        "dq_c1_y": "Ingresos Recibidos",
        "dq_c2_hdr": "#### Subregistro y Categorías Inespecíficas por Estado",
        "dq_c2_cap": "Compara los estados con mayor porcentaje de especialidad `Otras` o `Raza/Color No Declarada` (%), revelando brechas de registro.",
        "dq_c2_x": "Porcentaje (%)",
        "dq_c2_y": "Estado (UF)",
        "dq_c2_leg": "Indicador",
        "dq_c2_opt1": "Espec. Otras (%)",
        "dq_c2_opt2": "Raza No Declarada (%)",
        "dq_uf_tbl_hdr": "#### 🗺️ Diagnóstico de Completitud por Estado (UF)",
        "dq_uf_cols": {
            "uf": "Estado",
            "regiao": "Región",
            "internacoes": "Ingresos",
            "meses_ativos": "Meses Activos",
            "pct_raca_nd": "Raza No Decl. (%)",
            "pct_cid_ign": "CIE Ignorado (%)",
            "pct_esp_outras": "Espec. Otras (%)",
        },
        "dq_exp_uti_tit": "🔍 **Hallazgo Real del SIH/SUS: Casos donde `uti_mes_to > qt_diarias` ({n} casos / {pct:.2f}%)**",
        "dq_exp_uti_txt": "En los registros oficiales del SIH/SUS, las estancias de UCI facturadas (`uti_mes_to`) pueden superar `qt_diarias` por AIHs de continuación, traslados entre competencias o inconsistencias en origen. Detectar esto en Calidad de Datos demuestra por qué variables post-ingreso **no deben** usarse para predecir admisión ni reemplazar `qt_diarias`.",
        "dq_uti_cols": {
            "n_aih": "ID Anonimizado (AIH)",
            "uf": "Estado",
            "especialidade_leito": "Especialidad",
            "diag_princ": "CIE-10",
            "qt_diarias": "Días Totales",
            "uti_mes_to": "Días UCI Facturados",
            "diferenca_dias": "Exceso UCI (d)",
        },
        "dq_exp_null_tit": "🧹 **Auditoría Histórica de Nulos y Anti-Leakage del Esquema Original (113 Columnas)**",
        "dq_exp_null_txt": """
- **Diagnósticos Secundarios (`diagsec1`–`diagsec9` y `tpdisec1`–`9`):** presentaron **`89,65%` a `100%` de nulos** en el piloto completo, impidiendo usar índices automáticos de comorbilidad (Charlson/Elixhauser) en producción y quedando documentados en la Pestaña 5.
- **Campos Constantes o Redundantes (`ano_cmpt`, `uf_zi`, `dt_inter` cruda):** removidos o transformados en variables estructurales (`mes_internacao`, `internacao_fim_de_semana`) y claves de partición Hive.
- **Guardrail Anti-Leakage (`34` columnas post-alta):** campos como `cobranca`, `val_tot`, `dt_saida`, `proc_rea` y `uti_mes_*` solo existen tras el cierre de la estancia — bloqueados antes de entrenar la IA (Pestaña 3).
""",
        "ml_hdr": "### 🤖 Cómo Funciona la IA y Elección del Modelo (V4 Nacional)",
        "ml_intro": """Al momento del ingreso hospitalario, dos modelos predictivos operan en paralelo usando solo **8 datos de admisión** (edad, sexo, capítulo CIE-10, especialidad, carácter urgencia/electivo, complejidad, traslado intermunicipal y entrada en fin de semana):
1. **Modelo de Larga Estancia (≥ 15 días):** estima la probabilidad de hospitalización prolongada para apoyar el giro de camas y trabajo social.
2. **Modelo de Mortalidad Intrahospitalaria:** estima el riesgo vital agudo para priorizar monitoreo intensivo o cama de UCI.""",
        "ml_why": """🏆 **¿Por qué elegimos LightGBM como modelo oficial?**
- **Empate técnico en la cima con XGBoost:** Alcanzó excelente capacidad de separación (**ROC-AUC `0,8793`** en mortalidad y **`0,8160`** en larga estancia) junto a la mejor calibración (**Brier Score `0,0329` y `0,0480`**).
- **Mayor detección en larga estancia:** Identifica al **`83,92%`** de los casos prolongados y al **`87,58%`** de los casos críticos de mortalidad al ingreso.
- **Agilidad en producción:** Responde en milisegundos con mínimo consumo de memoria RAM — ideal para triaje en tiempo real y despliegue en la nube.""",
        "ml_guide": """📖 **Mini-Guía de Métricas (Lenguaje Claro):**
- **ROC-AUC (0 a 1):** Mide cuánto distingue el modelo entre pacientes graves y estables (`0,5` = azar; `> 0,80` = excelente).
- **PR-AUC:** Precisión enfocada en eventos poco frecuentes (fallecimiento y estancias largas).
- **Brier Score (menor es mejor):** Mide si la probabilidad prevista coincide con la realidad tras la **Calibración Isotónica** (ej.: 20% previsto equivale a ~20 casos reales cada 100 pacientes).
- **Ética y Cero *Data Leakage*:** Raza y mes no entran al modelo para evitar sesgos; variables post-alta fueron eliminadas.""",
        "ml_bench": "#### 📊 Comparativo Nacional (Prueba Ciega en `300.000` ingresos de los 27 Estados)",
        "ml_ob_sub": "##### 1. Mortalidad Intrahospitalaria *(Promedio Nacional: 3,85%)*",
        "ml_lg_sub": "##### 2. Larga Estancia ≥ 15 Días *(Promedio Nacional: 6,17%)*",
        "ml_cols": {"Modelo": "Modelo", "ROC_AUC": "ROC-AUC", "PR_AUC": "PR-AUC", "Brier_Calibrado": "Brier Score", "Limiar_Otimo_Pct": "Corte Alerta (%)", "Sensibilidade_Pct": "Detección (%)", "Especificidade_Pct": "Especif. (%)"},
        "ml_models_hdr": "#### 🧠 Cómo Funciona Cada Modelo — Pros y Contras",
        "ml_models_sub": "Resumen claro de cómo cada algoritmo evaluado en el comparativo nacional calcula el riesgo al ingreso:",
        "ml_models_cards": [
            {
                "title": "1. Regresión Logística *(Baseline Lineal)*",
                "how": "**Cómo funciona:** Modelo estadístico clásico que suma pesos fijos por cada variable de ingreso y aplica una función sigmoide para estimar la probabilidad del evento.",
                "pros": "**✅ Pros:** Muy rápida, coeficientes transparentes, consumo casi nulo de RAM y naturalmente bien calibrada (`Brier 0,0334 / 0,0493`).",
                "cons": "**⚠️ Contras:** Asume linealidad simple — no captura interacciones clínicas complejas (ej.: edad avanzada × capítulo CIE × urgencia), quedando atrás en Larga Estancia (`ROC-AUC 0,7805` y `PR-AUC 0,2916`)."
            },
            {
                "title": "2. Random Forest *(Ensemble Bagging)*",
                "how": "**Cómo funciona:** Entrena cientos de árboles de decisión independientes sobre submuestras aleatorias de datos y variables, promediando el voto de todo el bosque.",
                "pros": "**✅ Pros:** Captura relaciones no lineales de forma robusta frente a valores atípicos y logró la mayor especificidad nacional (`74,94%` en Mortalidad y `62,32%` en Larga Estancia).",
                "cons": "**⚠️ Contras:** Genera modelos pesados en RAM y en la nube, responde más lento entre árboles profundos y obtuvo menor precisión en eventos poco frecuentes (`PR-AUC 0,1994` Mortalidad)."
            },
            {
                "title": "3. XGBoost *(Gradient Boosting — Histograma)*",
                "how": "**Cómo funciona:** Entrena árboles de forma secuencial donde cada árbol corrige los errores residuales anteriores, usando histogramas numéricos y regularización L1/L2.",
                "pros": "**✅ Pros:** Colíder en capacidad de separación (`ROC-AUC 0,8795` Mortalidad y `0,8161` Larga Estancia), mayor detección en Mortalidad (`88,47%`) y mayor `PR-AUC` numérico (`0,2114 / 0,3427`).",
                "cons": "**⚠️ Contras:** Ligeramente más lento y pesado al entrenar 900 mil filas que LightGBM, además de requerir cuidado con pesos por desbalance (evidenciado en el colapso con SMOTE en V1)."
            },
            {
                "title": "4. LightGBM *(Gradient Boosting Leaf-Wise — Modelo Oficial V4)*",
                "how": "**Cómo funciona:** Algoritmo rápido por histogramas que expande primero las hojas con mayor ganancia de error (*leaf-wise*), reduciendo cómputo y memoria.",
                "pros": "**✅ Pros:** Empate técnico en la cima con XGBoost (`ROC-AUC 0,8793 / 0,8160`; `Brier 0,0329 / 0,0480` tras calibración), mayor sensibilidad en Larga Estancia (`83,92%`), mínima RAM y latencia en milisegundos.",
                "cons": "**⚠️ Contras:** El crecimiento *leaf-wise* puede sobreajustar bases pequeñas sin límite de hojas y requiere calibración isotónica más gráficos de importancia para explicabilidad clínica."
            },
        ],
        "ml_evid": "#### 📈 Evidencias Visuales y Explicabilidad (V4 Nacional)",
        "ml_img1": "Curvas ROC y PR: cuanto más cerca de la esquina superior, mayor precisión discriminativa.",
        "ml_img2": "Calibración Isotónica: curvas pegadas a la diagonal punteada prueban probabilidades realistas.",
        "ml_img3": "Factores con mayor peso en la IA (ej.: Edad, Medicina Interna e Ingreso de Urgencia).",
        "cli_hdr": "### 🏥 Panel de Gestión Clínica y Riesgo de Pacientes",
        "cli_sub": "Simulación hospitalaria de seguimiento de camas: muestra identificador anonimizado (`#AIH`), diagnóstico (`CIE-10`), días ingresados y dos **Alertas Predictivas** calculadas al ingreso.",
        "cli_exp": "⚙️ **Calibrar Umbrales de Alerta y Entender los Porcentajes**",
        "cli_exp_txt": "💡 **¿Por qué los cortes por defecto no son 50%?** En Brasil (2024), la tasa real de mortalidad es **3,85%** y la de larga estancia (≥15 días) es **6,17%**. Esperar un 50% al ingreso dejaría pasar casi todos los casos graves. Por eso, el corte **Amarillo/Naranja** parte del **Índice de Youden** (`4,0%` Mortalidad y `6,4%` Larga Estancia), mientras la zona **Roja** señala perfiles con **3x a 4x más riesgo que el promedio nacional** (`15,0%` y `20,0%`). Ajuste abajo según la ocupación del hospital:",
        "sl_ob": "⚠️ Cortes Riesgo de Mortalidad — Atención / Crítico (%):",
        "sl_lg": "🚩 Cortes Larga Estancia — Medio / Alto Riesgo (%):",
        "leg_lg": "🚩 **Larga Estancia (≥ 15 días):** `🟢 Bajo < {a:.1f}%` | `🟡 Medio {a:.1f}%–{b:.1f}%` | `🔴 Alto Riesgo ≥ {c:.1f}%`",
        "leg_ob": "⚠️ **Riesgo de Mortalidad:** `🟢 Estable < {a:.1f}%` | `🟠 Atención {a:.1f}%–{b:.1f}%` | `🔴 Crítico ≥ {c:.1f}%`",
        "sim_exp": "➕ **Simular Nuevo Ingreso de Paciente (Scorecard en Tiempo Real)**",
        "sim_idade": "Edad (años):",
        "sim_sexo": "Sexo Biológico:",
        "sim_espec": "Especialidad de Cama:",
        "sim_car": "Tipo de Ingreso:",
        "sim_cid": "Diagnóstico Principal (Capítulo CIE-10):",
        "sim_comp": "Complejidad:",
        "sim_transf": "¿Trasladado desde Otro Municipio?",
        "sim_fds": "¿Ingreso en Fin de Semana?",
        "sim_nao": "No",
        "sim_sim": "Sí",
        "sim_btn": "🚨 Calcular Score de Ingreso",
        "sim_res": "#### Resultado del Triaje (Probabilidades Calibradas Brasil V4):",
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
        "p_count": "Mostrando **{n}** pacientes muestreados:",
        "p_cols": {
            "id_anonimo": "ID Paciente",
            "uf": "Estado",
            "idade_anos": "Edad",
            "sexo_desc": "Sexo",
            "especialidade_leito": "Especialidad",
            "motivo_cid": "Motivo (CIE-10)",
            "qt_diarias": "Días Estancia",
            "uti_mes_to": "Días UCI",
            "flag_longa": "🚩 Larga Estancia",
            "flag_obito": "⚠️ Riesgo Óbito",
        },
        "fmt_idade": "%.0f años",
        "dic_hdr": "### 📖 Diccionario Completo SIH/SUS (113 Columnas Oficiales + Derivadas) y Glosario",
        "dic_sub": "Auditoría integral de cada campo del DATASUS, indicando explícitamente qué alimenta a la **IA**, qué queda solo en el **Perfil Epidemiológico** y qué fue **Descartado**.",
        "dic_search": "🔍 Buscar columna, bloque o significado:",
        "dic_ph": "Ej.: cobranca, diagsec, raca_cor, ROC-AUC, Brier...",
        "dic_filter_lbl": "Filtrar por Uso en el Proyecto:",
        "dic_filter_opts": [
            ("all", "Todas las Columnas (125)"),
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
        "dic_count": "Mostrando **{n}** variables del catálogo completo (`113` originales SIH/SUS + `12` derivadas).",
        "j_hdr": "### 🧭 Trayectoria del Proyecto: Proceso Creativo, Decisiones y Arquitectura (V1 → V4)",
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
- **Ingesta nacional (27 Estados / 324 archivos `.dbc`):** Pipeline paralelo vía FTP DATASUS + Arrow (`datasuspy`), comprimiendo **~20 GB** de `13.584.534` ingresos a **`186,06 MB`** en Parquet particionado Hive consultado con **DuckDB**.
- **¿Por qué muestrear `1.500.000` en entrenamiento y `30.000` en el panel web manteniendo el `100%` en la Pestaña 1?**
  - *Pestaña 1 (0% muestreo):* Creamos un **Cubo Agregado de `3,99 MB`** (`419.801` filas) con **100% de paridad exacta** de los `13.584.534` ingresos, `520.884` fallecimientos y `839.972` estancias largas.
  - *Entrenamiento V4 (`1,5M` = 900k entreno / 300k calibración / 300k prueba ciega):* Con 8 variables (~40 columnas codificadas), `900 mil` casos saturan la curva de aprendizaje de los 27 estados sin desbordar memoria RAM.
  - *Panel Cloud (`30.000` casos = `0,34 MB`):* Permite filtrar pacientes reales de todos los estados en milisegundos dentro de la cuota gratuita de Streamlit Cloud.
- **Calibración Isotónica y Umbrales de Youden:** Reemplazando `scale_pos_weight` extremo por *log-loss* nativo + **Calibración Isotónica** en `300k` casos, el **Brier Score bajó de `0,12–0,16` a `0,0329` (Mortalidad) y `0,0480` (Larga Estancia)**, logrando probabilidades reales y cortes por **Índice de Youden (`4,0%` / `6,4%`)** con deslizadores ajustables.
- **Producto Final:** Interfaz trilingüe (PT/EN/ES), filtros en cascada Región → Estado, diccionario completo de 113 columnas y despliegue online.
""",
        "j_evol_hdr": "#### 📊 Tabla Resumen de Evolución de Versiones (V1 → V4)",
        "j_tech_hdr": "#### 🛠️ Stack Tecnológico Utilizado y Por Qué Se Eligió",
        "j_tech_sub": "Arquitectura diseñada para procesar más de 13,5 millones de registros en hardware estándar y operar gratis en la nube con latencia de milisegundos:",
        "footer": "SUS Analytics Brasil 2024 • DuckDB + Parquet + LightGBM V4 Calibrado • Portafolio de Inteligencia en Salud",
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
        ])
    if lang == "es":
        return pd.DataFrame([
            {"Versión": "V1 (Baseline)", "Alcance": "Acre (56.459)", "Objetivos": "Solo Larga Est.", "Features": "7 (con fuga cobranca)", "Decisión / Hito Principal": "Detección de fuga en `cobranca` y corrección del colapso SMOTE + peso en XGBoost."},
            {"Versión": "V2 (Dual-Triage)", "Alcance": "Acre (56.459)", "Objetivos": "Larga Est. + Óbito", "Features": "10 (estrictas ingreso)", "Decisión / Hito Principal": "Eliminación de 24 cols financieras/post-alta; Matriz 2x2 y variables de traslado/cama."},
            {"Versión": "V3 (IA Ética)", "Alcance": "Acre (56.459)", "Objetivos": "Larga Est. + Óbito", "Features": "8 (clínicas y éticas)", "Decisión / Hito Principal": "Exclusión de `raca_cor` y `mes_internacao` de la IA (mantenidas solo en Equidad)."},
            {"Versión": "V4 (Nacional Cloud)", "Alcance": "Brasil 27 UFs (13,58M)", "Objetivos": "Larga Est. + Óbito", "Features": "8 (LightGBM calibrado)", "Decisión / Hito Principal": "Lakehouse DuckDB/Parquet, muestra 1,5M, Calibración Isotónica (Brier 0,033/0,048), app PT/EN/ES."},
        ])
    return pd.DataFrame([
        {"Versão": "V1 (Baseline)", "Escopo": "Acre (56.459)", "Alvos": "Apenas Longa Perm.", "Features": "7 (com leakage cobranca)", "Decisão / Marco Principal": "Descoberta do vazamento em `cobranca` e correção do colapso SMOTE + peso no XGBoost."},
        {"Versão": "V2 (Dual-Triage)", "Escopo": "Acre (56.459)", "Alvos": "Longa Perm. + Óbito", "Features": "10 (estritas de entrada)", "Decisão / Marco Principal": "Expurgo de 24 colunas financeiras/pós-alta; criação da Matriz 2x2 e flags geográficas/leito."},
        {"Versão": "V3 (IA Ética)", "Escopo": "Acre (56.459)", "Alvos": "Longa Perm. + Óbito", "Features": "8 (clínicas e éticas)", "Decisão / Marco Principal": "Remoção de `raca_cor` e `mes_internacao` da IA clínica (mantidas só na Aba 1 de Equidade)."},
        {"Versão": "V4 (Nacional Cloud)", "Escopo": "Brasil 27 UFs (13,58M)", "Alvos": "Longa Perm. + Óbito", "Features": "8 (LightGBM calibrado)", "Decisão / Marco Principal": "Lakehouse DuckDB/Parquet, amostra de 1,5M, Calibração Isotônica (Brier 0,033/0,048), app PT/EN/ES na nuvem."},
    ])


_TECH_STACK = [
    (
        "Python 3.11+ & Joblib",
        {"pt": "Linguagem Base & Serialização", "en": "Core Language & Serialization", "es": "Lenguaje Base y Serialización"},
        {"pt": "Orquestra ETL, modelagem, calibração e app web; Joblib empacota pipelines V4 em .joblib com cache instantâneo.", "en": "Unifies ETL, modeling, calibration, and web UI; Joblib packages V4 pipelines into .joblib for instant caching.", "es": "Unifica ETL, modelado, calibración y app web; Joblib empaqueta los pipelines V4 en .joblib con caché instantánea."},
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
        {"pt": "Isola OneHotEncoder + StandardScaler em Pipeline, treina baselines (Logística/RF) e aplica CalibratedClassifierCV isotônica.", "en": "Wraps OneHotEncoder + StandardScaler in Pipeline, runs baselines (Logistic/RF), and fits isotonic calibration.", "es": "Encapsula OneHotEncoder + StandardScaler en Pipeline, entrena baselines y ajusta Calibración Isotónica."},
        {"pt": "Impede data leakage no pré-processamento e derrubou o Brier Score de 0,12–0,16 para 0,0329 / 0,0480.", "en": "Prevents preprocessing leakage and cut probability error (Brier Score) down to 0.0329 / 0.0480.", "es": "Evita fugas en preprocesamiento y bajó el Brier Score de 0,12–0,16 a 0,0329 / 0,0480."},
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


def get_regras_qualidade_df(res_dq: dict, esperado_ufs: int, lang: str) -> tuple[pd.DataFrame, list[str]]:
    """Gera tabela auditável de Contrato de Dados (PT/EN/ES) a partir das checagens DuckDB ao vivo."""
    k = lang if lang in ("pt", "en", "es") else "pt"
    cols_map = {
        "pt": ("Dimensão", "Regra de Contrato / Expectativa", "Resultado Medido", "Status", "Impacto na Engenharia"),
        "en": ("Dimension", "Contract Rule / Expectation", "Measured Result", "Status", "Engineering Impact"),
        "es": ("Dimensión", "Regla de Contrato / Expectativa", "Resultado Medido", "Estado", "Impacto en Ingeniería"),
    }[k]

    esperado_parts = esperado_ufs * 12
    cov_ok = res_dq["particoes_uf_mes"] >= esperado_parts and expected_positive(res_dq["particoes_uf_mes"])
    ia_ok = res_dq["amostra_n"] > 0 and res_dq["null_ia_features"] == 0
    viol = res_dq["bad_diarias"] + res_dq["bad_idade"] + res_dq["bad_cid"]
    dom_ok = res_dq["amostra_n"] > 0 and viol == 0
    dem_ok = res_dq["pct_raca_nd"] < 1.0 and res_dq["pct_cid_ign"] < 1.0
    uti_ok = res_dq["uti_gt_diarias"] == 0
    aih_ok = res_dq["aihs_duplicadas"] == 0

    lbl_ok = {"pt": "APROVADO", "en": "PASS", "es": "APROBADO"}[k]
    lbl_warn = {"pt": "ATENÇÃO", "en": "WATCH", "es": "ATENCIÓN"}[k]
    lbl_alert = {"pt": "ANOMALIA SIH", "en": "SIH ANOMALY", "es": "ANOMALÍA SIH"}[k]
    lbl_info = {"pt": "INFORMATIVO", "en": "INFO", "es": "INFORMATIVO"}[k]

    regras = [
        (
            {"pt": "Cobertura", "en": "Coverage", "es": "Cobertura"}[k],
            {"pt": "Todas as UFs × 12 meses ativos no ano (sem lacunas de envio FTP)", "en": "All States × 12 active months present (no FTP gaps)", "es": "Todos los Estados × 12 meses activos (sin brechas FTP)"}[k],
            f"{res_dq['particoes_uf_mes']} / {esperado_parts} ({res_dq['ufs_cobertas']}/{esperado_ufs} UFs)",
            lbl_ok if cov_ok else lbl_alert,
            "dq_ok" if cov_ok else "dq_alert",
            {"pt": "Garante equilíbrio nacional na Aba 1 e evita viés regional.", "en": "Ensures balanced national denominator in Tab 1.", "es": "Garantiza equilibrio nacional sin sesgo regional."}[k],
        ),
        (
            {"pt": "Completude (IA)", "en": "Completeness (AI)", "es": "Completitud (IA)"}[k],
            {"pt": "Zero nulos nas 8 variáveis de entrada do modelo preditivo", "en": "Zero missing values across the 8 intake AI features", "es": "Cero nulos en las 8 variables de entrada de la IA"}[k],
            f"{_fmt_int_lang(res_dq['null_ia_features'], k)} / {_fmt_int_lang(res_dq['amostra_n'], k)}",
            lbl_ok if ia_ok else lbl_alert,
            "dq_ok" if ia_ok else "dq_alert",
            {"pt": "Mantém o LightGBM V4 operante na nuvem sem imputação frágil.", "en": "Keeps LightGBM V4 reliable in production without fragile imputation.", "es": "Mantiene LightGBM V4 estable en producción sin imputación frágil."}[k],
        ),
        (
            {"pt": "Validade Clínica", "en": "Clinical Validity", "es": "Validez Clínica"}[k],
            {"pt": "qt_diarias > 0, idade_anos em [0, 120] e diag_princ não vazio", "en": "qt_diarias > 0, idade_anos in [0, 120], non-empty diag_princ", "es": "qt_diarias > 0, idade_anos en [0, 120] y diag_princ válido"}[k],
            f"{_fmt_int_lang(viol, k)} ({res_dq['min_idade']:.0f}–{res_dq['max_idade']:.0f}, max {res_dq['max_diarias']}d)",
            lbl_ok if dom_ok else lbl_alert,
            "dq_ok" if dom_ok else "dq_alert",
            {"pt": "ETL filtra diárias inválidas/negativas antes de gerar o alvo.", "en": "ETL purges invalid stays before generating clinical targets.", "es": "ETL filtra estancias inválidas antes de generar los objetivos."}[k],
        ),
        (
            {"pt": "Completude Demog.", "en": "Demog. Completeness", "es": "Completitud Demog."}[k],
            {"pt": "Subnotificação de raça/cor (Nao_Declarada) e CID Ignorado < 1%", "en": "Not-declared race and unknown ICD share < 1%", "es": "Raza no declarada y CIE ignorado < 1%"}[k],
            f"{_fmt_pct_lang(res_dq['pct_raca_nd'], 2, k)} / {_fmt_pct_lang(res_dq['pct_cid_ign'], 3, k)}",
            lbl_ok if dem_ok else lbl_warn,
            "dq_ok" if dem_ok else "dq_warn",
            {"pt": "Monitora falhas de preenchimento por estado para equidade (Aba 1).", "en": "Tracks hospital reporting gaps by state for Tab 1 equity audits.", "es": "Monitorea subregistro por estado para auditoría de equidad."}[k],
        ),
        (
            {"pt": "Consistência", "en": "Consistency", "es": "Consistencia"}[k],
            {"pt": "Diárias de UTI faturadas <= Diárias totais (uti_mes_to <= qt_diarias)", "en": "Billed ICU days <= Total hospital days (uti_mes_to <= qt_diarias)", "es": "Días UCI facturados <= Días totales (uti_mes_to <= qt_diarias)"}[k],
            f"{_fmt_int_lang(res_dq['uti_gt_diarias'], k)} ({_fmt_pct_lang(res_dq['pct_uti_gt_diarias'], 2, k)})",
            lbl_ok if uti_ok else lbl_alert,
            "dq_ok" if uti_ok else "dq_alert",
            {"pt": "Anomalia real de faturamento: reforça bloqueio de campos UTI pós-alta na IA.", "en": "Real SIH billing quirk: justifies blocking ICU fields from intake AI.", "es": "Anomalía real de facturación: justifica excluir UCI post-alta de la IA."}[k],
        ),
        (
            {"pt": "Unicidade", "en": "Uniqueness", "es": "Unicidad"}[k],
            {"pt": "Chave n_aih na amostra/partição (AIHs de longa perm. podem repetir mês a mês)", "en": "Unique n_aih key (long-stay AIHs can repeat across billing months)", "es": "Clave n_aih (las AIH de larga estancia pueden repetirse por mes)"}[k],
            f"{_fmt_int_lang(res_dq['aihs_duplicadas'], k)} " + {"pt": "repetidas", "en": "repeated", "es": "repetidas"}[k],
            lbl_ok if aih_ok else lbl_info,
            "dq_ok" if aih_ok else "dq_info",
            {"pt": "Esperado no SIH/SUS (AIH tipo 5 / continuação); útil ao consolidar novos anos.", "en": "Expected in SIH/SUS continuation billing; key check for multi-year merges.", "es": "Esperado en SIH/SUS por prórrogas; clave al integrar nuevos años."}[k],
        ),
    ]

    rows = [
        {cols_map[0]: dim, cols_map[1]: regra, cols_map[2]: medido, cols_map[3]: status_lbl, cols_map[4]: impacto}
        for dim, regra, medido, status_lbl, _, impacto in regras
    ]
    status_keys = [r[4] for r in regras]
    return pd.DataFrame(rows), status_keys


def expected_positive(v: int) -> bool:
    return v > 0


def get_glossario_df(lang: str) -> pd.DataFrame:
    if lang == "en":
        return pd.DataFrame([
            {"Term": "SIH/SUS", "Domain": "Public Health", "Plain Explanation": "Brazil's Unified Health System Hospital Information System. Official public Ministry of Health database covering all publicly funded hospital admissions."},
            {"Term": "LightGBM", "Domain": "Machine Learning", "Plain Explanation": "Fast gradient-boosted decision tree algorithm. Selected for production due to top accuracy, strong recall, and millisecond scoring latency."},
            {"Term": "ROC-AUC", "Domain": "AI Metric", "Plain Explanation": "Ranges from 0.5 (random coin flip) to 1.0 (perfect). Measures how well the model assigns higher risk scores to severe cases than stable ones."},
            {"Term": "PR-AUC", "Domain": "AI Metric", "Plain Explanation": "Precision-Recall area under curve focused on rare events (mortality and long stays), evaluating detection quality without excessive false alarms."},
            {"Term": "Brier Score", "Domain": "Calibration", "Plain Explanation": "Measures probability calibration error (closer to 0 is better). Improved from ~0.13–0.16 (V3) down to 0.033–0.048 (V4)."},
            {"Term": "Isotonic Calibration", "Domain": "Machine Learning", "Plain Explanation": "Non-parametric calibration fitted on a dedicated holdout set ensuring that a 20% prediction corresponds to ~20 real cases out of 100."},
            {"Term": "Youden Index", "Domain": "Clinical Statistics", "Plain Explanation": "Optimal threshold maximizing Sensitivity + Specificity - 1, used as the baseline Yellow/Orange alert cutoff (4.0% and 6.4%)."},
            {"Term": "Sensitivity (Recall)", "Domain": "Epidemiology / AI", "Plain Explanation": "Share of true severe cases (deaths or 15+ day stays) successfully flagged right at admission (87.6% and 83.9%)."},
            {"Term": "Specificity", "Domain": "Epidemiology / AI", "Plain Explanation": "Share of standard non-severe hospitalizations accurately kept in the green zone without unnecessary alarms."},
            {"Term": "Data Leakage", "Domain": "Data Engineering", "Plain Explanation": "Using future post-admission information (such as final stay length or billing extension codes) to predict intake risk. Eliminated 100%."},
            {"Term": "DuckDB + Parquet", "Domain": "Data Engineering", "Plain Explanation": "Embedded analytical SQL engine and compressed columnar file format enabling sub-second queries over 13.58M rows without heavy servers."},
        ])
    if lang == "es":
        return pd.DataFrame([
            {"Término": "SIH/SUS", "Área": "Salud Pública", "Explicación Clara": "Sistema de Información Hospitalaria del SUS. Base pública oficial del Ministerio de Salud de Brasil con todos los ingresos hospitalarios financiados por el sistema público."},
            {"Término": "LightGBM", "Área": "Machine Learning", "Explicación Clara": "Algoritmo moderno basado en árboles de decisión rápidos. Elegido para producción por su alta precisión, gran detección y respuesta en milisegundos."},
            {"Término": "ROC-AUC", "Área": "Métrica de IA", "Explicación Clara": "Va de 0,5 (azar) a 1,0 (perfecto). Mide la capacidad del modelo para asignar mayor riesgo a los casos graves que a los estables."},
            {"Término": "PR-AUC", "Área": "Métrica de IA", "Explicación Clara": "Precisión enfocada en eventos poco frecuentes (mortalidad y estancias largas), evaluando aciertos sin exceso de falsas alarmas."},
            {"Término": "Brier Score", "Área": "Calibración", "Explicación Clara": "Mide el error entre la probabilidad prevista y la realidad (más cerca de 0 es mejor). Bajó de ~0,13–0,16 (V3) a 0,033–0,048 (V4)."},
            {"Término": "Calibración Isotónica", "Área": "Machine Learning", "Explicación Clara": "Técnica ajustada en un conjunto independiente que garantiza que una predicción del 20% corresponda a ~20 casos reales de cada 100 pacientes."},
            {"Término": "Índice de Youden", "Área": "Estadística Clínica", "Explicación Clara": "Punto matemático óptimo que equilibra Sensibilidad y Especificidad — usado como corte inicial de alerta amarilla/naranja (4,0% y 6,4%)."},
            {"Término": "Sensibilidad (Detección)", "Área": "Epidemiología / IA", "Explicación Clara": "Porcentaje de los casos graves reales (óbitos o estancias ≥15d) que el sistema logra detectar al ingreso (87,6% y 83,9%)."},
            {"Término": "Especificidad", "Área": "Epidemiología / IA", "Explicación Clara": "Porcentaje de ingresos normales que el sistema mantiene correctamente en zona verde sin alarmas innecesarias."},
            {"Término": "Data Leakage", "Área": "Ingeniería de Datos", "Explicación Clara": "Fuga de información del futuro (ej.: usar facturación o días finales para predecir el riesgo de entrada). Eliminado al 100% en este proyecto."},
            {"Término": "DuckDB + Parquet", "Área": "Ingeniería de Datos", "Explicación Clara": "Motor analítico SQL embebido y formato columnar comprimido que permiten consultar 13,58 millones de filas en segundos sin servidores pesados."},
        ])
    return pd.DataFrame([
        {"Termo": "SIH/SUS", "Área": "Saúde Pública", "Explicação Clara": "Sistema de Informações Hospitalares do SUS. Base oficial pública do Ministério da Saúde com todas as internações financiadas pelo SUS no Brasil."},
        {"Termo": "LightGBM", "Área": "Machine Learning", "Explicação Clara": "Algoritmo moderno baseado em árvores de decisão rápidas. Escolhido para produção pela alta precisão, ótima detecção e resposta em milissegundos."},
        {"Termo": "ROC-AUC", "Área": "Métrica de IA", "Explicação Clara": "Vai de 0,5 (chute aleatório) a 1,0 (perfeito). Mede a capacidade do modelo de dar um escore maior a quem realmente vai agravar do que a quem está estável."},
        {"Termo": "PR-AUC", "Área": "Métrica de IA", "Explicação Clara": "Precisão focada em eventos raros (como óbito e longa permanência), onde acertar o grupo minoritário sem gerar excesso de falsos alarmes é desafiador."},
        {"Termo": "Brier Score", "Área": "Calibração", "Explicação Clara": "Mede o erro entre a probabilidade prevista e a realidade (quanto mais perto de 0, melhor). Caiu de ~0,13–0,16 (V3) para 0,033–0,048 (V4)."},
        {"Termo": "Calibração Isotônica", "Área": "Machine Learning", "Explicação Clara": "Técnica aplicada em dados separados que garante que uma previsão de 20% realmente corresponda a ~20 casos reais a cada 100 pacientes."},
        {"Termo": "Limiar de Youden", "Área": "Estatística Clínica", "Explicação Clara": "Ponto matemático ideal que equilibra Sensibilidade e Especificidade — usado como linha de corte inicial para o alerta amarelo/laranja (4,0% e 6,4%)."},
        {"Termo": "Sensibilidade (Detecção)", "Área": "Epidemiologia / IA", "Explicação Clara": "De todos os pacientes que realmente morreram ou ficaram 15+ dias, quantos por cento o sistema conseguiu sinalizar na entrada (87,6% e 83,9%)."},
        {"Termo": "Especificidade", "Área": "Epidemiologia / IA", "Explicação Clara": "De todos os pacientes que tiveram alta normal, quantos por cento o sistema corretamente manteve no sinal verde."},
        {"Termo": "Data Leakage", "Área": "Engenharia de Dados", "Explicação Clara": "Vazamento de informação do futuro (ex.: usar cobrança ou número de dias finais para prever o risco de entrada). Eliminado 100% neste projeto."},
        {"Termo": "DuckDB + Parquet", "Área": "Engenharia de Dados", "Explicação Clara": "Banco analítico embutido e formato de armazenamento colunar compactado que permitem consultar 13,58 milhões de registros em segundos sem servidor pesado."},
    ])
