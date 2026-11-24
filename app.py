# -*- coding: utf-8 -*-
"""
🏥 SUS Analytics & Clinical Intelligence Platform
Aplicação Streamlit Multi-Abas Conectada ao Data Lake Parquet + DuckDB:
- Aba 1: Inteligência Epidemiológica & Equidade em Saúde (Multi-UF / Nacional)
- Aba 2: Inteligência de Dados, Modelos Preditivos & MLOps
- Aba 3: Cockpit de Gestão Clínica & Painel de Leitos (com Flags Preditivas)
"""

import os
import sys
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.join(BASE_DIR, "scripts") not in sys.path:
    sys.path.insert(0, os.path.join(BASE_DIR, "scripts"))

import duckdb_service as ds

# ==========================================
# 1. Configuração da Página e Estilo Visual
# ==========================================
st.set_page_config(
    page_title="SUS Analytics & Clinical Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .main-title {
        font-size: 2.15rem;
        font-weight: 800;
        color: #1e3a8a;
        letter-spacing: -0.02em;
        margin-bottom: 0.15rem;
    }
    .sub-title {
        font-size: 1.02rem;
        color: #475569;
        margin-bottom: 1.4rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-top: 4px solid #1e3a8a;
        border-radius: 10px;
        padding: 15px 12px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(15, 23, 42, 0.04);
    }
    .metric-card.red { border-top-color: #dc2626; }
    .metric-card.amber { border-top-color: #d97706; }
    .metric-card.green { border-top-color: #059669; }
    .metric-card h3 {
        margin: 0;
        font-size: 0.76rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 700;
    }
    .metric-card p {
        margin: 6px 0 0 0;
        font-size: 1.6rem;
        font-weight: 800;
        color: #0f172a;
    }
    .metric-card small {
        display: block;
        margin-top: 2px;
        font-size: 0.72rem;
        color: #64748b;
    }
    div[data-baseweb="tab-list"] {
        gap: 8px;
    }
    button[data-baseweb="tab"] {
        font-weight: 600;
        padding: 10px 16px;
        border-radius: 8px 8px 0 0;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Carregamento via DuckDB (Cacheado) e Modelos
# ==========================================
@st.cache_resource
def carregar_modelos():
    caminho_longa = os.path.join(BASE_DIR, "modelos", "modelo_admissao_longa_lightgbm_v4.joblib")
    caminho_obito = os.path.join(BASE_DIR, "modelos", "modelo_admissao_obito_lightgbm_v4.joblib")
    modelo_longa = joblib.load(caminho_longa) if os.path.exists(caminho_longa) else None
    modelo_obito = joblib.load(caminho_obito) if os.path.exists(caminho_obito) else None
    return modelo_longa, modelo_obito


@st.cache_data(ttl=300)
def carregar_metricas_v4():
    caminho_csv = os.path.join(BASE_DIR, "modelos", "metricas_v4_nacional.csv")
    if os.path.exists(caminho_csv):
        return pd.read_csv(caminho_csv)
    return pd.DataFrame()


@st.cache_data(ttl=300)
def get_filtros():
    return ds.listar_filtros_disponiveis()


@st.cache_data(ttl=300)
def get_kpis(regiao, uf):
    return ds.obter_kpis_globais(regiao=regiao, uf=uf)


@st.cache_data(ttl=300)
def get_contagem(regiao, uf, raca, espec, sexo, mes_range):
    return ds.contar_filtrado(regiao, uf, raca, espec, sexo, mes_range)


@st.cache_data(ttl=300)
def get_faixa_sexo(regiao, uf, raca, espec, sexo, mes_range):
    return ds.obter_distribuicao_faixa_sexo(regiao, uf, raca, espec, sexo, mes_range)


@st.cache_data(ttl=300)
def get_raca(regiao, uf, espec, sexo, mes_range):
    return ds.obter_equidade_raca(regiao, uf, espec, sexo, mes_range)


@st.cache_data(ttl=300)
def get_sazonalidade(regiao, uf, raca, espec, sexo):
    return ds.obter_sazonalidade_mensal(regiao, uf, raca, espec, sexo)


@st.cache_data(ttl=300)
def get_cid(regiao, uf, raca, espec, sexo, mes_range):
    return ds.obter_top_capitulos_cid(regiao, uf, raca, espec, sexo, mes_range)


@st.cache_data(ttl=300)
def get_resumo_ufs(regiao):
    return ds.obter_resumo_ufs(regiao=regiao)


@st.cache_data(ttl=300)
def get_amostra_cockpit(uf, espec, min_dias, busca):
    return ds.obter_amostra_cockpit(uf=uf, espec=espec, min_dias=min_dias, busca=busca, limite=300)


if not ds.parquet_disponivel():
    st.error("Nenhuma partição Parquet encontrada em `dados/parquet_nacional/`. Execute `scripts/pipeline_nacional_sih.py` primeiro.")
    st.stop()

modelo_longa, modelo_obito = carregar_modelos()
filtros = get_filtros()
kpis_gerais = get_kpis("Todas", "Todas")

# ==========================================
# Sidebar Executiva (Portfólio & Guia Rápido)
# ==========================================
with st.sidebar:
    st.markdown("## 🏥 SUS Analytics 2024")
    st.caption("Projeto de Portfólio em Ciência de Dados & Saúde Pública")
    st.markdown("---")
    st.markdown("**📌 Guia Rápido de Leitura:**")
    st.markdown("""
    - **📊 Aba 1 (Epidemiologia):** Para gestores públicos e saúde coletiva — quem interna, onde e por quê.
    - **🤖 Aba 2 (Modelos IA):** Para recrutadores técnicos — métricas, calibração e escolha do LightGBM.
    - **🏥 Aba 3 (Gestão Clínica):** Para médicos e enfermeiros — painel de leitos com alertas ajustáveis.
    - **📖 Aba 4 (Dicionário):** Para todos os públicos — significado de cada variável e termo técnico.
    """)
    st.markdown("---")
    st.markdown(f"**⚙️ Motor Analítico:**\n`{ds.modo_execucao()}`")
    st.markdown("**🧠 Produção:**\n`LightGBM V4 (Calibração Isotônica)`")
    st.markdown("---")
    st.info("⚖️ **IA Responsável:** Raça/cor e mês são mantidos na epidemiologia para equidade, mas excluídos dos modelos clínicos preditivos.")

# ==========================================
# 3. Header Principal e KPIs
# ==========================================
st.markdown('<div class="main-title">🏥 SUS Analytics & Clinical Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Plataforma de Inteligência em Saúde Pública (SIH/SUS Brasil 2024): perfil epidemiológico, previsão de riscos na admissão hospitalar e painel clínico de leitos.</div>', unsafe_allow_html=True)

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    st.markdown(f'<div class="metric-card"><h3>Internações Analisadas</h3><p>{kpis_gerais["total"]:,}</p><small>Microdados SIH/SUS 2024</small></div>'.replace(",", "."), unsafe_allow_html=True)
with kpi2:
    st.markdown(f'<div class="metric-card"><h3>Estados Cobertos</h3><p style="color:#1e3a8a;">{kpis_gerais["total_ufs"]} / 27 UFs</p><small>Cobertura Nacional</small></div>', unsafe_allow_html=True)
with kpi3:
    st.markdown(f'<div class="metric-card red"><h3>Média de Óbitos</h3><p style="color:#dc2626;">{kpis_gerais["taxa_obito"]:.2f}%</p><small>Mortalidade Hospitalar</small></div>', unsafe_allow_html=True)
with kpi4:
    st.markdown(f'<div class="metric-card amber"><h3>Internações ≥ 15 dias</h3><p style="color:#d97706;">{kpis_gerais["taxa_longa"]:.2f}%</p><small>Longa Permanência</small></div>', unsafe_allow_html=True)
with kpi5:
    st.markdown('<div class="metric-card green"><h3>Poder Preditivo (ROC-AUC)</h3><p style="color:#059669;">0,879 / 0,816</p><small>Óbito / Longa Perm. (V4)</small></div>', unsafe_allow_html=True)

st.write("")

# ==========================================
# 4. Navegação por Abas
# ==========================================
tab_epi, tab_ml, tab_gestao, tab_dic = st.tabs([
    "📊 1. Perfil Epidemiológico & Equidade",
    "🤖 2. Como a IA Funciona (Modelos & Métricas)",
    "🏥 3. Painel de Gestão Clínica & Leitos",
    "📖 4. Dicionário de Dados & Glossário"
])

# ==============================================================================
# ABA 1: INTELIGÊNCIA EPIDEMIOLÓGICA & EQUIDADE
# ==============================================================================
with tab_epi:
    st.markdown("### 📊 Panorama Epidemiológico das Internações Hospitalares")
    st.markdown("Explore quem utiliza os hospitais do SUS no Brasil em 2024, as principais causas de internação e diferenças entre regiões e perfis demográficos.")

    col_f1, col_f2, col_f3, col_f4, col_f5, col_f6 = st.columns(6)
    with col_f1:
        reg_sel = st.selectbox("Região:", ["Todas"] + filtros["regioes"], index=0)
    with col_f2:
        uf_sel = st.selectbox("Estado (UF):", ["Todas"] + filtros["ufs"], index=0)
    with col_f3:
        raca_sel = st.selectbox("Raça / Cor:", ["Todas"] + filtros["racas"], index=0)
    with col_f4:
        espec_sel = st.selectbox("Especialidade:", ["Todas"] + filtros["especialidades"], index=0)
    with col_f5:
        sexo_sel = st.selectbox("Sexo:", ["Todos", "Masculino", "Feminino"], index=0)
    with col_f6:
        mes_sel = st.slider("Mês (2024):", 1, 12, (1, 12))

    total_filtrado = get_contagem(reg_sel, uf_sel, raca_sel, espec_sel, sexo_sel, mes_sel)
    pct_total = (total_filtrado / kpis_gerais["total"] * 100) if kpis_gerais["total"] > 0 else 0
    st.info(f"🔎 Exibindo **{total_filtrado:,}** internações para os filtros selecionados ({pct_total:.1f}% do total nacional). *(Consultas processadas diretamente via motor DuckDB)*".replace(",", "."))

    # Linha 1 de Gráficos: Faixa Etária e Raça/Cor
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("#### Distribuição por Faixa Etária e Sexo")
        st.caption("Mostra o volume de internações em cada idade, separando homens e mulheres.")
        faixa_df = get_faixa_sexo(reg_sel, uf_sel, raca_sel, espec_sel, sexo_sel, mes_sel)
        fig_faixa = px.bar(
            faixa_df, x="faixa_etaria", y="contagem", color="sexo_desc",
            barmode="group", labels={"faixa_etaria": "Faixa Etária", "contagem": "Internações", "sexo_desc": "Sexo"},
            color_discrete_map={"Masculino": "#2563eb", "Feminino": "#ec4899"},
            template="plotly_white"
        )
        fig_faixa.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_faixa, use_container_width=True)

    with g2:
        st.markdown("#### Equidade: Mortalidade e Permanência por Raça/Cor")
        st.caption("Compara o tempo médio no hospital e o risco de óbito entre grupos raciais (tamanho da bolha = volume).")
        raca_metrics = get_raca(reg_sel, uf_sel, espec_sel, sexo_sel, mes_sel)
        fig_raca = px.scatter(
            raca_metrics, x="taxa_obito", y="media_diarias", size="total", color="raca_cor_desc",
            text="raca_cor_desc", labels={"taxa_obito": "Taxa de Óbito (%)", "media_diarias": "Média de Diárias"},
            template="plotly_white", size_max=40
        )
        fig_raca.update_traces(textposition="top center")
        fig_raca.update_layout(height=350, showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_raca, use_container_width=True)

    # Linha 2 de Gráficos: Sazonalidade Mensal e Capítulos CID-10
    g3, g4 = st.columns(2)
    with g3:
        st.markdown("#### Sazonalidade Mensal de Internações e Óbitos")
        st.caption("Acompanha como o volume de entradas (barras) e a mortalidade (linha) variam ao longo dos 12 meses.")
        mes_agg = get_sazonalidade(reg_sel, uf_sel, raca_sel, espec_sel, sexo_sel)
        fig_mes = go.Figure()
        fig_mes.add_trace(go.Bar(x=mes_agg["mes_internacao"], y=mes_agg["internacoes"], name="Internações", marker_color="#93c5fd"))
        fig_mes.add_trace(go.Scatter(x=mes_agg["mes_internacao"], y=mes_agg["taxa_mortalidade"], name="Taxa Óbito (%)", yaxis="y2", line=dict(color="#dc2626", width=3)))
        fig_mes.update_layout(
            height=350, template="plotly_white",
            yaxis=dict(title="Total de Internações"),
            yaxis2=dict(title="Mortalidade (%)", overlaying="y", side="right"),
            xaxis=dict(title="Mês de 2024", tickmode="linear", dtick=1),
            legend=dict(x=0.01, y=0.99),
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_mes, use_container_width=True)

    with g4:
        st.markdown("#### Top Causas por Capítulo da CID-10")
        st.caption("Principais motivos médicos das internações; passe o mouse para ver a taxa exata de óbito e longa permanência.")
        cid_df = get_cid(reg_sel, uf_sel, raca_sel, espec_sel, sexo_sel, mes_sel)
        fig_cid = px.bar(
            cid_df, y="capitulo_cid", x="total", orientation="h",
            color="taxa_obito", color_continuous_scale="Reds",
            custom_data=["taxa_obito", "taxa_longa"],
            labels={"capitulo_cid": "Capítulo CID-10", "total": "Internações", "taxa_obito": "Óbito (%)"},
            template="plotly_white"
        )
        fig_cid.update_traces(
            hovertemplate="<b>%{y}</b><br>Internações: %{x:,}<br>Taxa de Óbito: <b>%{customdata[0]:.2f}%</b><br>Longa Permanência: %{customdata[1]:.2f}%<extra></extra>"
        )
        fig_cid.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_cid, use_container_width=True)

    # Resumo comparativo entre UFs carregadas
    if len(filtros["ufs"]) > 1:
        st.markdown("#### 🗺️ Comparativo por Estado (UFs Carregadas)")
        df_ufs = get_resumo_ufs(reg_sel)
        st.dataframe(
            df_ufs,
            use_container_width=True,
            hide_index=True,
            column_config={
                "uf": "UF",
                "regiao": "Região",
                "internacoes": st.column_config.NumberColumn("Internações", format="%d"),
                "taxa_obito_pct": st.column_config.NumberColumn("Taxa Óbito (%)", format="%.2f%%"),
                "taxa_longa_pct": st.column_config.NumberColumn("Longa Perm. (%)", format="%.2f%%"),
                "media_diarias": st.column_config.NumberColumn("Média Diárias", format="%.2f d")
            }
        )

# ==============================================================================
# ABA 2: MODELAGEM PREDITIVA & MLOPS
# ==============================================================================
with tab_ml:
    st.markdown("### 🤖 Como a Inteligência Artificial Funciona & Escolha do Modelo (V4 Nacional)")
    st.markdown("""
    No momento em que o paciente entra no hospital, dois modelos preditivos operam juntos usando apenas **8 informações da admissão** (idade, sexo, motivo CID-10, especialidade do leito, caráter urgência/eletivo, complexidade, se veio transferido de outro município e se entrou no fim de semana):
    1. **Modelo de Longa Permanência (≥ 15 dias):** estima a chance de o paciente precisar de internação prolongada, apoiando o giro de leitos e o serviço social.
    2. **Modelo de Óbito Intra-hospitalar:** estima a gravidade de risco fatal para priorizar monitoramento intensivo ou UTI.
    """)

    col_gov1, col_gov2 = st.columns(2)
    with col_gov1:
        st.success("""
        🏆 **Por que escolhemos o LightGBM como modelo oficial?**
        - **Empate técnico no topo com o XGBoost:** Alcançou excelente poder de separação (**ROC-AUC de `0,8793`** para óbito e **`0,8160`** para longa permanência) com a melhor calibração empírica (**Brier Score de `0,0329` e `0,0480`**).
        - **Maior sensibilidade em longa permanência:** Detecta **`83,92%`** dos casos de internação prolongada e **`87,58%`** dos casos de alto risco de óbito.
        - **Agilidade em produção:** Responde em milissegundos consumindo pouca memória RAM — ideal para triagem em tempo real e publicação online, sendo mais rápido e leve que *Random Forest* e muito mais preciso que *Regressão Logística*.
        """)
    with col_gov2:
        st.info("""
        📖 **Mini-Guia das Métricas (Para todos os públicos):**
        - **ROC-AUC (0 a 1):** Mede o poder de distinção entre pacientes graves e estáveis (`0,5` = chute aleatório; acima de `0,80` = excelente).
        - **PR-AUC:** Precisão em eventos raros (óbito e longa permanência são minoria dos casos, exigindo avaliação rigorosa).
        - **Brier Score (quanto menor, melhor):** Mede se a probabilidade prevista reflete a realidade graças à **Calibração Isotônica** (ex.: se diz 20%, ~20 em 100 pacientes com aquele perfil realmente têm o desfecho).
        - **Ética & Sem *Data Leakage*:** `raca_cor` e `mes_internacao` não entram no modelo para evitar preconceito algorítmico; dados pós-alta também foram removidos para não "trapacear" na previsão.
        """)

    st.markdown("#### 📊 Comparativo Nacional (Teste Cego em `300.000` internações das 27 UFs)")
    df_m_v4 = carregar_metricas_v4()
    if not df_m_v4.empty:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("##### 1. Óbito Intra-hospitalar *(Média Nacional: 3,85%)*")
            st.dataframe(
                df_m_v4[df_m_v4["Desfecho"] == "Óbito Intra-hospitalar"][
                    ["Modelo", "ROC_AUC", "PR_AUC", "Brier_Calibrado", "Limiar_Otimo_Pct", "Sensibilidade_Pct", "Especificidade_Pct"]
                ].rename(columns={
                    "Brier_Calibrado": "Brier Score", "Limiar_Otimo_Pct": "Corte Alerta (%)",
                    "Sensibilidade_Pct": "Detecção (%)", "Especificidade_Pct": "Especif. (%)"
                }),
                use_container_width=True, hide_index=True
            )
        with col_m2:
            st.markdown("##### 2. Longa Permanência ≥ 15d *(Média Nacional: 6,17%)*")
            st.dataframe(
                df_m_v4[df_m_v4["Desfecho"] == "Longa Permanência"][
                    ["Modelo", "ROC_AUC", "PR_AUC", "Brier_Calibrado", "Limiar_Otimo_Pct", "Sensibilidade_Pct", "Especificidade_Pct"]
                ].rename(columns={
                    "Brier_Calibrado": "Brier Score", "Limiar_Otimo_Pct": "Corte Alerta (%)",
                    "Sensibilidade_Pct": "Detecção (%)", "Especificidade_Pct": "Especif. (%)"
                }),
                use_container_width=True, hide_index=True
            )

    st.markdown("---")
    st.markdown("#### 📈 Evidências Visuais e Explicabilidade (V4 Nacional)")
    col_img1, col_img2, col_img3 = st.columns(3)
    with col_img1:
        img_roc = os.path.join(BASE_DIR, "figuras", "fig_roc_pr_curvas_v4.png")
        if os.path.exists(img_roc):
            st.image(img_roc, caption="Curvas ROC e PR: quanto mais perto do topo/canto, maior a precisão do modelo.", use_container_width=True)
    with col_img2:
        img_cal = os.path.join(BASE_DIR, "figuras", "fig_calibracao_probabilidade_v4.png")
        if os.path.exists(img_cal):
            st.image(img_cal, caption="Calibração Isotônica: curvas coladas na linha pontilhada provam que o % previsto é realista.", use_container_width=True)
    with col_img3:
        img_imp = os.path.join(BASE_DIR, "figuras", "fig_importancias_admissao_v4.png")
        if os.path.exists(img_imp):
            st.image(img_imp, caption="Fatores que mais pesam na decisão da IA (ex.: Idade, Clínica Médica e Urgência).", use_container_width=True)

# ==============================================================================
# ABA 3: COCKPIT DE GESTÃO CLÍNICA & LEITOS
# ==============================================================================
with tab_gestao:
    st.markdown("### 🏥 Painel de Gestão Clínica e Risco de Pacientes")
    st.markdown("Simulação de um painel hospitalar real de acompanhamento de leitos: mostra o identificador anônimo da internação (`#AIH`), diagnóstico (`CID-10`), tempo já internado e duas **Flags Preditivas** calculadas na admissão.")

    with st.expander("⚙️ **Calibrar Cortes das Flags & Entender as Porcentagens**", expanded=True):
        st.caption(
            "💡 **Por que os cortes padrão não são 50%?** No Brasil (2024), a média real de óbito hospitalar é de **3,85%** e a de longa permanência (≥15 dias) é de **6,17%**. "
            "Se o hospital esperasse chegar a 50% de risco na porta de entrada, quase todos os casos graves passariam sem alerta. "
            "Por isso, o corte **Amarelo/Laranja** nasce do **Limiar de Youden** (`4,0%` para Óbito e `6,4%` para Longa Permanência — ponto estatístico que maximiza a detecção precoce) "
            "e o corte **Vermelho** marca perfis com **3x a 4x mais risco que a média nacional** (`15,0%` e `20,0%`). Ajuste abaixo conforme a lotação do seu hospital:"
        )
        sl_col1, sl_col2 = st.columns(2)
        with sl_col1:
            corte_obito = st.slider(
                "⚠️ Cortes Risco de Óbito — Atenção / Crítico (%):",
                min_value=1.0, max_value=50.0, value=(4.0, 15.0), step=0.5, format="%.1f%%"
            )
        with sl_col2:
            corte_longa = st.slider(
                "🚩 Cortes Longa Permanência — Médio / Alto Risco (%):",
                min_value=1.0, max_value=50.0, value=(6.4, 20.0), step=0.5, format="%.1f%%"
            )

    ob_med, ob_crit = corte_obito[0] / 100.0, corte_obito[1] / 100.0
    lg_med, lg_crit = corte_longa[0] / 100.0, corte_longa[1] / 100.0

    st.markdown(f"""
    - **🚩 Longa Permanência (≥ 15 dias):** `🟢 Baixo < {corte_longa[0]:.1f}%` | `🟡 Médio {corte_longa[0]:.1f}%–{max(corte_longa[1]-0.1, corte_longa[0]):.1f}%` | `🔴 Alto Risco ≥ {corte_longa[1]:.1f}%`
    - **⚠️ Risco de Óbito:** `🟢 Estável < {corte_obito[0]:.1f}%` | `🟠 Atenção {corte_obito[0]:.1f}%–{max(corte_obito[1]-0.1, corte_obito[0]):.1f}%` | `🔴 Crítico ≥ {corte_obito[1]:.1f}%`
    """)

    with st.expander("➕ **Simular Nova Admissão de Paciente (Scorecard em Tempo Real)**", expanded=False):
        c_sim1, c_sim2, c_sim3, c_sim4 = st.columns(4)
        with c_sim1:
            sim_idade = st.number_input("Idade (anos):", min_value=0.0, max_value=120.0, value=68.0, step=1.0)
            sim_sexo = st.selectbox("Sexo Biológico:", ["Masculino", "Feminino"], index=0)
        with c_sim2:
            sim_espec = st.selectbox("Especialidade do Leito:", ["Clinica_Medica", "Cirurgica", "Pediatrica", "Obstetrica", "Psiquiatrica"], index=0)
            sim_carater = st.selectbox("Caráter de Internação:", ["Urgencia_Emergencia", "Eletiva"], index=0)
        with c_sim3:
            capitulos_cid = [
                "X - Respiratorio", "IX - Circulatorio", "I - Infecciosas",
                "II - Neoplasias", "XI - Digestivo", "XIV - Geniturinario",
                "V - Mentais", "XV - Gravidez/Parto", "XIX - Lesoes"
            ]
            sim_cid = st.selectbox("Diagnóstico Principal (Capítulo CID-10):", capitulos_cid, index=0)
            sim_complex = st.selectbox("Complexidade:", ["Media_Complexidade", "Alta_Complexidade"], index=0)
        with c_sim4:
            sim_transf = st.radio("Paciente Transferido de Outro Município?", ["Não", "Sim"], index=1)
            sim_fds = st.radio("Admissão em Fim de Semana?", ["Não", "Sim"], index=0)

        if st.button("🚨 Calcular Score de Admissão"):
            input_dict = {
                "idade_anos": sim_idade,
                "sexo": "1" if sim_sexo == "Masculino" else "3",
                "capitulo_cid": sim_cid,
                "especialidade_leito": sim_espec,
                "carater_internacao": sim_carater,
                "complexidade": sim_complex,
                "transferido_intermun": "1" if sim_transf == "Sim" else "0",
                "internacao_fim_de_semana": "1" if sim_fds == "Sim" else "0"
            }
            df_in = pd.DataFrame([input_dict])
            p_longa_sim = float(modelo_longa.predict_proba(df_in)[0, 1]) if modelo_longa else 0.45
            p_obito_sim = float(modelo_obito.predict_proba(df_in)[0, 1]) if modelo_obito else 0.65

            st.markdown("#### Resultado da Triagem (Probabilidades Calibradas Brasil V4):")
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                badge_ob = "CRÍTICO" if p_obito_sim >= ob_crit else ("ATENÇÃO" if p_obito_sim >= ob_med else "ESTÁVEL")
                cor_ob = "#dc2626" if p_obito_sim >= ob_crit else ("#f59e0b" if p_obito_sim >= ob_med else "#16a34a")
                st.markdown(f"""
                <div style="background:{cor_ob}15; border-left: 5px solid {cor_ob}; padding: 12px; border-radius: 6px;">
                    <h4 style="margin:0; color:{cor_ob};">Risco de Óbito: {badge_ob}</h4>
                    <p style="margin:5px 0 0 0; font-size:1.6rem; font-weight:800; color:{cor_ob};">{p_obito_sim*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            with sc2:
                badge_lg = "ALTO RISCO" if p_longa_sim >= lg_crit else ("MÉDIO" if p_longa_sim >= lg_med else "BAIXO")
                cor_lg = "#dc2626" if p_longa_sim >= lg_crit else ("#f59e0b" if p_longa_sim >= lg_med else "#16a34a")
                st.markdown(f"""
                <div style="background:{cor_lg}15; border-left: 5px solid {cor_lg}; padding: 12px; border-radius: 6px;">
                    <h4 style="margin:0; color:{cor_lg};">Longa Permanência: {badge_lg}</h4>
                    <p style="margin:5px 0 0 0; font-size:1.6rem; font-weight:800; color:{cor_lg};">{p_longa_sim*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            with sc3:
                if p_longa_sim >= lg_crit and p_obito_sim >= ob_crit:
                    quadrante = "ZONA VERMELHA (CRÍTICA)"
                    acao = "🚨 Acionar UTI/Semi-Intensiva imediatamente. Suporte avançado de vida."
                elif p_obito_sim >= ob_crit:
                    quadrante = "ZONA DE EMERGÊNCIA AGUDA"
                    acao = "⚠️ Sala Vermelha: Monitoramento intensivo nas primeiras 48h."
                elif p_longa_sim >= lg_crit:
                    quadrante = "ZONA DE CRONICIDADE"
                    acao = "📋 Acionar Serviço Social e Fisioterapia para planejamento de alta precoce."
                else:
                    quadrante = "ZONA DE FLUXO RÁPIDO"
                    acao = "✅ Linha de cuidado convencional. Previsão de alta em até 48h."
                st.markdown(f"""
                <div style="background:#f1f5f9; border-left: 5px solid #0f172a; padding: 12px; border-radius: 6px;">
                    <h4 style="margin:0; color:#0f172a;">{quadrante}</h4>
                    <p style="margin:5px 0 0 0; font-size:0.95rem; font-weight:600; color:#334155;">{acao}</p>
                </div>
                """, unsafe_allow_html=True)

    st.write("")
    st.markdown("#### 📋 Painel de Pacientes Internados *(Amostra Dinâmica do Brasil)*")

    col_t1, col_t2, col_t3, col_t4, col_t5 = st.columns(5)
    with col_t1:
        filtro_uf = st.selectbox("Estado (UF):", ["Todas"] + filtros["ufs"], index=0, key="cockpit_uf")
    with col_t2:
        filtro_esp = st.selectbox("Especialidade:", ["Todos"] + filtros["especialidades"], index=0, key="cockpit_esp")
    with col_t3:
        filtro_risco = st.selectbox("Status de Gravidade:", ["Todos", "Crítico (Óbito Alto)", "Longa Permanência Elevada"], index=0)
    with col_t4:
        filtro_dias = st.slider("Dias Internado (Mínimo):", 0, 45, 0)
    with col_t5:
        busca_id = st.text_input("Buscar ID (AIH) ou CID:", "")

    amostra = get_amostra_cockpit(filtro_uf, filtro_esp, filtro_dias, busca_id)

    if amostra.empty:
        st.warning("Nenhum paciente encontrado para os filtros informados.")
    else:
        amostra = amostra.copy()
        df_mod = pd.DataFrame()
        df_mod["idade_anos"] = amostra["idade_anos"]
        for c in ["sexo", "capitulo_cid", "especialidade_leito", "carater_internacao", "complexidade", "transferido_intermun", "internacao_fim_de_semana"]:
            df_mod[c] = amostra[c].astype(str)

        if modelo_obito and modelo_longa:
            amostra["p_obito"] = modelo_obito.predict_proba(df_mod)[:, 1]
            amostra["p_longa"] = modelo_longa.predict_proba(df_mod)[:, 1]
        else:
            amostra["p_obito"] = 0.05
            amostra["p_longa"] = 0.10

        if filtro_risco == "Crítico (Óbito Alto)":
            amostra = amostra[amostra["p_obito"] >= ob_crit]
        elif filtro_risco == "Longa Permanência Elevada":
            amostra = amostra[amostra["p_longa"] >= lg_crit]

        amostra["id_anonimo"] = "#AIH-" + amostra["n_aih"].astype(str).str[-8:]
        amostra["sexo_desc"] = amostra["sexo"].astype(str).map({"1": "M", "3": "F"}).fillna("-")
        amostra["motivo_cid"] = amostra["diag_princ"].astype(str) + " • " + amostra["capitulo_cid"].astype(str)
        amostra["flag_obito"] = amostra["p_obito"].apply(
            lambda p: f"{'🔴 CRÍTICO' if p >= ob_crit else ('🟠 ATENÇÃO' if p >= ob_med else '🟢 ESTÁVEL')} ({p * 100:.1f}%)"
        )
        amostra["flag_longa"] = amostra["p_longa"].apply(
            lambda p: f"{'🔴 ALTO RISCO' if p >= lg_crit else ('🟡 MÉDIO' if p >= lg_med else '🟢 BAIXO')} ({p * 100:.1f}%)"
        )

        st.write(f"Exibindo **{len(amostra)}** pacientes amostrados:")

        colunas_visiveis = [
            "id_anonimo", "uf", "idade_anos", "sexo_desc", "especialidade_leito",
            "motivo_cid", "qt_diarias", "uti_mes_to", "flag_longa", "flag_obito"
        ]
        df_tabela = amostra[colunas_visiveis].rename(columns={
            "id_anonimo": "ID (AIH)",
            "uf": "UF",
            "idade_anos": "Idade",
            "sexo_desc": "Sexo",
            "especialidade_leito": "Especialidade",
            "motivo_cid": "Motivo (CID-10)",
            "qt_diarias": "Dias Internado",
            "uti_mes_to": "Diárias UTI",
            "flag_longa": "🚩 Longa Perm.",
            "flag_obito": "⚠️ Risco Óbito"
        })

        st.dataframe(
            df_tabela,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Idade": st.column_config.NumberColumn(format="%.0f anos"),
                "Dias Internado": st.column_config.NumberColumn(format="%d d"),
                "Diárias UTI": st.column_config.NumberColumn(format="%d d")
            }
        )

# ==============================================================================
# ABA 4: DICIONÁRIO DE DADOS & GLOSSÁRIO
# ==============================================================================
with tab_dic:
    st.markdown("### 📖 Dicionário de Dados & Glossário do Projeto")
    st.markdown("Guia de referência rápida traduzindo as colunas dos dados públicos do SUS e os termos de Ciência de Dados para todos os públicos.")

    busca_dic = st.text_input("🔍 Pesquisar variável ou termo técnico:", "", placeholder="Ex.: ROC-AUC, n_aih, Brier, transferido...")

    st.markdown("#### 🗂️ 1. Dicionário das Variáveis Hospitalares (SIH/SUS)")
    df_dicionario = pd.DataFrame([
        {"Variável": "n_aih", "Nome Amigável": "Número da AIH", "Significado": "Código oficial único da Autorização de Internação Hospitalar. Usado de forma anonimizada (#AIH-...) como identificador do paciente no painel.", "Uso no Projeto": "Identificador no Painel Clínico"},
        {"Variável": "uf / regiao", "Nome Amigável": "Estado / Região", "Significado": "Unidade da Federação (27 estados) e macrorregião brasileira do hospital onde ocorreu a internação.", "Uso no Projeto": "Filtros Epidemiológicos"},
        {"Variável": "idade_anos", "Nome Amigável": "Idade na Admissão", "Significado": "Idade do paciente convertida para anos contínuos (incluindo recém-nascidos em frações de ano).", "Uso no Projeto": "Feature da IA (Admissão)"},
        {"Variável": "sexo", "Nome Amigável": "Sexo Biológico", "Significado": "Sexo registrado na admissão hospitalar (1 = Masculino, 3 = Feminino).", "Uso no Projeto": "Feature da IA (Admissão)"},
        {"Variável": "diag_princ / capitulo_cid", "Nome Amigável": "Motivo (CID-10)", "Significado": "Código específico da Classificação Internacional de Doenças (ex.: J18 = Pneumonia) e seu grande capítulo médico.", "Uso no Projeto": "Feature da IA (Capítulo) & Painel"},
        {"Variável": "especialidade_leito", "Nome Amigável": "Especialidade do Leito", "Significado": "Tipo de enfermaria ou ala de internação (Clínica Médica, Cirúrgica, Obstétrica, Pediátrica, Psiquiátrica).", "Uso no Projeto": "Feature da IA (Admissão)"},
        {"Variável": "carater_internacao", "Nome Amigável": "Caráter da Entrada", "Significado": "Indica se o paciente entrou de forma programada (Eletiva) ou em situação aguda (Urgência/Emergência).", "Uso no Projeto": "Feature da IA (Admissão)"},
        {"Variável": "complexidade", "Nome Amigável": "Complexidade", "Significado": "Nível de densidade tecnológica exigido pelo procedimento principal (Média ou Alta Complexidade).", "Uso no Projeto": "Feature da IA (Admissão)"},
        {"Variável": "transferido_intermun", "Nome Amigável": "Transferência Intermunicipal", "Significado": "Indica (1 = Sim) quando o município de residência do paciente é diferente do município do hospital.", "Uso no Projeto": "Feature da IA (Admissão)"},
        {"Variável": "internacao_fim_de_semana", "Nome Amigável": "Entrada no Fim de Semana", "Significado": "Indica (1 = Sim) se a internação ocorreu no sábado ou domingo.", "Uso no Projeto": "Feature da IA (Admissão)"},
        {"Variável": "raca_cor_desc", "Nome Amigável": "Raça / Cor", "Significado": "Autodeclaração racial do paciente. Preservada na análise de equidade e removida da IA para evitar viés algorítmico.", "Uso no Projeto": "Apenas Epidemiologia (Ética)"},
        {"Variável": "mes_internacao", "Nome Amigável": "Mês da Internação", "Significado": "Mês do calendário (1 a 12). Usado na sazonalidade da Aba 1 e excluído da IA para evitar dependência de calendário.", "Uso no Projeto": "Apenas Epidemiologia"},
        {"Variável": "qt_diarias / uti_mes_to", "Nome Amigável": "Dias Internado / Diárias UTI", "Significado": "Quantidade total de dias de internação e diárias em leito de Terapia Intensiva.", "Uso no Projeto": "Alvo / Acompanhamento no Painel"},
        {"Variável": "Longa_Permanencia", "Nome Amigável": "Alvo: Longa Permanência", "Significado": "Variável binária indicando se a internação durou 15 dias ou mais (1 = Sim, 0 = Não).", "Uso no Projeto": "Alvo Preditivo 1"},
        {"Variável": "Obito", "Nome Amigável": "Alvo: Óbito Hospitalar", "Significado": "Variável binária indicando desfecho de óbito intra-hospitalar (1 = Sim, 0 = Não).", "Uso no Projeto": "Alvo Preditivo 2"},
    ])

    def render_wrapped_table(df_in: pd.DataFrame, widths: list[str]):
        import html as _html
        ths = "".join(
            f"<th style='width:{w}; background:#f1f5f9; color:#1e3a8a; font-weight:700; padding:10px 12px; text-align:left; border-bottom:2px solid #cbd5e1; font-size:0.88rem;'>{_html.escape(str(c))}</th>"
            for c, w in zip(df_in.columns, widths)
        )
        rows_html = ""
        for _, row in df_in.iterrows():
            tds = "".join(
                f"<td style='padding:10px 12px; border-bottom:1px solid #e2e8f0; vertical-align:top; white-space:normal; word-wrap:break-word; line-height:1.45; font-size:0.88rem; color:#0f172a;'>{_html.escape(str(val))}</td>"
                for val in row.values
            )
            rows_html += f"<tr>{tds}</tr>"
        table_html = (
            "<div style='overflow-x:auto; border:1px solid #e2e8f0; border-radius:8px; margin-bottom:1.2rem; background:#ffffff;'>"
            f"<table style='width:100%; border-collapse:collapse; table-layout:fixed;'><thead><tr>{ths}</tr></thead><tbody>{rows_html}</tbody></table>"
            "</div>"
        )
        st.markdown(table_html, unsafe_allow_html=True)

    if busca_dic:
        mask_d = df_dicionario.apply(lambda col: col.astype(str).str.contains(busca_dic, case=False, na=False)).any(axis=1)
        df_dic_show = df_dicionario[mask_d]
    else:
        df_dic_show = df_dicionario

    if df_dic_show.empty:
        st.info("Nenhuma variável encontrada para esse filtro.")
    else:
        render_wrapped_table(df_dic_show, ["18%", "18%", "46%", "18%"])

    st.markdown("#### 🧠 2. Glossário de Termos Técnicos (Ciência de Dados & Saúde)")
    df_glossario = pd.DataFrame([
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

    if busca_dic:
        mask_g = df_glossario.apply(lambda col: col.astype(str).str.contains(busca_dic, case=False, na=False)).any(axis=1)
        df_glo_show = df_glossario[mask_g]
    else:
        df_glo_show = df_glossario

    if df_glo_show.empty:
        st.info("Nenhum termo técnico encontrado para esse filtro.")
    else:
        render_wrapped_table(df_glo_show, ["20%", "18%", "62%"])

st.markdown("---")
st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 0.85rem;'>SUS Analytics Brasil 2024 • DuckDB + Parquet + LightGBM V4 Calibrado • Portfólio de Inteligência em Saúde</p>", unsafe_allow_html=True)
