# -*- coding: utf-8 -*-
"""
🏥 SUS Analytics & Clinical Intelligence Platform (PT / EN)
Aplicação Streamlit Multi-Abas Conectada ao Data Lake Parquet + DuckDB:
- Aba 1: Perfil Epidemiológico & Equidade em Saúde (27 UFs / 13,58M internações)
- Aba 2: Como a IA Funciona (Modelos V4 Calibrados & Escolha do LightGBM)
- Aba 3: Painel de Gestão Clínica & Leitos (Simulador + Flags Calibradas)
- Aba 4: Dicionário de Dados & Glossário
"""

import html as _html
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
from i18n import TEXTOS, get_dicionario_df, get_glossario_df, tr_series, tr_val

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
    st.error("Nenhuma partição Parquet encontrada em `dados/`. Execute `scripts/gerar_camada_deploy.py` primeiro.")
    st.stop()

modelo_longa, modelo_obito = carregar_modelos()
filtros = get_filtros()
kpis_gerais = get_kpis("Todas", "Todas")


def fmt_num(valor: int, lang: str) -> str:
    return f"{valor:,}" if lang == "en" else f"{valor:,}".replace(",", ".")


# ==========================================
# Sidebar Executiva + Seletor de Idioma
# ==========================================
with st.sidebar:
    idioma = st.radio(
        "🌐 Idioma / Language",
        options=["pt", "en"],
        format_func=lambda x: "🇧🇷 Português" if x == "pt" else "🇺🇸 English",
        horizontal=True,
        key="idioma_app"
    )
    t = TEXTOS[idioma]
    st.markdown("---")
    st.markdown(t["sidebar_title"])
    st.caption(t["sidebar_caption"])
    st.markdown(t["sidebar_resumo_hdr"])
    st.markdown(t["sidebar_resumo_body"])
    st.markdown("---")
    motor_str = t["modo_local"] if ds.parquet_bruto_disponivel() else t["modo_cloud"]
    st.markdown(f"{t['sidebar_motor']}\n`{motor_str}`")
    st.markdown(t["sidebar_prod"])
    st.markdown("---")
    st.info(t["sidebar_etica"])

# ==========================================
# 3. Header Principal e KPIs
# ==========================================
st.markdown(f'<div class="main-title">{t["main_title"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-title">{t["sub_title"]}</div>', unsafe_allow_html=True)

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    st.markdown(f'<div class="metric-card"><h3>{t["kpi1_lbl"]}</h3><p>{fmt_num(kpis_gerais["total"], idioma)}</p><small>{t["kpi1_sub"]}</small></div>', unsafe_allow_html=True)
with kpi2:
    st.markdown(f'<div class="metric-card"><h3>{t["kpi2_lbl"]}</h3><p style="color:#1e3a8a;">{kpis_gerais["total_ufs"]} / 27 UFs</p><small>{t["kpi2_sub"]}</small></div>', unsafe_allow_html=True)
with kpi3:
    ob_str = f"{kpis_gerais['taxa_obito']:.2f}%" if idioma == "en" else f"{kpis_gerais['taxa_obito']:.2f}%".replace(".", ",")
    st.markdown(f'<div class="metric-card red"><h3>{t["kpi3_lbl"]}</h3><p style="color:#dc2626;">{ob_str}</p><small>{t["kpi3_sub"]}</small></div>', unsafe_allow_html=True)
with kpi4:
    lg_str = f"{kpis_gerais['taxa_longa']:.2f}%" if idioma == "en" else f"{kpis_gerais['taxa_longa']:.2f}%".replace(".", ",")
    st.markdown(f'<div class="metric-card amber"><h3>{t["kpi4_lbl"]}</h3><p style="color:#d97706;">{lg_str}</p><small>{t["kpi4_sub"]}</small></div>', unsafe_allow_html=True)
with kpi5:
    roc_str = "0.879 / 0.816" if idioma == "en" else "0,879 / 0,816"
    st.markdown(f'<div class="metric-card green"><h3>{t["kpi5_lbl"]}</h3><p style="color:#059669;">{roc_str}</p><small>{t["kpi5_sub"]}</small></div>', unsafe_allow_html=True)

st.write("")

# ==========================================
# 4. Navegação por Abas
# ==========================================
tab_epi, tab_ml, tab_gestao, tab_dic = st.tabs(t["tabs"])

# ==============================================================================
# ABA 1: INTELIGÊNCIA EPIDEMIOLÓGICA & EQUIDADE
# ==============================================================================
with tab_epi:
    st.markdown(t["epi_hdr"])
    st.markdown(t["epi_sub"])

    col_f1, col_f2, col_f3, col_f4, col_f5, col_f6 = st.columns(6)
    with col_f1:
        reg_sel = st.selectbox(t["f_regiao"], ["Todas"] + filtros["regioes"], index=0, format_func=lambda x: tr_val(x, idioma))
    with col_f2:
        uf_sel = st.selectbox(t["f_uf"], ["Todas"] + filtros["ufs"], index=0, format_func=lambda x: tr_val(x, idioma))
    with col_f3:
        raca_sel = st.selectbox(t["f_raca"], ["Todas"] + filtros["racas"], index=0, format_func=lambda x: tr_val(x, idioma))
    with col_f4:
        espec_sel = st.selectbox(t["f_espec"], ["Todas"] + filtros["especialidades"], index=0, format_func=lambda x: tr_val(x, idioma))
    with col_f5:
        sexo_sel = st.selectbox(t["f_sexo"], ["Todos", "Masculino", "Feminino"], index=0, format_func=lambda x: tr_val(x, idioma))
    with col_f6:
        mes_sel = st.slider(t["f_mes"], 1, 12, (1, 12))

    total_filtrado = get_contagem(reg_sel, uf_sel, raca_sel, espec_sel, sexo_sel, mes_sel)
    pct_total = (total_filtrado / kpis_gerais["total"] * 100) if kpis_gerais["total"] > 0 else 0
    st.info(t["epi_info"].format(n=fmt_num(total_filtrado, idioma), pct=pct_total))

    # Linha 1 de Gráficos: Faixa Etária e Raça/Cor
    g1, g2 = st.columns(2)
    with g1:
        st.markdown(t["g1_hdr"])
        st.caption(t["g1_cap"])
        faixa_df = get_faixa_sexo(reg_sel, uf_sel, raca_sel, espec_sel, sexo_sel, mes_sel).copy()
        faixa_df["faixa_disp"] = tr_series(faixa_df["faixa_etaria"], idioma)
        faixa_df["sexo_disp"] = tr_series(faixa_df["sexo_desc"], idioma)
        cor_sexo = {tr_val("Masculino", idioma): "#2563eb", tr_val("Feminino", idioma): "#ec4899"}
        fig_faixa = px.bar(
            faixa_df, x="faixa_disp", y="contagem", color="sexo_disp",
            barmode="group", labels={"faixa_disp": t["g1_x"], "contagem": t["g1_y"], "sexo_disp": t["g1_c"]},
            color_discrete_map=cor_sexo,
            template="plotly_white"
        )
        fig_faixa.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_faixa, use_container_width=True)

    with g2:
        st.markdown(t["g2_hdr"])
        st.caption(t["g2_cap"])
        raca_metrics = get_raca(reg_sel, uf_sel, espec_sel, sexo_sel, mes_sel).copy()
        raca_metrics["raca_disp"] = tr_series(raca_metrics["raca_cor_desc"], idioma)
        fig_raca = px.scatter(
            raca_metrics, x="taxa_obito", y="media_diarias", size="total", color="raca_disp",
            text="raca_disp", labels={"taxa_obito": t["g2_x"], "media_diarias": t["g2_y"]},
            template="plotly_white", size_max=40
        )
        fig_raca.update_traces(textposition="top center")
        fig_raca.update_layout(height=350, showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_raca, use_container_width=True)

    # Linha 2 de Gráficos: Sazonalidade Mensal e Capítulos CID-10
    g3, g4 = st.columns(2)
    with g3:
        st.markdown(t["g3_hdr"])
        st.caption(t["g3_cap"])
        mes_agg = get_sazonalidade(reg_sel, uf_sel, raca_sel, espec_sel, sexo_sel)
        fig_mes = go.Figure()
        fig_mes.add_trace(go.Bar(x=mes_agg["mes_internacao"], y=mes_agg["internacoes"], name=t["g3_b"], marker_color="#93c5fd"))
        fig_mes.add_trace(go.Scatter(x=mes_agg["mes_internacao"], y=mes_agg["taxa_mortalidade"], name=t["g3_l"], yaxis="y2", line=dict(color="#dc2626", width=3)))
        fig_mes.update_layout(
            height=350, template="plotly_white",
            yaxis=dict(title=t["g3_y1"]),
            yaxis2=dict(title=t["g3_y2"], overlaying="y", side="right"),
            xaxis=dict(title=t["g3_x"], tickmode="linear", dtick=1),
            legend=dict(x=0.01, y=0.99),
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_mes, use_container_width=True)

    with g4:
        st.markdown(t["g4_hdr"])
        st.caption(t["g4_cap"])
        cid_df = get_cid(reg_sel, uf_sel, raca_sel, espec_sel, sexo_sel, mes_sel).copy()
        cid_df["cid_disp"] = tr_series(cid_df["capitulo_cid"], idioma)
        fig_cid = px.bar(
            cid_df, y="cid_disp", x="total", orientation="h",
            color="taxa_obito", color_continuous_scale="Reds",
            custom_data=["taxa_obito", "taxa_longa"],
            labels={"cid_disp": t["g4_y"], "total": t["g4_x"], "taxa_obito": t["g4_c"]},
            template="plotly_white"
        )
        fig_cid.update_traces(hovertemplate=t["g4_hover"])
        fig_cid.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_cid, use_container_width=True)

    if len(filtros["ufs"]) > 1:
        st.markdown(t["ufs_hdr"])
        df_ufs = get_resumo_ufs(reg_sel).copy()
        df_ufs["regiao"] = tr_series(df_ufs["regiao"], idioma)
        st.dataframe(
            df_ufs,
            use_container_width=True,
            hide_index=True,
            column_config={
                "uf": t["col_uf"],
                "regiao": t["col_reg"],
                "internacoes": st.column_config.NumberColumn(t["col_int"], format="%d"),
                "taxa_obito_pct": st.column_config.NumberColumn(t["col_ob_pct"], format="%.2f%%"),
                "taxa_longa_pct": st.column_config.NumberColumn(t["col_lg_pct"], format="%.2f%%"),
                "media_diarias": st.column_config.NumberColumn(t["col_med_d"], format="%.2f d")
            }
        )

# ==============================================================================
# ABA 2: COMO A IA FUNCIONA (MODELOS & MÉTRICAS)
# ==============================================================================
with tab_ml:
    st.markdown(t["ml_hdr"])
    st.markdown(t["ml_intro"])

    col_gov1, col_gov2 = st.columns(2)
    with col_gov1:
        st.success(t["ml_why"])
    with col_gov2:
        st.info(t["ml_guide"])

    st.markdown(t["ml_bench"])
    df_m_v4 = carregar_metricas_v4()
    if not df_m_v4.empty:
        col_m1, col_m2 = st.columns(2)
        cols_sel = ["Modelo", "ROC_AUC", "PR_AUC", "Brier_Calibrado", "Limiar_Otimo_Pct", "Sensibilidade_Pct", "Especificidade_Pct"]
        with col_m1:
            st.markdown(t["ml_ob_sub"])
            df_ob_m = df_m_v4[df_m_v4["Desfecho"] == "Óbito Intra-hospitalar"][cols_sel].rename(columns=t["ml_cols"])
            st.dataframe(df_ob_m, use_container_width=True, hide_index=True)
        with col_m2:
            st.markdown(t["ml_lg_sub"])
            df_lg_m = df_m_v4[df_m_v4["Desfecho"] == "Longa Permanência"][cols_sel].rename(columns=t["ml_cols"])
            st.dataframe(df_lg_m, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown(t["ml_evid"])
    col_img1, col_img2, col_img3 = st.columns(3)
    with col_img1:
        img_roc = os.path.join(BASE_DIR, "figuras", "fig_roc_pr_curvas_v4.png")
        if os.path.exists(img_roc):
            st.image(img_roc, caption=t["ml_img1"], use_container_width=True)
    with col_img2:
        img_cal = os.path.join(BASE_DIR, "figuras", "fig_calibracao_probabilidade_v4.png")
        if os.path.exists(img_cal):
            st.image(img_cal, caption=t["ml_img2"], use_container_width=True)
    with col_img3:
        img_imp = os.path.join(BASE_DIR, "figuras", "fig_importancias_admissao_v4.png")
        if os.path.exists(img_imp):
            st.image(img_imp, caption=t["ml_img3"], use_container_width=True)

# ==============================================================================
# ABA 3: PAINEL DE GESTÃO CLÍNICA & LEITOS
# ==============================================================================
with tab_gestao:
    st.markdown(t["cli_hdr"])
    st.markdown(t["cli_sub"])

    with st.expander(t["cli_exp"], expanded=True):
        st.caption(t["cli_exp_txt"])
        sl_col1, sl_col2 = st.columns(2)
        with sl_col1:
            corte_obito = st.slider(t["sl_ob"], min_value=1.0, max_value=50.0, value=(4.0, 15.0), step=0.5, format="%.1f%%")
        with sl_col2:
            corte_longa = st.slider(t["sl_lg"], min_value=1.0, max_value=50.0, value=(6.4, 20.0), step=0.5, format="%.1f%%")

    ob_med, ob_crit = corte_obito[0] / 100.0, corte_obito[1] / 100.0
    lg_med, lg_crit = corte_longa[0] / 100.0, corte_longa[1] / 100.0

    lg_mid_end = max(corte_longa[1] - 0.1, corte_longa[0])
    ob_mid_end = max(corte_obito[1] - 0.1, corte_obito[0])
    st.markdown(t["leg_lg"].format(a=corte_longa[0], b=lg_mid_end, c=corte_longa[1]))
    st.markdown(t["leg_ob"].format(a=corte_obito[0], b=ob_mid_end, c=corte_obito[1]))

    with st.expander(t["sim_exp"], expanded=False):
        c_sim1, c_sim2, c_sim3, c_sim4 = st.columns(4)
        with c_sim1:
            sim_idade = st.number_input(t["sim_idade"], min_value=0.0, max_value=120.0, value=68.0, step=1.0)
            sim_sexo = st.selectbox(t["sim_sexo"], ["Masculino", "Feminino"], index=0, format_func=lambda x: tr_val(x, idioma))
        with c_sim2:
            sim_espec = st.selectbox(t["sim_espec"], ["Clinica_Medica", "Cirurgica", "Pediatrica", "Obstetrica", "Psiquiatrica"], index=0, format_func=lambda x: tr_val(x, idioma))
            sim_carater = st.selectbox(t["sim_car"], ["Urgencia_Emergencia", "Eletiva"], index=0, format_func=lambda x: tr_val(x, idioma))
        with c_sim3:
            capitulos_cid = [
                "X - Respiratorio", "IX - Circulatorio", "I - Infecciosas",
                "II - Neoplasias", "XI - Digestivo", "XIV - Geniturinario",
                "V - Mentais", "XV - Gravidez/Parto", "XIX - Lesoes"
            ]
            sim_cid = st.selectbox(t["sim_cid"], capitulos_cid, index=0, format_func=lambda x: tr_val(x, idioma))
            sim_complex = st.selectbox(t["sim_comp"], ["Media_Complexidade", "Alta_Complexidade"], index=0, format_func=lambda x: tr_val(x, idioma))
        with c_sim4:
            sim_transf = st.radio(t["sim_transf"], [t["sim_nao"], t["sim_sim"]], index=1)
            sim_fds = st.radio(t["sim_fds"], [t["sim_nao"], t["sim_sim"]], index=0)

        if st.button(t["sim_btn"]):
            input_dict = {
                "idade_anos": sim_idade,
                "sexo": "1" if sim_sexo == "Masculino" else "3",
                "capitulo_cid": sim_cid,
                "especialidade_leito": sim_espec,
                "carater_internacao": sim_carater,
                "complexidade": sim_complex,
                "transferido_intermun": "1" if sim_transf == t["sim_sim"] else "0",
                "internacao_fim_de_semana": "1" if sim_fds == t["sim_sim"] else "0"
            }
            df_in = pd.DataFrame([input_dict])
            p_longa_sim = float(modelo_longa.predict_proba(df_in)[0, 1]) if modelo_longa else 0.45
            p_obito_sim = float(modelo_obito.predict_proba(df_in)[0, 1]) if modelo_obito else 0.65

            st.markdown(t["sim_res"])
            sc1, sc2, sc3 = st.columns(3)
            ob_lbls = t["badge_ob"]
            lg_lbls = t["badge_lg"]
            with sc1:
                badge_ob = ob_lbls[0] if p_obito_sim >= ob_crit else (ob_lbls[1] if p_obito_sim >= ob_med else ob_lbls[2])
                cor_ob = "#dc2626" if p_obito_sim >= ob_crit else ("#f59e0b" if p_obito_sim >= ob_med else "#16a34a")
                st.markdown(f"""
                <div style="background:{cor_ob}15; border-left: 5px solid {cor_ob}; padding: 12px; border-radius: 6px;">
                    <h4 style="margin:0; color:{cor_ob};">{t["sim_ob_tit"]}: {badge_ob}</h4>
                    <p style="margin:5px 0 0 0; font-size:1.6rem; font-weight:800; color:{cor_ob};">{p_obito_sim*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            with sc2:
                badge_lg = lg_lbls[0] if p_longa_sim >= lg_crit else (lg_lbls[1] if p_longa_sim >= lg_med else lg_lbls[2])
                cor_lg = "#dc2626" if p_longa_sim >= lg_crit else ("#f59e0b" if p_longa_sim >= lg_med else "#16a34a")
                st.markdown(f"""
                <div style="background:{cor_lg}15; border-left: 5px solid {cor_lg}; padding: 12px; border-radius: 6px;">
                    <h4 style="margin:0; color:{cor_lg};">{t["sim_lg_tit"]}: {badge_lg}</h4>
                    <p style="margin:5px 0 0 0; font-size:1.6rem; font-weight:800; color:{cor_lg};">{p_longa_sim*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            with sc3:
                if p_longa_sim >= lg_crit and p_obito_sim >= ob_crit:
                    quadrante, acao = t["quads"]["red"]
                elif p_obito_sim >= ob_crit:
                    quadrante, acao = t["quads"]["acute"]
                elif p_longa_sim >= lg_crit:
                    quadrante, acao = t["quads"]["chronic"]
                else:
                    quadrante, acao = t["quads"]["fast"]
                st.markdown(f"""
                <div style="background:#f1f5f9; border-left: 5px solid #0f172a; padding: 12px; border-radius: 6px;">
                    <h4 style="margin:0; color:#0f172a;">{quadrante}</h4>
                    <p style="margin:5px 0 0 0; font-size:0.95rem; font-weight:600; color:#334155;">{acao}</p>
                </div>
                """, unsafe_allow_html=True)

    st.write("")
    st.markdown(t["panel_hdr"])

    col_t1, col_t2, col_t3, col_t4, col_t5 = st.columns(5)
    with col_t1:
        filtro_uf = st.selectbox(t["p_uf"], ["Todas"] + filtros["ufs"], index=0, key="cockpit_uf", format_func=lambda x: tr_val(x, idioma))
    with col_t2:
        filtro_esp = st.selectbox(t["p_esp"], ["Todos"] + filtros["especialidades"], index=0, key="cockpit_esp", format_func=lambda x: tr_val(x, idioma))
    with col_t3:
        grav_idx = st.selectbox(t["p_grav"], [0, 1, 2], index=0, format_func=lambda i: t["p_grav_opts"][i])
    with col_t4:
        filtro_dias = st.slider(t["p_dias"], 0, 45, 0)
    with col_t5:
        busca_id = st.text_input(t["p_busca"], "")

    amostra = get_amostra_cockpit(filtro_uf, filtro_esp, filtro_dias, busca_id)

    if amostra.empty:
        st.warning(t["p_empty"])
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

        if grav_idx == 1:
            amostra = amostra[amostra["p_obito"] >= ob_crit]
        elif grav_idx == 2:
            amostra = amostra[amostra["p_longa"] >= lg_crit]

        ob_lbls = t["badge_ob"]
        lg_lbls = t["badge_lg"]
        amostra["id_anonimo"] = "#AIH-" + amostra["n_aih"].astype(str).str[-8:]
        amostra["sexo_desc"] = amostra["sexo"].astype(str).map({"1": "M", "3": "F"}).fillna("-")
        amostra["esp_disp"] = tr_series(amostra["especialidade_leito"], idioma)
        amostra["cid_disp"] = tr_series(amostra["capitulo_cid"], idioma)
        amostra["motivo_cid"] = amostra["diag_princ"].astype(str) + " • " + amostra["cid_disp"].astype(str)
        amostra["flag_obito"] = amostra["p_obito"].apply(
            lambda p: f"{'🔴 ' + ob_lbls[0] if p >= ob_crit else ('🟠 ' + ob_lbls[1] if p >= ob_med else '🟢 ' + ob_lbls[2])} ({p * 100:.1f}%)"
        )
        amostra["flag_longa"] = amostra["p_longa"].apply(
            lambda p: f"{'🔴 ' + lg_lbls[0] if p >= lg_crit else ('🟡 ' + lg_lbls[1] if p >= lg_med else '🟢 ' + lg_lbls[2])} ({p * 100:.1f}%)"
        )

        st.write(t["p_count"].format(n=len(amostra)))

        colunas_visiveis = [
            "id_anonimo", "uf", "idade_anos", "sexo_desc", "esp_disp",
            "motivo_cid", "qt_diarias", "uti_mes_to", "flag_longa", "flag_obito"
        ]
        p_cols_map = t["p_cols"]
        df_tabela = amostra[colunas_visiveis].rename(columns={
            "id_anonimo": p_cols_map["id_anonimo"],
            "uf": p_cols_map["uf"],
            "idade_anos": p_cols_map["idade_anos"],
            "sexo_desc": p_cols_map["sexo_desc"],
            "esp_disp": p_cols_map["especialidade_leito"],
            "motivo_cid": p_cols_map["motivo_cid"],
            "qt_diarias": p_cols_map["qt_diarias"],
            "uti_mes_to": p_cols_map["uti_mes_to"],
            "flag_longa": p_cols_map["flag_longa"],
            "flag_obito": p_cols_map["flag_obito"],
        })

        st.dataframe(
            df_tabela,
            use_container_width=True,
            hide_index=True,
            column_config={
                p_cols_map["idade_anos"]: st.column_config.NumberColumn(format=t["fmt_idade"]),
                p_cols_map["qt_diarias"]: st.column_config.NumberColumn(format="%d d"),
                p_cols_map["uti_mes_to"]: st.column_config.NumberColumn(format="%d d"),
            }
        )

# ==============================================================================
# ABA 4: DICIONÁRIO DE DADOS & GLOSSÁRIO
# ==============================================================================
with tab_dic:
    st.markdown(t["dic_hdr"])
    st.markdown(t["dic_sub"])

    busca_dic = st.text_input(t["dic_search"], "", placeholder=t["dic_ph"])

    def render_wrapped_table(df_in: pd.DataFrame, widths: list[str]):
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

    st.markdown(t["dic_s1"])
    df_dicionario = get_dicionario_df(idioma)
    if busca_dic:
        mask_d = df_dicionario.apply(lambda col: col.astype(str).str.contains(busca_dic, case=False, na=False)).any(axis=1)
        df_dic_show = df_dicionario[mask_d]
    else:
        df_dic_show = df_dicionario

    if df_dic_show.empty:
        st.info(t["dic_empty1"])
    else:
        render_wrapped_table(df_dic_show, ["18%", "18%", "46%", "18%"])

    st.markdown(t["dic_s2"])
    df_glossario = get_glossario_df(idioma)
    if busca_dic:
        mask_g = df_glossario.apply(lambda col: col.astype(str).str.contains(busca_dic, case=False, na=False)).any(axis=1)
        df_glo_show = df_glossario[mask_g]
    else:
        df_glo_show = df_glossario

    if df_glo_show.empty:
        st.info(t["dic_empty2"])
    else:
        render_wrapped_table(df_glo_show, ["20%", "18%", "62%"])

st.markdown("---")
st.markdown(f"<p style='text-align: center; color: #94a3b8; font-size: 0.85rem;'>{t['footer']}</p>", unsafe_allow_html=True)
