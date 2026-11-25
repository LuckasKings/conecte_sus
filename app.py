# -*- coding: utf-8 -*-
"""
🏥 SUS Analytics & Clinical Intelligence Platform (PT / EN / ES)
Aplicação Streamlit Multi-Abas Conectada ao Data Lake Parquet + DuckDB:
- Aba 1: Perfil Epidemiológico & Equidade em Saúde (27 UFs / 13,58M internações)
- Aba 2: Qualidade dos Dados & Contrato de Integridade
- Aba 3: Como a IA Funciona (Modelos V5 Calibrados com CID3 Híbrido, Benchmarks, Prós & Contras)
- Aba 4: Painel de Gestão Clínica & Leitos (Simulador CID3 + Flags Calibradas)
- Aba 5: Dicionário Completo SIH/SUS (113 Colunas + 13 Derivadas) & Glossário
- Aba 6: Jornada, Decisões, Arquitetura & Stack Tecnológica
"""

import html as _html
import json
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
from i18n import TEXTOS, get_dicionario_df, get_evolucao_df, get_glossario_df, get_qualidade_variaveis_df, get_tecnologias_df, tr_series, tr_val

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
        padding-top: 3.75rem;
        padding-bottom: 2rem;
    }
    .main-title {
        font-size: 2.15rem;
        font-weight: 800;
        color: #1e3a8a;
        letter-spacing: -0.02em;
        line-height: 1.25;
        padding-top: 0.2rem;
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
        padding: 14px 12px;
        min-height: 118px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        text-align: center;
        box-shadow: 0 2px 4px rgba(15, 23, 42, 0.04);
    }
    .metric-card.red { border-top-color: #dc2626; }
    .metric-card.amber { border-top-color: #d97706; }
    .metric-card.green { border-top-color: #059669; }
    .metric-card h3 {
        margin: 0;
        font-size: 0.71rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.03em;
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
    [data-testid="stTabs"] [role="tablist"] {
        gap: 12px !important;
        flex-wrap: wrap !important;
        overflow: visible !important;
        padding: 6px 2px 10px 2px !important;
    }
    [data-testid="stTabs"] [role="tablist"]::after,
    [data-testid="stTabs"] [data-testid="stTab"] .react-aria-SelectionIndicator {
        display: none !important;
    }
    [data-testid="stTabs"] [data-testid="stTab"] {
        height: auto !important;
        padding: 11px 20px !important;
        background: linear-gradient(180deg, #ffffff 0%, #f1f5f9 100%) !important;
        border: 1.5px solid #cbd5e1 !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 5px rgba(15, 23, 42, 0.05) !important;
        color: #334155 !important;
        font-weight: 600 !important;
        transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    [data-testid="stTabs"] [data-testid="stTab"]:hover,
    [data-testid="stTabs"] [data-testid="stTab"][data-hovered] {
        background: linear-gradient(180deg, #eff6ff 0%, #dbeafe 100%) !important;
        border-color: #60a5fa !important;
        color: #1e3a8a !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 12px rgba(37, 99, 235, 0.14) !important;
    }
    [data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"],
    [data-testid="stTabs"] [data-testid="stTab"][data-selected] {
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%) !important;
        border-color: #1e3a8a !important;
        box-shadow: 0 6px 16px rgba(30, 58, 138, 0.30) !important;
        transform: translateY(-1px) !important;
    }
    [data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] *,
    [data-testid="stTabs"] [data-testid="stTab"][data-selected] * {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Carregamento via DuckDB (Cacheado) e Modelos
# ==========================================
import features_v5  # Necessário para desserializar AdmissionV5Preprocessor dos artefatos .joblib


@st.cache_resource
def carregar_modelos():
    caminho_longa = os.path.join(BASE_DIR, "modelos", "modelo_admissao_longa_lightgbm_v5.joblib")
    caminho_obito = os.path.join(BASE_DIR, "modelos", "modelo_admissao_obito_lightgbm_v5.joblib")
    modelo_longa = joblib.load(caminho_longa) if os.path.exists(caminho_longa) else None
    modelo_obito = joblib.load(caminho_obito) if os.path.exists(caminho_obito) else None
    return modelo_longa, modelo_obito


@st.cache_data(ttl=300)
def carregar_metricas_v5():
    caminho_csv = os.path.join(BASE_DIR, "modelos", "metricas_v5_nacional.csv")
    if os.path.exists(caminho_csv):
        return pd.read_csv(caminho_csv)
    return pd.DataFrame()


@st.cache_data(ttl=300)
def carregar_top_cid3():
    caminho_json = os.path.join(BASE_DIR, "modelos", "top_cid3_por_capitulo_v5.json")
    if os.path.exists(caminho_json):
        with open(caminho_json, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


UFS_POR_REGIAO_BR = {
    "Centro-Oeste": ["DF", "GO", "MS", "MT"],
    "Nordeste": ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"],
    "Norte": ["AC", "AM", "AP", "PA", "RO", "RR", "TO"],
    "Sudeste": ["ES", "MG", "RJ", "SP"],
    "Sul": ["PR", "RS", "SC"],
}


@st.cache_data(ttl=300)
def get_filtros_v2():
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


@st.cache_data(ttl=300)
def get_qualidade_resumo(regiao, uf):
    return ds.obter_resumo_qualidade(regiao=regiao, uf=uf)


@st.cache_data(ttl=300)
def get_qualidade_uf(regiao):
    return ds.obter_qualidade_por_uf(regiao=regiao)


if not ds.parquet_disponivel():
    st.error("Nenhuma partição Parquet encontrada em `dados/`. Execute `scripts/gerar_camada_deploy.py` primeiro.")
    st.stop()

modelo_longa, modelo_obito = carregar_modelos()
top_cid3_por_cap = carregar_top_cid3()
filtros = get_filtros_v2()
kpis_gerais = get_kpis("Todas", "Todas")


def fmt_num(valor: int, lang: str) -> str:
    return f"{valor:,}" if lang == "en" else f"{valor:,}".replace(",", ".")


def render_wrapped_table(df_in: pd.DataFrame, widths: list[str], badge_col: str = None, status_keys: list[str] = None):
    badge_styles = {
        "ia_epi": ("#dcfce7", "#166534", "#bbf7d0"),
        "epi": ("#f3e8ff", "#6b21a8", "#e9d5ff"),
        "alvo_painel": ("#fef3c7", "#92400e", "#fde68a"),
        "nao_leakage": ("#fee2e2", "#991b1b", "#fecaca"),
        "nao_admin": ("#f1f5f9", "#475569", "#cbd5e1"),
        "dq_ok": ("#dcfce7", "#166534", "#bbf7d0"),
        "dq_warn": ("#fef3c7", "#92400e", "#fde68a"),
        "dq_info": ("#e0f2fe", "#075985", "#bae6fd"),
        "dq_alert": ("#fee2e2", "#991b1b", "#fecaca"),
    }
    ths = "".join(
        f"<th style='width:{w}; background:#f1f5f9; color:#1e3a8a; font-weight:700; padding:10px 12px; text-align:left; border-bottom:2px solid #cbd5e1; font-size:0.86rem;'>{_html.escape(str(c))}</th>"
        for c, w in zip(df_in.columns, widths)
    )
    rows_html = ""
    for row_i, (_, row) in enumerate(df_in.iterrows()):
        tds = ""
        s_key = status_keys[row_i] if status_keys is not None else None
        for col_name, val in zip(df_in.columns, row.values):
            esc_val = _html.escape(str(val))
            if badge_col and col_name == badge_col and s_key in badge_styles:
                bg, fg, bd = badge_styles[s_key]
                cell_content = f"<span style='display:inline-block; background:{bg}; color:{fg}; border:1px solid {bd}; padding:3px 8px; border-radius:6px; font-weight:700; font-size:0.78rem;'>{esc_val}</span>"
            else:
                cell_content = esc_val
            tds += f"<td style='padding:10px 12px; border-bottom:1px solid #e2e8f0; vertical-align:top; white-space:normal; word-wrap:break-word; line-height:1.45; font-size:0.86rem; color:#0f172a;'>{cell_content}</td>"
        rows_html += f"<tr>{tds}</tr>"
    table_html = (
        "<div style='overflow-x:auto; border:1px solid #e2e8f0; border-radius:8px; margin-bottom:1.2rem; background:#ffffff;'>"
        f"<table style='width:100%; border-collapse:collapse; table-layout:fixed;'><thead><tr>{ths}</tr></thead><tbody>{rows_html}</tbody></table>"
        "</div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)


# ==========================================
# Sidebar Executiva + Seletor de Idioma
# ==========================================
with st.sidebar:
    idioma_raw = st.segmented_control(
        "🌐 Idioma / Language",
        options=["pt", "en", "es"],
        format_func=lambda x: {"pt": "🇧🇷 PT", "en": "🇺🇸 EN", "es": "🇪🇸 ES"}[x],
        default="pt",
        selection_mode="single",
        key="idioma_app"
    )
    idioma = idioma_raw or "pt"
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
    roc_str = "0.908 / 0.848" if idioma == "en" else "0,908 / 0,848"
    st.markdown(f'<div class="metric-card green"><h3>{t["kpi5_lbl"]}</h3><p style="color:#059669;">{roc_str}</p><small>{t["kpi5_sub"]}</small></div>', unsafe_allow_html=True)

st.write("")

# ==========================================
# 4. Navegação por Abas
# ==========================================
tab_epi, tab_qual, tab_ml, tab_gestao, tab_dic, tab_jornada = st.tabs(t["tabs"])

# ==============================================================================
# ABA 1: INTELIGÊNCIA EPIDEMIOLÓGICA & EQUIDADE
# ==============================================================================
with tab_epi:
    st.markdown(t["epi_hdr"])
    st.markdown(t["epi_sub"])

    col_f1, col_f2, col_f3, col_f4, col_f5, col_f6 = st.columns(6)
    with col_f1:
        reg_sel = st.selectbox(t["f_regiao"], ["Todas"] + filtros["regioes"], index=0, format_func=lambda x: tr_val(x, idioma), key="epi_regiao")
    with col_f2:
        if reg_sel == "Todas":
            ufs_da_regiao = filtros["ufs"]
        else:
            ufs_oficiais = UFS_POR_REGIAO_BR.get(reg_sel, [])
            ufs_disponiveis = set(filtros["ufs"])
            ufs_da_regiao = [u for u in ufs_oficiais if u in ufs_disponiveis]
            if not ufs_da_regiao and "ufs_por_regiao" in filtros:
                ufs_da_regiao = filtros["ufs_por_regiao"].get(reg_sel, [])
        opcoes_uf = ["Todas"] + ufs_da_regiao
        if "epi_uf" in st.session_state and st.session_state["epi_uf"] not in opcoes_uf:
            st.session_state["epi_uf"] = "Todas"
        uf_sel = st.selectbox(t["f_uf"], opcoes_uf, index=0, format_func=lambda x: tr_val(x, idioma), key="epi_uf")
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
# ABA 2: QUALIDADE DOS DADOS & CONTRATO DE INTEGRIDADE
# ==============================================================================
with tab_qual:
    st.markdown(t["dq_hdr"])
    st.markdown(t["dq_sub"])

    dq_f1, dq_f2, _ = st.columns([1.2, 1.2, 2.6])
    with dq_f1:
        dq_reg = st.selectbox(t["f_regiao"], ["Todas"] + filtros["regioes"], index=0, format_func=lambda x: tr_val(x, idioma), key="dq_regiao")
    with dq_f2:
        if dq_reg == "Todas":
            dq_ufs_reg = filtros["ufs"]
        else:
            dq_ufs_of = UFS_POR_REGIAO_BR.get(dq_reg, [])
            dq_ufs_disp = set(filtros["ufs"])
            dq_ufs_reg = [u for u in dq_ufs_of if u in dq_ufs_disp] or filtros["ufs_por_regiao"].get(dq_reg, [])
        dq_opcoes_uf = ["Todas"] + dq_ufs_reg
        if "dq_uf" in st.session_state and st.session_state["dq_uf"] not in dq_opcoes_uf:
            st.session_state["dq_uf"] = "Todas"
        dq_uf = st.selectbox(t["f_uf"], dq_opcoes_uf, index=0, format_func=lambda x: tr_val(x, idioma), key="dq_uf")

    res_dq = get_qualidade_resumo(dq_reg, dq_uf)
    esperado_ufs = len(dq_ufs_reg) if dq_uf == "Todas" else 1
    esperado_parts = esperado_ufs * 12

    banner_key = "dq_banner_local" if res_dq["auditoria_fonte"] == "parquet" else "dq_banner_cloud"
    st.info(t[banner_key].format(n=fmt_num(res_dq["total"], idioma), na=fmt_num(res_dq["amostra_n"], idioma)))

    def fmt_dec(val: float, casas: int = 2) -> str:
        s = f"{val:.{casas}f}%"
        return s if idioma == "en" else s.replace(".", ",")

    dk1, dk2, dk3, dk4 = st.columns(4)
    with dk1:
        cor_cov = "#059669" if res_dq["particoes_uf_mes"] >= esperado_parts else "#d97706"
        st.markdown(
            f'<div class="metric-card green"><h3>{t["dq_kpi1_lbl"]}</h3>'
            f'<p style="color:{cor_cov};">{res_dq["particoes_uf_mes"]} / {esperado_parts}</p>'
            f'<small>{res_dq["ufs_cobertas"]}/{esperado_ufs} UFs • {res_dq["meses_cobertos"]}/12 • {t["dq_kpi1_sub"]}</small></div>',
            unsafe_allow_html=True,
        )
    with dk2:
        pct_ok_ia = 100.0 if res_dq["amostra_n"] > 0 and res_dq["null_ia_features"] == 0 else (
            ((res_dq["amostra_n"] - res_dq["null_ia_features"]) * 100.0 / res_dq["amostra_n"]) if res_dq["amostra_n"] > 0 else 0.0
        )
        st.markdown(
            f'<div class="metric-card green"><h3>{t["dq_kpi2_lbl"]}</h3>'
            f'<p style="color:#059669;">{fmt_dec(pct_ok_ia, 1)}</p>'
            f'<small>{fmt_num(res_dq["null_ia_features"], idioma)} nulos • {t["dq_kpi2_sub"]}</small></div>',
            unsafe_allow_html=True,
        )
    with dk3:
        viol_dom = res_dq["bad_diarias"] + res_dq["bad_idade"] + res_dq["bad_cid"]
        cor_dom = "#059669" if viol_dom == 0 else "#dc2626"
        cls_dom = "green" if viol_dom == 0 else "red"
        st.markdown(
            f'<div class="metric-card {cls_dom}"><h3>{t["dq_kpi3_lbl"]}</h3>'
            f'<p style="color:{cor_dom};">{fmt_num(viol_dom, idioma)}</p>'
            f'<small>{res_dq["min_idade"]:.0f}–{res_dq["max_idade"]:.0f} • {res_dq["min_diarias"]}–{res_dq["max_diarias"]}d • {t["dq_kpi3_sub"]}</small></div>',
            unsafe_allow_html=True,
        )
    with dk4:
        st.markdown(
            f'<div class="metric-card amber"><h3>{t["dq_kpi4_lbl"]}</h3>'
            f'<p style="color:#d97706;">{fmt_dec(res_dq["pct_raca_nd"], 2)} / {fmt_dec(res_dq["pct_esp_outras"], 2)}</p>'
            f'<small>CID Ign.: {fmt_dec(res_dq["pct_cid_ign"], 3)} • {t["dq_kpi4_sub"]}</small></div>',
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown(t["dq_contract_hdr"])
    st.caption(t["dq_contract_sub"])
    df_contrato, dq_status_keys = get_qualidade_variaveis_df(res_dq, esperado_ufs, idioma)
    render_wrapped_table(df_contrato, ["21%", "17%", "18%", "13%", "31%"], badge_col=t["dq_status_col"], status_keys=dq_status_keys)

    if len(filtros["ufs"]) > 1:
        df_dq_uf = get_qualidade_uf(dq_reg)
        if not df_dq_uf.empty:
            st.markdown(t["dq_uf_tbl_hdr"])
            df_dq_uf_disp = df_dq_uf.copy()
            df_dq_uf_disp["regiao"] = tr_series(df_dq_uf_disp["regiao"], idioma)
            dq_cols_map = t["dq_uf_cols"]
            st.dataframe(
                df_dq_uf_disp,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "uf": dq_cols_map["uf"],
                    "regiao": dq_cols_map["regiao"],
                    "internacoes": st.column_config.NumberColumn(dq_cols_map["internacoes"], format="%d"),
                    "meses_ativos": st.column_config.NumberColumn(dq_cols_map["meses_ativos"], format="%d / 12"),
                    "pct_raca_nd": st.column_config.NumberColumn(dq_cols_map["pct_raca_nd"], format="%.3f%%"),
                    "pct_cid_ign": st.column_config.NumberColumn(dq_cols_map["pct_cid_ign"], format="%.4f%%"),
                    "pct_esp_outras": st.column_config.NumberColumn(dq_cols_map["pct_esp_outras"], format="%.2f%%"),
                },
            )

    st.warning(t["dq_other_note"])

# ==============================================================================
# ABA 3: COMO A IA FUNCIONA (MODELOS & MÉTRICAS)
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
    df_m_v5 = carregar_metricas_v5()
    if not df_m_v5.empty:
        col_m1, col_m2 = st.columns(2)
        cols_sel = ["Modelo", "ROC_AUC", "PR_AUC", "Brier_Calibrado", "Limiar_Otimo_Pct", "Sensibilidade_Pct", "Especificidade_Pct"]
        with col_m1:
            st.markdown(t["ml_ob_sub"])
            df_ob_m = df_m_v5[df_m_v5["Desfecho"] == "Óbito Intra-hospitalar"][cols_sel].rename(columns=t["ml_cols"])
            st.dataframe(df_ob_m, use_container_width=True, hide_index=True)
        with col_m2:
            st.markdown(t["ml_lg_sub"])
            df_lg_m = df_m_v5[df_m_v5["Desfecho"] == "Longa Permanência"][cols_sel].rename(columns=t["ml_cols"])
            st.dataframe(df_lg_m, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown(t["ml_models_hdr"])
    st.caption(t["ml_models_sub"])
    cards_ml = t["ml_models_cards"]
    for row_idx in range(0, len(cards_ml), 2):
        mc1, mc2 = st.columns(2)
        for col_w, card in zip([mc1, mc2], cards_ml[row_idx:row_idx + 2]):
            with col_w:
                with st.container(border=True):
                    st.markdown(f"##### {card['title']}")
                    st.markdown(card["how"])
                    st.markdown(card["pros"])
                    st.markdown(card["cons"])

    st.markdown("---")
    st.markdown(t["ml_evid"])
    col_img1, col_img2, col_img3 = st.columns(3)
    with col_img1:
        img_roc = os.path.join(BASE_DIR, "figuras", "fig_roc_pr_curvas_v5.png")
        if os.path.exists(img_roc):
            st.image(img_roc, caption=t["ml_img1"], use_container_width=True)
    with col_img2:
        img_cal = os.path.join(BASE_DIR, "figuras", "fig_calibracao_probabilidade_v5.png")
        if os.path.exists(img_cal):
            st.image(img_cal, caption=t["ml_img2"], use_container_width=True)
    with col_img3:
        img_imp = os.path.join(BASE_DIR, "figuras", "fig_importancias_admissao_v5.png")
        if os.path.exists(img_imp):
            st.image(img_imp, caption=t["ml_img3"], use_container_width=True)

# ==============================================================================
# ABA 4: PAINEL DE GESTÃO CLÍNICA & LEITOS
# ==============================================================================
with tab_gestao:
    st.markdown(t["cli_hdr"])
    st.markdown(t["cli_sub"])

    with st.expander(t["cli_exp"], expanded=True):
        st.caption(t["cli_exp_txt"])
        sl_col1, sl_col2 = st.columns(2)
        with sl_col1:
            corte_obito = st.slider(t["sl_ob"], min_value=1.0, max_value=50.0, value=(4.1, 15.0), step=0.1, format="%.1f%%")
        with sl_col2:
            corte_longa = st.slider(t["sl_lg"], min_value=1.0, max_value=50.0, value=(6.3, 20.0), step=0.1, format="%.1f%%")

    ob_med, ob_crit = corte_obito[0] / 100.0, corte_obito[1] / 100.0
    lg_med, lg_crit = corte_longa[0] / 100.0, corte_longa[1] / 100.0

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
            opcoes_cid3 = ["_CAPITULO_"] + [item["cid3"] for item in top_cid3_por_cap.get(sim_cid, [])]
            counts_cid3 = {item["cid3"]: item["n"] for item in top_cid3_por_cap.get(sim_cid, [])}
            sim_cid3 = st.selectbox(
                t["sim_cid3"],
                opcoes_cid3,
                index=0,
                format_func=lambda x: t["sim_cid3_default"] if x == "_CAPITULO_" else f"{x} ({fmt_num(counts_cid3.get(x, 0), idioma)})",
                key="sim_cid3_sel",
            )
            sim_complex = st.selectbox(t["sim_comp"], ["Media_Complexidade", "Alta_Complexidade"], index=0, format_func=lambda x: tr_val(x, idioma))
        with c_sim4:
            sim_transf = st.radio(t["sim_transf"], [t["sim_nao"], t["sim_sim"]], index=1)
            sim_fds = st.radio(t["sim_fds"], [t["sim_nao"], t["sim_sim"]], index=0)

        if st.button(t["sim_btn"]):
            input_dict = {
                "idade_anos": sim_idade,
                "sexo": "1" if sim_sexo == "Masculino" else "3",
                "capitulo_cid": sim_cid,
                "cid3": sim_cid3,
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

    lg_mid_end = max(corte_longa[1] - 0.1, corte_longa[0])
    ob_mid_end = max(corte_obito[1] - 0.1, corte_obito[0])
    with st.container(border=True):
        mg1, mg2 = st.columns([1.05, 1.15])
        with mg1:
            st.markdown(t["cli_guide_hdr1"])
            st.markdown(t["leg_lg"].format(a=corte_longa[0], b=lg_mid_end, c=corte_longa[1]))
            st.markdown(t["leg_ob"].format(a=corte_obito[0], b=ob_mid_end, c=corte_obito[1]))
            st.caption(t["cli_guide_txt1"])
        with mg2:
            st.markdown(t["cli_guide_hdr2"])
            st.markdown(t["cli_guide_txt2"])

    amostra = get_amostra_cockpit(filtro_uf, filtro_esp, filtro_dias, busca_id)

    if amostra.empty:
        st.warning(t["p_empty"])
    else:
        amostra = amostra.copy()
        df_mod = pd.DataFrame()
        df_mod["idade_anos"] = amostra["idade_anos"]
        df_mod["cid3"] = (
            amostra["diag_princ"].fillna("_CAPITULO_").astype(str).str.strip().str[:3]
            .replace({"": "_CAPITULO_", "nan": "_CAPITULO_", "None": "_CAPITULO_", "IGN": "_CAPITULO_"})
        )
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

        mt = t["match_tokens"]
        real_ob_s = amostra["Obito"].fillna(0).astype(int) == 1 if "Obito" in amostra.columns else pd.Series(False, index=amostra.index)
        real_lg_s = amostra["qt_diarias"].fillna(0).astype(int) >= 15
        pred_ob_s = amostra["p_obito"] >= ob_med
        pred_lg_s = amostra["p_longa"] >= lg_med
        hit_ob_s = pred_ob_s == real_ob_s
        hit_lg_s = pred_lg_s == real_lg_s
        both_s = hit_ob_s & hit_lg_s

        def _fmt_match(r_ob: bool, r_lg: bool, p_ob_al: bool, p_lg_al: bool) -> str:
            h_ob = p_ob_al == r_ob
            h_lg = p_lg_al == r_lg
            if h_ob and h_lg:
                return f"✅ {mt['vp']}" if (r_ob or r_lg) else f"✅ {mt['vn']}"
            if h_ob ^ h_lg:
                tgt = mt["short_ob"] if not h_ob else mt["short_lg"]
                is_fp = p_ob_al if not h_ob else p_lg_al
                return f"✅ {mt['alert_ok']} ({tgt} {mt['fp_short']})" if is_fp else f"🔕 {tgt}: {mt['fn']}"
            if p_lg_al and p_ob_al:
                return f"✅ {mt['alert_ok']} ({mt['both_fp']})"
            if not p_lg_al and not p_ob_al:
                return f"❌ {mt['both_fn']}"
            lg_lbl = mt["fp_short"] if p_lg_al else mt["fn_short"]
            ob_lbl = mt["fp_short"] if p_ob_al else mt["fn_short"]
            return f"⚠️ {mt['short_lg']} {lg_lbl} | {mt['short_ob']} {ob_lbl}"

        amostra["desfecho_real"] = [mt["obito"] if r_ob else mt["alta"] for r_ob in real_ob_s]
        amostra["desfecho_match"] = [
            _fmt_match(r_ob, r_lg, p_ob_al, p_lg_al)
            for r_ob, r_lg, p_ob_al, p_lg_al in zip(real_ob_s, real_lg_s, pred_ob_s, pred_lg_s)
        ]

        if not amostra.empty:
            pct_both = float(both_s.mean() * 100)
            pct_lg = float(hit_lg_s.mean() * 100)
            pct_ob = float(hit_ob_s.mean() * 100)
            st.write(f"{t['p_count'].format(n=len(amostra))} ({t['p_match_summary'].format(both=pct_both, lg=pct_lg, ob=pct_ob)}):")
        else:
            st.write(t["p_count"].format(n=0) + ":")

        colunas_visiveis = [
            "id_anonimo", "uf", "idade_anos", "sexo_desc", "esp_disp",
            "motivo_cid", "qt_diarias", "uti_mes_to", "desfecho_real",
            "flag_longa", "flag_obito", "desfecho_match"
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
            "desfecho_real": p_cols_map["desfecho_real"],
            "flag_longa": p_cols_map["flag_longa"],
            "flag_obito": p_cols_map["flag_obito"],
            "desfecho_match": p_cols_map["desfecho_match"],
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
# ABA 5: DICIONÁRIO COMPLETO SIH/SUS (113 COLUNAS) & GLOSSÁRIO
# ==============================================================================
with tab_dic:
    st.markdown(t["dic_hdr"])
    st.markdown(t["dic_sub"])

    df_dicionario = get_dicionario_df(idioma)

    # Resumo quantitativo da auditoria de colunas
    c_d1, c_d2, c_d3, c_d4 = st.columns(4)
    n_ia = int((df_dicionario["_status_key"] == "ia_epi").sum())
    n_epi = int((df_dicionario["_status_key"] == "epi").sum())
    n_alvo = int((df_dicionario["_status_key"] == "alvo_painel").sum())
    n_nao = int(df_dicionario["_status_key"].isin(["nao_leakage", "nao_admin"]).sum())
    with c_d1:
        st.metric("Total (113 SIH + 13 Deriv.)", len(df_dicionario))
    with c_d2:
        lbl_ia = {"pt": "✅ IA + Epidemio", "en": "✅ AI + Epidemiology", "es": "✅ IA + Epidemiología"}[idioma]
        st.metric(lbl_ia, n_ia)
    with c_d3:
        lbl_epi = {"pt": "📊 Epidemio / Alvo", "en": "📊 Epidemio / Target", "es": "📊 Epidemio / Objetivo"}[idioma]
        st.metric(lbl_epi, f"{n_epi} + {n_alvo}")
    with c_d4:
        lbl_nao = {"pt": "⛔ Não Usadas", "en": "⛔ Not Used", "es": "⛔ No Usadas"}[idioma]
        st.metric(lbl_nao, n_nao)

    col_d1, col_d2 = st.columns([2, 1])
    with col_d1:
        busca_dic = st.text_input(t["dic_search"], "", placeholder=t["dic_ph"])
    with col_d2:
        filtro_opts = t["dic_filter_opts"]
        uso_sel = st.selectbox(
            t["dic_filter_lbl"],
            options=[k for k, _ in filtro_opts],
            format_func=lambda k: dict(filtro_opts)[k],
            index=0
        )

    st.markdown(t["dic_s1"])
    df_dic_show = df_dicionario.copy()
    if uso_sel != "all":
        df_dic_show = df_dic_show[df_dic_show["_status_key"] == uso_sel]
    if busca_dic:
        mask_d = df_dic_show.drop(columns=["_status_key"]).apply(
            lambda col: col.astype(str).str.contains(busca_dic, case=False, na=False)
        ).any(axis=1)
        df_dic_show = df_dic_show[mask_d]

    st.caption(t["dic_count"].format(n=len(df_dic_show)))
    if df_dic_show.empty:
        st.info(t["dic_empty1"])
    else:
        status_keys_list = df_dic_show["_status_key"].tolist()
        df_dic_clean = df_dic_show.drop(columns=["_status_key"])
        uso_col_name = {"pt": "Uso no Projeto", "en": "Usage Status", "es": "Uso en el Proyecto"}[idioma]
        render_wrapped_table(df_dic_clean, ["15%", "17%", "24%", "44%"], badge_col=uso_col_name, status_keys=status_keys_list)

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

# ==============================================================================
# ABA 6: JORNADA, DECISÕES & ARQUITETURA DO PROJETO
# ==============================================================================
with tab_jornada:
    st.markdown(t["j_hdr"])
    st.markdown(t["j_sub"])

    j1, j2 = st.columns(2)
    with j1:
        with st.container(border=True):
            st.markdown(f"#### {t['j_v1_tit']}")
            st.markdown(t["j_v1_txt"])
    with j2:
        with st.container(border=True):
            st.markdown(f"#### {t['j_v2_tit']}")
            st.markdown(t["j_v2_txt"])

    j3, j4 = st.columns(2)
    with j3:
        with st.container(border=True):
            st.markdown(f"#### {t['j_v3_tit']}")
            st.markdown(t["j_v3_txt"])
    with j4:
        with st.container(border=True):
            st.markdown(f"#### {t['j_v4_tit']}")
            st.markdown(t["j_v4_txt"])

    with st.container(border=True):
        st.markdown(f"#### {t['j_v5_tit']}")
        st.markdown(t["j_v5_txt"])

    st.markdown(t["j_evol_hdr"])
    df_evol = get_evolucao_df(idioma)
    render_wrapped_table(df_evol, ["16%", "17%", "16%", "18%", "33%"])

    st.markdown("---")
    st.markdown(t["j_tech_hdr"])
    st.caption(t["j_tech_sub"])
    df_tech = get_tecnologias_df(idioma)
    render_wrapped_table(df_tech, ["20%", "20%", "28%", "32%"])

st.markdown("---")
st.markdown(f"<p style='text-align: center; color: #94a3b8; font-size: 0.85rem;'>{t['footer']}</p>", unsafe_allow_html=True)
