# -*- coding: utf-8 -*-
"""
Camada de Serviço Analítico Híbrido com DuckDB:
- Aba 1 e KPIs: utiliza o cubo epidemiológico compacto (`dados/deploy/cubo_epidemiologico_brasil_2024.parquet`),
  preservando 100% de exatidão sobre os 13.584.534 registros em milissegundos (local e nuvem).
- Aba 3 (Cockpit Clínico): consulta os Parquets brutos locais (`dados/parquet_nacional/`) se presentes
  ou faz fallback automático para a amostra estratificada (`dados/deploy/amostra_cockpit_brasil_2024.parquet`) no deploy online.
"""

from pathlib import Path
import duckdb
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PARQUET_DIR = BASE_DIR / "dados" / "parquet_nacional"
PARQUET_GLOB = str(PARQUET_DIR / "**" / "*.parquet").replace("\\", "/")

DEPLOY_DIR = BASE_DIR / "dados" / "deploy"
CUBO_PATH = DEPLOY_DIR / "cubo_epidemiologico_brasil_2024.parquet"
AMOSTRA_PATH = DEPLOY_DIR / "amostra_cockpit_brasil_2024.parquet"


def parquet_bruto_disponivel() -> bool:
    return PARQUET_DIR.exists() and any(PARQUET_DIR.rglob("*.parquet"))


def parquet_disponivel() -> bool:
    return CUBO_PATH.exists() or parquet_bruto_disponivel()


def modo_execucao() -> str:
    if parquet_bruto_disponivel():
        return "Local Completo (13,58M Partições + Cubo)"
    if CUBO_PATH.exists():
        return "Cloud Otimizado (Cubo 13,58M + Amostra 30k)"
    return "Indisponível"


def _get_con():
    return duckdb.connect(database=":memory:")


def _build_where(regiao="Todas", uf="Todas", raca="Todas", espec="Todas", sexo="Todos", mes_range=(1, 12)):
    clauses = ["1=1"]
    params = []

    if regiao and regiao != "Todas":
        clauses.append("regiao = ?")
        params.append(regiao)
    if uf and uf != "Todas":
        clauses.append("uf = ?")
        params.append(uf)
    if raca and raca != "Todas":
        clauses.append("raca_cor_desc = ?")
        params.append(raca)
    if espec and espec != "Todas":
        clauses.append("especialidade_leito = ?")
        params.append(espec)
    if sexo and sexo != "Todos":
        sexo_cod = "1" if sexo == "Masculino" else "3"
        clauses.append("sexo = ?")
        params.append(sexo_cod)
    if mes_range:
        clauses.append("mes_internacao BETWEEN ? AND ?")
        params.extend([int(mes_range[0]), int(mes_range[1])])

    return " AND ".join(clauses), params


def _fonte_epidemiologia() -> str:
    if CUBO_PATH.exists():
        return f"read_parquet('{str(CUBO_PATH).replace(chr(92), '/')}')"
    return f"read_parquet('{PARQUET_GLOB}', hive_partitioning=true)"


def listar_filtros_disponiveis() -> dict:
    if not parquet_disponivel():
        return {"regioes": [], "ufs": [], "ufs_por_regiao": {}, "racas": [], "especialidades": []}

    fonte = _fonte_epidemiologia()
    con = _get_con()
    df_reg_uf = con.execute(f"SELECT DISTINCT regiao, uf FROM {fonte} ORDER BY regiao, uf").df()
    ufs_por_regiao = (
        df_reg_uf.groupby("regiao")["uf"].apply(lambda s: sorted(s.dropna().tolist())).to_dict()
    )
    return {
        "regioes": sorted(df_reg_uf["regiao"].dropna().unique().tolist()),
        "ufs": sorted(df_reg_uf["uf"].dropna().unique().tolist()),
        "ufs_por_regiao": ufs_por_regiao,
        "racas": con.execute(f"SELECT DISTINCT raca_cor_desc FROM {fonte} ORDER BY raca_cor_desc").df()["raca_cor_desc"].dropna().tolist(),
        "especialidades": con.execute(f"SELECT DISTINCT especialidade_leito FROM {fonte} ORDER BY especialidade_leito").df()["especialidade_leito"].dropna().tolist(),
    }


def obter_kpis_globais(regiao="Todas", uf="Todas") -> dict:
    if not parquet_disponivel():
        return {"total": 0, "taxa_obito": 0.0, "taxa_longa": 0.0, "total_ufs": 0, "gasto_mi": 0.0}

    where_sql, params = _build_where(regiao=regiao, uf=uf, mes_range=None)
    con = _get_con()

    if CUBO_PATH.exists():
        cubo_str = str(CUBO_PATH).replace("\\", "/")
        query = f"""
        SELECT
            CAST(COALESCE(SUM(internacoes), 0) AS BIGINT) AS total,
            COALESCE(SUM(obitos) * 100.0 / NULLIF(SUM(internacoes), 0), 0) AS taxa_obito,
            COALESCE(SUM(longas) * 100.0 / NULLIF(SUM(internacoes), 0), 0) AS taxa_longa,
            COUNT(DISTINCT uf) AS total_ufs,
            COALESCE(SUM(soma_val_tot) / 1e6, 0) AS gasto_mi
        FROM read_parquet('{cubo_str}')
        WHERE {where_sql}
        """
    else:
        query = f"""
        SELECT
            COUNT(*) AS total,
            COALESCE(AVG(Obito) * 100, 0) AS taxa_obito,
            COALESCE(AVG(Longa_Permanencia) * 100, 0) AS taxa_longa,
            COUNT(DISTINCT uf) AS total_ufs,
            COALESCE(SUM(val_tot) / 1e6, 0) AS gasto_mi
        FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=true)
        WHERE {where_sql}
        """
    row = con.execute(query, params).fetchone()
    return {
        "total": int(row[0]),
        "taxa_obito": float(row[1]),
        "taxa_longa": float(row[2]),
        "total_ufs": int(row[3]),
        "gasto_mi": float(row[4]),
    }


def contar_filtrado(regiao="Todas", uf="Todas", raca="Todas", espec="Todas", sexo="Todos", mes_range=(1, 12)) -> int:
    where_sql, params = _build_where(regiao, uf, raca, espec, sexo, mes_range)
    con = _get_con()
    if CUBO_PATH.exists():
        cubo_str = str(CUBO_PATH).replace("\\", "/")
        query = f"SELECT CAST(COALESCE(SUM(internacoes), 0) AS BIGINT) FROM read_parquet('{cubo_str}') WHERE {where_sql}"
    else:
        query = f"SELECT COUNT(*) FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=true) WHERE {where_sql}"
    return int(con.execute(query, params).fetchone()[0])


def obter_distribuicao_faixa_sexo(regiao="Todas", uf="Todas", raca="Todas", espec="Todas", sexo="Todos", mes_range=(1, 12)) -> pd.DataFrame:
    where_sql, params = _build_where(regiao, uf, raca, espec, sexo, mes_range)
    con = _get_con()
    if CUBO_PATH.exists():
        cubo_str = str(CUBO_PATH).replace("\\", "/")
        query = f"""
        SELECT
            faixa_etaria,
            CASE
                WHEN sexo = '1' THEN 'Masculino'
                WHEN sexo = '3' THEN 'Feminino'
                ELSE 'Outro'
            END AS sexo_desc,
            CAST(SUM(internacoes) AS BIGINT) AS contagem
        FROM read_parquet('{cubo_str}')
        WHERE {where_sql}
        GROUP BY faixa_etaria, sexo_desc
        """
    else:
        query = f"""
        SELECT
            CASE
                WHEN idade_anos < 1 THEN '<1 ano'
                WHEN idade_anos < 12 THEN '1-11 anos'
                WHEN idade_anos < 19 THEN '12-18 anos'
                WHEN idade_anos < 40 THEN '19-39 anos'
                WHEN idade_anos < 60 THEN '40-59 anos'
                WHEN idade_anos < 80 THEN '60-79 anos'
                ELSE '80+ anos'
            END AS faixa_etaria,
            CASE
                WHEN sexo = '1' THEN 'Masculino'
                WHEN sexo = '3' THEN 'Feminino'
                ELSE 'Outro'
            END AS sexo_desc,
            COUNT(*) AS contagem
        FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=true)
        WHERE {where_sql}
        GROUP BY faixa_etaria, sexo_desc
        """
    df = con.execute(query, params).df()
    ordem = ["<1 ano", "1-11 anos", "12-18 anos", "19-39 anos", "40-59 anos", "60-79 anos", "80+ anos"]
    df["faixa_etaria"] = pd.Categorical(df["faixa_etaria"], categories=ordem, ordered=True)
    return df.sort_values("faixa_etaria")


def obter_equidade_raca(regiao="Todas", uf="Todas", espec="Todas", sexo="Todos", mes_range=(1, 12)) -> pd.DataFrame:
    where_sql, params = _build_where(regiao=regiao, uf=uf, raca="Todas", espec=espec, sexo=sexo, mes_range=mes_range)
    con = _get_con()
    if CUBO_PATH.exists():
        cubo_str = str(CUBO_PATH).replace("\\", "/")
        query = f"""
        SELECT
            raca_cor_desc,
            CAST(SUM(internacoes) AS BIGINT) AS total,
            SUM(obitos) * 100.0 / NULLIF(SUM(internacoes), 0) AS taxa_obito,
            SUM(soma_diarias) * 1.0 / NULLIF(SUM(internacoes), 0) AS media_diarias
        FROM read_parquet('{cubo_str}')
        WHERE {where_sql}
        GROUP BY raca_cor_desc
        """
    else:
        query = f"""
        SELECT
            raca_cor_desc,
            COUNT(*) AS total,
            AVG(Obito) * 100 AS taxa_obito,
            AVG(qt_diarias) AS media_diarias
        FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=true)
        WHERE {where_sql}
        GROUP BY raca_cor_desc
        """
    return con.execute(query, params).df()


def obter_sazonalidade_mensal(regiao="Todas", uf="Todas", raca="Todas", espec="Todas", sexo="Todos") -> pd.DataFrame:
    where_sql, params = _build_where(regiao=regiao, uf=uf, raca=raca, espec=espec, sexo=sexo, mes_range=(1, 12))
    con = _get_con()
    if CUBO_PATH.exists():
        cubo_str = str(CUBO_PATH).replace("\\", "/")
        query = f"""
        SELECT
            mes_internacao,
            CAST(SUM(internacoes) AS BIGINT) AS internacoes,
            CAST(SUM(obitos) AS BIGINT) AS obitos,
            SUM(obitos) * 100.0 / NULLIF(SUM(internacoes), 0) AS taxa_mortalidade
        FROM read_parquet('{cubo_str}')
        WHERE {where_sql}
        GROUP BY mes_internacao
        ORDER BY mes_internacao
        """
    else:
        query = f"""
        SELECT
            mes_internacao,
            COUNT(*) AS internacoes,
            SUM(Obito) AS obitos,
            AVG(Obito) * 100 AS taxa_mortalidade
        FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=true)
        WHERE {where_sql}
        GROUP BY mes_internacao
        ORDER BY mes_internacao
        """
    return con.execute(query, params).df()


def obter_top_capitulos_cid(regiao="Todas", uf="Todas", raca="Todas", espec="Todas", sexo="Todos", mes_range=(1, 12)) -> pd.DataFrame:
    where_sql, params = _build_where(regiao, uf, raca, espec, sexo, mes_range)
    con = _get_con()
    if CUBO_PATH.exists():
        cubo_str = str(CUBO_PATH).replace("\\", "/")
        query = f"""
        SELECT
            capitulo_cid,
            CAST(SUM(internacoes) AS BIGINT) AS total,
            SUM(longas) * 100.0 / NULLIF(SUM(internacoes), 0) AS taxa_longa,
            SUM(obitos) * 100.0 / NULLIF(SUM(internacoes), 0) AS taxa_obito
        FROM read_parquet('{cubo_str}')
        WHERE {where_sql}
        GROUP BY capitulo_cid
        ORDER BY total DESC
        LIMIT 8
        """
    else:
        query = f"""
        SELECT
            capitulo_cid,
            COUNT(*) AS total,
            AVG(Longa_Permanencia) * 100 AS taxa_longa,
            AVG(Obito) * 100 AS taxa_obito
        FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=true)
        WHERE {where_sql}
        GROUP BY capitulo_cid
        ORDER BY total DESC
        LIMIT 8
        """
    df = con.execute(query, params).df()
    return df.sort_values("total", ascending=True)


def obter_resumo_ufs(regiao="Todas") -> pd.DataFrame:
    where_sql, params = _build_where(regiao=regiao, mes_range=None)
    con = _get_con()
    if CUBO_PATH.exists():
        cubo_str = str(CUBO_PATH).replace("\\", "/")
        query = f"""
        SELECT
            uf,
            regiao,
            CAST(SUM(internacoes) AS BIGINT) AS internacoes,
            ROUND(SUM(obitos) * 100.0 / NULLIF(SUM(internacoes), 0), 2) AS taxa_obito_pct,
            ROUND(SUM(longas) * 100.0 / NULLIF(SUM(internacoes), 0), 2) AS taxa_longa_pct,
            ROUND(SUM(soma_diarias) * 1.0 / NULLIF(SUM(internacoes), 0), 2) AS media_diarias
        FROM read_parquet('{cubo_str}')
        WHERE {where_sql}
        GROUP BY uf, regiao
        ORDER BY internacoes DESC
        """
    else:
        query = f"""
        SELECT
            uf,
            regiao,
            COUNT(*) AS internacoes,
            ROUND(AVG(Obito) * 100, 2) AS taxa_obito_pct,
            ROUND(AVG(Longa_Permanencia) * 100, 2) AS taxa_longa_pct,
            ROUND(AVG(qt_diarias), 2) AS media_diarias
        FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=true)
        WHERE {where_sql}
        GROUP BY uf, regiao
        ORDER BY internacoes DESC
        """
    return con.execute(query, params).df()


def obter_amostra_cockpit(uf="Todas", espec="Todos", min_dias=0, busca="", limite=300) -> pd.DataFrame:
    clauses = ["1=1"]
    params = []
    if uf and uf != "Todas":
        clauses.append("uf = ?")
        params.append(uf)
    if espec and espec != "Todos":
        clauses.append("especialidade_leito = ?")
        params.append(espec)
    if min_dias and min_dias > 0:
        clauses.append("qt_diarias >= ?")
        params.append(int(min_dias))
    if busca and busca.strip():
        clauses.append("(LOWER(n_aih) LIKE ? OR LOWER(diag_princ) LIKE ? OR LOWER(capitulo_cid) LIKE ?)")
        termo = f"%{busca.strip().lower()}%"
        params.extend([termo, termo, termo])

    where_sql = " AND ".join(clauses)
    con = _get_con()

    if parquet_bruto_disponivel():
        fonte_cockpit = f"read_parquet('{PARQUET_GLOB}', hive_partitioning=true)"
    elif AMOSTRA_PATH.exists():
        amostra_str = str(AMOSTRA_PATH).replace("\\", "/")
        fonte_cockpit = f"read_parquet('{amostra_str}')"
    else:
        return pd.DataFrame()

    query = f"""
    SELECT * FROM (
        SELECT
            n_aih,
            uf,
            regiao,
            idade_anos,
            sexo,
            especialidade_leito,
            diag_princ,
            capitulo_cid,
            carater_internacao,
            complexidade,
            transferido_intermun,
            internacao_fim_de_semana,
            qt_diarias,
            uti_mes_to
        FROM {fonte_cockpit}
        WHERE {where_sql}
    ) AS filtrado
    USING SAMPLE {int(limite)} ROWS (reservoir, 42)
    """
    return con.execute(query, params).df()


def _fonte_registro_detalhado() -> tuple[str, str]:
    if parquet_bruto_disponivel():
        return f"read_parquet('{PARQUET_GLOB}', hive_partitioning=true)", "parquet"
    if AMOSTRA_PATH.exists():
        amostra_str = str(AMOSTRA_PATH).replace("\\", "/")
        return f"read_parquet('{amostra_str}')", "amostra"
    return "", "indisponivel"


def obter_resumo_qualidade(regiao="Todas", uf="Todas") -> dict:
    """Computa cobertura, completude e validade focadas nos dados do Perfil Epidemiológico e Treino da IA."""
    vazio = {
        "total": 0, "ufs_cobertas": 0, "meses_cobertos": 0, "particoes_uf_mes": 0,
        "qtd_raca_nd": 0, "pct_raca_nd": 0.0, "pct_sexo_outro": 0.0,
        "pct_cid_ign": 0.0, "pct_esp_outras": 0.0,
        "amostra_n": 0, "auditoria_fonte": "indisponivel", "null_ia_features": 0,
        "bad_diarias": 0, "bad_idade": 0, "bad_cid": 0, "bad_sexo": 0,
        "null_carater": 0, "null_complex": 0, "null_transf": 0, "null_fds": 0,
        "min_idade": 0.0, "max_idade": 0.0, "min_diarias": 0, "max_diarias": 0,
    }
    if not parquet_disponivel():
        return vazio

    con = _get_con()
    where_sql, params = _build_where(regiao=regiao, uf=uf, mes_range=None)

    if CUBO_PATH.exists():
        cubo_str = str(CUBO_PATH).replace("\\", "/")
        q_cubo = f"""
        SELECT
            CAST(COALESCE(SUM(internacoes), 0) AS BIGINT) AS total,
            COUNT(DISTINCT uf) AS ufs_cobertas,
            COUNT(DISTINCT mes_internacao) AS meses_cobertos,
            COUNT(DISTINCT uf || '-' || CAST(mes_internacao AS VARCHAR)) AS particoes_uf_mes,
            CAST(COALESCE(SUM(CASE WHEN raca_cor_desc = 'Nao_Declarada' THEN internacoes ELSE 0 END), 0) AS BIGINT) AS qtd_raca_nd,
            COALESCE(SUM(CASE WHEN raca_cor_desc = 'Nao_Declarada' THEN internacoes ELSE 0 END) * 100.0 / NULLIF(SUM(internacoes), 0), 0.0) AS pct_raca_nd,
            COALESCE(SUM(CASE WHEN sexo NOT IN ('1', '3') THEN internacoes ELSE 0 END) * 100.0 / NULLIF(SUM(internacoes), 0), 0.0) AS pct_sexo_outro,
            COALESCE(SUM(CASE WHEN capitulo_cid = 'Ignorado' THEN internacoes ELSE 0 END) * 100.0 / NULLIF(SUM(internacoes), 0), 0.0) AS pct_cid_ign,
            COALESCE(SUM(CASE WHEN especialidade_leito = 'Outras' THEN internacoes ELSE 0 END) * 100.0 / NULLIF(SUM(internacoes), 0), 0.0) AS pct_esp_outras
        FROM read_parquet('{cubo_str}')
        WHERE {where_sql}
        """
    else:
        q_cubo = f"""
        SELECT
            COUNT(*) AS total,
            COUNT(DISTINCT uf) AS ufs_cobertas,
            COUNT(DISTINCT mes_internacao) AS meses_cobertos,
            COUNT(DISTINCT uf || '-' || CAST(mes_internacao AS VARCHAR)) AS particoes_uf_mes,
            SUM(CASE WHEN raca_cor_desc = 'Nao_Declarada' THEN 1 ELSE 0 END) AS qtd_raca_nd,
            COALESCE(AVG(CASE WHEN raca_cor_desc = 'Nao_Declarada' THEN 100.0 ELSE 0.0 END), 0.0) AS pct_raca_nd,
            COALESCE(AVG(CASE WHEN sexo NOT IN ('1', '3') THEN 100.0 ELSE 0.0 END), 0.0) AS pct_sexo_outro,
            COALESCE(AVG(CASE WHEN capitulo_cid = 'Ignorado' THEN 100.0 ELSE 0.0 END), 0.0) AS pct_cid_ign,
            COALESCE(AVG(CASE WHEN especialidade_leito = 'Outras' THEN 100.0 ELSE 0.0 END), 0.0) AS pct_esp_outras
        FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=true)
        WHERE {where_sql}
        """
    r_c = con.execute(q_cubo, params).fetchone()

    res = dict(vazio)
    res.update({
        "total": int(r_c[0]),
        "ufs_cobertas": int(r_c[1]),
        "meses_cobertos": int(r_c[2]),
        "particoes_uf_mes": int(r_c[3]),
        "qtd_raca_nd": int(r_c[4]),
        "pct_raca_nd": float(r_c[5]),
        "pct_sexo_outro": float(r_c[6]),
        "pct_cid_ign": float(r_c[7]),
        "pct_esp_outras": float(r_c[8]),
    })

    fonte_det, modo_det = _fonte_registro_detalhado()
    if fonte_det:
        det_clauses = ["1=1"]
        det_params = []
        if regiao and regiao != "Todas":
            det_clauses.append("regiao = ?")
            det_params.append(regiao)
        if uf and uf != "Todas":
            det_clauses.append("uf = ?")
            det_params.append(uf)
        det_where = " AND ".join(det_clauses)
        q_det = f"""
        SELECT
            COUNT(*) AS amostra_n,
            CAST(COALESCE(SUM(
                CASE WHEN idade_anos IS NULL OR sexo IS NULL OR capitulo_cid IS NULL
                          OR especialidade_leito IS NULL OR carater_internacao IS NULL
                          OR complexidade IS NULL OR transferido_intermun IS NULL
                          OR internacao_fim_de_semana IS NULL
                     THEN 1 ELSE 0 END
            ), 0) AS BIGINT) AS null_ia_features,
            CAST(COALESCE(SUM(CASE WHEN qt_diarias <= 0 OR qt_diarias IS NULL THEN 1 ELSE 0 END), 0) AS BIGINT) AS bad_diarias,
            CAST(COALESCE(SUM(CASE WHEN idade_anos < 0 OR idade_anos > 120 OR idade_anos IS NULL THEN 1 ELSE 0 END), 0) AS BIGINT) AS bad_idade,
            CAST(COALESCE(SUM(CASE WHEN diag_princ IS NULL OR TRIM(diag_princ) IN ('', 'None', 'nan') THEN 1 ELSE 0 END), 0) AS BIGINT) AS bad_cid,
            CAST(COALESCE(SUM(CASE WHEN sexo NOT IN ('1', '3') OR sexo IS NULL THEN 1 ELSE 0 END), 0) AS BIGINT) AS bad_sexo,
            CAST(COALESCE(SUM(CASE WHEN carater_internacao IS NULL OR TRIM(CAST(carater_internacao AS VARCHAR)) = '' THEN 1 ELSE 0 END), 0) AS BIGINT) AS null_carater,
            CAST(COALESCE(SUM(CASE WHEN complexidade IS NULL OR TRIM(CAST(complexidade AS VARCHAR)) = '' THEN 1 ELSE 0 END), 0) AS BIGINT) AS null_complex,
            CAST(COALESCE(SUM(CASE WHEN transferido_intermun IS NULL THEN 1 ELSE 0 END), 0) AS BIGINT) AS null_transf,
            CAST(COALESCE(SUM(CASE WHEN internacao_fim_de_semana IS NULL THEN 1 ELSE 0 END), 0) AS BIGINT) AS null_fds,
            COALESCE(MIN(idade_anos), 0.0) AS min_idade,
            COALESCE(MAX(idade_anos), 0.0) AS max_idade,
            COALESCE(MIN(qt_diarias), 0) AS min_diarias,
            COALESCE(MAX(qt_diarias), 0) AS max_diarias
        FROM {fonte_det}
        WHERE {det_where}
        """
        r_d = con.execute(q_det, det_params).fetchone()
        n_det = int(r_d[0])
        res.update({
            "amostra_n": n_det,
            "auditoria_fonte": modo_det,
            "null_ia_features": int(r_d[1]),
            "bad_diarias": int(r_d[2]),
            "bad_idade": int(r_d[3]),
            "bad_cid": int(r_d[4]),
            "bad_sexo": int(r_d[5]),
            "null_carater": int(r_d[6]),
            "null_complex": int(r_d[7]),
            "null_transf": int(r_d[8]),
            "null_fds": int(r_d[9]),
            "min_idade": float(r_d[10]),
            "max_idade": float(r_d[11]),
            "min_diarias": int(r_d[12]),
            "max_diarias": int(r_d[13]),
        })
    return res


def obter_qualidade_por_uf(regiao="Todas") -> pd.DataFrame:
    """Ranking por UF de completude e categorias ignoradas/não declaradas."""
    if not parquet_disponivel():
        return pd.DataFrame()
    where_sql, params = _build_where(regiao=regiao, mes_range=None)
    con = _get_con()
    if CUBO_PATH.exists():
        cubo_str = str(CUBO_PATH).replace("\\", "/")
        query = f"""
        SELECT
            uf,
            regiao,
            CAST(SUM(internacoes) AS BIGINT) AS internacoes,
            COUNT(DISTINCT mes_internacao) AS meses_ativos,
            ROUND(SUM(CASE WHEN raca_cor_desc = 'Nao_Declarada' THEN internacoes ELSE 0 END) * 100.0 / NULLIF(SUM(internacoes), 0), 3) AS pct_raca_nd,
            ROUND(SUM(CASE WHEN capitulo_cid = 'Ignorado' THEN internacoes ELSE 0 END) * 100.0 / NULLIF(SUM(internacoes), 0), 4) AS pct_cid_ign,
            ROUND(SUM(CASE WHEN especialidade_leito = 'Outras' THEN internacoes ELSE 0 END) * 100.0 / NULLIF(SUM(internacoes), 0), 2) AS pct_esp_outras
        FROM read_parquet('{cubo_str}')
        WHERE {where_sql}
        GROUP BY uf, regiao
        ORDER BY pct_raca_nd DESC, pct_esp_outras DESC, internacoes DESC
        """
    else:
        query = f"""
        SELECT
            uf,
            regiao,
            COUNT(*) AS internacoes,
            COUNT(DISTINCT mes_internacao) AS meses_ativos,
            ROUND(AVG(CASE WHEN raca_cor_desc = 'Nao_Declarada' THEN 100.0 ELSE 0.0 END), 3) AS pct_raca_nd,
            ROUND(AVG(CASE WHEN capitulo_cid = 'Ignorado' THEN 100.0 ELSE 0.0 END), 4) AS pct_cid_ign,
            ROUND(AVG(CASE WHEN especialidade_leito = 'Outras' THEN 100.0 ELSE 0.0 END), 2) AS pct_esp_outras
        FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=true)
        WHERE {where_sql}
        GROUP BY uf, regiao
        ORDER BY pct_raca_nd DESC, pct_esp_outras DESC, internacoes DESC
        """
    return con.execute(query, params).df()
