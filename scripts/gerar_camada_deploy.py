# -*- coding: utf-8 -*-
"""
Gerador da Camada Leve de Deploy Multi-Ano (Nuvem / Streamlit Cloud):
Lê todas as partições Hive (`ano=*/uf=*/mes=*/dados.parquet`) via DuckDB e gera:
1. dados/deploy/cubo_epidemiologico_brasil_2024.parquet (~4-12 MB):
   Agregado exato multi-ano preservando 100% das contagens de UFs, anos, meses e filtros da Aba 1.
2. dados/deploy/amostra_cockpit_brasil_2024.parquet (~0.35 MB):
   Amostra representativa de 30.000 internações reais multi-ano (inclui `ano` e `Obito`) para o Cockpit (Aba 4).
"""

import time
from pathlib import Path
import duckdb

BASE_DIR = Path(__file__).resolve().parent.parent
PARQUET_GLOB = str(BASE_DIR / "dados" / "parquet_nacional" / "**" / "*.parquet").replace("\\", "/")
DEPLOY_DIR = BASE_DIR / "dados" / "deploy"
DEPLOY_DIR.mkdir(parents=True, exist_ok=True)


def gerar():
    t0 = time.time()
    con = duckdb.connect(database=":memory:")

    cubo_path = str(DEPLOY_DIR / "cubo_epidemiologico_brasil_2024.parquet").replace("\\", "/")
    amostra_path = str(DEPLOY_DIR / "amostra_cockpit_brasil_2024.parquet").replace("\\", "/")

    print("1. Gerando cubo epidemiológico exato multi-ano -> agregado...")
    q_cubo = f"""
    COPY (
        SELECT
            CAST(ano AS INTEGER) AS ano,
            regiao,
            uf,
            raca_cor_desc,
            especialidade_leito,
            sexo,
            mes_internacao,
            CASE
                WHEN idade_anos < 1 THEN '<1 ano'
                WHEN idade_anos < 12 THEN '1-11 anos'
                WHEN idade_anos < 19 THEN '12-18 anos'
                WHEN idade_anos < 40 THEN '19-39 anos'
                WHEN idade_anos < 60 THEN '40-59 anos'
                WHEN idade_anos < 80 THEN '60-79 anos'
                ELSE '80+ anos'
            END AS faixa_etaria,
            capitulo_cid,
            COUNT(*) AS internacoes,
            SUM(Obito) AS obitos,
            SUM(Longa_Permanencia) AS longas,
            SUM(qt_diarias) AS soma_diarias,
            SUM(val_tot) AS soma_val_tot
        FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=true)
        GROUP BY 1, 2, 3, 4, 5, 6, 7, 8, 9
    ) TO '{cubo_path}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    con.execute(q_cubo)

    print("2. Gerando amostra estratificada de 30.000 internações multi-ano para o Cockpit Clínico...")
    q_amostra = f"""
    COPY (
        SELECT
            CAST(ano AS INTEGER) AS ano,
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
            uti_mes_to,
            Obito
        FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=true)
        USING SAMPLE 30000 ROWS (reservoir, 42)
    ) TO '{amostra_path}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    con.execute(q_amostra)

    # Auditoria de paridade multi-ano
    row_raw = con.execute(
        f"SELECT COUNT(*), CAST(SUM(Obito) AS BIGINT), CAST(SUM(Longa_Permanencia) AS BIGINT) FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=true)"
    ).fetchone()
    row_cubo = con.execute(
        f"SELECT CAST(SUM(internacoes) AS BIGINT), CAST(SUM(obitos) AS BIGINT), CAST(SUM(longas) AS BIGINT), COUNT(*) FROM read_parquet('{cubo_path}')"
    ).fetchone()
    row_amostra = con.execute(
        f"SELECT COUNT(*), COUNT(DISTINCT uf), COUNT(DISTINCT ano) FROM read_parquet('{amostra_path}')"
    ).fetchone()
    por_ano = con.execute(
        f"""
        SELECT ano, CAST(SUM(internacoes) AS BIGINT) AS n, CAST(SUM(obitos) AS BIGINT) AS ob, CAST(SUM(longas) AS BIGINT) AS lg
        FROM read_parquet('{cubo_path}')
        GROUP BY ano ORDER BY ano
        """
    ).fetchall()

    mb_cubo = (DEPLOY_DIR / "cubo_epidemiologico_brasil_2024.parquet").stat().st_size / (1024 * 1024)
    mb_amostra = (DEPLOY_DIR / "amostra_cockpit_brasil_2024.parquet").stat().st_size / (1024 * 1024)

    print("=" * 78)
    print(f"PARIDADE BRUTO vs CUBO ({time.time()-t0:.2f}s):")
    print(f"  Bruto Total: Internações={row_raw[0]:,} | Óbitos={row_raw[1]:,} | Longas={row_raw[2]:,}")
    print(f"  Cubo Deploy: Internações={row_cubo[0]:,} | Óbitos={row_cubo[1]:,} | Longas={row_cubo[2]:,} (Linhas no cubo: {row_cubo[3]:,})")
    for a, n, ob, lg in por_ano:
        print(f"    • Ano {a}: {n:,} internações | {ob:,} óbitos ({ob*100/n:.2f}%) | {lg:,} longas ({lg*100/n:.2f}%)")
    print(f"  Amostra Cockpit: {row_amostra[0]:,} casos | {row_amostra[1]} UFs | {row_amostra[2]} anos")
    print(f"  Tamanho em disco: Cubo = {mb_cubo:.2f} MB | Amostra = {mb_amostra:.2f} MB (Total: {mb_cubo+mb_amostra:.2f} MB)")
    print("=" * 78)
    assert row_raw == row_cubo[:3], "ERRO DE PARIDADE ENTRE BRUTO E CUBO!"


if __name__ == "__main__":
    gerar()
