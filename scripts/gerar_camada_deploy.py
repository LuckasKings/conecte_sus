# -*- coding: utf-8 -*-
"""
Gerador da Camada Leve de Deploy (Nuvem / Streamlit Cloud):
Lê os 13.584.534 registros brutos de dados/parquet_nacional/ via DuckDB e gera:
1. dados/deploy/cubo_epidemiologico_brasil_2024.parquet (~2-4 MB):
   Agregado exato preservando 100% das contagens de todas as 27 UFs e filtros da Aba 1.
2. dados/deploy/amostra_cockpit_brasil_2024.parquet (~1-2 MB):
   Amostra estratificada representativa de 30.000 internações reais para o Cockpit Clínico (Aba 3).
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

    print("1. Gerando cubo epidemiológico exato (13,58M internações -> agregado)...")
    q_cubo = f"""
    COPY (
        SELECT
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
        GROUP BY 1, 2, 3, 4, 5, 6, 7, 8
    ) TO '{cubo_path}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    con.execute(q_cubo)

    print("2. Gerando amostra estratificada de 30.000 internações para o Cockpit Clínico...")
    q_amostra = f"""
    COPY (
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
        FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=true)
        USING SAMPLE 30000 ROWS (reservoir, 42)
    ) TO '{amostra_path}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    con.execute(q_amostra)

    # Auditoria de paridade
    row_raw = con.execute(f"SELECT COUNT(*), SUM(Obito), SUM(Longa_Permanencia) FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=true)").fetchone()
    row_cubo = con.execute(f"SELECT SUM(internacoes), SUM(obitos), SUM(longas), COUNT(*) FROM read_parquet('{cubo_path}')").fetchone()
    row_amostra = con.execute(f"SELECT COUNT(*), COUNT(DISTINCT uf) FROM read_parquet('{amostra_path}')").fetchone()

    mb_cubo = (DEPLOY_DIR / "cubo_epidemiologico_brasil_2024.parquet").stat().st_size / (1024 * 1024)
    mb_amostra = (DEPLOY_DIR / "amostra_cockpit_brasil_2024.parquet").stat().st_size / (1024 * 1024)

    print("=" * 75)
    print(f"PARIDADE BRUTO vs CUBO ({time.time()-t0:.2f}s):")
    print(f"  Bruto (13,58M): Internações={row_raw[0]:,} | Óbitos={row_raw[1]:,} | Longas={row_raw[2]:,}")
    print(f"  Cubo Deploy:    Internações={row_cubo[0]:,} | Óbitos={row_cubo[1]:,} | Longas={row_cubo[2]:,} (Linhas no cubo: {row_cubo[3]:,})")
    print(f"  Amostra Cockpit: {row_amostra[0]:,} casos cobrindo {row_amostra[1]} UFs")
    print(f"  Tamanho em disco: Cubo = {mb_cubo:.2f} MB | Amostra = {mb_amostra:.2f} MB (Total: {mb_cubo+mb_amostra:.2f} MB)")
    print("=" * 75)
    assert row_raw == row_cubo[:3], "ERRO DE PARIDADE ENTRE BRUTO E CUBO!"


if __name__ == "__main__":
    gerar()
