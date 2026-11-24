# -*- coding: utf-8 -*-
"""
Motor Analítico Local via DuckDB sobre Parquet Particionado (Hive: ano/uf/mes).
Permite consultar milhões de internações do SIH/SUS em milissegundos sem carregar tudo na RAM.
"""

from pathlib import Path
import time
import duckdb

BASE_DIR = Path(__file__).resolve().parent.parent
PARQUET_GLOB = str(BASE_DIR / "dados" / "parquet_nacional" / "**" / "*.parquet").replace("\\", "/")


def consultar_resumo():
    arquivos = list((BASE_DIR / "dados" / "parquet_nacional").rglob("*.parquet"))
    if not arquivos:
        print("Nenhum arquivo Parquet encontrado em dados/parquet_nacional/.")
        return

    tamanho_mb = sum(f.stat().st_size for f in arquivos) / (1024 * 1024)
    print("=" * 75)
    print(f"🦆 CONSULTA ANALÍTICA DUCKDB | {len(arquivos)} partições Parquet ({tamanho_mb:.2f} MB em disco)")
    print("=" * 75)

    con = duckdb.connect(database=":memory:")
    t0 = time.time()

    # 1. Resumo por UF
    query_uf = f"""
    SELECT
        uf,
        regiao,
        COUNT(*) AS total_internacoes,
        ROUND(AVG(Obito) * 100, 2) AS taxa_obito_pct,
        ROUND(AVG(Longa_Permanencia) * 100, 2) AS taxa_longa_pct,
        ROUND(AVG(qt_diarias), 2) AS media_diarias,
        ROUND(SUM(val_tot) / 1e6, 2) AS gasto_milhoes_brl
    FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=true)
    GROUP BY uf, regiao
    ORDER BY total_internacoes DESC
    """
    df_uf = con.execute(query_uf).df()
    print("\n1. RESUMO POR ESTADO (UF):")
    print(df_uf.to_string(index=False))

    # 2. Top 5 Capítulos CID-10
    query_cid = f"""
    SELECT
        capitulo_cid,
        COUNT(*) AS internacoes,
        ROUND(AVG(Obito) * 100, 2) AS obito_pct,
        ROUND(AVG(Longa_Permanencia) * 100, 2) AS longa_pct
    FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=true)
    GROUP BY capitulo_cid
    ORDER BY internacoes DESC
    LIMIT 6
    """
    df_cid = con.execute(query_cid).df()
    print("\n2. TOP 6 CAPÍTULOS CID-10:")
    print(df_cid.to_string(index=False))

    # 3. Equidade por Raça/Cor
    query_raca = f"""
    SELECT
        raca_cor_desc,
        COUNT(*) AS internacoes,
        ROUND(AVG(Obito) * 100, 2) AS obito_pct,
        ROUND(AVG(qt_diarias), 2) AS media_diarias
    FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=true)
    GROUP BY raca_cor_desc
    ORDER BY internacoes DESC
    """
    df_raca = con.execute(query_raca).df()
    print("\n3. EQUIDADE POR RAÇA/COR:")
    print(df_raca.to_string(index=False))

    print(f"\n⚡ Consultas executadas via DuckDB em {time.time() - t0:.3f}s!")
    print("=" * 75)


if __name__ == "__main__":
    consultar_resumo()
