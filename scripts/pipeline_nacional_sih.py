# -*- coding: utf-8 -*-
"""
Pipeline Escalável de Ingestão e Transformação Nacional — SIH/SUS (27 UFs)
Fase 2: Download via FTP DataSUS -> Leitura Arrow (datasuspy) ->
        Padronização Clínica & Anti-Leakage -> Parquet Particionado (Hive: ano/uf/mes).

Uso:
  # Testar em piloto (ex: Acre local já existente + Rondônia meses 1 e 2):
  python scripts/pipeline_nacional_sih.py --ufs AC RO --meses 1 2

  # Rodar Brasil inteiro (27 UFs x 12 meses = 324 arquivos):
  python scripts/pipeline_nacional_sih.py --ano 2024
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import ftplib
import os
import shutil
import threading
import time
from pathlib import Path

PRINT_LOCK = threading.Lock()

def log_safe(msg: str):
    with PRINT_LOCK:
        print(msg, flush=True)

import datasuspy
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).resolve().parent.parent
DADOS_DIR = BASE_DIR / "dados"
PARQUET_DIR = DADOS_DIR / "parquet_nacional"
TMP_DBC_DIR = DADOS_DIR / "tmp_dbc"

FTP_HOST = "ftp.datasus.gov.br"
FTP_DIR = "/dissemin/publicos/SIHSUS/200801_/Dados/"

TODAS_UFS = [
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO",
    "MA", "MG", "MS", "MT", "PA", "PB", "PE", "PI", "PR",
    "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO"
]

REGIOES_UF = {
    "AC": "Norte", "AM": "Norte", "AP": "Norte", "PA": "Norte", "RO": "Norte", "RR": "Norte", "TO": "Norte",
    "AL": "Nordeste", "BA": "Nordeste", "CE": "Nordeste", "MA": "Nordeste", "PB": "Nordeste",
    "PE": "Nordeste", "PI": "Nordeste", "RN": "Nordeste", "SE": "Nordeste",
    "DF": "Centro-Oeste", "GO": "Centro-Oeste", "MS": "Centro-Oeste", "MT": "Centro-Oeste",
    "ES": "Sudeste", "MG": "Sudeste", "RJ": "Sudeste", "SP": "Sudeste",
    "PR": "Sul", "RS": "Sul", "SC": "Sul"
}

COLUNAS_BRUTAS_ESSENCIAIS = [
    "n_aih", "cnes", "munic_res", "munic_mov", "dt_inter", "dt_saida",
    "diag_princ", "espec", "car_int", "complex", "sexo", "idade", "cod_idade",
    "raca_cor", "qt_diarias", "morte", "uti_mes_to", "val_tot", "mes_cmpt", "ano_cmpt"
]

MAPA_ESPEC = {
    1: "Cirurgica",
    2: "Obstetrica",
    3: "Clinica_Medica",
    4: "Pediatrica",
    5: "Hospital_Dia",
    7: "Psiquiatrica"
}

MAPA_CAR_INT = {
    1: "Eletiva",
    2: "Urgencia_Emergencia"
}

MAPA_COMPLEX = {
    2: "Media_Complexidade",
    3: "Alta_Complexidade"
}

MAPA_RACA = {
    1: "Branca",
    2: "Preta",
    3: "Parda",
    4: "Amarela",
    5: "Indigena",
    99: "Nao_Declarada",
    0: "Nao_Declarada"
}


def capitulo_cid(cid: str) -> str:
    if pd.isna(cid) or str(cid).strip() == "":
        return "Ignorado"
    letra = str(cid).strip()[0].upper()
    mapa = {
        "A": "I - Infecciosas", "B": "I - Infecciosas",
        "C": "II - Neoplasias", "D": "III - Sangue",
        "E": "IV - Endocrinas", "F": "V - Mentais",
        "G": "VI - Nervoso", "H": "VII/VIII - Olho/Ouvido",
        "I": "IX - Circulatorio", "J": "X - Respiratorio",
        "K": "XI - Digestivo", "L": "XII - Pele",
        "M": "XIII - Osteomuscular", "N": "XIV - Geniturinario",
        "O": "XV - Gravidez/Parto", "P": "XVI - Perinatal",
        "Q": "XVII - Congenitas", "R": "XVIII - Sintomas",
        "S": "XIX - Lesoes", "T": "XIX - Lesoes",
        "V": "XX - Causas externas", "W": "XX - Causas externas",
        "X": "XX - Causas externas", "Y": "XX - Causas externas",
        "Z": "XXI - Fatores de saude",
    }
    return mapa.get(letra, "Ignorado")


def idade_para_anos_vec(idade: pd.Series, cod: pd.Series) -> pd.Series:
    """Cálculo vetorizado de idade em anos contínuos (alta performance para milhões de linhas)."""
    res = pd.Series(np.nan, index=idade.index, dtype="float64")
    mask_anos = cod.isin([0, 4, 5])
    mask_meses = cod == 1
    mask_dias = cod == 2
    mask_horas = cod == 3

    res.loc[mask_anos] = idade.loc[mask_anos]
    res.loc[mask_meses] = idade.loc[mask_meses] / 12.0
    res.loc[mask_dias] = idade.loc[mask_dias] / 365.25
    res.loc[mask_horas] = idade.loc[mask_horas] / (365.25 * 24.0)
    return res


def baixar_dbc(nome_arquivo: str, destino: Path, tentativas: int = 4) -> bool:
    """Baixa arquivo .dbc do FTP oficial do DataSUS com retry e backoff (escrita atômica)."""
    destino_tmp = destino.with_suffix(".dbc.tmp")
    for tentativa in range(1, tentativas + 1):
        try:
            with ftplib.FTP(FTP_HOST, timeout=120) as ftp:
                ftp.login()
                ftp.cwd(FTP_DIR)
                with open(destino_tmp, "wb") as f:
                    ftp.retrbinary(f"RETR {nome_arquivo}", f.write)
            destino_tmp.replace(destino)
            return True
        except Exception as e:
            if destino_tmp.exists():
                destino_tmp.unlink()
            log_safe(f"    [!] Tentativa {tentativa}/{tentativas} falhou para {nome_arquivo}: {e}")
            time.sleep(3 * tentativa)
    return False


def transformar_para_parquet(caminho_dbc: Path, uf: str, ano: int, mes: int, destino_parquet: Path) -> int:
    """Lê .dbc via Arrow, seleciona colunas essenciais, aplica regras V3 e grava Parquet ZSTD."""
    tabela_arrow = datasuspy.read_dbc(str(caminho_dbc))

    # Normalizar nomes de colunas para minúsculo no Arrow antes de converter
    tabela_arrow = tabela_arrow.rename_columns([c.lower() for c in tabela_arrow.column_names])
    colunas_presentes = [c for c in COLUNAS_BRUTAS_ESSENCIAIS if c in tabela_arrow.column_names]
    tabela_enxuta = tabela_arrow.select(colunas_presentes)
    df = tabela_enxuta.to_pandas()

    # Garantir colunas ausentes caso algum estado tenha schema diferente
    for col in COLUNAS_BRUTAS_ESSENCIAIS:
        if col not in df.columns:
            df[col] = np.nan

    # Conversões numéricas
    cols_num = ["qt_diarias", "idade", "cod_idade", "morte", "car_int", "espec", "complex", "raca_cor", "uti_mes_to", "val_tot"]
    for col in cols_num:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["idade_anos"] = idade_para_anos_vec(df["idade"], df["cod_idade"])

    # Filtrar registros inválidos (mesmo critério do piloto AC)
    df = df.dropna(subset=["qt_diarias", "morte", "idade_anos"]).copy()
    df = df[df["qt_diarias"] > 0].copy()

    # Padronização de features clínicas e epidemiológicas
    df["regiao"] = REGIOES_UF.get(uf, "Desconhecida")
    df["capitulo_cid"] = df["diag_princ"].apply(capitulo_cid)
    df["diag_princ"] = df["diag_princ"].astype(str).str.strip()
    df["especialidade_leito"] = df["espec"].map(MAPA_ESPEC).fillna("Outras")
    df["carater_internacao"] = df["car_int"].map(MAPA_CAR_INT).fillna("Urgencia_Emergencia")
    df["complexidade"] = df["complex"].map(MAPA_COMPLEX).fillna("Media_Complexidade")
    df["raca_cor_desc"] = df["raca_cor"].map(MAPA_RACA).fillna("Nao_Declarada")
    df["sexo"] = df["sexo"].astype(str).str.strip()
    df["munic_res"] = df["munic_res"].astype(str).str.strip()
    df["munic_mov"] = df["munic_mov"].astype(str).str.strip()
    df["transferido_intermun"] = (df["munic_res"] != df["munic_mov"]).astype("int8")

    dt_inter = pd.to_datetime(df["dt_inter"], format="%Y%m%d", errors="coerce")
    df["mes_internacao"] = dt_inter.dt.month.fillna(mes).astype("int8")
    df["internacao_fim_de_semana"] = (dt_inter.dt.day_of_week >= 5).astype("int8")

    # Targets
    df["Longa_Permanencia"] = (df["qt_diarias"] >= 15).astype("int8")
    df["Obito"] = (df["morte"] == 1).astype("int8")

    # Tipos compactos para economia extrema de memória e disco
    df["n_aih"] = df["n_aih"].astype(str).str.strip()
    df["cnes"] = df["cnes"].astype(str).str.strip()
    df["qt_diarias"] = df["qt_diarias"].astype("int32")
    df["uti_mes_to"] = df["uti_mes_to"].fillna(0).astype("int16")
    df["val_tot"] = df["val_tot"].fillna(0.0).astype("float32")
    df["idade_anos"] = df["idade_anos"].astype("float32")

    colunas_saida = [
        "n_aih", "cnes", "regiao", "munic_res", "munic_mov",
        "idade_anos", "sexo", "raca_cor_desc", "diag_princ", "capitulo_cid",
        "especialidade_leito", "carater_internacao", "complexidade",
        "transferido_intermun", "internacao_fim_de_semana", "mes_internacao",
        "qt_diarias", "uti_mes_to", "val_tot", "Longa_Permanencia", "Obito"
    ]

    df_out = df[colunas_saida]
    destino_parquet.parent.mkdir(parents=True, exist_ok=True)
    tmp_parquet = destino_parquet.with_suffix(".parquet.tmp")
    tabela_final = pa.Table.from_pandas(df_out, preserve_index=False)
    pq.write_table(tabela_final, str(tmp_parquet), compression="zstd")
    tmp_parquet.replace(destino_parquet)

    return len(df_out)


def processar_unidade(uf: str, ano: int, mes: int, manter_dbc: bool = False) -> tuple[str, int, str, int]:
    """Executa download (ou reuso local) e conversão de um arquivo UF/Mês."""
    aa = str(ano)[-2:]
    nome_dbc = f"RD{uf}{aa}{mes:02d}.dbc"
    destino_parquet = PARQUET_DIR / f"ano={ano}" / f"uf={uf}" / f"mes={mes:02d}" / "dados.parquet"

    if destino_parquet.exists():
        return ("PULADO", 0, uf, mes)

    # Verificar se .dbc já existe em dados/ (ex: arquivos AC já baixados) ou tmp_dbc/
    dbc_local_dados = DADOS_DIR / nome_dbc
    dbc_tmp = TMP_DBC_DIR / nome_dbc
    baixado_agora = False

    if dbc_local_dados.exists():
        caminho_dbc = dbc_local_dados
    elif dbc_tmp.exists():
        caminho_dbc = dbc_tmp
    else:
        TMP_DBC_DIR.mkdir(parents=True, exist_ok=True)
        ok = baixar_dbc(nome_dbc, dbc_tmp)
        if not ok:
            return ("ERRO_FTP", 0, uf, mes)
        caminho_dbc = dbc_tmp
        baixado_agora = True

    try:
        qtd = transformar_para_parquet(caminho_dbc, uf, ano, mes, destino_parquet)
        if baixado_agora and not manter_dbc and dbc_tmp.exists():
            dbc_tmp.unlink()
        return ("OK", qtd, uf, mes)
    except Exception as e:
        log_safe(f"    [!] Erro ao processar {nome_dbc}: {e}")
        return ("ERRO_PROC", 0, uf, mes)


def main():
    parser = argparse.ArgumentParser(description="Pipeline SIH/SUS Nacional -> Parquet Particionado")
    parser.add_argument("--ufs", nargs="+", default=TODAS_UFS, help="Lista de siglas de UFs (padrão: todas 27)")
    parser.add_argument("--ano", type=int, default=2024, help="Ano de competência (padrão: 2024)")
    parser.add_argument("--meses", nargs="+", type=int, default=list(range(1, 13)), help="Meses 1..12")
    parser.add_argument("--workers", type=int, default=4, help="Número de downloads/conversões simultâneos (padrão: 4)")
    parser.add_argument("--manter-dbc", action="store_true", help="Manter arquivos .dbc baixados após conversão")
    args = parser.parse_args()

    ufs = [u.upper() for u in args.ufs]
    tarefas = [(uf, mes) for uf in ufs for mes in args.meses]
    total_tarefas = len(tarefas)
    print("=" * 70, flush=True)
    print(f"PIPELINE NACIONAL SIH/SUS {args.ano} -> PARQUET HIVE (Workers: {args.workers})", flush=True)
    print(f"UFs selecionadas ({len(ufs)}): {', '.join(ufs)}", flush=True)
    print(f"Meses: {args.meses} | Total de arquivos-alvo: {total_tarefas}", flush=True)
    print(f"Destino Parquet: {PARQUET_DIR}", flush=True)
    print("=" * 70, flush=True)

    total_registros = 0
    stats = {"OK": 0, "PULADO": 0, "ERRO_FTP": 0, "ERRO_PROC": 0}
    inicio = time.time()
    concluidos_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futuros = {
            executor.submit(processar_unidade, uf, args.ano, mes, args.manter_dbc): (uf, mes)
            for uf, mes in tarefas
        }
        for futuro in as_completed(futuros):
            status, qtd, uf, mes = futuro.result()
            stats[status] += 1
            total_registros += qtd
            concluidos_count += 1
            if status == "OK":
                log_safe(f"  [{concluidos_count}/{total_tarefas}] ✓ {uf} {args.ano}/{mes:02d}: {qtd:,} registros salvos em Parquet.")
            elif status == "PULADO":
                log_safe(f"  [{concluidos_count}/{total_tarefas}] ⏭ {uf} {args.ano}/{mes:02d}: Parquet já existente.")
            else:
                log_safe(f"  [{concluidos_count}/{total_tarefas}] ✗ {uf} {args.ano}/{mes:02d}: {status}")

    # Limpar diretório temporário se vazio
    if TMP_DBC_DIR.exists() and not any(TMP_DBC_DIR.iterdir()):
        shutil.rmtree(TMP_DBC_DIR, ignore_errors=True)

    duracao = time.time() - inicio
    print("\n" + "=" * 70)
    print("RESUMO DA EXECUÇÃO")
    print(f"Concluídos: {stats['OK']} | Pulados: {stats['PULADO']} | Erros FTP: {stats['ERRO_FTP']} | Erros Proc: {stats['ERRO_PROC']}")
    print(f"Novos registros processados: {total_registros:,}")
    print(f"Tempo total: {duracao:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
