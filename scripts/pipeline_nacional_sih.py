# -*- coding: utf-8 -*-
"""
Pipeline Incremental Multi-Ano de Ingestão Nacional — SIH/SUS (27 UFs)
- Suporta múltiplos anos (ex.: --anos 2024 2025 2026) via FTP oficial DataSUS.
- Controle estritamente incremental e idempotente por manifesto (`dados/historico_cargas.json`)
  e partição Hive (`ano/uf/mes`), além de deduplicação pela chave natural (`n_aih`, `dt_inter`, `cnes`).
- Grava resumo auditado da última carga em `dados/deploy/resumo_ultima_carga.json` para a Aba 2.

Uso:
  python scripts/pipeline_nacional_sih.py --anos 2025 2026 --workers 6
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import ftplib
import json
import re
import shutil
import threading
import time
from pathlib import Path

import datasuspy
import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PRINT_LOCK = threading.Lock()


def log_safe(msg: str):
    with PRINT_LOCK:
        print(msg, flush=True)


BASE_DIR = Path(__file__).resolve().parent.parent
DADOS_DIR = BASE_DIR / "dados"
PARQUET_DIR = DADOS_DIR / "parquet_nacional"
TMP_DBC_DIR = DADOS_DIR / "tmp_dbc"
DEPLOY_DIR = DADOS_DIR / "deploy"
HISTORICO_PATH = DADOS_DIR / "historico_cargas.json"
RESUMO_CARGA_PATH = DEPLOY_DIR / "resumo_ultima_carga.json"

FTP_HOST = "ftp.datasus.gov.br"
FTP_DIR = "/dissemin/publicos/SIHSUS/200801_/Dados/"

TODAS_UFS = [
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO",
    "MA", "MG", "MS", "MT", "PA", "PB", "PE", "PI", "PR",
    "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO",
]

REGIOES_UF = {
    "AC": "Norte", "AM": "Norte", "AP": "Norte", "PA": "Norte", "RO": "Norte", "RR": "Norte", "TO": "Norte",
    "AL": "Nordeste", "BA": "Nordeste", "CE": "Nordeste", "MA": "Nordeste", "PB": "Nordeste",
    "PE": "Nordeste", "PI": "Nordeste", "RN": "Nordeste", "SE": "Nordeste",
    "DF": "Centro-Oeste", "GO": "Centro-Oeste", "MS": "Centro-Oeste", "MT": "Centro-Oeste",
    "ES": "Sudeste", "MG": "Sudeste", "RJ": "Sudeste", "SP": "Sudeste",
    "PR": "Sul", "RS": "Sul", "SC": "Sul",
}

COLUNAS_BRUTAS_ESSENCIAIS = [
    "n_aih", "cnes", "munic_res", "munic_mov", "dt_inter", "dt_saida",
    "diag_princ", "espec", "car_int", "complex", "sexo", "idade", "cod_idade",
    "raca_cor", "qt_diarias", "morte", "uti_mes_to", "val_tot", "mes_cmpt", "ano_cmpt",
]

MAPA_ESPEC = {
    1: "Cirurgica",
    2: "Obstetrica",
    3: "Clinica_Medica",
    4: "Pediatrica",
    5: "Hospital_Dia",
    7: "Psiquiatrica",
}

MAPA_CAR_INT = {
    1: "Eletiva",
    2: "Urgencia_Emergencia",
}

MAPA_COMPLEX = {
    2: "Media_Complexidade",
    3: "Alta_Complexidade",
}

MAPA_RACA = {
    1: "Branca",
    2: "Preta",
    3: "Parda",
    4: "Amarela",
    5: "Indigena",
    99: "Nao_Declarada",
    0: "Nao_Declarada",
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
    """Cálculo vetorizado de idade em anos contínuos."""
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


def _write_json_atomic(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def _carregar_historico() -> dict:
    if HISTORICO_PATH.exists():
        try:
            with open(HISTORICO_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "particoes" in data:
                data.setdefault("execucoes", [])
                return data
        except Exception:
            pass
    return {"particoes": {}, "execucoes": []}


def sincronizar_manifesto_com_disco(historico: dict) -> int:
    """Registra partições Parquet já existentes no disco ainda ausentes do manifesto."""
    particoes = historico["particoes"]
    novas_idx = 0
    padrao = re.compile(r"ano=(\d{4})[/\\]uf=([A-Z]{2})[/\\]mes=(\d{2})[/\\]dados\.parquet$")
    pendentes = []
    if PARQUET_DIR.exists():
        for pq_path in PARQUET_DIR.rglob("dados.parquet"):
            m = padrao.search(str(pq_path))
            if not m:
                continue
            ano, uf, mes = int(m.group(1)), m.group(2), int(m.group(3))
            chave = f"{ano}-{uf}-{mes:02d}"
            if chave not in particoes:
                pendentes.append((chave, ano, uf, mes, pq_path))

    if not pendentes:
        return 0

    con = duckdb.connect(database=":memory:")
    for chave, ano, uf, mes, pq_path in pendentes:
        pq_sql = str(pq_path).replace("\\", "/")
        row = con.execute(
            f"""
            SELECT
                COUNT(*) AS n,
                CAST(COALESCE(SUM(Obito), 0) AS BIGINT) AS obitos,
                CAST(COALESCE(SUM(Longa_Permanencia), 0) AS BIGINT) AS longas
            FROM read_parquet('{pq_sql}')
            """
        ).fetchone()
        n_reg = int(row[0])
        particoes[chave] = {
            "ano": ano,
            "uf": uf,
            "mes": mes,
            "arquivo_dbc": f"RD{uf}{str(ano)[-2:]}{mes:02d}.dbc",
            "n_bruto": n_reg,
            "n_aceito": n_reg,
            "n_descartado_filtro": 0,
            "n_dedup": 0,
            "obitos": int(row[1]),
            "longas": int(row[2]),
            "origem": "indexado_disco",
            "data_ingestao_utc": datetime.now(timezone.utc).isoformat(),
        }
        novas_idx += 1
    con.close()
    return novas_idx


def listar_arquivos_ftp_disponiveis() -> set[str] | None:
    """Consulta única no FTP do DataSUS para descobrir competências realmente publicadas."""
    try:
        with ftplib.FTP(FTP_HOST, timeout=60) as ftp:
            ftp.login()
            ftp.cwd(FTP_DIR)
            nomes = ftp.nlst("RD*.dbc")
        return {n.rsplit("/", 1)[-1].upper() for n in nomes}
    except Exception as e:
        log_safe(f"  [!] Aviso: falha ao pré-listar diretório FTP ({e}). Seguindo com tentativa direta.")
        return None


def baixar_dbc(nome_arquivo: str, destino: Path, tentativas: int = 4) -> bool:
    """Baixa arquivo .dbc do FTP oficial do DataSUS com retry e escrita atômica."""
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


def transformar_para_parquet(caminho_dbc: Path, uf: str, ano: int, mes: int, destino_parquet: Path) -> dict:
    """Lê .dbc via Arrow, audita qualidade/dedup, aplica regras clínicas e grava Parquet ZSTD."""
    tabela_arrow = datasuspy.read_dbc(str(caminho_dbc))
    tabela_arrow = tabela_arrow.rename_columns([c.lower() for c in tabela_arrow.column_names])
    colunas_presentes = [c for c in COLUNAS_BRUTAS_ESSENCIAIS if c in tabela_arrow.column_names]
    df = tabela_arrow.select(colunas_presentes).to_pandas()

    for col in COLUNAS_BRUTAS_ESSENCIAIS:
        if col not in df.columns:
            df[col] = np.nan

    n_bruto = len(df)
    cols_num = ["qt_diarias", "idade", "cod_idade", "morte", "car_int", "espec", "complex", "raca_cor", "uti_mes_to", "val_tot"]
    for col in cols_num:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["idade_anos"] = idade_para_anos_vec(df["idade"], df["cod_idade"])

    mask_idade_val = df["idade_anos"].notna() & (df["idade_anos"] >= 0) & (df["idade_anos"] <= 120)
    mask_diarias_val = df["qt_diarias"].notna() & (df["qt_diarias"] > 0)
    mask_morte_val = df["morte"].notna()
    bad_idade_pre = int((~mask_idade_val).sum())
    bad_diarias_pre = int((~mask_diarias_val).sum())

    df = df.loc[mask_idade_val & mask_diarias_val & mask_morte_val].copy()
    n_pos_filtro = len(df)
    n_descartado_filtro = int(n_bruto - n_pos_filtro)

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

    dt_inter_raw = df["dt_inter"].astype(str).str.strip()
    dt_inter = pd.to_datetime(dt_inter_raw, format="%Y%m%d", errors="coerce")
    df["mes_internacao"] = dt_inter.dt.month.fillna(mes).astype("int8")
    df["internacao_fim_de_semana"] = (dt_inter.dt.day_of_week >= 5).astype("int8")

    df["Longa_Permanencia"] = (df["qt_diarias"] >= 15).astype("int8")
    df["Obito"] = (df["morte"] == 1).astype("int8")

    df["n_aih"] = df["n_aih"].astype(str).str.strip()
    df["cnes"] = df["cnes"].astype(str).str.strip()
    df["_dt_inter_key"] = dt_inter_raw

    # Deduplicação por chave natural dentro da partição mensal
    antes_dedup = len(df)
    df = df.drop_duplicates(subset=["n_aih", "_dt_inter_key", "cnes"], keep="last").copy()
    n_dedup = int(antes_dedup - len(df))

    df["qt_diarias"] = df["qt_diarias"].astype("int32")
    df["uti_mes_to"] = df["uti_mes_to"].fillna(0).astype("int16")
    df["val_tot"] = df["val_tot"].fillna(0.0).astype("float32")
    df["idade_anos"] = df["idade_anos"].astype("float32")

    colunas_saida = [
        "n_aih", "cnes", "regiao", "munic_res", "munic_mov",
        "idade_anos", "sexo", "raca_cor_desc", "diag_princ", "capitulo_cid",
        "especialidade_leito", "carater_internacao", "complexidade",
        "transferido_intermun", "internacao_fim_de_semana", "mes_internacao",
        "qt_diarias", "uti_mes_to", "val_tot", "Longa_Permanencia", "Obito",
    ]
    df_out = df[colunas_saida]

    destino_parquet.parent.mkdir(parents=True, exist_ok=True)
    tmp_parquet = destino_parquet.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pandas(df_out, preserve_index=False), str(tmp_parquet), compression="zstd")
    tmp_parquet.replace(destino_parquet)

    return {
        "n_bruto": int(n_bruto),
        "n_aceito": int(len(df_out)),
        "n_descartado_filtro": n_descartado_filtro,
        "n_dedup": n_dedup,
        "obitos": int(df_out["Obito"].sum()),
        "longas": int(df_out["Longa_Permanencia"].sum()),
        "qtd_raca_nd": int((df_out["raca_cor_desc"] == "Nao_Declarada").sum()),
        "qtd_esp_outras": int((df_out["especialidade_leito"] == "Outras").sum()),
        "qtd_cid_ign": int((df_out["capitulo_cid"] == "Ignorado").sum()),
        "bad_sexo": int((~df_out["sexo"].isin(["1", "3"])).sum()),
        "bad_idade_pre": bad_idade_pre,
        "bad_diarias_pre": bad_diarias_pre,
    }


def processar_unidade(
    uf: str,
    ano: int,
    mes: int,
    particoes_existentes: set[str],
    ftp_disponiveis: set[str] | None,
    manter_dbc: bool = False,
) -> dict:
    """Executa carga estritamente incremental de uma unidade (ano, uf, mes)."""
    chave = f"{ano}-{uf}-{mes:02d}"
    aa = str(ano)[-2:]
    nome_dbc = f"RD{uf}{aa}{mes:02d}.dbc"
    destino_parquet = PARQUET_DIR / f"ano={ano}" / f"uf={uf}" / f"mes={mes:02d}" / "dados.parquet"

    if destino_parquet.exists() or chave in particoes_existentes:
        return {"status": "PULADO", "chave": chave, "ano": ano, "uf": uf, "mes": mes, "nome_dbc": nome_dbc}

    dbc_local_dados = DADOS_DIR / nome_dbc
    dbc_tmp = TMP_DBC_DIR / nome_dbc
    baixado_agora = False

    if dbc_local_dados.exists():
        caminho_dbc = dbc_local_dados
    elif dbc_tmp.exists():
        caminho_dbc = dbc_tmp
    else:
        if ftp_disponiveis is not None and nome_dbc.upper() not in ftp_disponiveis:
            return {"status": "AUSENTE_FTP", "chave": chave, "ano": ano, "uf": uf, "mes": mes, "nome_dbc": nome_dbc}
        TMP_DBC_DIR.mkdir(parents=True, exist_ok=True)
        if not baixar_dbc(nome_dbc, dbc_tmp):
            return {"status": "ERRO_FTP", "chave": chave, "ano": ano, "uf": uf, "mes": mes, "nome_dbc": nome_dbc}
        caminho_dbc = dbc_tmp
        baixado_agora = True

    try:
        m = transformar_para_parquet(caminho_dbc, uf, ano, mes, destino_parquet)
        if baixado_agora and not manter_dbc and dbc_tmp.exists():
            dbc_tmp.unlink()
        return {"status": "OK", "chave": chave, "ano": ano, "uf": uf, "mes": mes, "nome_dbc": nome_dbc, **m}
    except Exception as e:
        log_safe(f"    [!] Erro ao processar {nome_dbc}: {e}")
        return {"status": "ERRO_PROC", "chave": chave, "ano": ano, "uf": uf, "mes": mes, "nome_dbc": nome_dbc}


def _consolidar_totais_banco(particoes: dict) -> dict:
    total_reg = sum(int(v.get("n_aceito", 0)) for v in particoes.values())
    total_ob = sum(int(v.get("obitos", 0)) for v in particoes.values())
    total_lg = sum(int(v.get("longas", 0)) for v in particoes.values())
    anos_disp = sorted({int(v["ano"]) for v in particoes.values()})
    comps = sorted({f"{int(v['ano'])}/{int(v['mes']):02d}" for v in particoes.values()})
    return {
        "total_particoes": len(particoes),
        "total_registros": int(total_reg),
        "total_obitos": int(total_ob),
        "total_longas": int(total_lg),
        "anos_disponiveis": anos_disp,
        "range_competencias": f"{comps[0]} → {comps[-1]}" if comps else "-",
        "competencias_ativas": len(comps),
    }


def _auditar_lote_base_2024(historico: dict) -> dict | None:
    """Consolida métricas reais da carga base 2024 a partir das partições locais para compor histórico."""
    parts_2024 = [v for v in historico.get("particoes", {}).values() if int(v.get("ano", 0)) == 2024]
    if not parts_2024:
        return None
    dir_2024 = PARQUET_DIR / "ano=2024"
    pct_raca, pct_esp, pct_cid, bad_sexo = 0.0, 0.0, 0.0, 0
    if dir_2024.exists():
        glob_24 = str(dir_2024 / "**" / "*.parquet").replace("\\", "/")
        con = duckdb.connect(database=":memory:")
        r = con.execute(
            f"""
            SELECT
                COALESCE(AVG(CASE WHEN raca_cor_desc = 'Nao_Declarada' THEN 100.0 ELSE 0.0 END), 0.0),
                COALESCE(AVG(CASE WHEN especialidade_leito = 'Outras' THEN 100.0 ELSE 0.0 END), 0.0),
                COALESCE(AVG(CASE WHEN capitulo_cid = 'Ignorado' THEN 100.0 ELSE 0.0 END), 0.0),
                CAST(COALESCE(SUM(CASE WHEN sexo NOT IN ('1', '3') THEN 1 ELSE 0 END), 0) AS BIGINT)
            FROM read_parquet('{glob_24}', hive_partitioning=true)
            """
        ).fetchone()
        con.close()
        pct_raca, pct_esp, pct_cid, bad_sexo = round(float(r[0]), 3), round(float(r[1]), 3), round(float(r[2]), 4), int(r[3])

    aceitos = sum(int(v.get("n_aceito", 0)) for v in parts_2024)
    obitos = sum(int(v.get("obitos", 0)) for v in parts_2024)
    longas = sum(int(v.get("longas", 0)) for v in parts_2024)
    brutos = sum(int(v.get("n_bruto", v.get("n_aceito", 0))) for v in parts_2024)
    desc = sum(int(v.get("n_descartado_filtro", 0)) for v in parts_2024)
    dedup = sum(int(v.get("n_dedup", 0)) for v in parts_2024)
    datas = sorted(str(v.get("data_ingestao_utc", ""))[:10] for v in parts_2024 if v.get("data_ingestao_utc"))
    dt_lbl = f"{datas[0]} 00:00:00" if datas else "2026-11-24 00:00:00"
    n_ac = max(aceitos, 1)
    return {
        "data_execucao_local": dt_lbl,
        "anos_novos": [2024],
        "ufs_novas_count": len({v.get("uf") for v in parts_2024}),
        "competencias_novas_count": len({int(v.get("mes", 0)) for v in parts_2024}),
        "range_lote_novo": "2024/01 → 2024/12",
        "qtd_particoes_novas": len(parts_2024),
        "registros_brutos": brutos,
        "registros_aceitos": aceitos,
        "registros_descartados_filtro": desc,
        "registros_deduplicados": dedup,
        "obitos_lote": obitos,
        "longas_lote": longas,
        "taxa_obito_lote_pct": round(obitos * 100.0 / n_ac, 2),
        "taxa_longa_lote_pct": round(longas * 100.0 / n_ac, 2),
        "qualidade_lote": {
            "pct_raca_nd": pct_raca,
            "pct_esp_outras": pct_esp,
            "pct_cid_ign": pct_cid,
            "bad_sexo_pos_filtro": bad_sexo,
            "bad_idade_pre_filtro": 0,
            "bad_diarias_pre_filtro": desc,
        },
        "origem": "Carga Base Nacional 2024",
    }


def _compactar_lote_historico(ev: dict) -> dict:
    aceitos = int(ev.get("registros_aceitos", 0))
    n_ac = max(aceitos, 1)
    ob = int(ev.get("obitos_lote", 0))
    lg = int(ev.get("longas_lote", 0))
    return {
        "data_execucao_local": ev.get("data_execucao_local", "-"),
        "anos_novos": ev.get("anos_novos", []),
        "ufs_novas_count": int(ev.get("ufs_novas_count", 0)),
        "competencias_novas_count": int(ev.get("competencias_novas_count", 0)),
        "range_lote_novo": ev.get("range_lote_novo", "-"),
        "qtd_particoes_novas": int(ev.get("qtd_particoes_novas", 0)),
        "registros_brutos": int(ev.get("registros_brutos", 0)),
        "registros_aceitos": aceitos,
        "registros_descartados_filtro": int(ev.get("registros_descartados_filtro", 0)),
        "registros_deduplicados": int(ev.get("registros_deduplicados", 0)),
        "obitos_lote": ob,
        "longas_lote": lg,
        "taxa_obito_lote_pct": ev.get("taxa_obito_lote_pct", round(ob * 100.0 / n_ac, 2)),
        "taxa_longa_lote_pct": ev.get("taxa_longa_lote_pct", round(lg * 100.0 / n_ac, 2)),
        "qualidade_lote": ev.get("qualidade_lote", {}),
        "origem": ev.get("origem", "Incremental FTP DataSUS"),
    }


def _montar_historico_cargas_deploy(historico: dict, resumo_previo: dict) -> list[dict]:
    itens = []
    vistos = set()
    candidatos = []

    base_24 = _auditar_lote_base_2024(historico)
    if base_24:
        candidatos.append(base_24)

    for prev_item in resumo_previo.get("historico_cargas", []):
        if int(prev_item.get("qtd_particoes_novas", 0)) > 0:
            candidatos.append(prev_item)

    for ev in historico.get("execucoes", []):
        if int(ev.get("qtd_particoes_novas", 0)) > 0:
            candidatos.append(_compactar_lote_historico(ev))

    for item in candidatos:
        chave = (item.get("range_lote_novo"), int(item.get("qtd_particoes_novas", 0)), int(item.get("registros_aceitos", 0)))
        if chave not in vistos:
            vistos.add(chave)
            itens.append(_compactar_lote_historico(item))

    itens.sort(key=lambda x: (str(x.get("data_execucao_local", "")), str(x.get("range_lote_novo", ""))))
    return itens[-50:]


def main():
    anos_padrao = list(range(2024, max(2026, datetime.now().year) + 1))
    parser = argparse.ArgumentParser(description="Pipeline Incremental Multi-Ano SIH/SUS -> Parquet Hive")
    parser.add_argument("--ufs", nargs="+", default=TODAS_UFS, help="Lista de siglas de UFs (padrão: 27)")
    parser.add_argument("--anos", nargs="+", type=int, default=anos_padrao, help="Anos alvo")
    parser.add_argument("--ano", type=int, default=None, help="Alias opcional para um único ano")
    parser.add_argument("--meses", nargs="+", type=int, default=list(range(1, 13)), help="Meses 1..12")
    parser.add_argument("--workers", type=int, default=6, help="Downloads/conversões em paralelo")
    parser.add_argument("--manter-dbc", action="store_true", help="Manter arquivos .dbc baixados após conversão")
    parser.add_argument("--reindexar-manifesto", action="store_true", help="Apenas sincronizar manifesto com o disco sem acessar FTP")
    parser.add_argument("--atualizar-resumo", action="store_true", help="Regerar resumo_ultima_carga.json com historico completo sem acessar FTP")
    args = parser.parse_args()

    anos = [args.ano] if args.ano is not None else sorted(set(args.anos))
    ufs = [u.upper() for u in args.ufs]
    meses = sorted(set(args.meses))

    historico = _carregar_historico()
    idx_novas = sincronizar_manifesto_com_disco(historico)
    if idx_novas:
        print(f"Manifesto sincronizado com {idx_novas} partições já existentes no disco.", flush=True)

    if args.reindexar_manifesto or args.atualizar_resumo:
        _write_json_atomic(HISTORICO_PATH, historico)
        if args.atualizar_resumo:
            resumo_previo = {}
            if RESUMO_CARGA_PATH.exists():
                try:
                    with open(RESUMO_CARGA_PATH, "r", encoding="utf-8") as f:
                        resumo_previo = json.load(f)
                except Exception:
                    resumo_previo = {}
            totais_banco = _consolidar_totais_banco(historico["particoes"])
            hist_deploy = _montar_historico_cargas_deploy(historico, resumo_previo)
            ultimo_lote = hist_deploy[-1] if hist_deploy else resumo_previo.get("ultimo_lote_com_dados", {})
            payload_resumo = {
                "ultima_verificacao_local": resumo_previo.get("ultima_verificacao_local", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                "ultima_verificacao_particoes_puladas": int(resumo_previo.get("ultima_verificacao_particoes_puladas", totais_banco["total_particoes"])),
                "ultima_verificacao_particoes_novas": int(resumo_previo.get("ultima_verificacao_particoes_novas", 0)),
                "ultimo_lote_com_dados": ultimo_lote,
                "historico_cargas": hist_deploy,
                "totais_banco": totais_banco,
            }
            _write_json_atomic(RESUMO_CARGA_PATH, payload_resumo)
            print(f"Resumo deploy atualizado ({len(hist_deploy)} lotes em historico_cargas).", flush=True)
        print(f"Manifesto salvo ({len(historico['particoes'])} partições). Saindo.", flush=True)
        return

    particoes_existentes = set(historico["particoes"].keys())
    tarefas = [(ano, uf, mes) for ano in anos for uf in ufs for mes in meses]
    pendentes_ftp = any(
        not (PARQUET_DIR / f"ano={a}" / f"uf={u}" / f"mes={m:02d}" / "dados.parquet").exists()
        and f"{a}-{u}-{m:02d}" not in particoes_existentes
        for a, u, m in tarefas
    )

    ftp_disponiveis = listar_arquivos_ftp_disponiveis() if pendentes_ftp else None

    print("=" * 78, flush=True)
    print(f"PIPELINE INCREMENTAL SIH/SUS {anos} -> PARQUET HIVE (Workers: {args.workers})", flush=True)
    print(f"UFs: {len(ufs)} | Meses: {meses} | Tarefas avaliadas: {len(tarefas)}", flush=True)
    print(f"Partições já presentes (puladas sem duplicar): {len(particoes_existentes)}", flush=True)
    print("=" * 78, flush=True)

    stats = {"OK": 0, "PULADO": 0, "AUSENTE_FTP": 0, "ERRO_FTP": 0, "ERRO_PROC": 0}
    lote_regs = {
        "brutos": 0, "aceitos": 0, "descartados_filtro": 0, "dedup": 0,
        "obitos": 0, "longas": 0,
        "qtd_raca_nd": 0, "qtd_esp_outras": 0, "qtd_cid_ign": 0, "bad_sexo": 0,
        "bad_idade_pre": 0, "bad_diarias_pre": 0,
    }
    anos_novos = set()
    ufs_novas = set()
    comps_novas = set()

    inicio = time.time()
    concluidos = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futuros = {
            executor.submit(processar_unidade, uf, ano, mes, particoes_existentes, ftp_disponiveis, args.manter_dbc): (ano, uf, mes)
            for ano, uf, mes in tarefas
        }
        for futuro in as_completed(futuros):
            r = futuro.result()
            st_code = r["status"]
            stats[st_code] += 1
            concluidos += 1
            if st_code == "OK":
                agora_utc = datetime.now(timezone.utc).isoformat()
                historico["particoes"][r["chave"]] = {
                    "ano": r["ano"],
                    "uf": r["uf"],
                    "mes": r["mes"],
                    "arquivo_dbc": r["nome_dbc"],
                    "n_bruto": r["n_bruto"],
                    "n_aceito": r["n_aceito"],
                    "n_descartado_filtro": r["n_descartado_filtro"],
                    "n_dedup": r["n_dedup"],
                    "obitos": r["obitos"],
                    "longas": r["longas"],
                    "origem": "ftp_incremental",
                    "data_ingestao_utc": agora_utc,
                }
                lote_regs["brutos"] += r["n_bruto"]
                lote_regs["aceitos"] += r["n_aceito"]
                lote_regs["descartados_filtro"] += r["n_descartado_filtro"]
                lote_regs["dedup"] += r["n_dedup"]
                lote_regs["obitos"] += r["obitos"]
                lote_regs["longas"] += r["longas"]
                lote_regs["qtd_raca_nd"] += r["qtd_raca_nd"]
                lote_regs["qtd_esp_outras"] += r["qtd_esp_outras"]
                lote_regs["qtd_cid_ign"] += r["qtd_cid_ign"]
                lote_regs["bad_sexo"] += r["bad_sexo"]
                lote_regs["bad_idade_pre"] += r["bad_idade_pre"]
                lote_regs["bad_diarias_pre"] += r["bad_diarias_pre"]
                anos_novos.add(r["ano"])
                ufs_novas.add(r["uf"])
                comps_novas.add(f"{r['ano']}/{r['mes']:02d}")
                log_safe(
                    f"  [{concluidos}/{len(tarefas)}] ✓ {r['uf']} {r['ano']}/{r['mes']:02d}: "
                    f"+{r['n_aceito']:,} aceitos (bruto {r['n_bruto']:,}, desc {r['n_descartado_filtro']}, dedup {r['n_dedup']})."
                )
            elif st_code in ("ERRO_FTP", "ERRO_PROC"):
                log_safe(f"  [{concluidos}/{len(tarefas)}] ✗ {r['uf']} {r['ano']}/{r['mes']:02d}: {st_code}")

    if TMP_DBC_DIR.exists() and not any(TMP_DBC_DIR.iterdir()):
        shutil.rmtree(TMP_DBC_DIR, ignore_errors=True)

    duracao = round(time.time() - inicio, 1)
    totais_banco = _consolidar_totais_banco(historico["particoes"])
    comps_sorted = sorted(comps_novas)
    n_ac = max(lote_regs["aceitos"], 1)

    resumo_exec = {
        "data_execucao_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_execucao_utc": datetime.now(timezone.utc).isoformat(),
        "duracao_segundos": duracao,
        "anos_solicitados": anos,
        "qtd_particoes_novas": stats["OK"],
        "qtd_particoes_puladas": stats["PULADO"],
        "qtd_particoes_ausentes_ftp": stats["AUSENTE_FTP"],
        "qtd_particoes_erro": stats["ERRO_FTP"] + stats["ERRO_PROC"],
        "anos_novos": sorted(anos_novos),
        "ufs_novas_count": len(ufs_novas),
        "competencias_novas_count": len(comps_sorted),
        "range_lote_novo": f"{comps_sorted[0]} → {comps_sorted[-1]}" if comps_sorted else "-",
        "registros_brutos": lote_regs["brutos"],
        "registros_aceitos": lote_regs["aceitos"],
        "registros_descartados_filtro": lote_regs["descartados_filtro"],
        "registros_deduplicados": lote_regs["dedup"],
        "obitos_lote": lote_regs["obitos"],
        "longas_lote": lote_regs["longas"],
        "taxa_obito_lote_pct": round(lote_regs["obitos"] * 100.0 / n_ac, 2) if lote_regs["aceitos"] else 0.0,
        "taxa_longa_lote_pct": round(lote_regs["longas"] * 100.0 / n_ac, 2) if lote_regs["aceitos"] else 0.0,
        "qualidade_lote": {
            "pct_raca_nd": round(lote_regs["qtd_raca_nd"] * 100.0 / n_ac, 3) if lote_regs["aceitos"] else 0.0,
            "pct_esp_outras": round(lote_regs["qtd_esp_outras"] * 100.0 / n_ac, 3) if lote_regs["aceitos"] else 0.0,
            "pct_cid_ign": round(lote_regs["qtd_cid_ign"] * 100.0 / n_ac, 4) if lote_regs["aceitos"] else 0.0,
            "bad_sexo_pos_filtro": lote_regs["bad_sexo"],
            "bad_idade_pre_filtro": lote_regs["bad_idade_pre"],
            "bad_diarias_pre_filtro": lote_regs["bad_diarias_pre"],
        },
        "origem": "Incremental FTP DataSUS",
        "totais_banco": totais_banco,
    }

    historico["execucoes"].append(resumo_exec)
    historico["execucoes"] = historico["execucoes"][-50:]
    _write_json_atomic(HISTORICO_PATH, historico)

    # Preserva métricas e histórico dos lotes reais para exibição auditável na Aba 2
    resumo_previo = {}
    if RESUMO_CARGA_PATH.exists():
        try:
            with open(RESUMO_CARGA_PATH, "r", encoding="utf-8") as f:
                resumo_previo = json.load(f)
        except Exception:
            resumo_previo = {}

    hist_deploy = _montar_historico_cargas_deploy(historico, resumo_previo)
    ultimo_lote_efetivo = hist_deploy[-1] if hist_deploy else resumo_previo.get("ultimo_lote_com_dados", resumo_exec)
    payload_resumo = {
        "ultima_verificacao_local": resumo_exec["data_execucao_local"],
        "ultima_verificacao_particoes_puladas": stats["PULADO"],
        "ultima_verificacao_particoes_novas": stats["OK"],
        "ultimo_lote_com_dados": ultimo_lote_efetivo,
        "historico_cargas": hist_deploy,
        "totais_banco": totais_banco,
    }
    _write_json_atomic(RESUMO_CARGA_PATH, payload_resumo)

    print("\n" + "=" * 78, flush=True)
    print("RESUMO DA CARGA INCREMENTAL SIH/SUS", flush=True)
    print(
        f"Partições -> Novas: {stats['OK']} | Puladas (já no banco): {stats['PULADO']} | "
        f"Ausentes FTP: {stats['AUSENTE_FTP']} | Erros: {stats['ERRO_FTP'] + stats['ERRO_PROC']}",
        flush=True,
    )
    print(
        f"Registros lote -> Brutos: {lote_regs['brutos']:,} | Aceitos: {lote_regs['aceitos']:,} | "
        f"Descartados filtro: {lote_regs['descartados_filtro']:,} | Deduplicados: {lote_regs['dedup']:,}",
        flush=True,
    )
    print(
        f"Base consolidada -> {totais_banco['total_particoes']} partições | "
        f"{totais_banco['total_registros']:,} internações ({totais_banco['range_competencias']})",
        flush=True,
    )
    print(f"Tempo total: {duracao:.1f}s | Manifesto: {HISTORICO_PATH}", flush=True)
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
