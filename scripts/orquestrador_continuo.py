# -*- coding: utf-8 -*-
"""
Orquestrador Contínuo & Gatilho Híbrido MLOps (Cloud & Local):
1. Executa pipeline incremental no DataSUS (`pipeline_nacional_sih.py`) buscando apenas partições inéditas.
2. Atualiza a camada deploy (`gerar_camada_deploy.py`) sempre que entrarem novas internações (atualizando Aba 1, 2 e 4).
3. Audita drift clínico e progresso de volume/competências frente a `modelos/politica_mlops.json`.
4. Dispara `retreinar_e_promover.py` (Champion vs. Challenger) somente quando o gatilho híbrido aprovar ou via `--forcar-retreino`.

Uso:
  python scripts/orquestrador_continuo.py
  python scripts/orquestrador_continuo.py --apenas-politica
  python scripts/orquestrador_continuo.py --forcar-retreino
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys

import duckdb
import joblib
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import features_v5  # noqa: E402, F401

PARQUET_DIR = BASE_DIR / "dados" / "parquet_nacional"
PARQUET_GLOB = str(PARQUET_DIR / "**" / "*.parquet").replace("\\", "/")
RESUMO_CARGA_PATH = BASE_DIR / "dados" / "deploy" / "resumo_ultima_carga.json"
MODELOS_DIR = BASE_DIR / "modelos"
POLITICA_PATH = MODELOS_DIR / "politica_mlops.json"
REGISTRO_PATH = MODELOS_DIR / "registro_modelos_v5.json"
LIMIARES_PATH = MODELOS_DIR / "config_limiares_v5.json"

BASE_CAT_COLS = [
    "sexo",
    "capitulo_cid",
    "especialidade_leito",
    "carater_internacao",
    "complexidade",
    "transferido_intermun",
    "internacao_fim_de_semana",
]
INPUT_COLS = ["idade_anos", "cid3"] + BASE_CAT_COLS


def _read_json(path: Path, default=None):
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {} if default is None else default


def _write_json_atomic(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def _parse_fim_janela(texto_range: str) -> tuple[int, int] | None:
    pares = re.findall(r"(\d{4})/(\d{2})", str(texto_range or ""))
    if not pares:
        return None
    a, m = pares[-1]
    return int(a), int(m)


def _auditar_drift_recente(comps_monitoradas: list[tuple[int, int]], politica: dict) -> dict:
    """Mede prevalência, ROC-AUC e Brier Score do Champion ativo em competências recentes."""
    limiares = _read_json(LIMIARES_PATH, {})
    m_lg_path = MODELOS_DIR / "modelo_admissao_longa_lightgbm_v5.joblib"
    m_ob_path = MODELOS_DIR / "modelo_admissao_obito_lightgbm_v5.joblib"
    if not comps_monitoradas or not PARQUET_DIR.exists() or not m_lg_path.exists() or not m_ob_path.exists():
        return {"amostra_n": 0, "drift_detectado": False, "motivos_drift": []}

    comp_sql = ", ".join(f"({int(a)}, {int(m)})" for a, m in comps_monitoradas)
    cols_sel = """
        idade_anos,
        COALESCE(NULLIF(SUBSTR(TRIM(diag_princ), 1, 3), ''), '_CAPITULO_') AS cid3,
        sexo, capitulo_cid, especialidade_leito, carater_internacao,
        complexidade, transferido_intermun, internacao_fim_de_semana,
        Longa_Permanencia, Obito
    """
    con = duckdb.connect(database=":memory:")
    df_mon = con.execute(
        f"""
        SELECT * FROM (
            SELECT {cols_sel}
            FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=true)
            WHERE (CAST(ano AS INTEGER), CAST(mes AS INTEGER)) IN ({comp_sql})
        ) sub_mon
        USING SAMPLE 60000 ROWS (reservoir, 42)
        """
    ).df()
    con.close()

    if df_mon.empty or len(df_mon) < 500:
        return {"amostra_n": int(len(df_mon)), "drift_detectado": False, "motivos_drift": []}

    for c in BASE_CAT_COLS + ["cid3"]:
        df_mon[c] = df_mon[c].astype(str)

    X_mon = df_mon[INPUT_COLS]
    y_lg = df_mon["Longa_Permanencia"].astype(int)
    y_ob = df_mon["Obito"].astype(int)

    mod_lg = joblib.load(m_lg_path)
    mod_ob = joblib.load(m_ob_path)
    p_lg = mod_lg.predict_proba(X_mon)[:, 1]
    p_ob = mod_ob.predict_proba(X_mon)[:, 1]

    prev_ob_pct = float(y_ob.mean() * 100.0)
    prev_lg_pct = float(y_lg.mean() * 100.0)
    roc_ob = float(roc_auc_score(y_ob, p_ob)) if y_ob.nunique() > 1 else 0.0
    roc_lg = float(roc_auc_score(y_lg, p_lg)) if y_lg.nunique() > 1 else 0.0
    brier_ob = float(brier_score_loss(y_ob, p_ob))
    brier_lg = float(brier_score_loss(y_lg, p_lg))

    ref_ob_pct = float(limiares.get("Obito", {}).get("prevalencia_nacional", 0.0377) * 100.0)
    ref_lg_pct = float(limiares.get("Longa_Permanencia", {}).get("prevalencia_nacional", 0.0606) * 100.0)
    d_ob_pp = abs(prev_ob_pct - ref_ob_pct)
    d_lg_pp = abs(prev_lg_pct - ref_lg_pct)

    motivos = []
    if d_ob_pp >= float(politica.get("drift_max_delta_pp_obito", 0.50)):
        motivos.append(f"Desvio prevalência Óbito ({d_ob_pp:.2f} p.p.)")
    if d_lg_pp >= float(politica.get("drift_max_delta_pp_longa", 0.80)):
        motivos.append(f"Desvio prevalência Longa Perm. ({d_lg_pp:.2f} p.p.)")
    if roc_ob and roc_ob < float(politica.get("drift_roc_min_obito", 0.8950)):
        motivos.append(f"ROC-AUC Óbito abaixo do piso ({roc_ob:.4f})")
    if roc_lg and roc_lg < float(politica.get("drift_roc_min_longa", 0.8300)):
        motivos.append(f"ROC-AUC Longa abaixo do piso ({roc_lg:.4f})")
    if brier_ob > float(politica.get("drift_brier_max_obito", 0.0330)):
        motivos.append(f"Brier Óbito degradado ({brier_ob:.4f})")
    if brier_lg > float(politica.get("drift_brier_max_longa", 0.0500)):
        motivos.append(f"Brier Longa degradado ({brier_lg:.4f})")

    fmt_rng = f"{comps_monitoradas[0][0]}/{comps_monitoradas[0][1]:02d} → {comps_monitoradas[-1][0]}/{comps_monitoradas[-1][1]:02d}"
    return {
        "janela_monitorada": fmt_rng,
        "amostra_n": int(len(df_mon)),
        "taxa_obito_pct": round(prev_ob_pct, 2),
        "taxa_longa_pct": round(prev_lg_pct, 2),
        "delta_obito_pp": round(d_ob_pp, 2),
        "delta_longa_pp": round(d_lg_pp, 2),
        "roc_obito": round(roc_ob, 4),
        "roc_longa": round(roc_lg, 4),
        "brier_obito": round(brier_ob, 4),
        "brier_longa": round(brier_lg, 4),
        "drift_detectado": bool(motivos),
        "motivos_drift": motivos,
    }


def _avaliar_politica_retreino(forcar_retreino: bool = False) -> tuple[bool, dict]:
    politica = _read_json(POLITICA_PATH, {
        "min_novas_competencias": 3,
        "min_novos_registros": 3_000_000,
        "min_dias_entre_retreinos": 30,
        "estrategia_retreino": "acumulado",
        "teste_meses": 3,
    })
    registro = _read_json(REGISTRO_PATH, {"historico_execucoes": []})

    if not PARQUET_DIR.exists():
        estado = {
            "ultima_checagem_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "novas_competencias_acumuladas": 0,
            "meta_competencias": int(politica.get("min_novas_competencias", 3)),
            "novos_registros_acumulados": 0,
            "meta_registros": int(politica.get("min_novos_registros", 3_000_000)),
            "dias_desde_retreino": 0,
            "meta_dias": int(politica.get("min_dias_entre_retreinos", 30)),
            "drift_detectado": False,
            "decisao_politica": "AGUARDANDO_DADOS",
            "motivo_politica": "Base Parquet local não encontrada.",
            "telemetria_monitorada": {},
        }
        return False, estado

    con = duckdb.connect(database=":memory:")
    comps_df = con.execute(
        f"""
        SELECT CAST(ano AS INTEGER) AS ano, CAST(mes AS INTEGER) AS mes, COUNT(*) AS n
        FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=true)
        GROUP BY 1, 2
        ORDER BY 1, 2
        """
    ).df()
    con.close()

    todas_comps = [(int(r["ano"]), int(r["mes"]), int(r["n"])) for _, r in comps_df.iterrows()]
    ult_av = registro.get("ultima_avaliacao") or {}
    champ = registro.get("champion_atual") or {}

    cutoff_eval = _parse_fim_janela(ult_av.get("janela_teste_temporal")) or _parse_fim_janela(champ.get("base_treino")) or (2024, 12)
    comps_novas = [(a, m, n) for (a, m, n) in todas_comps if (a, m) > cutoff_eval]
    qtd_novas_comps = len(comps_novas)
    qtd_novos_regs = sum(n for _, _, n in comps_novas)

    atualizado_str = str(ult_av.get("data_local") or champ.get("atualizado_em") or "")
    dias_passados = 999
    if atualizado_str:
        try:
            dt_ult = datetime.strptime(atualizado_str[:19], "%Y-%m-%d %H:%M:%S")
            dias_passados = max(0, (datetime.now() - dt_ult).days)
        except Exception:
            try:
                dt_ult = datetime.strptime(atualizado_str[:10], "%Y-%m-%d")
                dias_passados = max(0, (datetime.now() - dt_ult).days)
            except Exception:
                dias_passados = 0

    min_comps = int(politica.get("min_novas_competencias", 3))
    min_regs = int(politica.get("min_novos_registros", 3_000_000))
    min_dias = int(politica.get("min_dias_entre_retreinos", 30))

    # Monitora saúde nos meses recém-chegados após último corte; se ainda 0 novos, audita janela recente
    janela_mon = [(a, m) for a, m, _ in comps_novas] if comps_novas else [(a, m) for a, m, _ in todas_comps[-3:]]
    tele_drift = _auditar_drift_recente(janela_mon, politica)
    drift_acionavel = bool(comps_novas and tele_drift.get("drift_detectado"))

    volume_ok = (qtd_novas_comps >= min_comps) or (qtd_novos_regs >= min_regs)
    cooldown_ok = dias_passados >= min_dias

    if forcar_retreino:
        deve_retreinar = True
        decisao = "DISPARADO_MANUAL"
        motivo = "Retreinamento forçado manualmente via parâmetro --forcar-retreino."
    elif drift_acionavel:
        deve_retreinar = True
        decisao = "DISPARADO_DRIFT"
        motivo = "Drift clínico detectado nos lotes inéditos: " + "; ".join(tele_drift["motivos_drift"])
    elif volume_ok and cooldown_ok:
        deve_retreinar = True
        decisao = "DISPARADO_VOLUME"
        motivo = f"Meta atingida ({qtd_novas_comps}/{min_comps} competências, {qtd_novos_regs:,} registros, {dias_passados}d >= {min_dias}d)."
    elif volume_ok and not cooldown_ok:
        deve_retreinar = False
        decisao = "AGUARDANDO_COOLDOWN"
        motivo = f"Volume atingido ({qtd_novas_comps} comp.), aguardando cooldown ({dias_passados}/{min_dias} dias)."
    else:
        deve_retreinar = False
        decisao = "MONITORANDO_ESTAVEL"
        motivo = (
            f"Aguardando acúmulo pós {cutoff_eval[0]}/{cutoff_eval[1]:02d}: "
            f"{qtd_novas_comps}/{min_comps} competências inéditas ({qtd_novos_regs:,}/{min_regs:,} casos) • Sem drift."
        )

    estado = {
        "ultima_checagem_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cutoff_ultima_avaliacao": f"{cutoff_eval[0]}/{cutoff_eval[1]:02d}",
        "novas_competencias_acumuladas": qtd_novas_comps,
        "meta_competencias": min_comps,
        "novos_registros_acumulados": qtd_novos_regs,
        "meta_registros": min_regs,
        "dias_desde_retreino": dias_passados,
        "meta_dias": min_dias,
        "drift_detectado": drift_acionavel,
        "decisao_politica": decisao,
        "motivo_politica": motivo,
        "telemetria_monitorada": tele_drift,
    }
    return deve_retreinar, estado


def _salvar_politica_estado(estado: dict):
    reg = _read_json(REGISTRO_PATH, {"historico_execucoes": []})
    reg["politica_estado"] = estado
    _write_json_atomic(REGISTRO_PATH, reg)


def _run_step(cmd: list[str], label: str):
    print(f"\n>>> [{label}] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=BASE_DIR, check=True)


def main():
    parser = argparse.ArgumentParser(description="Orquestrador Contínuo SIH/SUS + Gatilho Híbrido MLOps")
    parser.add_argument("--ufs", nargs="+", default=None)
    parser.add_argument("--anos", nargs="+", type=int, default=None)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--pular-ftp", action="store_true", help="Não consultar FTP (apenas atualizar resumo/política)")
    parser.add_argument("--apenas-politica", action="store_true", help="Apenas avaliar gatilho/drift sem rodar ingestão ou retreino")
    parser.add_argument("--forcar-deploy", action="store_true", help="Forçar reconstrução de cubo/amostra deploy")
    parser.add_argument("--forcar-retreino", action="store_true", help="Forçar execução do retreino Champion vs. Challenger")
    args = parser.parse_args()

    novas_particoes = 0
    if not args.apenas_politica:
        if args.pular_ftp:
            _run_step([sys.executable, "-u", "scripts/pipeline_nacional_sih.py", "--atualizar-resumo"], "Resumo Local")
        else:
            cmd_ing = [sys.executable, "-u", "scripts/pipeline_nacional_sih.py", "--workers", str(args.workers)]
            if args.anos:
                cmd_ing += ["--anos"] + [str(a) for a in args.anos]
            if args.ufs:
                cmd_ing += ["--ufs"] + args.ufs
            _run_step(cmd_ing, "Ingestão Incremental DataSUS")

        resumo = _read_json(RESUMO_CARGA_PATH, {})
        novas_particoes = int(resumo.get("ultima_verificacao_particoes_novas", 0))
        if novas_particoes > 0 or args.forcar_deploy:
            _run_step([sys.executable, "-u", "scripts/gerar_camada_deploy.py"], "Camada Deploy Multi-Ano")
        else:
            print("\n>>> [Camada Deploy] 0 novas partições ingeridas — cubo e amostra já atualizados.", flush=True)

    deve_retreinar, estado_politica = _avaliar_politica_retreino(forcar_retreino=args.forcar_retreino)
    _salvar_politica_estado(estado_politica)

    print("\n" + "=" * 78, flush=True)
    print(f"POLÍTICA MLOPS AUTOMÁTICA -> {estado_politica['decisao_politica']}", flush=True)
    print(f"Motivo: {estado_politica['motivo_politica']}", flush=True)
    tele = estado_politica.get("telemetria_monitorada", {})
    if tele.get("amostra_n"):
        print(
            f"Monitor ({tele.get('janela_monitorada')}, n={tele['amostra_n']:,}) -> "
            f"Óbito={tele['taxa_obito_pct']:.2f}% (ROC={tele['roc_obito']:.4f}, Brier={tele['brier_obito']:.4f}) | "
            f"Longa={tele['taxa_longa_pct']:.2f}% (ROC={tele['roc_longa']:.4f}, Brier={tele['brier_longa']:.4f})",
            flush=True,
        )
    print("=" * 78, flush=True)

    if deve_retreinar and not args.apenas_politica:
        politica = _read_json(POLITICA_PATH, {})
        estrategia = str(politica.get("estrategia_retreino", "acumulado"))
        teste_meses = str(int(politica.get("teste_meses", 3)))
        _run_step(
            [sys.executable, "-u", "scripts/retreinar_e_promover.py", "--estrategia", estrategia, "--teste-meses", teste_meses],
            "Champion vs. Challenger",
        )
        # Atualiza estado pós-retreino com o novo cutoff
        _, estado_pos = _avaliar_politica_retreino(forcar_retreino=False)
        _salvar_politica_estado(estado_pos)


if __name__ == "__main__":
    main()
