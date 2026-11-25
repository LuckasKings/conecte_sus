# -*- coding: utf-8 -*-
"""
Retreinamento Contínuo & Promoção Champion vs. Challenger — SIH/SUS V5 Nacional
Protocolo MLOps sem Data Leakage Temporal:
1. Separa as competências mais recentes como Teste Cego Temporal (`--teste-meses`, padrão 3 meses).
2. Extrai Treino (até 2,4M) e Calibração Isotônica (300k) apenas de competências ANTERIORES
   (suporta estratégia `--estrategia acumulado | rolling | recalibrar`).
3. Avalia o modelo Champion atual (produção) e o Challenger candidato no MESMO Teste Cego Temporal.
4. Só promove o Challenger para produção (`modelos/*.joblib`, métricas, limiares e figuras) se ele
   superar o Champion nos critérios objetivos (ROC-AUC, PR-AUC e Brier Score calibrado),
   mantendo backup em `modelos/archive/` e registro auditável em `modelos/registro_modelos_v5.json`.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time
import warnings

import duckdb
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from matplotlib import rcParams
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from features_v5 import BASE_CAT_COLS, AdmissionV5Preprocessor  # noqa: E402

warnings.filterwarnings("ignore")

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
rcParams["font.size"] = 11
rcParams["axes.linewidth"] = 1.4
rcParams["axes.edgecolor"] = "#334155"
rcParams["axes.grid"] = False
rcParams["figure.facecolor"] = "white"
rcParams["axes.facecolor"] = "white"
rcParams["savefig.dpi"] = 300
rcParams["savefig.bbox"] = "tight"

BASE_DIR = SCRIPTS_DIR.parent
PARQUET_GLOB = str(BASE_DIR / "dados" / "parquet_nacional" / "**" / "*.parquet").replace("\\", "/")
FIGURAS_DIR = BASE_DIR / "figuras"
MODELOS_DIR = BASE_DIR / "modelos"
ARCHIVE_DIR = MODELOS_DIR / "archive"
REGISTRO_PATH = MODELOS_DIR / "registro_modelos_v5.json"

INPUT_COLS = ["idade_anos", "cid3"] + BASE_CAT_COLS


def _comp_sql(comps: list[tuple[int, int]]) -> str:
    return ", ".join(f"({int(a)}, {int(m)})" for a, m in comps)


def _avaliar_classificador(modelo, X_te: pd.DataFrame, y_te: pd.Series, limiar: float) -> dict:
    p = modelo.predict_proba(X_te)[:, 1]
    roc_auc = float(roc_auc_score(y_te, p))
    pr_auc = float(average_precision_score(y_te, p))
    brier_cal = float(brier_score_loss(y_te, p))
    y_pred = (p >= limiar).astype(int)
    cm = confusion_matrix(y_te, y_pred)
    tn, fp, fn, tp = cm.ravel()
    rec = float(recall_score(y_te, y_pred, zero_division=0))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    acc = float(accuracy_score(y_te, y_pred))
    prec = float(precision_score(y_te, y_pred, zero_division=0))
    f1 = float(f1_score(y_te, y_pred, zero_division=0))
    fpr, tpr, _ = roc_curve(y_te, p)
    prec_c, rec_c, _ = precision_recall_curve(y_te, p)
    prob_true, prob_pred = calibration_curve(y_te, p, n_bins=10)
    return {
        "probs": p,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_cal": brier_cal,
        "limiar": float(limiar),
        "sens": rec,
        "spec": spec,
        "acc": acc,
        "prec": prec,
        "f1": f1,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "fpr": fpr, "tpr": tpr,
        "prec_curve": prec_c, "rec_curve": rec_c,
        "prob_true_cal": prob_true, "prob_pred_cal": prob_pred,
    }


def _deve_promover(cmp: dict) -> tuple[bool, str]:
    """Regra objetiva Champion vs Challenger em ambos os desfechos no teste cego temporal."""
    melhorias = 0
    regressoes = []
    for t_id in ("Obito", "Longa_Permanencia"):
        ch = cmp[t_id]["champion"]
        cl = cmp[t_id]["challenger"]
        d_roc = cl["roc_auc"] - ch["roc_auc"]
        d_pr = cl["pr_auc"] - ch["pr_auc"]
        d_brier = cl["brier_cal"] - ch["brier_cal"]
        d_sens = (cl["sens"] - ch["sens"]) * 100.0

        if d_roc < -0.0015 or d_pr < -0.0020 or d_brier > 0.0010 or d_sens < -3.0:
            regressoes.append(f"{t_id} (ΔROC={d_roc:+.4f}, ΔPR={d_pr:+.4f}, ΔBrier={d_brier:+.4f}, ΔSens={d_sens:+.2f}pp)")
        elif d_roc >= 0.0005 or d_pr >= 0.0010 or d_brier <= -0.0003:
            melhorias += 1

    if regressoes:
        return False, "Champion mantido devido a regressão em: " + "; ".join(regressoes)
    if melhorias >= 1:
        return True, f"Challenger promovido com melhoria significativa em {melhorias}/2 desfechos sem regressões."
    return False, "Champion mantido: diferença dentro da margem de empate técnico (sem ganho suficiente para troca)."


def _treinar_challenger_target(
    t_id: str,
    estrategia: str,
    champion_model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
):
    if estrategia == "recalibrar" and champion_model is not None:
        base = getattr(champion_model, "estimator", None)
        if base is None and hasattr(champion_model, "calibrated_classifiers_") and champion_model.calibrated_classifiers_:
            base = champion_model.calibrated_classifiers_[0].estimator
        if base is not None:
            calibrator = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic")
            calibrator.fit(X_cal, y_cal)
            p_cal = calibrator.predict_proba(X_cal)[:, 1]
            fpr_c, tpr_c, thr_c = roc_curve(y_cal, p_cal)
            limiar = float(thr_c[np.argmax(tpr_c - fpr_c)])
            return calibrator, limiar, np.zeros(len(X_cal)), 0.0

    clf = LGBMClassifier(
        n_estimators=220, learning_rate=0.07, num_leaves=63, min_child_samples=100,
        subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42, verbose=-1,
    )
    pipe = Pipeline([
        ("preprocessor", AdmissionV5Preprocessor(top_n_cid3=200, bayes_m=50.0, n_folds=5, random_state=42)),
        ("classifier", clf),
    ])
    pipe.fit(X_train, y_train)
    calibrator = CalibratedClassifierCV(FrozenEstimator(pipe), method="isotonic")
    calibrator.fit(X_cal, y_cal)
    p_cal = calibrator.predict_proba(X_cal)[:, 1]
    fpr_c, tpr_c, thr_c = roc_curve(y_cal, p_cal)
    limiar = float(thr_c[np.argmax(tpr_c - fpr_c)])
    return calibrator, limiar, pipe, 0.0


def _gerar_benchmark_e_figuras_promocao(
    challengers: dict,
    X_train: pd.DataFrame,
    X_cal: pd.DataFrame,
    X_test: pd.DataFrame,
    desfechos: dict,
    estrato_tr: pd.Series,
    top_cid3_por_cap: dict,
    range_desc: str,
):
    """Atualiza benchmark 4 modelos, metricas_v5_nacional.csv, limiares e as 3 figuras do dashboard."""
    _, idx_rf = train_test_split(np.arange(len(X_train)), test_size=min(250_000, len(X_train) - 1), stratify=estrato_tr, random_state=42)
    cores = {
        "Logistic Regression": "#64748b",
        "Random Forest": "#10b981",
        "XGBoost": "#f59e0b",
        "LightGBM": "#2563eb",
    }

    metricas_rows = []
    curvas = {"Longa_Permanencia": {}, "Obito": {}}
    importancias = {"Longa_Permanencia": {}, "Obito": {}}
    limiares_producao = {}

    for t_id, t_cfg in desfechos.items():
        y_tr, y_ca, y_te = t_cfg["y_train"], t_cfg["y_cal"], t_cfg["y_test"]
        modelos_extra = {
            "Logistic Regression": LogisticRegression(C=1.0, max_iter=500, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=80, max_depth=14, min_samples_leaf=50, n_jobs=-1, random_state=42),
            "XGBoost": XGBClassifier(
                n_estimators=220, learning_rate=0.07, max_depth=6, subsample=0.8,
                colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=42, eval_metric="logloss",
            ),
        }

        all_results = []
        for mod_name, clf_base in modelos_extra.items():
            t_m = time.time()
            print(f"    -> Benchmark {t_id}: treinando {mod_name}...", flush=True)
            pipe = Pipeline([
                ("preprocessor", AdmissionV5Preprocessor(top_n_cid3=200, bayes_m=50.0, n_folds=5, random_state=42)),
                ("classifier", clf_base),
            ])
            if mod_name == "Random Forest":
                pipe.fit(X_train.iloc[idx_rf], y_tr.iloc[idx_rf])
            else:
                pipe.fit(X_train, y_tr)
            p_raw = pipe.predict_proba(X_test)[:, 1]
            brier_raw = float(brier_score_loss(y_te, p_raw))
            cal = CalibratedClassifierCV(FrozenEstimator(pipe), method="isotonic")
            cal.fit(X_cal, y_ca)
            p_cal_fit = cal.predict_proba(X_cal)[:, 1]
            fpr_c, tpr_c, thr_c = roc_curve(y_ca, p_cal_fit)
            lim = float(thr_c[np.argmax(tpr_c - fpr_c)])
            ev = _avaliar_classificador(cal, X_test, y_te, lim)
            ev["brier_raw"] = brier_raw
            print(f"       {mod_name} concluído em {time.time()-t_m:.1f}s (ROC={ev['roc_auc']:.4f} | PR={ev['pr_auc']:.4f})", flush=True)
            all_results.append((mod_name, ev, pipe))

        # LightGBM Challenger já treinado
        chall_cal, chall_lim, chall_pipe = challengers[t_id]["model"], challengers[t_id]["limiar"], challengers[t_id]["pipe"]
        if chall_pipe is not None and hasattr(chall_pipe, "predict_proba"):
            brier_raw_lgbm = float(brier_score_loss(y_te, chall_pipe.predict_proba(X_test)[:, 1]))
        else:
            brier_raw_lgbm = challengers[t_id]["eval"]["brier_cal"]
        ev_lgbm = challengers[t_id]["eval"]
        ev_lgbm["brier_raw"] = brier_raw_lgbm
        all_results.append(("LightGBM", ev_lgbm, chall_pipe))

        for mod_name, ev, pipe_obj in all_results:
            curvas[t_id][mod_name] = ev
            if mod_name == "LightGBM":
                if pipe_obj is not None and hasattr(pipe_obj, "named_steps"):
                    prep = pipe_obj.named_steps["preprocessor"]
                    clf = pipe_obj.named_steps["classifier"]
                    importancias[t_id]["LightGBM"] = ([str(f) for f in prep.get_feature_names_out()], clf.feature_importances_)
                limiares_producao[t_id] = {
                    "limiar_youden": round(ev["limiar"], 4),
                    "prevalencia_nacional": round(float(y_te.mean()), 4),
                    "roc_auc": round(ev["roc_auc"], 4),
                    "pr_auc": round(ev["pr_auc"], 4),
                    "brier_cal": round(ev["brier_cal"], 4),
                }

            metricas_rows.append({
                "Desfecho": t_cfg["nome"],
                "Modelo": mod_name,
                "ROC_AUC": round(ev["roc_auc"], 4),
                "PR_AUC": round(ev["pr_auc"], 4),
                "Brier_Raw": round(ev["brier_raw"], 4),
                "Brier_Calibrado": round(ev["brier_cal"], 4),
                "Limiar_Otimo_Pct": round(ev["limiar"] * 100, 2),
                "Sensibilidade_Pct": round(ev["sens"] * 100, 2),
                "Especificidade_Pct": round(ev["spec"] * 100, 2),
                "Acuracia_Pct": round(ev["acc"] * 100, 2),
                "Precisao_Pct": round(ev["prec"] * 100, 2),
                "F1_Score_Pct": round(ev["f1"] * 100, 2),
                "TN": ev["tn"], "FP": ev["fp"], "FN": ev["fn"], "TP": ev["tp"],
            })

    pd.DataFrame(metricas_rows).to_csv(MODELOS_DIR / "metricas_v5_nacional.csv", index=False)
    with open(MODELOS_DIR / "config_limiares_v5.json", "w", encoding="utf-8") as f:
        json.dump(limiares_producao, f, indent=2, ensure_ascii=False)
    with open(MODELOS_DIR / "top_cid3_por_capitulo_v5.json", "w", encoding="utf-8") as f:
        json.dump(top_cid3_por_cap, f, indent=2, ensure_ascii=False)

    # Salvar LightGBM Challengers promovidos
    joblib.dump(challengers["Longa_Permanencia"]["model"], MODELOS_DIR / "modelo_admissao_longa_lightgbm_v5.joblib", compress=3)
    joblib.dump(challengers["Obito"]["model"], MODELOS_DIR / "modelo_admissao_obito_lightgbm_v5.joblib", compress=3)

    # Figura 1: ROC / PR
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for row_idx, (t_id, t_nome) in enumerate([("Obito", "Óbito Intra-hospitalar"), ("Longa_Permanencia", "Longa Permanência (≥15d)")]):
        ax_roc, ax_pr = axes[row_idx, 0], axes[row_idx, 1]
        for mod_name, c_data in curvas[t_id].items():
            ax_roc.plot(c_data["fpr"], c_data["tpr"], lw=2.2, color=cores[mod_name], label=f"{mod_name} (AUC = {c_data['roc_auc']:.4f})")
            ax_pr.plot(c_data["rec_curve"], c_data["prec_curve"], lw=2.2, color=cores[mod_name], label=f"{mod_name} (PR-AUC = {c_data['pr_auc']:.4f})")
        ax_roc.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6)
        ax_roc.set_title(f"Curva ROC V5 — {t_nome} ({range_desc})", fontweight="bold")
        ax_roc.set_xlabel("Taxa de Falso Positivo (1 - Especificidade)")
        ax_roc.set_ylabel("Sensibilidade (Recall)")
        ax_roc.legend(loc="lower right", frameon=True, fontsize=9)

        prev_rate = float(desfechos[t_id]["y_test"].mean())
        ax_pr.axhline(prev_rate, color="black", linestyle="--", lw=1.2, alpha=0.6, label=f"Baseline ({prev_rate:.2%})")
        ax_pr.set_title(f"Curva Precision-Recall V5 — {t_nome} ({range_desc})", fontweight="bold")
        ax_pr.set_xlabel("Sensibilidade (Recall)")
        ax_pr.set_ylabel("Precisão (VPP)")
        ax_pr.legend(loc="upper right", frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURAS_DIR / "fig_roc_pr_curvas_v5.png")
    plt.close()

    # Figura 2: Calibração
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for idx, (t_id, t_nome) in enumerate([("Obito", "Óbito Intra-hospitalar"), ("Longa_Permanencia", "Longa Permanência (≥15d)")]):
        ax = axes[idx]
        ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Calibração Perfeita")
        for mod_name, c_data in curvas[t_id].items():
            ax.plot(c_data["prob_pred_cal"], c_data["prob_true_cal"], marker="o", lw=2.2, color=cores[mod_name], label=f"{mod_name} (Brier = {c_data['brier_cal']:.4f})")
        ax.set_title(f"Calibração Isotônica V5 — {t_nome}", fontweight="bold")
        ax.set_xlabel("Probabilidade Média Prevista")
        ax.set_ylabel("Fração Real Observada")
        ax.legend(loc="upper left", frameon=True, fontsize=9)
        ax.set_xlim(0, 0.75)
        ax.set_ylim(0, 0.75)
    plt.tight_layout()
    plt.savefig(FIGURAS_DIR / "fig_calibracao_probabilidade_v5.png")
    plt.close()

    # Figura 3: Importâncias
    if "LightGBM" in importancias["Obito"] and "LightGBM" in importancias["Longa_Permanencia"]:
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        for idx, (t_id, t_nome) in enumerate([("Obito", "Óbito Intra-hospitalar"), ("Longa_Permanencia", "Longa Permanência (≥15d)")]):
            ax = axes[idx]
            fnames, imps = importancias[t_id]["LightGBM"]
            df_imp = pd.DataFrame({"Feature": fnames, "Importance": imps}).sort_values("Importance", ascending=True).tail(15)
            ax.barh(df_imp["Feature"], df_imp["Importance"], color="#2563eb" if idx == 1 else "#dc2626", alpha=0.88)
            ax.set_title(f"Top 15 Variáveis (LightGBM V5 Brasil) — {t_nome}", fontweight="bold")
            ax.set_xlabel("Importância (Splits)")
        plt.tight_layout()
        plt.savefig(FIGURAS_DIR / "fig_importancias_admissao_v5.png")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Retreino Temporal e Promoção Champion vs Challenger V5")
    parser.add_argument("--estrategia", choices=["acumulado", "rolling", "recalibrar"], default="acumulado")
    parser.add_argument("--rolling-meses", type=int, default=18, help="Janela de meses (se estrategia=rolling)")
    parser.add_argument("--teste-meses", type=int, default=3, help="Meses mais recentes reservados para Teste Cego Temporal")
    parser.add_argument("--max-treino", type=int, default=2_400_000)
    parser.add_argument("--max-cal", type=int, default=300_000)
    parser.add_argument("--max-teste", type=int, default=300_000)
    parser.add_argument("--dry-run", action="store_true", help="Avalia sem substituir os artefatos de produção")
    args = parser.parse_args()

    t0 = time.time()
    con = duckdb.connect(database=":memory:")

    comps_df = con.execute(
        f"""
        SELECT CAST(ano AS INTEGER) AS ano, CAST(mes AS INTEGER) AS mes, COUNT(*) AS n
        FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=true)
        GROUP BY 1, 2
        ORDER BY 1, 2
        """
    ).df()
    todas_comps = [(int(r["ano"]), int(r["mes"])) for _, r in comps_df.iterrows()]
    assert len(todas_comps) > args.teste_meses, f"Competências insuficientes ({len(todas_comps)}) para split temporal!"

    test_comps = todas_comps[-args.teste_meses:]
    prior_all = todas_comps[:-args.teste_meses]
    train_comps = prior_all[-args.rolling_meses:] if args.estrategia == "rolling" else prior_all

    fmt_rng = lambda cs: f"{cs[0][0]}/{cs[0][1]:02d} → {cs[-1][0]}/{cs[-1][1]:02d}"
    print("=" * 80, flush=True)
    print(f"MLOPS RETREINAMENTO & CHAMPION vs CHALLENGER ({args.estrategia.upper()})", flush=True)
    print(f"Janela Treino/Cal ({len(train_comps)} comps): {fmt_rng(train_comps)}", flush=True)
    print(f"Teste Cego Temporal ({len(test_comps)} comps): {fmt_rng(test_comps)}", flush=True)
    print("=" * 80, flush=True)

    # Top-20 CID3 por capítulo atualizado com todas as competências disponíveis
    df_top_cap = con.execute(
        f"""
        WITH contagem AS (
            SELECT capitulo_cid, COALESCE(NULLIF(SUBSTR(TRIM(diag_princ), 1, 3), ''), '_CAPITULO_') AS cid3, COUNT(*) AS n
            FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=true)
            GROUP BY 1, 2
        ),
        ranqueado AS (
            SELECT capitulo_cid, cid3, n, ROW_NUMBER() OVER (PARTITION BY capitulo_cid ORDER BY n DESC) AS rk
            FROM contagem WHERE cid3 != '_CAPITULO_'
        )
        SELECT capitulo_cid, cid3, CAST(n AS BIGINT) AS n FROM ranqueado WHERE rk <= 20 ORDER BY capitulo_cid, n DESC
        """
    ).df()
    top_cid3_por_cap = {
        str(cap): [{"cid3": str(r["cid3"]), "n": int(r["n"])} for _, r in grp.iterrows()]
        for cap, grp in df_top_cap.groupby("capitulo_cid")
    }

    cols_sel = """
        idade_anos,
        COALESCE(NULLIF(SUBSTR(TRIM(diag_princ), 1, 3), ''), '_CAPITULO_') AS cid3,
        sexo, capitulo_cid, especialidade_leito, carater_internacao,
        complexidade, transferido_intermun, internacao_fim_de_semana,
        Longa_Permanencia, Obito
    """

    n_sample_tr_cal = args.max_treino + args.max_cal
    df_tr_cal = con.execute(
        f"""
        SELECT * FROM (
            SELECT {cols_sel}
            FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=true)
            WHERE (CAST(ano AS INTEGER), CAST(mes AS INTEGER)) IN ({_comp_sql(train_comps)})
        ) sub_tr
        USING SAMPLE {int(n_sample_tr_cal)} ROWS (reservoir, 42)
        """
    ).df()

    df_te = con.execute(
        f"""
        SELECT * FROM (
            SELECT {cols_sel}
            FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=true)
            WHERE (CAST(ano AS INTEGER), CAST(mes AS INTEGER)) IN ({_comp_sql(test_comps)})
        ) sub_te
        USING SAMPLE {int(args.max_teste)} ROWS (reservoir, 42)
        """
    ).df()
    con.close()

    for df_part in (df_tr_cal, df_te):
        for c in BASE_CAT_COLS + ["cid3"]:
            df_part[c] = df_part[c].astype(str)

    estrato_tc = df_tr_cal["Obito"].astype(str) + "_" + df_tr_cal["Longa_Permanencia"].astype(str)
    cal_size = min(args.max_cal, max(50_000, int(0.12 * len(df_tr_cal))))
    idx_tr, idx_ca = train_test_split(np.arange(len(df_tr_cal)), test_size=cal_size, stratify=estrato_tc, random_state=42)

    X_train, X_cal, X_test = df_tr_cal[INPUT_COLS].iloc[idx_tr], df_tr_cal[INPUT_COLS].iloc[idx_ca], df_te[INPUT_COLS]
    desfechos = {
        "Longa_Permanencia": {
            "nome": "Longa Permanência",
            "y_train": df_tr_cal["Longa_Permanencia"].astype(int).iloc[idx_tr],
            "y_cal": df_tr_cal["Longa_Permanencia"].astype(int).iloc[idx_ca],
            "y_test": df_te["Longa_Permanencia"].astype(int),
        },
        "Obito": {
            "nome": "Óbito Intra-hospitalar",
            "y_train": df_tr_cal["Obito"].astype(int).iloc[idx_tr],
            "y_cal": df_tr_cal["Obito"].astype(int).iloc[idx_ca],
            "y_test": df_te["Obito"].astype(int),
        },
    }
    print(f"  ✓ Amostras -> Treino: {len(X_train):,} | Calibração: {len(X_cal):,} | Teste Temporal: {len(X_test):,}", flush=True)

    with open(MODELOS_DIR / "config_limiares_v5.json", "r", encoding="utf-8") as f:
        cfg_champ = json.load(f)

    champ_paths = {
        "Longa_Permanencia": MODELOS_DIR / "modelo_admissao_longa_lightgbm_v5.joblib",
        "Obito": MODELOS_DIR / "modelo_admissao_obito_lightgbm_v5.joblib",
    }

    comparativo = {}
    challengers = {}

    for t_id, t_cfg in desfechos.items():
        m_champ = joblib.load(champ_paths[t_id])
        lim_champ = float(cfg_champ[t_id]["limiar_youden"])
        ev_champ = _avaliar_classificador(m_champ, X_test, t_cfg["y_test"], lim_champ)

        m_chall, lim_chall, pipe_chall, _ = _treinar_challenger_target(
            t_id, args.estrategia, m_champ, X_train, t_cfg["y_train"], X_cal, t_cfg["y_cal"]
        )
        ev_chall = _avaliar_classificador(m_chall, X_test, t_cfg["y_test"], lim_chall)

        challengers[t_id] = {"model": m_chall, "limiar": lim_chall, "pipe": pipe_chall, "eval": ev_chall}
        comparativo[t_id] = {
            "champion": {k: round(ev_champ[k], 4) for k in ("roc_auc", "pr_auc", "brier_cal", "limiar", "sens", "spec")},
            "challenger": {k: round(ev_chall[k], 4) for k in ("roc_auc", "pr_auc", "brier_cal", "limiar", "sens", "spec")},
        }
        print(
            f"\n[{t_id}] Teste Temporal ({fmt_rng(test_comps)}):\n"
            f"  Champion   -> ROC={ev_champ['roc_auc']:.4f} | PR={ev_champ['pr_auc']:.4f} | Brier={ev_champ['brier_cal']:.4f} | Sens={ev_champ['sens']*100:.2f}%\n"
            f"  Challenger -> ROC={ev_chall['roc_auc']:.4f} | PR={ev_chall['pr_auc']:.4f} | Brier={ev_chall['brier_cal']:.4f} | Sens={ev_chall['sens']*100:.2f}%",
            flush=True,
        )

    promover, motivo = _deve_promover(comparativo)
    print(f"\n⚖️ DECISÃO CHAMPION vs CHALLENGER: {'PROMOVIDO' if promover else 'CHAMPION MANTIDO'} — {motivo}", flush=True)

    status_acao = "PROMOVIDO" if (promover and not args.dry_run) else ("APROVADO_DRY_RUN" if promover else "CHAMPION_MANTIDO")

    if promover and not args.dry_run:
        ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        bkp_dir = ARCHIVE_DIR / f"champion_backup_{ts_tag}"
        bkp_dir.mkdir(parents=True, exist_ok=True)
        for fname in (
            "modelo_admissao_longa_lightgbm_v5.joblib",
            "modelo_admissao_obito_lightgbm_v5.joblib",
            "metricas_v5_nacional.csv",
            "config_limiares_v5.json",
        ):
            src = MODELOS_DIR / fname
            if src.exists():
                shutil.copy2(src, bkp_dir / fname)

        print("Atualizando modelos de produção, benchmark completo e figuras 300 DPI...", flush=True)
        _gerar_benchmark_e_figuras_promocao(
            challengers, X_train, X_cal, X_test, desfechos,
            estrato_tc.iloc[idx_tr], top_cid3_por_cap, f"Brasil {fmt_rng(todas_comps)}"
        )
        print(f"  ✓ Backup do Champion anterior salvo em: {bkp_dir}", flush=True)

    # Registrar histórico MLOps (tracked em Git para exibição na Aba 3)
    reg_prev = {"historico_execucoes": []}
    if REGISTRO_PATH.exists():
        try:
            with open(REGISTRO_PATH, "r", encoding="utf-8") as f:
                reg_prev = json.load(f)
        except Exception:
            reg_prev = {"historico_execucoes": []}

    run_entry = {
        "data_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_utc": datetime.now(timezone.utc).isoformat(),
        "estrategia": args.estrategia,
        "janela_treino_cal": fmt_rng(train_comps),
        "janela_teste_temporal": fmt_rng(test_comps),
        "n_treino": int(len(X_train)),
        "n_cal": int(len(X_cal)),
        "n_teste_temporal": int(len(X_test)),
        "decisao": status_acao,
        "promovido": bool(promover and not args.dry_run),
        "motivo": motivo,
        "comparativo": comparativo,
        "duracao_segundos": round(time.time() - t0, 1),
    }

    historico_runs = reg_prev.get("historico_execucoes", [])
    historico_runs.append(run_entry)
    historico_runs = historico_runs[-15:]

    champion_atual = (
        {
            "versao": "V5 Contínuo" if (promover and not args.dry_run) else reg_prev.get("champion_atual", {}).get("versao", "V5 Nacional (2024)"),
            "base_treino": fmt_rng(train_comps),
            "atualizado_em": run_entry["data_local"],
        }
        if (promover and not args.dry_run)
        else reg_prev.get("champion_atual", {"versao": "V5 Nacional (Base 2024)", "base_treino": "2024/01 → 2024/12", "atualizado_em": "2026-11-25"})
    )

    payload_reg = {
        "champion_atual": champion_atual,
        "ultima_avaliacao": run_entry,
        "historico_execucoes": historico_runs,
    }
    tmp_reg = REGISTRO_PATH.with_suffix(".json.tmp")
    with open(tmp_reg, "w", encoding="utf-8") as f:
        json.dump(payload_reg, f, indent=2, ensure_ascii=False)
    tmp_reg.replace(REGISTRO_PATH)

    print(f"Registro MLOps atualizado em: {REGISTRO_PATH} ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
