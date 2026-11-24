# -*- coding: utf-8 -*-
"""
Treinamento e Calibração Nacional V4 (SIH/SUS Brasil 2024 - 27 UFs):
- Coorte estratificada nacional de 1.500.000 internações extraída via DuckDB (Parquet Hive).
- Split rigoroso: 900.000 Treino | 300.000 Calibração Isotônica | 300.000 Teste Cego.
- 4 Algoritmos: Regressão Logística, Random Forest, XGBoost (hist) e LightGBM.
- Dois desfechos concomitantes:
  1. Longa Permanência (>= 15 dias)
  2. Óbito Intra-hospitalar
- Calibração Isotônica dedicada para probabilidades clínicas reais (Brier Score otimizado)
  e cálculo do limiar clínico ótimo (Youden's J) na amostra de calibração.
"""

import json
import os
import time
import warnings
from pathlib import Path

import duckdb
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from matplotlib import rcParams
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# Configurações visuais (300 DPI, padrão científico)
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

BASE_DIR = Path(__file__).resolve().parent.parent
PARQUET_GLOB = str(BASE_DIR / "dados" / "parquet_nacional" / "**" / "*.parquet").replace("\\", "/")
FIGURAS_DIR = BASE_DIR / "figuras"
MODELOS_DIR = BASE_DIR / "modelos"
FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
MODELOS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_NUM = ["idade_anos"]
FEATURES_CAT = [
    "sexo",
    "capitulo_cid",
    "especialidade_leito",
    "carater_internacao",
    "complexidade",
    "transferido_intermun",
    "internacao_fim_de_semana",
]

print("=" * 80)
print("TREINAMENTO E CALIBRAÇÃO NACIONAL V4 — SIH/SUS BRASIL 2024 (27 UFs)")
print("=" * 80)

t_inicio = time.time()

# 1. Extração via DuckDB (1.500.000 internações amostradas uniformemente dos 13,58M)
print("\n[1/5] Extraindo coorte estratificada nacional (1.500.000 registros) via DuckDB...")
con = duckdb.connect(database=":memory:")
query = f"""
SELECT
    uf,
    idade_anos,
    sexo,
    capitulo_cid,
    especialidade_leito,
    carater_internacao,
    complexidade,
    transferido_intermun,
    internacao_fim_de_semana,
    Longa_Permanencia,
    Obito
FROM read_parquet('{PARQUET_GLOB}', hive_partitioning=true)
USING SAMPLE 1500000 ROWS (reservoir, 42)
"""
df = con.execute(query).df()
con.close()

for c in FEATURES_CAT:
    df[c] = df[c].astype(str)

print(f"  ✓ Coorte carregada: {len(df):,} internações cobrindo {df['uf'].nunique()} UFs em {time.time() - t_inicio:.1f}s")
print(f"  • Prevalência Longa Permanência: {df['Longa_Permanencia'].mean():.2%}")
print(f"  • Prevalência Óbito:             {df['Obito'].mean():.2%}")

# 2. Split Triplo Estratificado: Treino (900k) / Calibração (300k) / Teste Cego (300k)
estrato = df["Obito"].astype(str) + "_" + df["Longa_Permanencia"].astype(str)
X = df[FEATURES_NUM + FEATURES_CAT]
y_longa = df["Longa_Permanencia"].astype(int)
y_obito = df["Obito"].astype(int)

idx_all = np.arange(len(df))
idx_train_cal, idx_test = train_test_split(idx_all, test_size=0.20, stratify=estrato, random_state=42)
idx_train, idx_cal = train_test_split(idx_train_cal, test_size=0.25, stratify=estrato.iloc[idx_train_cal], random_state=42)

X_train, X_cal, X_test = X.iloc[idx_train], X.iloc[idx_cal], X.iloc[idx_test]
print(f"  ✓ Splits -> Treino: {len(X_train):,} | Calibração: {len(X_cal):,} | Teste Cego: {len(X_test):,}")

# Subamostra para Random Forest (250k linhas de treino para manter execução rápida)
rng = np.random.RandomState(42)
idx_rf = rng.choice(len(X_train), size=250_000, replace=False)

# Pré-processador compacto float32
def criar_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), FEATURES_NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32), FEATURES_CAT),
        ]
    )

desfechos = {
    "Longa_Permanencia": {
        "nome": "Longa Permanência",
        "y_train": y_longa.iloc[idx_train],
        "y_cal": y_longa.iloc[idx_cal],
        "y_test": y_longa.iloc[idx_test],
    },
    "Obito": {
        "nome": "Óbito Intra-hospitalar",
        "y_train": y_obito.iloc[idx_train],
        "y_cal": y_obito.iloc[idx_cal],
        "y_test": y_obito.iloc[idx_test],
    },
}

def criar_modelos(scale_w: float):
    # Treinamos com scale_pos_weight moderado/nativo + calibração isotônica para garantir discriminação E probabilidade real
    return {
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=500, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=80, max_depth=12, min_samples_leaf=50, n_jobs=-1, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=180, learning_rate=0.08, max_depth=6, subsample=0.8,
            colsample_bytree=0.8, tree_method="hist", n_jobs=-1,
            random_state=42, eval_metric="logloss"
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200, learning_rate=0.08, num_leaves=63, min_child_samples=100,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42, verbose=-1
        ),
    }

metricas_rows = []
curvas = {"Longa_Permanencia": {}, "Obito": {}}
cms = {"Longa_Permanencia": {}, "Obito": {}}
importancias = {"Longa_Permanencia": {}, "Obito": {}}
limiares_producao = {}

# 3. Treinamento, Calibração Isotônica e Avaliação em Teste Cego
for target_id, t_cfg in desfechos.items():
    print(f"\n{'=' * 80}")
    print(f"[2/5] TREINANDO E CALIBRANDO DESFECHO: {t_cfg['nome'].upper()}")
    print(f"{'=' * 80}")

    y_tr = t_cfg["y_train"]
    y_ca = t_cfg["y_cal"]
    y_te = t_cfg["y_test"]
    scale_w = (len(y_tr) - y_tr.sum()) / y_tr.sum()

    modelos = criar_modelos(scale_w)

    for mod_name, clf_base in modelos.items():
        t_m = time.time()
        pipe = Pipeline([
            ("preprocessor", criar_preprocessor()),
            ("classifier", clf_base),
        ])

        if mod_name == "Random Forest":
            pipe.fit(X_train.iloc[idx_rf], y_tr.iloc[idx_rf])
        else:
            pipe.fit(X_train, y_tr)

        # Probabilidades antes da calibração (para comparação no gráfico)
        p_test_raw = pipe.predict_proba(X_test)[:, 1]
        brier_raw = brier_score_loss(y_te, p_test_raw)

        # Calibração Isotônica dedicada no split X_cal (sem tocar em X_test!)
        calibrator = CalibratedClassifierCV(FrozenEstimator(pipe), method="isotonic")
        calibrator.fit(X_cal, y_ca)

        p_cal_fit = calibrator.predict_proba(X_cal)[:, 1]
        p_test_cal = calibrator.predict_proba(X_test)[:, 1]

        # Limiar ótimo clínico (Índice de Youden: maximiza Sensibilidade + Especificidade - 1) em X_cal
        fpr_c, tpr_c, thr_c = roc_curve(y_ca, p_cal_fit)
        idx_youden = np.argmax(tpr_c - fpr_c)
        limiar_otimo = float(thr_c[idx_youden])

        # Métricas no Teste Cego (300.000 internações nunca vistas!)
        roc_auc = roc_auc_score(y_te, p_test_cal)
        pr_auc = average_precision_score(y_te, p_test_cal)
        brier_cal = brier_score_loss(y_te, p_test_cal)

        y_pred = (p_test_cal >= limiar_otimo).astype(int)
        cm = confusion_matrix(y_te, y_pred)
        tn, fp, fn, tp = cm.ravel()
        acc = accuracy_score(y_te, y_pred)
        prec = precision_score(y_te, y_pred, zero_division=0)
        rec = recall_score(y_te, y_pred)
        spec = tn / (tn + fp)
        f1 = f1_score(y_te, y_pred, zero_division=0)

        cms[target_id][mod_name] = (cm, limiar_otimo)

        fpr, tpr, _ = roc_curve(y_te, p_test_cal)
        prec_curve, rec_curve, _ = precision_recall_curve(y_te, p_test_cal)
        prob_true_raw, prob_pred_raw = calibration_curve(y_te, p_test_raw, n_bins=10)
        prob_true_cal, prob_pred_cal = calibration_curve(y_te, p_test_cal, n_bins=10)

        curvas[target_id][mod_name] = {
            "fpr": fpr, "tpr": tpr,
            "prec_curve": prec_curve, "rec_curve": rec_curve,
            "prob_true_raw": prob_true_raw, "prob_pred_raw": prob_pred_raw,
            "prob_true_cal": prob_true_cal, "prob_pred_cal": prob_pred_cal,
            "roc_auc": roc_auc, "pr_auc": pr_auc, "brier_cal": brier_cal,
        }

        # Importância das variáveis (do estimador base)
        prep = pipe.named_steps["preprocessor"]
        clf = pipe.named_steps["classifier"]
        fnames = [f.replace("num__", "").replace("cat__", "") for f in prep.get_feature_names_out()]
        if hasattr(clf, "feature_importances_"):
            imps = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            imps = np.abs(clf.coef_[0])
        else:
            imps = np.zeros(len(fnames))
        importancias[target_id][mod_name] = (fnames, imps)

        # Persistir modelo calibrado V4
        slug = mod_name.lower().replace(" ", "_")
        save_path = MODELOS_DIR / f"modelo_admissao_{'longa' if target_id == 'Longa_Permanencia' else 'obito'}_{slug}_v4.joblib"
        joblib.dump(calibrator, save_path)

        if mod_name == "LightGBM":
            limiares_producao[target_id] = {
                "limiar_youden": round(limiar_otimo, 4),
                "prevalencia_nacional": round(float(y_te.mean()), 4),
                "roc_auc": round(roc_auc, 4),
                "pr_auc": round(pr_auc, 4),
                "brier_cal": round(brier_cal, 4),
            }

        print(
            f"  ✓ [{mod_name:<19}] ({time.time()-t_m:4.1f}s) | ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | "
            f"Brier: {brier_raw:.4f}→{brier_cal:.4f} | Limiar*: {limiar_otimo*100:4.1f}% | "
            f"Sens: {rec*100:5.2f}% | Spec: {spec*100:5.2f}% | Acc: {acc*100:5.2f}%"
        )

        metricas_rows.append({
            "Desfecho": t_cfg["nome"],
            "Modelo": mod_name,
            "ROC_AUC": round(roc_auc, 4),
            "PR_AUC": round(pr_auc, 4),
            "Brier_Raw": round(brier_raw, 4),
            "Brier_Calibrado": round(brier_cal, 4),
            "Limiar_Otimo_Pct": round(limiar_otimo * 100, 2),
            "Sensibilidade_Pct": round(rec * 100, 2),
            "Especificidade_Pct": round(spec * 100, 2),
            "Acuracia_Pct": round(acc * 100, 2),
            "Precisao_Pct": round(prec * 100, 2),
            "F1_Score_Pct": round(f1 * 100, 2),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        })

# Salvar tabela de métricas e limiares de produção
df_metricas = pd.DataFrame(metricas_rows)
df_metricas.to_csv(MODELOS_DIR / "metricas_v4_nacional.csv", index=False)
with open(MODELOS_DIR / "config_limiares_v4.json", "w", encoding="utf-8") as f:
    json.dump(limiares_producao, f, indent=2, ensure_ascii=False)

print("\n[3/5] Tabela Consolidada (Teste Cego Nacional — 300.000 internações):")
print(df_metricas.to_string(index=False))

# =============================================================
# 4. GERAÇÃO DAS FIGURAS V4 EM 300 DPI
# =============================================================
print("\n[4/5] Gerando figuras diagnósticas V4 em 300 DPI...")
cores = {
    "Logistic Regression": "#64748b",
    "Random Forest": "#10b981",
    "XGBoost": "#f59e0b",
    "LightGBM": "#2563eb",
}

# Figura 1: Curvas ROC e Precision-Recall (V4 Nacional)
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
for row_idx, (t_id, t_nome) in enumerate([("Obito", "Óbito Intra-hospitalar"), ("Longa_Permanencia", "Longa Permanência (≥15d)")]):
    ax_roc = axes[row_idx, 0]
    ax_pr = axes[row_idx, 1]
    for mod_name, c_data in curvas[t_id].items():
        ax_roc.plot(c_data["fpr"], c_data["tpr"], lw=2.2, color=cores[mod_name],
                    label=f"{mod_name} (AUC = {c_data['roc_auc']:.4f})")
        ax_pr.plot(c_data["rec_curve"], c_data["prec_curve"], lw=2.2, color=cores[mod_name],
                   label=f"{mod_name} (PR-AUC = {c_data['pr_auc']:.4f})")
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6)
    ax_roc.set_title(f"Curva ROC — {t_nome} (Brasil 2024)", fontweight="bold")
    ax_roc.set_xlabel("Taxa de Falso Positivo (1 - Especificidade)")
    ax_roc.set_ylabel("Sensibilidade (Recall)")
    ax_roc.legend(loc="lower right", frameon=True, fontsize=9)

    prev_rate = float(desfechos[t_id]["y_test"].mean())
    ax_pr.axhline(prev_rate, color="black", linestyle="--", lw=1.2, alpha=0.6, label=f"Baseline ({prev_rate:.2%})")
    ax_pr.set_title(f"Curva Precision-Recall — {t_nome} (Brasil 2024)", fontweight="bold")
    ax_pr.set_xlabel("Sensibilidade (Recall)")
    ax_pr.set_ylabel("Precisão (VPP)")
    ax_pr.legend(loc="upper right", frameon=True, fontsize=9)

plt.tight_layout()
plt.savefig(FIGURAS_DIR / "fig_roc_pr_curvas_v4.png")
plt.close()

# Figura 2: Calibração de Probabilidades V4 (Diagonal Perfeita)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for idx, (t_id, t_nome) in enumerate([("Obito", "Óbito Intra-hospitalar"), ("Longa_Permanencia", "Longa Permanência (≥15d)")]):
    ax = axes[idx]
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Calibração Perfeita")
    for mod_name, c_data in curvas[t_id].items():
        ax.plot(c_data["prob_pred_cal"], c_data["prob_true_cal"], marker="o", lw=2.2, color=cores[mod_name],
                label=f"{mod_name} (Brier = {c_data['brier_cal']:.4f})")
    ax.set_title(f"Calibração Isotônica — {t_nome}", fontweight="bold")
    ax.set_xlabel("Probabilidade Média Prevista")
    ax.set_ylabel("Fração Real Observada")
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    ax.set_xlim(0, 0.75)
    ax.set_ylim(0, 0.75)

plt.tight_layout()
plt.savefig(FIGURAS_DIR / "fig_calibracao_probabilidade_v4.png")
plt.close()

# Figura 3: Importâncias das Variáveis (LightGBM V4 Nacional)
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
for idx, (t_id, t_nome) in enumerate([("Obito", "Óbito Intra-hospitalar"), ("Longa_Permanencia", "Longa Permanência (≥15d)")]):
    ax = axes[idx]
    fnames, imps = importancias[t_id]["LightGBM"]
    df_imp = pd.DataFrame({"Feature": fnames, "Importance": imps}).sort_values("Importance", ascending=True).tail(12)
    ax.barh(df_imp["Feature"], df_imp["Importance"], color="#2563eb" if idx == 1 else "#dc2626", alpha=0.88)
    ax.set_title(f"Top 12 Variáveis (LightGBM Brasil) — {t_nome}", fontweight="bold")
    ax.set_xlabel("Importância (Splits)")

plt.tight_layout()
plt.savefig(FIGURAS_DIR / "fig_importancias_admissao_v4.png")
plt.close()

# Figura 4: Matrizes de Confusão V4 no Limiar Clínico Ótimo
for t_id, t_nome, fname_out in [
    ("Obito", "Óbito Intra-hospitalar", "fig_matrizes_confusao_obito_v4.png"),
    ("Longa_Permanencia", "Longa Permanência (≥15d)", "fig_matrizes_confusao_longa_v4.png"),
]:
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for idx, mod_name in enumerate(["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]):
        ax = axes[idx]
        cm, thr = cms[t_id][mod_name]
        cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
        ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
        for i in range(2):
            for j in range(2):
                cor_txt = "white" if cm_pct[i, j] > 50 else "black"
                ax.text(j, i, f"{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)", ha="center", va="center", color=cor_txt, fontsize=9, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Neg", "Pos"])
        ax.set_yticklabels(["Real Neg", "Real Pos"])
        ax.set_title(f"{mod_name}\n(Limiar: {thr*100:.1f}%)", fontsize=10, fontweight="bold")
    fig.suptitle(f"Matrizes de Confusão V4 (Teste Cego Brasil = 300k) — {t_nome}", fontsize=13, fontweight="bold", y=1.04)
    plt.tight_layout()
    plt.savefig(FIGURAS_DIR / fname_out)
    plt.close()

print(f"\n[5/5] Concluído com sucesso em {time.time() - t_inicio:.1f}s!")
print(f"  • Modelos V4 salvos em: {MODELOS_DIR}")
print(f"  • Figuras V4 salvas em: {FIGURAS_DIR}")
