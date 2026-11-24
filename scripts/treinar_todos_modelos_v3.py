# -*- coding: utf-8 -*-
"""
Treinamento Abrangente dos 4 Modelos Definitivos V3 (SIH/SUS Acre 2024):
- Logistic Regression
- Random Forest
- XGBoost (corrigido sem SMOTE)
- LightGBM

Dois desfechos concomitantes:
1. Longa Permanência (>= 15 dias)
2. Óbito Intra-hospitalar

Gera todas as métricas reais em 10 dobras estratificadas e atualiza todas as figuras em figuras/.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             roc_curve, precision_recall_curve, confusion_matrix,
                             accuracy_score, precision_score, recall_score, f1_score)
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configurações visuais (Padrão USP/Esalq)
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
rcParams['font.size'] = 11
rcParams['axes.linewidth'] = 1.5
rcParams['axes.edgecolor'] = 'black'
rcParams['axes.grid'] = False
rcParams['figure.facecolor'] = 'white'
rcParams['axes.facecolor'] = 'white'
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DADOS_PATH = os.path.join(BASE_DIR, 'dados', 'sih_modelagem_admissao_v2.csv')
FIGURAS_DIR = os.path.join(BASE_DIR, 'figuras')
MODELOS_DIR = os.path.join(BASE_DIR, 'modelos')

os.makedirs(FIGURAS_DIR, exist_ok=True)
os.makedirs(MODELOS_DIR, exist_ok=True)

print("="*75)
print("INICIANDO TREINAMENTO ABRANGENTE DOS 4 MODELOS DEFINITIVOS V3")
print("="*75)

df = pd.read_csv(DADOS_PATH)
total = len(df)

features_num = ['idade_anos']
features_cat = [
    'sexo',
    'capitulo_cid',
    'especialidade_leito',
    'carater_internacao',
    'complexidade',
    'transferido_intermun',
    'internacao_fim_de_semana'
]

for col in features_cat:
    df[col] = df[col].astype(str)

X = df[features_num + features_cat]
y_longa = df['Longa_Permanencia']
y_obito = df['Obito']

print(f"Coorte Total: {total:,} internações")
print(f"Features de Admissão ({len(features_num) + len(features_cat)}): {features_num + features_cat}")
print(f"Desfecho 1: Longa Permanência = {y_longa.sum():,} ({y_longa.mean():.2%}) | Razão = {(total-y_longa.sum())/y_longa.sum():.2f}:1")
print(f"Desfecho 2: Óbito Hospitalar  = {y_obito.sum():,} ({y_obito.mean():.2%}) | Razão = {(total-y_obito.sum())/y_obito.sum():.2f}:1")

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), features_num),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), features_cat)
    ])

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

weight_longa = (total - y_longa.sum()) / y_longa.sum()
weight_obito = (total - y_obito.sum()) / y_obito.sum()

desfechos_configs = {
    'Longa_Permanencia': {
        'nome': 'Longa Permanência',
        'y': y_longa,
        'weight': weight_longa,
        'modelos': {
            'Logistic Regression': LogisticRegression(C=0.5, max_iter=1000, random_state=42, class_weight='balanced'),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42, class_weight='balanced'),
            'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.08, max_depth=5, subsample=0.8,
                                     colsample_bytree=0.8, scale_pos_weight=weight_longa, n_jobs=-1,
                                     random_state=42, eval_metric='logloss'),
            'LightGBM': LGBMClassifier(n_estimators=100, learning_rate=0.08, num_leaves=31,
                                       scale_pos_weight=weight_longa, subsample=0.8, colsample_bytree=0.8,
                                       n_jobs=-1, random_state=42, verbose=-1)
        }
    },
    'Obito': {
        'nome': 'Óbito Intra-hospitalar',
        'y': y_obito,
        'weight': weight_obito,
        'modelos': {
            'Logistic Regression': LogisticRegression(C=0.5, max_iter=1000, random_state=42, class_weight='balanced'),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42, class_weight='balanced'),
            'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.08, max_depth=5, subsample=0.8,
                                     colsample_bytree=0.8, scale_pos_weight=weight_obito, n_jobs=-1,
                                     random_state=42, eval_metric='logloss'),
            'LightGBM': LGBMClassifier(n_estimators=100, learning_rate=0.08, num_leaves=31,
                                       scale_pos_weight=weight_obito, subsample=0.8, colsample_bytree=0.8,
                                       n_jobs=-1, random_state=42, verbose=-1)
        }
    }
}

tabela_geral_metricas = []
curvas_completas = {}
cms_completas = {}
importancias_completas = {}

for target_id, t_cfg in desfechos_configs.items():
    t_nome = t_cfg['nome']
    y_target = t_cfg['y']
    curvas_completas[target_id] = {}
    cms_completas[target_id] = {}
    importancias_completas[target_id] = {}

    print(f"\n{'='*70}")
    print(f"PROCESSANDO DESFECHO: {t_nome.upper()}")
    print(f"{'='*70}")

    for mod_name, model_instance in t_cfg['modelos'].items():
        print(f"\n--> Treinando e avaliando: {mod_name}...")
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model_instance)
        ])

        # 1. 10-Fold CV predict_proba
        y_proba = cross_val_predict(pipe, X, y_target, cv=cv, method='predict_proba')[:, 1]
        y_pred = (y_proba >= 0.50).astype(int)

        # 2. Métricas
        roc_auc = roc_auc_score(y_target, y_proba)
        pr_auc = average_precision_score(y_target, y_proba)
        brier = brier_score_loss(y_target, y_proba)

        cm = confusion_matrix(y_target, y_pred)
        tn, fp, fn, tp = cm.ravel()
        acc = accuracy_score(y_target, y_pred)
        prec = precision_score(y_target, y_pred, zero_division=0)
        rec = recall_score(y_target, y_pred)
        spec = tn / (tn + fp)
        f1 = f1_score(y_target, y_pred, zero_division=0)

        cms_completas[target_id][mod_name] = cm

        fpr, tpr, _ = roc_curve(y_target, y_proba)
        prec_curve, rec_curve, _ = precision_recall_curve(y_target, y_proba)
        prob_true, prob_pred = calibration_curve(y_target, y_proba, n_bins=10)

        curvas_completas[target_id][mod_name] = {
            'y_proba': y_proba,
            'fpr': fpr, 'tpr': tpr,
            'prec_curve': prec_curve, 'rec_curve': rec_curve,
            'prob_true': prob_true, 'prob_pred': prob_pred
        }

        # 3. Fit completo para importância e persistência
        pipe.fit(X, y_target)
        save_path = os.path.join(MODELOS_DIR, f"modelo_{target_id}_{mod_name.lower().replace(' ', '_')}_v3.joblib")
        joblib.dump(pipe, save_path)

        clf = pipe.named_steps['classifier']
        prep = pipe.named_steps['preprocessor']
        fnames = prep.get_feature_names_out()
        fnames_clean = [f.replace('num__', '').replace('cat__', '') for f in fnames]

        if hasattr(clf, 'feature_importances_'):
            imps = clf.feature_importances_
        elif hasattr(clf, 'coef_'):
            imps = np.abs(clf.coef_[0])
        else:
            imps = np.zeros(len(fnames_clean))
        importancias_completas[target_id][mod_name] = (fnames_clean, imps)

        print(f"  [{mod_name}] ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | Sensibilidade (Recall): {rec*100:.2f}% | Especificidade: {spec*100:.2f}% | Acurácia: {acc*100:.2f}% | Brier: {brier:.4f}")
        print(f"  Matriz Confusão: TN={tn:,} | FP={fp:,} | FN={fn:,} | TP={tp:,}")

        tabela_geral_metricas.append({
            'Desfecho': t_nome,
            'Modelo': mod_name,
            'ROC_AUC': roc_auc,
            'PR_AUC': pr_auc,
            'Brier_Score': brier,
            'Acuracia': acc,
            'Sensibilidade': rec,
            'Especificidade': spec,
            'Precisao': prec,
            'F1_Score': f1,
            'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp
        })

df_metricas_final = pd.DataFrame(tabela_geral_metricas)
print("\n" + "="*85)
print("TABELA COMPLETA DE DESEMPENHO DOS 4 MODELOS (V3):")
print("="*85)
print(df_metricas_final.to_string(index=False))

# =============================================================
# 4. GERAÇÃO DE TODAS AS FIGURAS ATUALIZADAS
# =============================================================
print("\n--> Gerando figuras de diagnóstico em figuras/...")

cores_modelos = {
    'Logistic Regression': '#1f77b4',
    'Random Forest': '#ff7f0e',
    'XGBoost': '#2ca02c',
    'LightGBM': '#d62728'
}

# FIGURA 1: PAINEL DE MATRIZES DE CONFUSÃO - LONGA PERMANÊNCIA (2x2)
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
axes = axes.flatten()
for ax, (m_name, cm) in zip(axes, cms_completas['Longa_Permanencia'].items()):
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}", ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black',
                    fontsize=12, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Curta', 'Longa'], fontsize=10)
    ax.set_yticklabels(['Curta', 'Longa'], fontsize=10)
    ax.set_xlabel('Predito', fontsize=10)
    ax.set_ylabel('Real', fontsize=10)
    ax.set_title(f"{m_name} (Longa)", fontsize=11, fontweight='bold')
    for spine in ax.spines.values(): spine.set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(FIGURAS_DIR, 'fig_matrizes_confusao_longa_v3.png'), dpi=300)
plt.close()
print("Salvo: figuras/fig_matrizes_confusao_longa_v3.png")

# FIGURA 2: PAINEL DE MATRIZES DE CONFUSÃO - ÓBITO (2x2)
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
axes = axes.flatten()
for ax, (m_name, cm) in zip(axes, cms_completas['Obito'].items()):
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}", ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black',
                    fontsize=12, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Vivo', 'Óbito'], fontsize=10)
    ax.set_yticklabels(['Vivo', 'Óbito'], fontsize=10)
    ax.set_xlabel('Predito', fontsize=10)
    ax.set_ylabel('Real', fontsize=10)
    ax.set_title(f"{m_name} (Óbito)", fontsize=11, fontweight='bold')
    for spine in ax.spines.values(): spine.set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(FIGURAS_DIR, 'fig_matrizes_confusao_obito_v3.png'), dpi=300)
plt.close()
print("Salvo: figuras/fig_matrizes_confusao_obito_v3.png")

# FIGURA 3: CURVAS ROC E PRECISION-RECALL (4 MODELOS x 2 DESFECHOS - PAINEL 2x2)
fig, axes = plt.subplots(2, 2, figsize=(13, 11))

# ROC Longa
ax = axes[0, 0]
for m_name in ['Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM']:
    d = curvas_completas['Longa_Permanencia'][m_name]
    auc_v = df_metricas_final[(df_metricas_final['Desfecho']=='Longa Permanência') & (df_metricas_final['Modelo']==m_name)]['ROC_AUC'].values[0]
    ax.plot(d['fpr'], d['tpr'], label=f"{m_name} (AUC={auc_v:.3f})", color=cores_modelos[m_name], linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax.set_title('ROC: Longa Permanência (Modelos V3)', fontsize=11, fontweight='bold')
ax.set_xlabel('Taxa Falsos Positivos', fontsize=10)
ax.set_ylabel('Sensibilidade (Recall)', fontsize=10)
ax.legend(loc='lower right', fontsize=9)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# PR Longa
ax = axes[0, 1]
b_longa = y_longa.mean()
for m_name in ['Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM']:
    d = curvas_completas['Longa_Permanencia'][m_name]
    pr_v = df_metricas_final[(df_metricas_final['Desfecho']=='Longa Permanência') & (df_metricas_final['Modelo']==m_name)]['PR_AUC'].values[0]
    ax.plot(d['rec_curve'], d['prec_curve'], label=f"{m_name} (PR={pr_v:.3f})", color=cores_modelos[m_name], linewidth=2)
ax.axhline(b_longa, color='gray', linestyle='--', label=f'Baseline ({b_longa:.1%})')
ax.set_title('Precision-Recall: Longa Permanência', fontsize=11, fontweight='bold')
ax.set_xlabel('Recall (Sensibilidade)', fontsize=10)
ax.set_ylabel('Precisão', fontsize=10)
ax.legend(loc='upper right', fontsize=9)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# ROC Óbito
ax = axes[1, 0]
for m_name in ['Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM']:
    d = curvas_completas['Obito'][m_name]
    auc_v = df_metricas_final[(df_metricas_final['Desfecho']=='Óbito Intra-hospitalar') & (df_metricas_final['Modelo']==m_name)]['ROC_AUC'].values[0]
    ax.plot(d['fpr'], d['tpr'], label=f"{m_name} (AUC={auc_v:.3f})", color=cores_modelos[m_name], linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax.set_title('ROC: Óbito Intra-hospitalar (Modelos V3)', fontsize=11, fontweight='bold')
ax.set_xlabel('Taxa Falsos Positivos', fontsize=10)
ax.set_ylabel('Sensibilidade (Recall)', fontsize=10)
ax.legend(loc='lower right', fontsize=9)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# PR Óbito
ax = axes[1, 1]
b_obito = y_obito.mean()
for m_name in ['Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM']:
    d = curvas_completas['Obito'][m_name]
    pr_v = df_metricas_final[(df_metricas_final['Desfecho']=='Óbito Intra-hospitalar') & (df_metricas_final['Modelo']==m_name)]['PR_AUC'].values[0]
    ax.plot(d['rec_curve'], d['prec_curve'], label=f"{m_name} (PR={pr_v:.3f})", color=cores_modelos[m_name], linewidth=2)
ax.axhline(b_obito, color='gray', linestyle='--', label=f'Baseline ({b_obito:.1%})')
ax.set_title('Precision-Recall: Óbito Intra-hospitalar', fontsize=11, fontweight='bold')
ax.set_xlabel('Recall (Sensibilidade)', fontsize=10)
ax.set_ylabel('Precisão', fontsize=10)
ax.legend(loc='upper right', fontsize=9)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(FIGURAS_DIR, 'fig_roc_pr_curvas_v3.png'), dpi=300)
plt.close()
print("Salvo: figuras/fig_roc_pr_curvas_v3.png")

# FIGURA 4: CURVAS DE CALIBRAÇÃO (PAINEL 1x2)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
for m_name in ['Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM']:
    d = curvas_completas['Longa_Permanencia'][m_name]
    ax1.plot(d['prob_pred'], d['prob_true'], 's-', label=m_name, color=cores_modelos[m_name], linewidth=1.5, markersize=4)
ax1.plot([0, 1], [0, 1], 'k--', label='Perfeita Calibração')
ax1.set_title('Calibração: Longa Permanência', fontsize=11, fontweight='bold')
ax1.set_xlabel('Probabilidade Média Predita', fontsize=10)
ax1.set_ylabel('Fração Real Positivos', fontsize=10)
ax1.legend(fontsize=9)
ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

for m_name in ['Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM']:
    d = curvas_completas['Obito'][m_name]
    ax2.plot(d['prob_pred'], d['prob_true'], 's-', label=m_name, color=cores_modelos[m_name], linewidth=1.5, markersize=4)
ax2.plot([0, 1], [0, 1], 'k--', label='Perfeita Calibração')
ax2.set_title('Calibração: Óbito Intra-hospitalar', fontsize=11, fontweight='bold')
ax2.set_xlabel('Probabilidade Média Predita', fontsize=10)
ax2.set_ylabel('Fração Real Positivos', fontsize=10)
ax2.legend(fontsize=9)
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(FIGURAS_DIR, 'fig_calibracao_probabilidade_v3.png'), dpi=300)
plt.close()
print("Salvo: figuras/fig_calibracao_probabilidade_v3.png")

# FIGURA 5: IMPORTÂNCIA CLÍNICA AGREGADA (LIGHTGBM V3)
def agregar_dimensao(fnames, imps):
    grupos = {
        'Idade do Paciente': 0.0,
        'Diagnóstico CID-10': 0.0,
        'Especialidade do Leito': 0.0,
        'Caráter da Internação': 0.0,
        'Transferência Intermunicipal': 0.0,
        'Complexidade': 0.0,
        'Sexo': 0.0,
        'Internação Fim de Semana': 0.0
    }
    for f, imp in zip(fnames, imps):
        if 'idade_anos' in f: grupos['Idade do Paciente'] += imp
        elif 'capitulo_cid' in f: grupos['Diagnóstico CID-10'] += imp
        elif 'especialidade_leito' in f: grupos['Especialidade do Leito'] += imp
        elif 'carater_internacao' in f: grupos['Caráter da Internação'] += imp
        elif 'transferido_intermun' in f: grupos['Transferência Intermunicipal'] += imp
        elif 'complexidade' in f: grupos['Complexidade'] += imp
        elif 'sexo' in f: grupos['Sexo'] += imp
        elif 'internacao_fim_de_semana' in f: grupos['Internação Fim de Semana'] += imp
    return pd.Series(grupos).sort_values(ascending=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
s_longa = agregar_dimensao(*importancias_completas['Longa_Permanencia']['LightGBM'])
s_obito = agregar_dimensao(*importancias_completas['Obito']['LightGBM'])

ax1.barh(s_longa.index, s_longa.values, color='#2b5c8f', edgecolor='black', linewidth=0.5)
ax1.set_xlabel('Importância Agregada (Splits)', fontsize=10)
ax1.set_title('Dimensões Clínicas: Longa Permanência (LightGBM)', fontsize=11, fontweight='bold')
ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

ax2.barh(s_obito.index, s_obito.values, color='#c0392b', edgecolor='black', linewidth=0.5)
ax2.set_xlabel('Importância Agregada (Splits)', fontsize=10)
ax2.set_title('Dimensões Clínicas: Óbito Intra-hospitalar (LightGBM)', fontsize=11, fontweight='bold')
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(FIGURAS_DIR, 'fig_importancias_admissao_v3.png'), dpi=300)
plt.close()
print("Salvo: figuras/fig_importancias_admissao_v3.png")

print("\n" + "="*75)
print("TODOS OS 4 MODELOS FORAM TREINADOS E AS FIGURAS FORAM ATUALIZADAS!")
print("="*75)
