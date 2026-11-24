# -*- coding: utf-8 -*-
"""
Modelagem Definitiva V3 (Clínica e Ética) e Preparação da Base Epidemiológica:
- Target 1: Longa Permanência (>= 15 dias)
- Target 2: Óbito Intra-hospitalar

Features Preditoras Estritas (Sem mes_internacao e Sem raca_cor):
- Numéricas: idade_anos
- Categóricas: sexo, capitulo_cid, especialidade_leito, carater_internacao,
               complexidade, transferido_intermun, internacao_fim_de_semana

Base Epidemiológica:
- Contém todas as variáveis demográficas, raça/cor, sazonalidade e geografia
  para a futura geração do Dashboard de Vigilância em Saúde.
"""
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
from lightgbm import LGBMClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             roc_curve, precision_recall_curve, confusion_matrix,
                             accuracy_score, precision_score, recall_score, f1_score)
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configurações de plotagem do projeto
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

# =============================================================
# 1. CARREGAMENTO DOS DADOS E PREPARAÇÃO DA BASE EPIDEMIOLÓGICA
# =============================================================
print("--> Carregando dados completos de admissão...")
df_completo = pd.read_csv("sih_modelagem_admissao_v2.csv")
total = len(df_completo)

# Criar e salvar a Base Epidemiológica Especializada
cols_epidemio = [
    'idade_anos', 'sexo', 'raca_cor_desc', 'capitulo_cid',
    'especialidade_leito', 'carater_internacao', 'complexidade',
    'transferido_intermun', 'mes_internacao', 'internacao_fim_de_semana',
    'Longa_Permanencia', 'Obito', 'qt_diarias'
]
df_epidemio = df_completo[cols_epidemio].copy()
df_epidemio.to_csv("sih_perfil_epidemiologico_acre.csv", index=False)
print("💾 Base Epidemiológica consolidada e salva: sih_perfil_epidemiologico_acre.csv")

# =============================================================
# 2. DEFINIÇÃO DAS FEATURES CLÍNICAS PURAS (V3 - MODELAGEM)
# =============================================================
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
    df_completo[col] = df_completo[col].astype(str)

X = df_completo[features_num + features_cat]
y_longa = df_completo['Longa_Permanencia']
y_obito = df_completo['Obito']

print(f"\nFeatures do Modelo V3 ({len(features_num) + len(features_cat)}):")
print(f"  Numéricas: {features_num}")
print(f"  Categóricas: {features_cat}")
print(f"  (mes_internacao e raca_cor expurgadas do vetor de admissão)")

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), features_num),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), features_cat)
    ])

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

weight_longa = (total - y_longa.sum()) / y_longa.sum()
weight_obito = (total - y_obito.sum()) / y_obito.sum()

modelos_configs = {
    'Longa_Permanencia': {
        'target_name': 'Longa Permanência',
        'y': y_longa,
        'weight': weight_longa,
        'Logistic Regression': LogisticRegression(C=0.5, max_iter=1000, random_state=42, class_weight='balanced'),
        'LightGBM': LGBMClassifier(n_estimators=100, learning_rate=0.08, num_leaves=31,
                                   scale_pos_weight=weight_longa, subsample=0.8, colsample_bytree=0.8,
                                   random_state=42, verbose=-1)
    },
    'Obito': {
        'target_name': 'Óbito Intra-hospitalar',
        'y': y_obito,
        'weight': weight_obito,
        'Logistic Regression': LogisticRegression(C=0.5, max_iter=1000, random_state=42, class_weight='balanced'),
        'LightGBM': LGBMClassifier(n_estimators=100, learning_rate=0.08, num_leaves=31,
                                   scale_pos_weight=weight_obito, subsample=0.8, colsample_bytree=0.8,
                                   random_state=42, verbose=-1)
    }
}

resultados_v3 = []
curvas_dict_v3 = {}
modelos_treinados_v3 = {}
importancias_dict_v3 = {}

# =============================================================
# 3. VALIDAÇÃO CRUZADA E TREINAMENTO V3
# =============================================================
for target_key, config in modelos_configs.items():
    target_label = config['target_name']
    y_target = config['y']
    print(f"\n{'='*65}")
    print(f"AVALIANDO DESFECHO (V3): {target_label.upper()}")
    print(f"{'='*65}")

    curvas_dict_v3[target_key] = {}
    importancias_dict_v3[target_key] = {}

    for mod_name in ['Logistic Regression', 'LightGBM']:
        print(f"\n---> Treinando {mod_name} para {target_label}...")
        base_estimator = config[mod_name]
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', base_estimator)
        ])

        y_proba = cross_val_predict(pipe, X, y_target, cv=cv, method='predict_proba')[:, 1]
        y_pred = (y_proba >= 0.50).astype(int)

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

        fpr, tpr, _ = roc_curve(y_target, y_proba)
        prec_curve, rec_curve, _ = precision_recall_curve(y_target, y_proba)
        prob_true, prob_pred = calibration_curve(y_target, y_proba, n_bins=10)

        curvas_dict_v3[target_key][mod_name] = {
            'y_proba': y_proba,
            'fpr': fpr, 'tpr': tpr,
            'prec_curve': prec_curve, 'rec_curve': rec_curve,
            'prob_true': prob_true, 'prob_pred': prob_pred
        }

        pipe.fit(X, y_target)
        modelos_treinados_v3[f"{target_key}_{mod_name}"] = pipe

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

        importancias_dict_v3[target_key][mod_name] = (fnames_clean, imps)

        print(f"Métricas ({mod_name} - {target_label} - V3):")
        print(f"  ROC-AUC:       {roc_auc:.4f}")
        print(f"  PR-AUC:        {pr_auc:.4f}")
        print(f"  Brier Score:   {brier:.4f}")
        print(f"  Acurácia:      {acc*100:.2f}%")
        print(f"  Sensibilidade: {rec*100:.2f}% (Recall)")
        print(f"  Especificidade:{spec*100:.2f}%")
        print(f"  Precisão:      {prec*100:.2f}%")
        print(f"  F1-Score:      {f1*100:.2f}%")
        print(f"  Matriz Confusão: TN={tn:,} | FP={fp:,} | FN={fn:,} | TP={tp:,}")

        resultados_v3.append({
            'Desfecho': target_label,
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

df_res_v3 = pd.DataFrame(resultados_v3)
print("\n" + "="*80)
print("TABELA CONSOLIDADA DE RESULTADOS V3 (CLÍNICO E ÉTICO):")
print("="*80)
print(df_res_v3.to_string(index=False))

# =============================================================
# 4. GERAÇÃO DOS GRÁFICOS V3 (300 DPI)
# =============================================================
print("\n--> Gerando figuras diagnósticas V3...")

# FIGURA 1: CURVAS ROC E PRECISION-RECALL (V3)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# ROC Longa
ax = axes[0, 0]
for mod_name, cor in [('Logistic Regression', '#1f77b4'), ('LightGBM', '#d62728')]:
    d = curvas_dict_v3['Longa_Permanencia'][mod_name]
    auc_val = df_res_v3[(df_res_v3['Desfecho']=='Longa Permanência') & (df_res_v3['Modelo']==mod_name)]['ROC_AUC'].values[0]
    ax.plot(d['fpr'], d['tpr'], label=f"{mod_name} (AUC = {auc_val:.3f})", color=cor, linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.6)
ax.set_title('ROC: Longa Permanência (Modelo V3 Clínico)', fontsize=11, fontweight='bold')
ax.set_xlabel('Taxa de Falsos Positivos', fontsize=10)
ax.set_ylabel('Sensibilidade (Recall)', fontsize=10)
ax.legend(loc='lower right', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# PR Longa
ax = axes[0, 1]
baseline_longa = y_longa.mean()
for mod_name, cor in [('Logistic Regression', '#1f77b4'), ('LightGBM', '#d62728')]:
    d = curvas_dict_v3['Longa_Permanencia'][mod_name]
    pr_val = df_res_v3[(df_res_v3['Desfecho']=='Longa Permanência') & (df_res_v3['Modelo']==mod_name)]['PR_AUC'].values[0]
    ax.plot(d['rec_curve'], d['prec_curve'], label=f"{mod_name} (PR-AUC = {pr_val:.3f})", color=cor, linewidth=2)
ax.axhline(baseline_longa, color='gray', linestyle='--', label=f'Baseline ({baseline_longa:.1%})')
ax.set_title('Precision-Recall: Longa Permanência (Modelo V3)', fontsize=11, fontweight='bold')
ax.set_xlabel('Recall (Sensibilidade)', fontsize=10)
ax.set_ylabel('Precisão', fontsize=10)
ax.legend(loc='upper right', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ROC Óbito
ax = axes[1, 0]
for mod_name, cor in [('Logistic Regression', '#1f77b4'), ('LightGBM', '#d62728')]:
    d = curvas_dict_v3['Obito'][mod_name]
    auc_val = df_res_v3[(df_res_v3['Desfecho']=='Óbito Intra-hospitalar') & (df_res_v3['Modelo']==mod_name)]['ROC_AUC'].values[0]
    ax.plot(d['fpr'], d['tpr'], label=f"{mod_name} (AUC = {auc_val:.3f})", color=cor, linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.6)
ax.set_title('ROC: Óbito Intra-hospitalar (Modelo V3 Clínico)', fontsize=11, fontweight='bold')
ax.set_xlabel('Taxa de Falsos Positivos', fontsize=10)
ax.set_ylabel('Sensibilidade (Recall)', fontsize=10)
ax.legend(loc='lower right', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# PR Óbito
ax = axes[1, 1]
baseline_obito = y_obito.mean()
for mod_name, cor in [('Logistic Regression', '#1f77b4'), ('LightGBM', '#d62728')]:
    d = curvas_dict_v3['Obito'][mod_name]
    pr_val = df_res_v3[(df_res_v3['Desfecho']=='Óbito Intra-hospitalar') & (df_res_v3['Modelo']==mod_name)]['PR_AUC'].values[0]
    ax.plot(d['rec_curve'], d['prec_curve'], label=f"{mod_name} (PR-AUC = {pr_val:.3f})", color=cor, linewidth=2)
ax.axhline(baseline_obito, color='gray', linestyle='--', label=f'Baseline ({baseline_obito:.1%})')
ax.set_title('Precision-Recall: Óbito Intra-hospitalar (Modelo V3)', fontsize=11, fontweight='bold')
ax.set_xlabel('Recall (Sensibilidade)', fontsize=10)
ax.set_ylabel('Precisão', fontsize=10)
ax.legend(loc='upper right', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('fig_roc_pr_curvas_v3.png', dpi=300)
plt.close()
print("Salvo: fig_roc_pr_curvas_v3.png")

# FIGURA 2: TOP IMPORTÂNCIAS COMPARATIVAS (LIGHTGBM V3)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Agrupar importâncias para mostrar a força consolidada das variáveis
def agregar_importancias(fnames, imps):
    grupos = {
        'Idade do Paciente': 0,
        'Diagnóstico CID-10': 0,
        'Especialidade do Leito': 0,
        'Caráter da Internação': 0,
        'Transferência Intermunicipal': 0,
        'Complexidade': 0,
        'Sexo': 0,
        'Internação Fim de Semana': 0
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
    
    s = pd.Series(grupos).sort_values(ascending=True)
    return s

imp_series_longa = agregar_importancias(*importancias_dict_v3['Longa_Permanencia']['LightGBM'])
imp_series_obito = agregar_importancias(*importancias_dict_v3['Obito']['LightGBM'])

ax1.barh(imp_series_longa.index, imp_series_longa.values, color='#2b5c8f', edgecolor='black', linewidth=0.5)
ax1.set_xlabel('Importância Agregada (Splits)', fontsize=10)
ax1.set_title('Importância por Grupo Clínico: Longa Permanência', fontsize=11, fontweight='bold')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

ax2.barh(imp_series_obito.index, imp_series_obito.values, color='#c0392b', edgecolor='black', linewidth=0.5)
ax2.set_xlabel('Importância Agregada (Splits)', fontsize=10)
ax2.set_title('Importância por Grupo Clínico: Óbito Intra-hospitalar', fontsize=11, fontweight='bold')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('fig_importancias_admissao_v3.png', dpi=300)
plt.close()
print("Salvo: fig_importancias_admissao_v3.png (Agrupada por Dimensão Clínica)")

# =============================================================
# 5. SALVAR MODELOS DEFINITIVOS V3
# =============================================================
joblib.dump(modelos_treinados_v3['Longa_Permanencia_LightGBM'], 'modelo_admissao_longa_lightgbm_v3.joblib')
joblib.dump(modelos_treinados_v3['Obito_LightGBM'], 'modelo_admissao_obito_lightgbm_v3.joblib')
joblib.dump(modelos_treinados_v3['Longa_Permanencia_Logistic Regression'], 'modelo_admissao_longa_lr_v3.joblib')
joblib.dump(modelos_treinados_v3['Obito_Logistic Regression'], 'modelo_admissao_obito_lr_v3.joblib')
print("\n💾 Modelos V3 definitivos salvos (.joblib)")
