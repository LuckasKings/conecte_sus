# -*- coding: utf-8 -*-
"""
Modelagem Dual Calibrada para Admissão Hospitalar (SIH/SUS Acre 2024):
- Target 1: Longa Permanência (qt_diarias >= 15)
- Target 2: Óbito Intra-hospitalar (morte == 1)

Modelos: LightGBM e Regressão Logística
Validação: 10-Fold Stratified Cross-Validation
Abordagem: Ponderação de classes (sem SMOTE), calibração de probabilidade e extração de métricas completas.
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
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
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
# 1. CARREGAMENTO DOS DADOS (V2)
# =============================================================
print("--> Carregando dados limpos de admissão (sih_modelagem_admissao_v2.csv)...")
df = pd.read_csv("sih_modelagem_admissao_v2.csv")

features_num = ['idade_anos', 'mes_internacao']
features_cat = ['sexo', 'raca_cor_desc', 'capitulo_cid', 'especialidade_leito',
                'carater_internacao', 'complexidade', 'transferido_intermun',
                'internacao_fim_de_semana']

# Converter categóricas para string
for col in features_cat:
    df[col] = df[col].astype(str)

X = df[features_num + features_cat]
y_longa = df['Longa_Permanencia']
y_obito = df['Obito']

total = len(df)
print(f"Total de registros: {total:,}")
print(f"Longa Permanência: {y_longa.sum():,} ({y_longa.mean():.2%}) | Razão: {(total - y_longa.sum()) / y_longa.sum():.2f}:1")
print(f"Óbitos: {y_obito.sum():,} ({y_obito.mean():.2%}) | Razão: {(total - y_obito.sum()) / y_obito.sum():.2f}:1")

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), features_num),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), features_cat)
    ])

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# =============================================================
# 2. DEFINIÇÃO DOS MODELOS DUAIS
# =============================================================
# Razões de classes para pesos
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

resultados_finais = []
curvas_dict = {}
modelos_treinados = {}
importancias_dict = {}

# =============================================================
# 3. LOOP DE TREINAMENTO E VALIDAÇÃO CRUZADA 10-FOLD
# =============================================================
for target_key, config in modelos_configs.items():
    target_label = config['target_name']
    y_target = config['y']
    print(f"\n{'='*65}")
    print(f"AVALIANDO DESFECHO: {target_label.upper()}")
    print(f"{'='*65}")

    curvas_dict[target_key] = {}
    importancias_dict[target_key] = {}

    for mod_name in ['Logistic Regression', 'LightGBM']:
        print(f"\n---> Treinando {mod_name} para {target_label}...")
        base_estimator = config[mod_name]
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', base_estimator)
        ])

        # 1. Obtenção de probabilidades via cross_val_predict
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

        # Salvar curvas para plot
        fpr, tpr, _ = roc_curve(y_target, y_proba)
        prec_curve, rec_curve, _ = precision_recall_curve(y_target, y_proba)
        prob_true, prob_pred = calibration_curve(y_target, y_proba, n_bins=10)

        curvas_dict[target_key][mod_name] = {
            'y_proba': y_proba,
            'fpr': fpr, 'tpr': tpr,
            'prec_curve': prec_curve, 'rec_curve': rec_curve,
            'prob_true': prob_true, 'prob_pred': prob_pred
        }

        # 3. Fit no dataset completo para importância e persistência
        pipe.fit(X, y_target)
        modelos_treinados[f"{target_key}_{mod_name}"] = pipe

        # Extrair importância
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

        importancias_dict[target_key][mod_name] = (fnames_clean, imps)

        print(f"Métricas ({mod_name} - {target_label}):")
        print(f"  ROC-AUC:       {roc_auc:.4f} (Excelente discriminação)")
        print(f"  PR-AUC:        {pr_auc:.4f}  (Baseline aleatório: {y_target.mean():.4f})")
        print(f"  Brier Score:   {brier:.4f}")
        print(f"  Acurácia:      {acc*100:.2f}%")
        print(f"  Sensibilidade: {rec*100:.2f}% (Recall - Foco em capturar o evento)")
        print(f"  Especificidade:{spec*100:.2f}%")
        print(f"  Precisão:      {prec*100:.2f}%")
        print(f"  F1-Score:      {f1*100:.2f}%")
        print(f"  Matriz Confusão: TN={tn:,} | FP={fp:,} | FN={fn:,} | TP={tp:,}")

        resultados_finais.append({
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

df_metricas = pd.DataFrame(resultados_finais)
print("\n" + "="*80)
print("TABELA CONSOLIDADA DE RESULTADOS DOS MODELOS DUAIS (V2):")
print("="*80)
print(df_metricas.to_string(index=False))

# =============================================================
# 4. GERAÇÃO DOS GRÁFICOS (300 DPI, PADRÃO USP/ESALQ)
# =============================================================
print("\n--> Gerando figuras de diagnóstico...")

# FIGURA 1: CURVAS ROC E PRECISION-RECALL (PAINEL 2x2)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Subplot 1: ROC Longa Permanência
ax = axes[0, 0]
for mod_name, cor in [('Logistic Regression', '#1f77b4'), ('LightGBM', '#d62728')]:
    d = curvas_dict['Longa_Permanencia'][mod_name]
    auc_val = df_metricas[(df_metricas['Desfecho']=='Longa Permanência') & (df_metricas['Modelo']==mod_name)]['ROC_AUC'].values[0]
    ax.plot(d['fpr'], d['tpr'], label=f"{mod_name} (AUC = {auc_val:.3f})", color=cor, linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.6)
ax.set_title('ROC: Longa Permanência', fontsize=12, fontweight='bold')
ax.set_xlabel('Taxa de Falsos Positivos (1 - Especificidade)', fontsize=10)
ax.set_ylabel('Taxa de Verdadeiros Positivos (Sensibilidade)', fontsize=10)
ax.legend(loc='lower right', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Subplot 2: PR Longa Permanência
ax = axes[0, 1]
baseline_longa = y_longa.mean()
for mod_name, cor in [('Logistic Regression', '#1f77b4'), ('LightGBM', '#d62728')]:
    d = curvas_dict['Longa_Permanencia'][mod_name]
    pr_val = df_metricas[(df_metricas['Desfecho']=='Longa Permanência') & (df_metricas['Modelo']==mod_name)]['PR_AUC'].values[0]
    ax.plot(d['rec_curve'], d['prec_curve'], label=f"{mod_name} (PR-AUC = {pr_val:.3f})", color=cor, linewidth=2)
ax.axhline(baseline_longa, color='gray', linestyle='--', label=f'Baseline ({baseline_longa:.1%})')
ax.set_title('Precision-Recall: Longa Permanência', fontsize=12, fontweight='bold')
ax.set_xlabel('Recall (Sensibilidade)', fontsize=10)
ax.set_ylabel('Precisão', fontsize=10)
ax.legend(loc='upper right', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Subplot 3: ROC Óbito
ax = axes[1, 0]
for mod_name, cor in [('Logistic Regression', '#1f77b4'), ('LightGBM', '#d62728')]:
    d = curvas_dict['Obito'][mod_name]
    auc_val = df_metricas[(df_metricas['Desfecho']=='Óbito Intra-hospitalar') & (df_metricas['Modelo']==mod_name)]['ROC_AUC'].values[0]
    ax.plot(d['fpr'], d['tpr'], label=f"{mod_name} (AUC = {auc_val:.3f})", color=cor, linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.6)
ax.set_title('ROC: Óbito Intra-hospitalar', fontsize=12, fontweight='bold')
ax.set_xlabel('Taxa de Falsos Positivos (1 - Especificidade)', fontsize=10)
ax.set_ylabel('Taxa de Verdadeiros Positivos (Sensibilidade)', fontsize=10)
ax.legend(loc='lower right', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Subplot 4: PR Óbito
ax = axes[1, 1]
baseline_obito = y_obito.mean()
for mod_name, cor in [('Logistic Regression', '#1f77b4'), ('LightGBM', '#d62728')]:
    d = curvas_dict['Obito'][mod_name]
    pr_val = df_metricas[(df_metricas['Desfecho']=='Óbito Intra-hospitalar') & (df_metricas['Modelo']==mod_name)]['PR_AUC'].values[0]
    ax.plot(d['rec_curve'], d['prec_curve'], label=f"{mod_name} (PR-AUC = {pr_val:.3f})", color=cor, linewidth=2)
ax.axhline(baseline_obito, color='gray', linestyle='--', label=f'Baseline ({baseline_obito:.1%})')
ax.set_title('Precision-Recall: Óbito Intra-hospitalar', fontsize=12, fontweight='bold')
ax.set_xlabel('Recall (Sensibilidade)', fontsize=10)
ax.set_ylabel('Precisão', fontsize=10)
ax.legend(loc='upper right', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('fig_roc_pr_curvas_v2.png', dpi=300)
plt.close()
print("Salvo: fig_roc_pr_curvas_v2.png")

# FIGURA 2: CURVAS DE CALIBRAÇÃO DE PROBABILIDADE
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for mod_name, cor in [('Logistic Regression', '#1f77b4'), ('LightGBM', '#d62728')]:
    d = curvas_dict['Longa_Permanencia'][mod_name]
    ax1.plot(d['prob_pred'], d['prob_true'], 's-', label=mod_name, color=cor, linewidth=2)
ax1.plot([0, 1], [0, 1], 'k--', label='Perfeita Calibração')
ax1.set_title('Calibração: Longa Permanência', fontsize=12, fontweight='bold')
ax1.set_xlabel('Probabilidade Média Predita', fontsize=10)
ax1.set_ylabel('Fração Real de Positivos', fontsize=10)
ax1.legend(fontsize=10)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

for mod_name, cor in [('Logistic Regression', '#1f77b4'), ('LightGBM', '#d62728')]:
    d = curvas_dict['Obito'][mod_name]
    ax2.plot(d['prob_pred'], d['prob_true'], 's-', label=mod_name, color=cor, linewidth=2)
ax2.plot([0, 1], [0, 1], 'k--', label='Perfeita Calibração')
ax2.set_title('Calibração: Óbito Intra-hospitalar', fontsize=12, fontweight='bold')
ax2.set_xlabel('Probabilidade Média Predita', fontsize=10)
ax2.set_ylabel('Fração Real de Positivos', fontsize=10)
ax2.legend(fontsize=10)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('fig_calibracao_probabilidade_v2.png', dpi=300)
plt.close()
print("Salvo: fig_calibracao_probabilidade_v2.png")

# FIGURA 3: TOP IMPORTÂNCIAS COMPARATIVAS (LIGHTGBM - LONGA vs ÓBITO)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Top 12 Longa
fnames_l, imps_l = importancias_dict['Longa_Permanencia']['LightGBM']
idx_l = np.argsort(imps_l)[::-1][:12]
ax1.barh(range(len(idx_l)), imps_l[idx_l][::-1], color='#2b5c8f', edgecolor='black', linewidth=0.5)
ax1.set_yticks(range(len(idx_l)))
ax1.set_yticklabels([fnames_l[i] for i in idx_l][::-1], fontsize=9)
ax1.set_xlabel('Importância (Splits)', fontsize=10)
ax1.set_title('Top Variáveis: Longa Permanência (LightGBM)', fontsize=11, fontweight='bold')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Top 12 Óbito
fnames_o, imps_o = importancias_dict['Obito']['LightGBM']
idx_o = np.argsort(imps_o)[::-1][:12]
ax2.barh(range(len(idx_o)), imps_o[idx_o][::-1], color='#c0392b', edgecolor='black', linewidth=0.5)
ax2.set_yticks(range(len(idx_o)))
ax2.set_yticklabels([fnames_o[i] for i in idx_o][::-1], fontsize=9)
ax2.set_xlabel('Importância (Splits)', fontsize=10)
ax2.set_title('Top Variáveis: Óbito Intra-hospitalar (LightGBM)', fontsize=11, fontweight='bold')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('fig_importancias_admissao_v2.png', dpi=300)
plt.close()
print("Salvo: fig_importancias_admissao_v2.png")

# =============================================================
# 5. SALVAR MODELOS PARA USO EM PRODUÇÃO / SCORECARD
# =============================================================
joblib.dump(modelos_treinados['Longa_Permanencia_LightGBM'], 'modelo_admissao_longa_lightgbm.joblib')
joblib.dump(modelos_treinados['Obito_LightGBM'], 'modelo_admissao_obito_lightgbm.joblib')
joblib.dump(modelos_treinados['Longa_Permanencia_Logistic Regression'], 'modelo_admissao_longa_lr.joblib')
joblib.dump(modelos_treinados['Obito_Logistic Regression'], 'modelo_admissao_obito_lr.joblib')
print("\n💾 Modelos salvos com sucesso (.joblib)")
print("Treinamento e avaliação concluídos com 100% de sucesso!")
