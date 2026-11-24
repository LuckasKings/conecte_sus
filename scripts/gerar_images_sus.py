# -*- coding: utf-8 -*-
"""
Gera, para TODOS os modelos avaliados:
  - Gráfico de importância das variáveis (top 15)
  - Matriz de confusão individual
  - Painel comparativo das matrizes de confusão
Utilizando dados reais do SIH/SUS (Acre, 2024).
"""
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# =============================================================
# CONFIGURAÇÕES DE PLOTAGEM (Manual USP/Esalq)
# =============================================================
rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 11
rcParams['axes.linewidth'] = 1.5
rcParams['axes.edgecolor'] = 'black'
rcParams['axes.grid'] = False
rcParams['figure.facecolor'] = 'white'
rcParams['axes.facecolor'] = 'white'
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'

# =============================================================
# 1. CARREGAMENTO E PRÉ-PROCESSAMENTO
# =============================================================
print("Carregando dados do SIH/SUS...")
df = pd.read_csv('sih_modelagem.csv')

# Definindo X e y (removendo o target e a variável qt_diarias que originou o target)
X = df.drop(['Longa_Permanencia', 'qt_diarias'], axis=1)
y = df['Longa_Permanencia']

# Definindo as features de acordo com o script de preparação
numeric_features = ['idade_anos']
categorical_features = ['sexo', 'capitulo_cid', 'complex', 'cobranca']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# =============================================================
# 2. DEFINIÇÃO DOS MODELOS
# =============================================================
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=42, use_label_encoder=False,
                              eval_metric='logloss', scale_pos_weight=18), # Ajuste p/ desbalanceamento
    'LightGBM': LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=31,
                                max_depth=-1, random_state=42, verbose=-1, class_weight='balanced')
}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# =============================================================
# 3. FUNÇÃO PARA CALCULAR IMPORTÂNCIA DAS VARIÁVEIS
# =============================================================
def get_feature_importance(pipeline, model_name):
    classifier = pipeline.named_steps['classifier']
    preprocessor_fitted = pipeline.named_steps['preprocessor']
    feature_names = preprocessor_fitted.get_feature_names_out()
    feature_names_clean = [f.replace('num__', '').replace('cat__', '') for f in feature_names]

    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
    elif hasattr(classifier, 'coef_'):
        importances = np.abs(classifier.coef_[0])
    else:
        raise ValueError(f"Modelo {model_name} não possui atributo de importância.")
    return feature_names_clean, importances

# =============================================================
# 4. LOOP PRINCIPAL – TREINAR, CALCULAR IMPORTÂNCIA, MATRIZ
# =============================================================
confusion_matrices = {}
importances_dict = {}
predictions_dict = {}

for name, model in models.items():
    print(f"\n=== Processando {name} ===")
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', model)
    ])

    # Ajuste no dataset completo (para importância e matriz de confusão)
    pipeline.fit(X, y)

    # ----- Importância das variáveis -----
    feature_names, importances = get_feature_importance(pipeline, name)
    importances_dict[name] = (feature_names, importances)

    # ----- Matriz de confusão via cross_val_predict -----
    y_pred = cross_val_predict(pipeline, X, y, cv=cv)
    cm = confusion_matrix(y, y_pred)
    confusion_matrices[name] = cm
    predictions_dict[name] = y_pred

    print(f"  Matriz de confusão:\n{cm}")
    print(f"  Relatório:\n{classification_report(y, y_pred, target_names=['Curta','Longa'])}")

# =============================================================
# 5. FIGURAS – IMPORTÂNCIA DAS VARIÁVEIS (uma por modelo)
# =============================================================
cores_imp = {
    'Logistic Regression': '#1f77b4',
    'Random Forest': '#ff7f0e',
    'XGBoost': '#2ca02c',
    'LightGBM': '#d62728'
}

for name, (feature_names, importances) in importances_dict.items():
    indices = np.argsort(importances)[::-1][:15]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(len(top_importances)), top_importances[::-1],
            color=cores_imp[name], edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(top_importances)))
    ax.set_yticklabels(top_features[::-1], fontsize=10)
    ax.set_xlabel('Importância', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fname = f'fig_importancia_{name.lower().replace(" ", "_")}_sus.png'
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"✅ {fname} gerada")

# =============================================================
# 6. FIGURAS – MATRIZ DE CONFUSÃO (uma por modelo)
# =============================================================
for name, cm in confusion_matrices.items():
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black',
                    fontsize=14, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Curta', 'Longa'], fontsize=11)
    ax.set_yticklabels(['Curta', 'Longa'], fontsize=11)
    ax.set_xlabel('Predito', fontsize=11)
    ax.set_ylabel('Real', fontsize=11)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    fname = f'fig_matriz_confusao_{name.lower().replace(" ", "_")}_sus.png'
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"✅ {fname} gerada")

# =============================================================
# 7. FIGURA – PAINEL COMPARATIVO DAS MATRIZES DE CONFUSÃO
# =============================================================
fig, axes = plt.subplots(2, 2, figsize=(11, 10))
axes = axes.flatten()

for ax, (name, cm) in zip(axes, confusion_matrices.items()):
    ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black',
                    fontsize=13, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Curta', 'Longa'], fontsize=10)
    ax.set_yticklabels(['Curta', 'Longa'], fontsize=10)
    ax.set_xlabel('Predito', fontsize=10)
    ax.set_ylabel('Real', fontsize=10)
    ax.set_title(name, fontsize=12, fontweight='bold')
    for spine in ax.spines.values():
        spine.set_visible(False)

plt.tight_layout()
plt.savefig('fig_matrizes_confusao_todos_sus.png', dpi=300)
plt.close()
print("✅ fig_matrizes_confusao_todos_sus.png gerada (painel comparativo)")

# =============================================================
# 8. FIGURA – PAINEL COMPARATIVO DAS IMPORTÂNCIAS
# =============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.flatten()

for ax, (name, (feature_names, importances)) in zip(axes, importances_dict.items()):
    indices = np.argsort(importances)[::-1][:10]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    ax.barh(range(len(top_importances)), top_importances[::-1],
            color=cores_imp[name], edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(top_importances)))
    ax.set_yticklabels(top_features[::-1], fontsize=9)
    ax.set_xlabel('Importância', fontsize=10)
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('fig_importancias_todos_sus.png', dpi=300)
plt.close()
print("✅ fig_importancias_todos_sus.png gerada (painel comparativo)")