# -*- coding: utf-8 -*-
"""
Script para geracao de graficos da EDA e avaliacao completa dos modelos
SIH/SUS Acre 2024 (Usando apenas matplotlib e scikit-learn/imblearn)
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
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Configuracoes de plotagem do projeto
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
# 1. CARREGAMENTO DOS DADOS
# =============================================================
print("--> Carregando bases...")
df_bruto = pd.read_csv("sih_acre_2024_completo.csv", dtype=str, low_memory=False)
df_model = pd.read_csv("sih_modelagem.csv")

# Conversao na base bruta para correlacao
colunas_num_bruto = ['qt_diarias', 'val_tot', 'val_sh', 'val_sp', 'val_uti',
                     'idade', 'cod_idade', 'morte', 'diar_acom']
for col in colunas_num_bruto:
    if col in df_bruto.columns:
        df_bruto[col] = pd.to_numeric(df_bruto[col], errors='coerce')

def idade_para_anos(idade, cod):
    if pd.isna(idade) or pd.isna(cod): return np.nan
    cod = int(cod)
    if cod in (0, 4, 5): return float(idade)
    if cod == 1: return idade / 12.0
    if cod == 2: return idade / 365.25
    if cod == 3: return idade / (365.25 * 24)
    return np.nan

df_bruto['idade_anos'] = df_bruto.apply(lambda r: idade_para_anos(r['idade'], r['cod_idade']), axis=1)

# =============================================================
# 2. GRAFICO 1: DISTRIBUICAO DA VARIAVEL ALVO
# =============================================================
print("\n--> Gerando fig_distribuicao_target.png...")
counts = df_model['Longa_Permanencia'].value_counts().sort_index()
total = len(df_model)
pcts = (counts / total) * 100

fig, ax = plt.subplots(figsize=(7, 5))
cores = ['#2b5c8f', '#d95f02']
barras = ax.bar(['Curta (< 15 dias)', 'Longa (>= 15 dias)'], counts, color=cores, edgecolor='black', width=0.55)

for barra, cnt, pct in zip(barras, counts, pcts):
    altura = barra.get_height()
    ax.text(barra.get_x() + barra.get_width()/2., altura + total*0.015,
            f'{cnt:,}\n({pct:.2f}%)',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylim(0, max(counts) * 1.15)
ax.set_ylabel('Número de Internações', fontsize=11, fontweight='bold')
ax.set_title('Distribuição da Permanência Hospitalar (SIH/SUS Acre 2024)', fontsize=12, fontweight='bold', pad=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('fig_distribuicao_target.png', dpi=300)
plt.close()
print("Salvo: fig_distribuicao_target.png")

# =============================================================
# 3. GRAFICO 2: CORRELACAO DAS VARIAVEIS NUMERICAS (HEATMAP)
# =============================================================
print("\n--> Gerando fig_correlacao_heatmap.png...")
vars_corr = {
    'idade_anos': 'Idade',
    'qt_diarias': 'Diárias',
    'val_tot': 'Val. Total',
    'val_sh': 'Serv. Hosp.',
    'val_sp': 'Serv. Prof.',
    'val_uti': 'Val. UTI',
    'diar_acom': 'Diar. Acomp.',
    'morte': 'Óbito'
}
df_sub_corr = df_bruto[list(vars_corr.keys())].dropna(subset=['idade_anos', 'qt_diarias']).rename(columns=vars_corr)
corr_matrix = df_sub_corr.corr()

fig, ax = plt.subplots(figsize=(8.5, 7))
cax = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
cbar = fig.colorbar(cax, shrink=0.8)
cbar.ax.tick_params(labelsize=10)

labels = list(corr_matrix.columns)
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10, fontweight='bold')
ax.set_yticklabels(labels, fontsize=10, fontweight='bold')

for i in range(len(labels)):
    for j in range(len(labels)):
        val = corr_matrix.iloc[i, j]
        text_color = "white" if abs(val) > 0.45 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=10, fontweight='bold')

ax.set_title('Matriz de Correlação das Variáveis Numéricas (SIH/SUS Acre 2024)', fontsize=12, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('fig_correlacao_heatmap.png', dpi=300)
plt.close()
print("Salvo: fig_correlacao_heatmap.png")

# =============================================================
# 4. GRAFICO 3: BOXPLOTS DAS VARIAVEIS NUMERICAS POR CLASSE
# =============================================================
print("\n--> Gerando fig_boxplots_classe.png...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

dados_idade_curta = df_model[df_model['Longa_Permanencia'] == 0]['idade_anos'].values
dados_idade_longa = df_model[df_model['Longa_Permanencia'] == 1]['idade_anos'].values

dados_diar_curta = df_model[df_model['Longa_Permanencia'] == 0]['qt_diarias'].values
dados_diar_longa = df_model[df_model['Longa_Permanencia'] == 1]['qt_diarias'].values

bp1 = ax1.boxplot([dados_idade_curta, dados_idade_longa],
                  patch_artist=True, widths=0.45,
                  tick_labels=['Curta (<15d)', 'Longa (>=15d)'],
                  flierprops=dict(marker='o', markersize=2, alpha=0.3, color='gray'))

cores_box = ['#2b5c8f', '#d95f02']
for patch, color in zip(bp1['boxes'], cores_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
for median in bp1['medians']:
    median.set(color='black', linewidth=1.8)

ax1.set_ylabel('Idade do Paciente (anos)', fontsize=11, fontweight='bold')
ax1.set_title('Distribuição de Idade por Permanência', fontsize=12, fontweight='bold')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

bp2 = ax2.boxplot([dados_diar_curta, dados_diar_longa],
                  patch_artist=True, widths=0.45,
                  tick_labels=['Curta (<15d)', 'Longa (>=15d)'],
                  flierprops=dict(marker='o', markersize=2, alpha=0.3, color='gray'))

for patch, color in zip(bp2['boxes'], cores_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
for median in bp2['medians']:
    median.set(color='black', linewidth=1.8)

ax2.set_yscale('log')
ax2.set_ylabel('Diárias Hospitalares (escala log)', fontsize=11, fontweight='bold')
ax2.set_title('Distribuição de Diárias por Permanência', fontsize=12, fontweight='bold')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('fig_boxplots_classe.png', dpi=300)
plt.close()
print("Salvo: fig_boxplots_classe.png")

# =============================================================
# 5. TREINAMENTO, VALIDAÇÃO CRUZADA E EXTRAÇÃO DE MÉTRICAS
# =============================================================
print("\n--> Executando avaliacao dos 4 modelos via 10-Fold Stratified CV...")
X = df_model.drop(['Longa_Permanencia', 'qt_diarias'], axis=1)
y = df_model['Longa_Permanencia']

numeric_features = ['idade_anos']
categorical_features = ['sexo', 'capitulo_cid', 'complex', 'cobranca']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=42, eval_metric='logloss', scale_pos_weight=18),
    'LightGBM': LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=31,
                                max_depth=-1, random_state=42, verbose=-1, class_weight='balanced')
}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

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
        raise ValueError(f"Modelo {model_name} nao possui atributo de importancia.")
    return feature_names_clean, importances

tabela_metricas = []
top_features_dict = {}

for name, model in models.items():
    print(f"\n==========================================")
    print(f"Modelo: {name}")
    print(f"==========================================")
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', model)
    ])

    # Fit para importancia
    pipeline.fit(X, y)
    fnames, imps = get_feature_importance(pipeline, name)
    idx_sorted = np.argsort(imps)[::-1]
    top_features_dict[name] = [(fnames[i], imps[i]) for i in idx_sorted[:10]]
    print(f"Top 5 features mais importantes ({name}):")
    for f, imp in top_features_dict[name][:5]:
        print(f"  - {f}: {imp:.4f}")

    # CV Predict
    y_pred = cross_val_predict(pipeline, X, y, cv=cv)
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, pos_label=1)
    rec = recall_score(y, y_pred, pos_label=1)
    f1 = f1_score(y, y_pred, pos_label=1)
    spec = tn / (tn + fp)

    print(f"\nMatriz de Confusao:")
    print(f"  TN (Curta predita Curta): {tn:,}")
    print(f"  FP (Curta predita Longa): {fp:,}")
    print(f"  FN (Longa predita Curta): {fn:,}")
    print(f"  TP (Longa predita Longa): {tp:,}")
    print(f"\nMetricas:")
    print(f"  Acuracia:       {acc*100:.2f}%")
    print(f"  Precisao (L):   {prec*100:.2f}%")
    print(f"  Recall (L):     {rec*100:.2f}%")
    print(f"  Especificidade: {spec*100:.2f}%")
    print(f"  F1-Score (L):   {f1*100:.2f}%")

    tabela_metricas.append({
        'Modelo': name,
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
        'Acuracia': acc,
        'Precisao_Longa': prec,
        'Recall_Longa': rec,
        'Especificidade_Curta': spec,
        'F1_Longa': f1
    })

df_res = pd.DataFrame(tabela_metricas)
print("\n" + "="*70)
print("TABELA COMPARATIVA FINAL DOS MODELOS:")
print("="*70)
print(df_res.to_string(index=False))
