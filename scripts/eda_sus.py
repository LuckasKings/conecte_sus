# -*- coding: utf-8 -*-
"""
Script de Analise Exploratoria de Dados (EDA) - SIH/SUS Acre 2024
"""
import pandas as pd
import numpy as np

print("="*60)
print("1. PERFIL DA BASE BRUTA: sih_acre_2024_completo.csv")
print("="*60)
df_bruto = pd.read_csv("sih_acre_2024_completo.csv", dtype=str, low_memory=False)
print(f"Dimensoes brutas: {df_bruto.shape[0]:,} linhas, {df_bruto.shape[1]} colunas")

# Conversao basica para analisar missings e tipos
colunas_num = ['qt_diarias', 'val_tot', 'val_sh', 'val_sp', 'val_uti',
               'idade', 'cod_idade', 'morte', 'complex', 'cobranca',
               'marca_uti', 'diar_acom']
for col in colunas_num:
    if col in df_bruto.columns:
        df_bruto[col] = pd.to_numeric(df_bruto[col], errors='coerce')

missings = df_bruto.isnull().sum()
pct_missing = (missings / len(df_bruto)) * 100
df_miss = pd.DataFrame({'n_missing': missings, 'pct_missing': pct_missing})
df_miss_relevant = df_miss[df_miss['n_missing'] > 0].sort_values(by='pct_missing', ascending=False)
print("\nPercentual de missing por coluna (colunas com missing):")
print(df_miss_relevant.head(25))

print("\n" + "="*60)
print("2. PERFIL DA BASE DE MODELAGEM: sih_modelagem.csv")
print("="*60)
df_mod = pd.read_csv("sih_modelagem.csv")
print(f"Dimensoes modelagem: {df_mod.shape[0]:,} linhas, {df_mod.shape[1]} colunas")
print("\nTipos das colunas na base de modelagem:")
print(df_mod.dtypes)

print("\nNulos na base de modelagem:")
print(df_mod.isnull().sum())

print("\n" + "="*60)
print("3. DISTRIBUICAO DO TARGET: Longa_Permanencia (0 = Curta, 1 = Longa)")
print("="*60)
contagem_target = df_mod['Longa_Permanencia'].value_counts()
prop_target = df_mod['Longa_Permanencia'].value_counts(normalize=True) * 100
for classe, nome in [(0, 'Curta (<15 dias)'), (1, 'Longa (>=15 dias)')]:
    print(f"Classe {classe} ({nome}): {contagem_target[classe]:,} casos ({prop_target[classe]:.2f}%)")
razao_desbal = contagem_target[0] / contagem_target[1]
print(f"Razao de desbalanceamento: {razao_desbal:.2f}:1 (Curta:Longa)")

print("\n" + "="*60)
print("4. ESTATISTICAS DESCRITIVAS DAS NUMERICAS POR CLASSE")
print("="*60)
num_cols = ['idade_anos', 'qt_diarias']
for col in num_cols:
    print(f"\n--- Estatisticas de {col} por classe ---")
    desc = df_mod.groupby('Longa_Permanencia')[col].agg(['count', 'mean', 'std', 'median', lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75), 'min', 'max'])
    desc.columns = ['Count', 'Media', 'Desvio', 'Mediana', 'Q25', 'Q75', 'Min', 'Max']
    print(desc)

print("\n" + "="*60)
print("5. VARIAVEIS CATEGORICAS VS CLASSE ALVO")
print("="*60)
cat_cols = ['sexo', 'complex', 'cobranca', 'capitulo_cid']
for col in cat_cols:
    print(f"\n--- Cruzamento de {col} com Longa_Permanencia ---")
    ct = pd.crosstab(df_mod[col], df_mod['Longa_Permanencia'], margins=True)
    ct_prop = pd.crosstab(df_mod[col], df_mod['Longa_Permanencia'], normalize='index') * 100
    res = pd.concat([ct[0].rename('Curta_N'), ct[1].rename('Longa_N'), ct['All'].rename('Total_N'),
                     ct_prop[1].rename('%_Longa')], axis=1).sort_values(by='Total_N', ascending=False)
    print(res.head(15))
