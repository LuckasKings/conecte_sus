# -*- coding: utf-8 -*-
"""
Prepara dados SIH/SUS para modelagem de longa permanência.
"""
import pandas as pd
import numpy as np

# ============================================================
# 1. CARREGAMENTO
# ============================================================
df = pd.read_csv("sih_acre_2024_completo.csv", dtype=str, low_memory=False)
print(f"Registros brutos: {len(df):,}")

# ============================================================
# 2. CONVERSÃO DE TIPOS
# ============================================================
colunas_num = ['qt_diarias', 'val_tot', 'val_sh', 'val_sp', 'val_uti',
               'idade', 'cod_idade', 'morte', 'complex', 'cobranca',
               'marca_uti', 'diar_acom']
for col in colunas_num:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# ============================================================
# 3. IDADE EM ANOS
# ============================================================
def idade_para_anos(idade, cod):
    if pd.isna(idade) or pd.isna(cod):
        return np.nan
    cod = int(cod)
    if cod in (0, 4, 5): return float(idade)
    if cod == 1: return idade / 12.0
    if cod == 2: return idade / 365.25
    if cod == 3: return idade / (365.25 * 24)
    return np.nan

df['idade_anos'] = df.apply(lambda r: idade_para_anos(r['idade'], r['cod_idade']), axis=1)

# ============================================================
# 4. CAPÍTULO CID-10
# ============================================================
def capitulo_cid(cid):
    if pd.isna(cid) or str(cid).strip() == '':
        return 'Ignorado'
    letra = str(cid)[0].upper()
    mapa = {
        'A':'I - Infecciosas', 'B':'I - Infecciosas',
        'C':'II - Neoplasias', 'D':'III - Sangue',
        'E':'IV - Endocrinas', 'F':'V - Mentais',
        'G':'VI - Nervoso', 'H':'VII/VIII - Olho/Ouvido',
        'I':'IX - Circulatorio', 'J':'X - Respiratorio',
        'K':'XI - Digestivo', 'L':'XII - Pele',
        'M':'XIII - Osteomuscular', 'N':'XIV - Geniturinario',
        'O':'XV - Gravidez/Parto', 'P':'XVI - Perinatal',
        'Q':'XVII - Congenitas', 'R':'XVIII - Sintomas',
        'S':'XIX - Lesoes', 'T':'XIX - Lesoes',
        'V':'XX - Causas externas', 'W':'XX - Causas externas',
        'X':'XX - Causas externas', 'Y':'XX - Causas externas',
        'Z':'XXI - Fatores de saude',
    }
    return mapa.get(letra, 'Ignorado')

df['capitulo_cid'] = df['diag_princ'].apply(capitulo_cid)

# ============================================================
# 5. TARGET: LONGA PERMANÊNCIA (>= 15 dias)
# ============================================================
df = df.dropna(subset=['qt_diarias'])
df = df[df['qt_diarias'] > 0]
df['Longa_Permanencia'] = (df['qt_diarias'] >= 15).astype(int)

print(f"\n📊 Distribuição do target:")
print(df['Longa_Permanencia'].value_counts())
print(f"Proporção longa permanência: {df['Longa_Permanencia'].mean():.1%}")

# ============================================================
# 6. FEATURES PARA MODELAGEM (SEM LEAKAGE)
# ============================================================
# Features conhecidas NA ADMISSÃO:
features_num = ['idade_anos']
features_cat = ['sexo', 'capitulo_cid', 'complex', 'cobranca']

# NOTA: 'val_tot' e 'morte' são conhecidas só ao final -> excluídas
# Se quiser testar cenário com leakage, adicione-as depois

cols = features_num + features_cat + ['Longa_Permanencia', 'qt_diarias']
df_model = df[cols].dropna()

# Converter categóricas para string (evita problemas com OneHotEncoder)
for c in features_cat:
    df_model[c] = df_model[c].astype(str)

print(f"\n✅ DataFrame final: {df_model.shape}")
print(df_model.head())
print(f"\n📈 Distribuição final do target:")
print(df_model['Longa_Permanencia'].value_counts(normalize=True))

df_model.to_csv("sih_modelagem.csv", index=False)
print("\n💾 sih_modelagem.csv salvo")