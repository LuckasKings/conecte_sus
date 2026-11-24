# -*- coding: utf-8 -*-
"""
Prepara dados do SIH/SUS Acre 2024 para Predição Conjunta na Admissão:
- Target 1: Longa Permanência (qt_diarias >= 15)
- Target 2: Óbito Intra-hospitalar (morte == 1)

Critério: Uso estrito de variáveis disponíveis no momento da entrada do paciente,
eliminando vazamentos de dados (data leakage) e ruídos administrativos.
"""
import pandas as pd
import numpy as np

print("="*65)
print("1. CARREGAMENTO DOS DADOS BRUTOS (SIH Acre 2024)")
print("="*65)
df = pd.read_csv("sih_acre_2024_completo.csv", dtype=str, low_memory=False)
print(f"Registros brutos carregados: {len(df):,}")

# ============================================================
# 2. CONVERSÃO DE TIPOS E TRATAMENTO TEMPORAL
# ============================================================
colunas_num = ['qt_diarias', 'idade', 'cod_idade', 'morte', 'car_int', 'espec', 'complex', 'raca_cor']
for col in colunas_num:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 3. IDADE EM ANOS CONTÍNUOS
def idade_para_anos(idade, cod):
    if pd.isna(idade) or pd.isna(cod): return np.nan
    cod = int(cod)
    if cod in (0, 4, 5): return float(idade)
    if cod == 1: return idade / 12.0
    if cod == 2: return idade / 365.25
    if cod == 3: return idade / (365.25 * 24)
    return np.nan

df['idade_anos'] = df.apply(lambda r: idade_para_anos(r['idade'], r['cod_idade']), axis=1)

# 4. CAPÍTULO CID-10
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

# 5. MAPEAMENTO DE ESPECIALIDADE DO LEITO
mapa_espec = {
    1: 'Cirurgica',
    2: 'Obstetrica',
    3: 'Clinica_Medica',
    4: 'Pediatrica',
    5: 'Hospital_Dia',
    7: 'Psiquiatrica'
}
df['especialidade_leito'] = df['espec'].map(mapa_espec).fillna('Outras')

# 6. CARÁTER DA INTERNAÇÃO (Urgência vs Eletiva)
mapa_car_int = {
    1: 'Eletiva',
    2: 'Urgencia_Emergencia'
}
df['carater_internacao'] = df['car_int'].map(mapa_car_int).fillna('Urgencia_Emergencia')

# 7. COMPLEXIDADE
mapa_complex = {
    2: 'Media_Complexidade',
    3: 'Alta_Complexidade'
}
df['complexidade'] = df['complex'].map(mapa_complex).fillna('Media_Complexidade')

# 8. RAÇA/COR
mapa_raca = {
    1: 'Branca',
    2: 'Preta',
    3: 'Parda',
    4: 'Amarela',
    5: 'Indigena',
    99: 'Nao_Declarada',
    0: 'Nao_Declarada'
}
df['raca_cor_desc'] = df['raca_cor'].map(mapa_raca).fillna('Nao_Declarada')

# 9. VULNERABILIDADE GEOGRÁFICA: TRANSFERÊNCIA INTERMUNICIPAL
df['transferido_intermun'] = (df['munic_res'].astype(str).str.strip() != df['munic_mov'].astype(str).str.strip()).astype(int)

# 10. SAZONALIDADE E TEMPORALIDADE DA ENTRADA
df['dt_inter_dt'] = pd.to_datetime(df['dt_inter'], format='%Y%m%d', errors='coerce')
df['mes_internacao'] = df['dt_inter_dt'].dt.month.fillna(0).astype(int)
df['internacao_fim_de_semana'] = (df['dt_inter_dt'].dt.dayofweek >= 5).astype(int)

# 11. DEFINIÇÃO DOS TARGETS (Desfechos)
df = df.dropna(subset=['qt_diarias', 'morte', 'idade_anos'])
df = df[df['qt_diarias'] > 0]

df['Longa_Permanencia'] = (df['qt_diarias'] >= 15).astype(int)
df['Obito'] = (df['morte'] == 1).astype(int)

# ============================================================
# 12. CONSOLIDAÇÃO DA BASE DE MODELAGEM V2 (SEM DATA LEAKAGE)
# ============================================================
features_admissao = [
    'idade_anos',
    'sexo',
    'raca_cor_desc',
    'capitulo_cid',
    'especialidade_leito',
    'carater_internacao',
    'complexidade',
    'transferido_intermun',
    'internacao_fim_de_semana',
    'mes_internacao'
]

targets = ['Longa_Permanencia', 'Obito']
metadados_apoio = ['qt_diarias'] # apenas para consulta/apoio analítico, não feature

colunas_finais = features_admissao + targets + metadados_apoio
df_final = df[colunas_finais].copy()

# Garantir tipos categóricos em string
for col in ['sexo', 'raca_cor_desc', 'capitulo_cid', 'especialidade_leito', 'carater_internacao', 'complexidade']:
    df_final[col] = df_final[col].astype(str)

print("\n" + "="*65)
print("2. RESUMO E ESTATÍSTICAS DA NOVA BASE (V2)")
print("="*65)
print(f"Total de registros na base final: {len(df_final):,}")
print(f"Total de colunas: {df_final.shape[1]}")
print(f"\nFeatures de Admissão ({len(features_admissao)}): {features_admissao}")
print(f"Targets ({len(targets)}): {targets}")

print("\n--- DISTRIBUIÇÃO DOS TARGETS ---")
print(f"Longa Permanência (1 = >=15d):")
print(df_final['Longa_Permanencia'].value_counts())
print(f"Proporção Longa: {df_final['Longa_Permanencia'].mean():.2%}")

print(f"\nÓbito Intra-hospitalar (1 = Sim):")
print(df_final['Obito'].value_counts())
print(f"Proporção Óbito: {df_final['Obito'].mean():.2%}")

print("\n--- CRUZAMENTO DOS DESFECHOS (MATRIZ DE RISCO 2x2) ---")
cruzamento = pd.crosstab(df_final['Longa_Permanencia'], df_final['Obito'],
                         rownames=['Longa Permanência'], colnames=['Óbito'], margins=True)
print(cruzamento)
print("\nPercentual relativo de cada quadrante:")
print((cruzamento / len(df_final) * 100).round(2))

# Salvar base V2
df_final.to_csv("sih_modelagem_admissao_v2.csv", index=False)
print(f"\n💾 Nova base de admissão salva com sucesso: sih_modelagem_admissao_v2.csv")
