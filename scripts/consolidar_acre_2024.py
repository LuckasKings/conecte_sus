# -*- coding: utf-8 -*-
import glob
import datasuspy
import pandas as pd

arquivos = sorted(glob.glob("RDAC24*.dbc"))
print(f"Lendo {len(arquivos)} arquivos...")

partes = []
for arq in arquivos:
    print(f"  → {arq}")
    partes.append(datasuspy.read_dbc(arq).to_pandas())

df = pd.concat(partes, ignore_index=True)

print(f"\n✅ Shape consolidado: {df.shape}")
print(f"Colunas: {len(df.columns)}")

df.to_csv("sih_acre_2024_completo.csv", index=False)
print("💾 sih_acre_2024_completo.csv salvo")