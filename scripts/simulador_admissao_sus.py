# -*- coding: utf-8 -*-
"""
Simulador e Scorecard de Admissão Hospitalar SUS Acre (Versão V3 - Ética e Clínica):
Recebe os dados preenchidos no acolhimento do paciente (sem mês e sem raça/cor) e calcula:
- P(Longa Permanência)
- P(Óbito Intra-hospitalar)
- Classificação na Matriz de Risco 2x2
- Ação assistencial e de regulação imediata
"""
import os
import pandas as pd
import numpy as np
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LONGA = os.path.join(BASE_DIR, 'modelos', 'modelo_admissao_longa_lightgbm_v3.joblib')
DEFAULT_OBITO = os.path.join(BASE_DIR, 'modelos', 'modelo_admissao_obito_lightgbm_v3.joblib')

class ScorecardAdmissaoSUS:
    def __init__(self, modelo_longa_path=None, modelo_obito_path=None):
        modelo_longa_path = modelo_longa_path or DEFAULT_LONGA
        modelo_obito_path = modelo_obito_path or DEFAULT_OBITO
        self.modelo_longa = joblib.load(modelo_longa_path)
        self.modelo_obito = joblib.load(modelo_obito_path)
        self.features_num = ['idade_anos']
        self.features_cat = ['sexo', 'capitulo_cid', 'especialidade_leito',
                             'carater_internacao', 'complexidade', 'transferido_intermun',
                             'internacao_fim_de_semana']

    def prever_paciente(self, dados_paciente: dict) -> dict:
        """
        Recebe dicionário com dados de admissão e retorna o scorecard de risco.
        """
        df_input = pd.DataFrame([dados_paciente])
        
        # Filtrar e formatar tipos exatos esperados pelo Pipeline V3
        df_model = pd.DataFrame()
        for c in self.features_num:
            df_model[c] = pd.to_numeric(df_input[c], errors='coerce')
        for c in self.features_cat:
            df_model[c] = df_input[c].astype(str)

        # Predições de probabilidade dos modelos V3
        p_longa = float(self.modelo_longa.predict_proba(df_model)[0, 1])
        p_obito = float(self.modelo_obito.predict_proba(df_model)[0, 1])

        # Categorização de Risco Operacional
        risco_longa = "ALTO" if p_longa >= 0.50 else ("MÉDIO" if p_longa >= 0.30 else "BAIXO")
        risco_obito = "ALTO" if p_obito >= 0.50 else ("MÉDIO" if p_obito >= 0.25 else "BAIXO")

        # Classificação no Quadrante da Matriz de Decisão 2x2
        is_longa = p_longa >= 0.50
        is_obito = p_obito >= 0.50

        if is_longa and is_obito:
            quadrante = "ZONA VERMELHA (CRÍTICA)"
            cor = "#c0392b"
            acao = "Prioridade Máxima: Acionar UTI/Semi-Intensiva, suporte respiratório/hemodinâmico e equipe de cuidados avançados."
        elif not is_longa and is_obito:
            quadrante = "ZONA DE EMERGÊNCIA AGUDA"
            cor = "#d35400"
            acao = "Sala Vermelha / Emergência: Alto risco de desfecho hiperagudo. Monitoramento contínuo nas primeiras 48h (risco iminente de choque/PCR)."
        elif is_longa and not is_obito:
            quadrante = "ZONA DE CRONICIDADE"
            cor = "#f39c12"
            acao = "Planejamento de Leito: Baixa letalidade imediata, mas alta chance de retenção prolongada. Acionar Serviço Social, RAPS ou Reabilitação nas primeiras 24h."
        else:
            quadrante = "ZONA DE FLUXO RÁPIDO"
            cor = "#27ae60"
            acao = "Linha de Cuidado Padrão: Protocolo de alta precoce (24h a 48h). Risco mínimo de intercorrências graves."

        return {
            'nome_paciente': dados_paciente.get('nome', 'Paciente Sem Identificação'),
            'p_longa': p_longa,
            'risco_longa': risco_longa,
            'p_obito': p_obito,
            'risco_obito': risco_obito,
            'quadrante': quadrante,
            'cor_alerta': cor,
            'acao_recomendada': acao
        }

if __name__ == '__main__':
    print("="*75)
    print("SIMULADOR DE TRIAGEM HOSPITALAR NA ADMISSÃO V3 (CLÍNICO E ÉTICO)")
    print("="*75)
    
    scorecard = ScorecardAdmissaoSUS()

    casos_teste = [
        {
            'nome': 'Caso 1: Seu Raimundo (76 anos, Clínica Médica, Insuf. Respiratória, Urgência, Interior)',
            'idade_anos': 76.0,
            'sexo': '1',
            'capitulo_cid': 'X - Respiratorio',
            'especialidade_leito': 'Clinica_Medica',
            'carater_internacao': 'Urgencia_Emergencia',
            'complexidade': 'Alta_Complexidade',
            'transferido_intermun': '1',
            'internacao_fim_de_semana': '1'
        },
        {
            'nome': 'Caso 2: Dona Maria (24 anos, Obstetrícia, Parto Normal Eletivo, Capital)',
            'idade_anos': 24.0,
            'sexo': '3',
            'capitulo_cid': 'XV - Gravidez/Parto',
            'especialidade_leito': 'Obstetrica',
            'carater_internacao': 'Eletiva',
            'complexidade': 'Media_Complexidade',
            'transferido_intermun': '0',
            'internacao_fim_de_semana': '0'
        },
        {
            'nome': 'Caso 3: Francisco (38 anos, Psiquiatria, Transtorno Mental, Urgência)',
            'idade_anos': 38.0,
            'sexo': '1',
            'capitulo_cid': 'V - Mentais',
            'especialidade_leito': 'Psiquiatrica',
            'carater_internacao': 'Urgencia_Emergencia',
            'complexidade': 'Media_Complexidade',
            'transferido_intermun': '0',
            'internacao_fim_de_semana': '0'
        },
        {
            'nome': 'Caso 4: Carlos (62 anos, Clínica Médica, Ap. Circulatório/IAM Choque, Urgência)',
            'idade_anos': 62.0,
            'sexo': '1',
            'capitulo_cid': 'IX - Circulatorio',
            'especialidade_leito': 'Clinica_Medica',
            'carater_internacao': 'Urgencia_Emergencia',
            'complexidade': 'Alta_Complexidade',
            'transferido_intermun': '0',
            'internacao_fim_de_semana': '1'
        }
    ]

    for caso in casos_teste:
        res = scorecard.prever_paciente(caso)
        print(f"\n📋 PACIENTE: {res['nome_paciente']}")
        print(f"   • Score de Risco Longa Permanência: {res['p_longa']*100:.1f}% [{res['risco_longa']}]")
        print(f"   • Score de Risco Óbito:             {res['p_obito']*100:.1f}% [{res['risco_obito']}]")
        print(f"   • ESTRATIFICAÇÃO:                   {res['quadrante']}")
        print(f"   • CONDUTA ASSISTENCIAL:             {res['acao_recomendada']}")
        print("-" * 75)
