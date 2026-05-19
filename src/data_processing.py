"""
Módulo de Processamento de Dados (data_processing.py)
Extração, limpeza e filtragem agnóstica à interface.
"""

import pandas as pd
import numpy as np
import os

def carregar_dados(ficheiro_obj, nome_ficheiro: str) -> pd.DataFrame:
    """Carrega dataset genérico para Análise Exploratória."""
    _, extensao = os.path.splitext(nome_ficheiro.lower())
    try:
        if extensao in ['.xlsx', '.xls']:
            df = pd.read_excel(ficheiro_obj)
        else:
            if hasattr(ficheiro_obj, 'seek'): ficheiro_obj.seek(0)
            try:
                df = pd.read_csv(ficheiro_obj, sep=';', decimal=',', engine='python')
                if len(df.columns) < 2: raise ValueError
            except:
                if hasattr(ficheiro_obj, 'seek'): ficheiro_obj.seek(0)
                df = pd.read_csv(ficheiro_obj, sep=',', decimal='.', engine='python')
        return df.dropna(how='all')
    except Exception as e:
        raise RuntimeError(f"Erro ao ler '{nome_ficheiro}': {str(e)}")

def ler_e_limpar_dados(arquivo_obj, nome_arquivo: str, fator_conversao: float = 1.0) -> pd.DataFrame:
    """Carrega dados estritos (2 colunas) para Avaliação de Modelos."""
    df = carregar_dados(arquivo_obj, nome_arquivo)
    
    # Isolamento vetorial: Assume-se que a 1ª coluna é Referência (x) e a 2ª é Previsão (y)
    df_limpo = df.iloc[:, :2].copy()
    if df_limpo.shape[1] < 2:
        raise ValueError("O ficheiro deve conter pelo menos duas colunas (Real e Predito).")
        
    df_limpo.columns = ['Referencia', 'Previsto']
    df_limpo['Referencia'] = pd.to_numeric(df_limpo['Referencia'], errors='coerce')
    df_limpo['Previsto'] = pd.to_numeric(df_limpo['Previsto'], errors='coerce')
    df_limpo = df_limpo.dropna()
    
    if len(df_limpo) == 0:
        raise ValueError("Após a sanitização, não restaram tensores numéricos válidos.")
        
    df_limpo['Referencia'] = df_limpo['Referencia'] * fator_conversao
    df_limpo['Previsto'] = df_limpo['Previsto'] * fator_conversao
    return df_limpo

def aplicar_filtro_dinamico(df: pd.DataFrame, coluna: str, regra: str, valor: any) -> pd.DataFrame:
    """Filtra subconjuntos para a aba exploratória."""
    if coluna not in df.columns: return df
    try:
        if regra == "Valores Exatos" and isinstance(valor, list):
            return df[df[coluna].isin(valor)]
        elif regra == "Menor ou igual (<=)":
            return df[df[coluna] <= float(valor)]
        elif regra == "Maior ou igual (>=)":
            return df[df[coluna] >= float(valor)]
        elif regra == "Começa com (Prefixo)":
            return df[df[coluna].astype(str).str.startswith(str(valor), na=False)]
        elif regra == "Contém (Texto)":
            return df[df[coluna].astype(str).str.contains(str(valor), na=False, case=False)]
        return df
    except Exception as e:
        raise ValueError(f"Erro ao aplicar o filtro: {str(e)}")
