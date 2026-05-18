# --------------------------------------------------------------
# src/data_processing.py
# --------------------------------------------------------------
"""
Módulo de Processamento de Dados (data_processing.py)
StructStat: Avaliação de modelos estruturais.

Responsável pela extração (leitura de CSV/XLSX), sanitização (remoção de strings 
residuais e NaNs) e transformação (aplicação de fatores de escala) dos dados brutos.
"""

import pandas as pd
import numpy as np
import io
import os

def ler_e_limpar_dados(arquivo_obj, nome_arquivo: str, fator_conversao: float = 1.0) -> pd.DataFrame:
    """
    Processa um ficheiro de resultados experimentais/numéricos, garantindo
    um DataFrame limpo, estritamente numérico e dimensionado corretamente.
    
    Parâmetros:
    -----------
    arquivo_obj : str ou file-like object
        Caminho local do ficheiro ou objeto em memória (Streamlit UploadedFile).
    nome_arquivo : str
        Nome do ficheiro (necessário para inferir a extensão .csv ou .xlsx).
    fator_conversao : float, opcional (default=1.0)
        Multiplicador escalar para conversão de unidades (ex: converter N para kN).
        
    Retorno:
    --------
    pd.DataFrame
        DataFrame contendo duas colunas limpas ['Referencia', 'Previsto'].
        
    Exceções:
    ---------
    ValueError / RuntimeError: Se o ficheiro estiver corrompido ou sem dados numéricos.
    """
    _, extensao = os.path.splitext(nome_arquivo.lower())
    df = None
    
    try:
        if extensao in ['.xlsx', '.xls']:
            df = pd.read_excel(arquivo_obj)
        else:
            # Tratamento de ficheiros CSV
            # Muitos sistemas de aquisição em PT/BR usam ';' como separador e ',' como decimal.
            # Se a leitura falhar, assume-se o padrão internacional da literatura ('.' e ',').
            
            # Se for um objeto em memória (Streamlit), garante que o cursor está no início
            if hasattr(arquivo_obj, 'seek'):
                arquivo_obj.seek(0)
                
            try:
                df = pd.read_csv(arquivo_obj, sep=';', decimal=',', engine='python')
            except pd.errors.ParserError:
                if hasattr(arquivo_obj, 'seek'):
                    arquivo_obj.seek(0)
                df = pd.read_csv(arquivo_obj, sep=',', decimal='.', engine='python')
        
        if df is None or df.empty:
            raise ValueError("O ficheiro encontra-se vazio ou tem um formato ilegível.")

        # Isolamento vetorial: Assume-se que a 1ª coluna é Referência (x) e a 2ª é Previsão (y)
        df_limpo = df.iloc[:, :2].copy()
        
        if df_limpo.shape[1] < 2:
            raise ValueError("O ficheiro deve conter pelo menos duas colunas.")
            
        df_limpo.columns = ['Referencia', 'Previsto']
        
        # Sanitização: Força a conversão para Float. 
        # Caracteres de texto (ex: cabeçalhos duplos, unidades na célula) tornam-se NaN.
        df_limpo['Referencia'] = pd.to_numeric(df_limpo['Referencia'], errors='coerce')
        df_limpo['Previsto'] = pd.to_numeric(df_limpo['Previsto'], errors='coerce')
        
        # Eliminação rigorosa de linhas incompletas (Graus de liberdade perdidos)
        df_limpo = df_limpo.dropna()
        
        if len(df_limpo) == 0:
            raise ValueError("Após a sanitização, não restaram tensores numéricos válidos.")
            
        # Transformação Tensorial: Aplicação linear do fator de conversão (N -> kN, etc.)
        df_limpo['Referencia'] = df_limpo['Referencia'] * fator_conversao
        df_limpo['Previsto'] = df_limpo['Previsto'] * fator_conversao
        
        return df_limpo
        
    except Exception as e:
        raise RuntimeError(f"Falha na extração de '{nome_arquivo}': {str(e)}")