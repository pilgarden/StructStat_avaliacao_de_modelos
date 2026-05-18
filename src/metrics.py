# --------------------------------------------------------------
# src/metrics.py
# --------------------------------------------------------------
"""
Módulo de Métricas (metrics.py)
StructStat: Avaliação de modelos estruturais.

Isola a lógica matemática para avaliação de desempenho de modelos preditivos 
face a resultados experimentais (referência).
"""

import numpy as np
from sklearn.metrics import (
    r2_score, 
    mean_absolute_percentage_error, 
    mean_absolute_error, 
    root_mean_squared_error
)
from scipy.stats import pearsonr

def calcular_metricas(y_true: np.ndarray, y_pred: np.ndarray, 
                      nome_arquivo: str = "Desconhecido", p: int = 1) -> dict:
    """
    Calcula um conjunto exaustivo de métricas estatísticas para avaliar 
    a aderência de um modelo estrutural aos dados experimentais.
    
    Parâmetros:
    -----------
    y_true : np.ndarray
        Vetor com os resultados experimentais/referência.
    y_pred : np.ndarray
        Vetor com os resultados previstos pelo modelo analítico/numérico.
    nome_arquivo : str
        Identificador do conjunto de dados para rastreabilidade.
    p : int
        Número de variáveis preditoras no modelo (usado no R² ajustado). 
        Para regressão linear simples ou comparação direta 1:1, p=1.
        
    Retorno:
    --------
    dict
        Dicionário contendo todas as métricas calculadas.
        
    Notas Científicas:
    ------------------
    - R² e Pearson (r): Avaliam a correlação linear.
    - RMSE e MAE: Avaliam a magnitude do erro na mesma dimensão dos dados.
    - MaxAE: Fundamental na verificação de Estados Limites (maior desvio pontual).
    - Bias: Erro médio. Valores > 0 indicam que o modelo superestima a capacidade; 
            < 0 indicam que o modelo é conservador (subestima).
    """
    
    # Garantir que os inputs são arrays do NumPy para operações vetoriais
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    
    # Validação de amostra mínima
    if n < 3:
        raise ValueError(f"[{nome_arquivo}] Amostra insuficiente (n={n}) para significância estatística.")
    
    # 1. Métricas de Correlação e Variância
    r2 = r2_score(y_true, y_pred)
    
    # R² Ajustado penaliza a adição de preditores que não melhoram o modelo.
    r2_ajustado = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
    
    # r de Pearson mede estritamente a linearidade (-1 a 1)
    pearson_corr, p_value = pearsonr(y_true, y_pred)
    
    # 2. Métricas de Magnitude de Erro (Dimensionais)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Máximo Erro Absoluto (Pior cenário de previsão)
    max_ae = np.max(np.abs(y_true - y_pred))
    
    # Bias (Viés médio de previsão)
    bias = np.mean(y_pred - y_true)
    
    # 3. Métricas Percentuais e Relativas (Adimensionais)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    
    # Coeficiente de Variação do Erro (RMSE normalizado pela média da referência)
    media_referencia = np.mean(y_true)
    cv = (rmse / media_referencia) * 100 if media_referencia != 0 else np.nan
    
    # Construção do dicionário de saída
    resultados = {
        'Arquivo': nome_arquivo,
        'N (Amostras)': n,
        'R² (%)': r2 * 100,
        'R² Ajust. (%)': r2_ajustado * 100,
        'Pearson (r)': pearson_corr,
        'RMSE': rmse,
        'MAE': mae,
        'Max Erro': max_ae,
        'Bias': bias,
        'MAPE (%)': mape,
        'CV (%)': cv
    }
    
    return resultados