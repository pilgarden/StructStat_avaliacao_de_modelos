"""
Módulo de Métricas (metrics.py)
StructStat: Avaliação de modelos estruturais.
"""

import numpy as np
from sklearn.metrics import (
    r2_score, 
    mean_absolute_percentage_error, 
    mean_absolute_error, 
    root_mean_squared_error
)
from scipy.stats import pearsonr, f

def calcular_metricas(y_true: np.ndarray, y_pred: np.ndarray, 
                      nome_arquivo: str = "Desconhecido", p: int = 1) -> dict:
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    
    if n < 3:
        raise ValueError(f"[{nome_arquivo}] Amostra insuficiente (n={n}) para significância estatística.")
    
    # 1. Métricas de Correlação e Variância
    r2 = r2_score(y_true, y_pred)
    r2_ajustado = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
    
    # Indicador de Overfitting (Queda de Desempenho por Complexidade)
    # Rastreia o quanto a inserção de variáveis (p) inflacionou artificialmente o R2.
    overfitting = (r2 - r2_ajustado) * 100
    
    # Pearson
    pearson_corr, _ = pearsonr(y_true, y_pred)
    
    # 2. Estatística F e Valor-p (Significância Global do Modelo)
    # F = (MSR / MSE). Se R2 for 1.0, F tende ao infinito.
    if r2 >= 1.0:
        f_stat = np.inf
        p_value = 0.0
    elif r2 <= 0.0:
        f_stat = 0.0
        p_value = 1.0
    else:
        f_stat = (r2 / p) / ((1 - r2) / (n - p - 1))
        p_value = f.sf(f_stat, p, n - p - 1) # sf = Survival Function (1 - CDF)
    
    # 3. Métricas de Magnitude de Erro
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    max_ae = np.max(np.abs(y_true - y_pred))
    bias = np.mean(y_pred - y_true)
    
    # 4. Métricas Percentuais
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    media_referencia = np.mean(y_true)
    cv = (rmse / media_referencia) * 100 if media_referencia != 0 else np.nan
    
    resultados = {
        'Arquivo': nome_arquivo,
        'N (Amostras)': n,
        'R² (%)': r2 * 100,
        'R² Ajust. (%)': r2_ajustado * 100,
        'Overfitting (%)': overfitting,
        'Est. F': f_stat,
        'Valor-p': p_value,
        'Pearson (r)': pearson_corr,
        'RMSE': rmse,
        'MAE': mae,
        'Max Erro': max_ae,
        'Bias': bias,
        'MAPE (%)': mape,
        'CV (%)': cv
    }
    
    return resultados
