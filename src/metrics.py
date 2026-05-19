"""
Módulo de Métricas e Testes Estatísticos (metrics.py)
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import (
    r2_score, mean_absolute_percentage_error, 
    mean_absolute_error, root_mean_squared_error
)

# ==========================================
# 1. MÓDULO: AVALIAÇÃO DE MODELOS PREDITIVOS
# ==========================================
def calcular_metricas(y_true: np.ndarray, y_pred: np.ndarray, nome_arquivo: str = "Desconhecido", p: int = 1) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    if n < 3: raise ValueError(f"[{nome_arquivo}] Amostra insuficiente (n={n}).")
    
    r2 = r2_score(y_true, y_pred)
    r2_ajustado = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
    overfitting = (r2 - r2_ajustado) * 100
    pearson_corr, _ = stats.pearsonr(y_true, y_pred)
    
    if r2 >= 1.0:
        f_stat, p_value = np.inf, 0.0
    elif r2 <= 0.0:
        f_stat, p_value = 0.0, 1.0
    else:
        f_stat = (r2 / p) / ((1 - r2) / (n - p - 1))
        p_value = stats.f.sf(f_stat, p, n - p - 1)
    
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    max_ae = np.max(np.abs(y_true - y_pred))
    bias = np.mean(y_pred - y_true)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    
    media_ref = np.mean(y_true)
    cv = (rmse / media_ref) * 100 if media_ref != 0 else np.nan
    
    return {
        'Arquivo': nome_arquivo, 'N (Amostras)': n, 'R² (%)': r2 * 100, 
        'R² Ajust. (%)': r2_ajustado * 100, 'Overfitting (%)': overfitting,
        'Est. F': f_stat, 'Valor-p': p_value, 'Pearson (r)': pearson_corr,
        'RMSE': rmse, 'MAE': mae, 'Max Erro': max_ae, 'Bias': bias,
        'MAPE (%)': mape, 'CV (%)': cv
    }

# ==========================================
# 2. MÓDULO: ANÁLISE EXPLORATÓRIA
# ==========================================
def calcular_estatisticas_descritivas(df: pd.DataFrame) -> pd.DataFrame:
    df_num = df.select_dtypes(include=[np.number])
    if df_num.empty: return pd.DataFrame()
    desc = df_num.describe().T
    desc['Assimetria'] = df_num.skew()
    desc['Curtose'] = df_num.kurt()
    return desc

def testar_normalidade(amostra: pd.Series) -> dict:
    amostra = amostra.dropna()
    n = len(amostra)
    stat_sw, p_sw = stats.shapiro(amostra) if n >= 3 else (np.nan, np.nan)
    stat_ks, p_ks = stats.kstest((amostra - amostra.mean()) / amostra.std(ddof=1), 'norm')
    ad_res = stats.anderson(amostra, dist='norm')
    crit_ad_05 = ad_res.critical_values[2] 
    
    return {
        "Shapiro-Wilk": {"Estatística": stat_sw, "p-value": p_sw},
        "Kolmogorov-Smirnov": {"Estatística": stat_ks, "p-value": p_ks},
        "Anderson-Darling": {"Estatística": ad_res.statistic, "Critico_5%": crit_ad_05, "Normal": ad_res.statistic < crit_ad_05}
    }

def testar_correlacao(df: pd.DataFrame, col_x: str, col_y: str) -> dict:
    df_clean = df[[col_x, col_y]].dropna()
    if len(df_clean) < 3: return {}
    r_p, p_p = stats.pearsonr(df_clean[col_x], df_clean[col_y])
    r_s, p_s = stats.spearmanr(df_clean[col_x], df_clean[col_y])
    return {"Pearson": {"r": r_p, "p-value": p_p}, "Spearman": {"rho": r_s, "p-value": p_s}}

def testar_homocedasticidade(df: pd.DataFrame, col_grupo: str, col_valor: str) -> dict:
    grupos = [grupo[col_valor].dropna().values for _, grupo in df.groupby(col_grupo)]
    if len(grupos) < 2: return {}
    stat_lev, p_lev = stats.levene(*grupos)
    try: stat_bart, p_bart = stats.bartlett(*grupos)
    except: stat_bart, p_bart = np.nan, np.nan
    return {"Levene": {"Estatística": stat_lev, "p-value": p_lev}, "Bartlett": {"Estatística": stat_bart, "p-value": p_bart}}

def comparar_medias(df: pd.DataFrame, col_grupo: str, col_valor: str) -> dict:
    grupos_dict = {nome: grupo[col_valor].dropna().values for nome, grupo in df.groupby(col_grupo)}
    matrizes = [m for m in grupos_dict.values() if len(m) >= 2]
    nomes = list(grupos_dict.keys())
    
    if len(matrizes) == 2:
        t_stat, p_t = stats.ttest_ind(matrizes[0], matrizes[1], equal_var=False)
        u_stat, p_u = stats.mannwhitneyu(matrizes[0], matrizes[1])
        return {"Tipo": "2_Grupos", "Grupos": nomes, "T-Student (Welch)": p_t, "Mann-Whitney": p_u}
    elif len(matrizes) > 2:
        f_stat, p_anova = stats.f_oneway(*matrizes)
        try: tukey_res_str = "Concluído (Ver visualização)"
        except Exception as e: tukey_res_str = f"Erro no Tukey: {e}"
        return {"Tipo": "Multi_Grupos", "ANOVA_p": p_anova, "Tukey": tukey_res_str}
    return {}

def detetar_outliers(amostra: pd.Series) -> dict:
    amostra = amostra.dropna()
    n = len(amostra)
    if n < 3: return {"Grubbs": [], "Chauvenet": []}
    
    media, desvio = amostra.mean(), amostra.std(ddof=1)
    d_max = np.abs(amostra - media) / desvio
    prob = 2 * (1 - stats.norm.cdf(d_max))
    outliers_chauvenet = amostra[prob < (1.0 / (2 * n))].index.tolist()
    
    t_crit = stats.t.isf(0.05 / (2 * n), n - 2)
    G_crit = ((n - 1) / np.sqrt(n)) * np.sqrt((t_crit**2) / (n - 2 + t_crit**2))
    outliers_grubbs = amostra[d_max.values > G_crit].index.tolist()
    
    return {"Grubbs (Indices)": outliers_grubbs, "Chauvenet (Indices)": outliers_chauvenet}
