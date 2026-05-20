import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats  # ESSENCIAL PARA O TESTE DE GRUBBS
from statsmodels.stats.outliers_influence import variance_inflation_factor
from SALib.sample import saltelli
from SALib.analyze import sobol
import streamlit as st

def check_multicollinearity(df_X: pd.DataFrame):
    cols_to_use = [c for c in df_X.columns if 'const' not in c.lower()]
    X = sm.add_constant(df_X[cols_to_use])
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    try:
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    except:
        vif_data["VIF"] = np.nan
    return vif_data.sort_values(by="VIF", ascending=False)

def check_homoscedasticity(y_true, y_pred):
    residuals = y_true - y_pred
    exog = sm.add_constant(y_pred)
    test = sm.stats.diagnostic.het_breuschpagan(residuals, exog)
    return {"Estatística LM": test[0], "p-valor (LM)": test[1], "Estatística F": test[2], "p-valor (F)": test[3]}

@st.cache_data(show_spinner=False)
def run_sobol_sensitivity(_model_func, problem, num_samples=1024):
    try:
        param_values = saltelli.sample(problem, num_samples)
        Y = np.array([_model_func(p) for p in param_values])
        Si = sobol.analyze(problem, Y, print_to_console=False)
        return Si
    except Exception:
        return None

def detect_outliers_grubbs(data):
    # Agora o 'stats' será encontrado aqui
    n = len(data)
    if n <= 2: return 0, False, 0
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    abs_diff = np.abs(data - mean)
    max_diff = np.max(abs_diff)
    max_idx = np.argmax(abs_diff)
    g_stat = max_diff / std
    t_dist = stats.t.ppf(1 - 0.05 / (2 * n), n - 2)
    g_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_dist**2 / (n - 2 + t_dist**2))
    is_outlier = g_stat > g_crit
    return g_stat, is_outlier, max_idx
