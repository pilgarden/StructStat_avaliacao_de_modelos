import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats  # Adicione esta linha
from statsmodels.stats.outliers_influence import variance_inflation_factor
from SALib.sample import saltelli
from SALib.analyze import sobol
import streamlit as st

def check_multicollinearity(df_X: pd.DataFrame):
    """Calcula o VIF para as variáveis independentes."""
    cols_to_use = [c for c in df_X.columns if 'const' not in c.lower()]
    X = sm.add_constant(df_X[cols_to_use])
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data.sort_values(by="VIF", ascending=False)

def check_homoscedasticity(y_true, y_pred):
    """Executa o Teste de Breusch-Pagan nos resíduos."""
    residuals = y_true - y_pred
    exog = sm.add_constant(y_pred)
    test = sm.stats.diagnostic.het_breuschpagan(residuals, exog)
    return {
        "Estatística LM": test[0],
        "p-valor (LM)": test[1],
        "Estatística F": test[2],
        "p-valor (F)": test[3]
    }

@st.cache_data(show_spinner=False)
def run_sobol_sensitivity(_model_func, problem, num_samples=1024):
    """
    Análise de Sensibilidade Global Sobol com cache para evitar recálculos.
    O prefixo '_' em '_model_func' indica ao Streamlit para ignorar o hash desta função.
    """
    try:
        # Gera a matriz de amostras via Saltelli
        param_values = saltelli.sample(problem, num_samples)
        
        # Executa a função do modelo para cada combinação de parâmetros
        Y = np.array([_model_func(p) for p in param_values])
        
        # Calcula os índices de Sobol
        Si = sobol.analyze(problem, Y, print_to_console=False)
        return Si
    except Exception as e:
        import streamlit as st
        st.error(f"Erro na análise de Sobol: {e}")
        return None

def detect_outliers_grubbs(data):
    # Agora a variável 'stats' será reconhecida
    n = len(data)
    if n <= 2: return None, None, None
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
