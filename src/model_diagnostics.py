import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from SALib.sample import saltelli
from SALib.analyze import sobol
import streamlit as st

def check_multicollinearity(df_X: pd.DataFrame):
    """
    Calcula o VIF. Remove a constante automaticamente se ela já existir 
    para evitar erros de matriz singular.
    """
    # Remove 'const' se já existir para evitar erro
    cols_to_use = [c for c in df_X.columns if 'const' not in c.lower()]
    X = sm.add_constant(df_X[cols_to_use])
    
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    
    # Cálculo com try-except para evitar erro se houver correlação perfeita
    try:
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    except Exception as e:
        st.error(f"Erro no cálculo do VIF: {e}. Verifique se há variáveis perfeitamente colineares.")
        vif_data["VIF"] = np.nan
        
    return vif_data.sort_values(by="VIF", ascending=False)

def check_homoscedasticity(y_true, y_pred):
    """
    Teste de Breusch-Pagan. Retorna um dicionário formatado.
    """
    residuals = y_true - y_pred
    exog = sm.add_constant(y_pred)
    
    # O teste de Breusch-Pagan retorna: (lm, lm_pvalue, fvalue, f_pvalue)
    test = sm.stats.diagnostic.het_breuschpagan(residuals, exog)
    
    return {
        "Estatística BP": test[0],
        "p-valor (LM)": test[1],
        "Estatística F": test[2],
        "p-valor (F)": test[3]
    }

@st.cache_data(show_spinner=False)
def run_sobol_sensitivity(model_func, problem, num_samples=1024):
    """
    Análise de Sensibilidade Global Sobol com cache para evitar recálculos.
    """
    try:
        param_values = saltelli.sample(problem, num_samples)
        # Executa a função do modelo para cada combinação de parâmetros
        Y = np.array([model_func(p) for p in param_values])
        Si = sobol.analyze(problem, Y, print_to_console=False)
        return Si
    except Exception as e:
        st.error(f"Erro na análise de Sobol: {e}")
        return None
