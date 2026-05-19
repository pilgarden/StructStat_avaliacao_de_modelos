"""
Módulo de Diagnóstico Avançado de Modelos (model_diagnostics.py)
StructStat: Análise de Homocedasticidade, Multicolinearidade e Sensibilidade.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from SALib.analyze import sobol
from SALib.sample import saltelli

def analisar_homocedasticidade(model_results):
    """
    Realiza o Teste de Breusch-Pagan para verificar a constância da variância dos resíduos.
    """
    # Hipótese Nula (H0): Homocedasticidade presente.
    # p-value < 0.05 sugere Heterocedasticidade.
    from statsmodels.stats.diagnostic import het_breuschpagan
    resid = model_results.resid
    exog = sm.add_constant(model_results.model.exog)
    lm, p_value, f, f_p = het_breuschpagan(resid, exog)
    return {"p_value": p_value, "lm_stat": lm, "homocedastico": p_value > 0.05}

def calcular_vif(df_exog):
    """
    Calcula o Variance Inflation Factor (VIF) para detectar multicolinearidade.
    VIF > 10 indica multicolinearidade severa que deve ser corrigida.
    """
    vif_data = pd.DataFrame()
    vif_data["Variável"] = df_exog.columns
    vif_data["VIF"] = [variance_inflation_factor(df_exog.values, i) for i in range(df_exog.shape[1])]
    return vif_data

def executar_sensibilidade_sobol(problema, func, N=1024):
    """
    Sensibilidade Global (Sobol).
    problema: Dicionário da SALib (nomes, bounds, num_vars).
    func: Função do modelo que recebe um array de amostras.
    """
    param_values = saltelli.sample(problema, N)
    Y = np.array([func(val) for val in param_values])
    Si = sobol.analyze(problema, Y)
    return Si
