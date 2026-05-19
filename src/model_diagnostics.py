import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from SALib.sample import saltelli
from SALib.analyze import sobol

def check_multicollinearity(df_X):
    """
    Calcula o Fator de Inflação da Variância (VIF).
    VIF > 10 indica multicolinearidade severa.
    """
    X = sm.add_constant(df_X)
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

def check_homoscedasticity(y_true, y_pred):
    """
    Testes de Homocedasticidade (Resíduos vs Previsto).
    Retorna o p-valor do teste de Breusch-Pagan.
    """
    residuals = y_true - y_pred
    # Breusch-Pagan: H0 é homocedasticidade
    name = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
    test = sm.stats.diagnostic.het_breuschpagan(residuals, sm.add_constant(y_pred))
    return dict(zip(name, test))

def run_sobol_sensitivity(model_func, problem):
    """
    Análise de Sensibilidade Global Sobol.
    problem: dict com 'num_vars', 'names', 'bounds'
    """
    param_values = saltelli.sample(problem, 1024)
    Y = np.array([model_func(p) for p in param_values])
    Si = sobol.analyze(problem, Y, print_to_console=False)
    return Si
