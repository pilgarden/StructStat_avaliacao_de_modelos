"""
Agrega métricas e resultados do diagnóstico avançado para relatório PDF/UI.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import r2_score

from src.metrics import calcular_metricas
from src.model_diagnostics import (
    check_homoscedasticity,
    check_multicollinearity,
    detect_outliers_grubbs,
)


def compute_diagnostics_report(
    df: pd.DataFrame,
    alvo: str,
    previsto: str,
    preditores: list[str],
    dataset_name: str = "Dataset",
    sobol_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Calcula todas as métricas usadas nas abas de diagnóstico."""
    df_clean = df[[alvo, previsto]].dropna().copy()
    y_true = df_clean[alvo].values
    y_pred = df_clean[previsto].values
    residuos = df_clean[alvo] - df_clean[previsto]

    report: dict[str, Any] = {
        "meta": {
            "dataset": dataset_name,
            "gerado_em": datetime.now().strftime("%d/%m/%Y %H:%M"),
            "alvo": alvo,
            "previsto": previsto,
            "preditores": list(preditores),
            "n_amostras": len(df_clean),
        },
        "metricas": {},
        "residuos": {},
        "outliers": {},
        "vif": None,
        "pareceres": {},
        "sobol": sobol_df,
    }

    if len(df_clean) < 3:
        report["erro"] = "Amostra insuficiente (mínimo 3 observações)."
        return report

    try:
        report["metricas"] = calcular_metricas(
            y_true, y_pred, nome_arquivo=dataset_name, p=max(len(preditores), 1)
        )
    except Exception as e:
        report["metricas"] = {"R² (%)": r2_score(y_true, y_pred) * 100, "Nota": str(e)}

    report["metricas"]["R2_linear"] = float(r2_score(y_true, y_pred))

    # Resíduos e testes
    df_clean["Residuos"] = residuos
    df_clean["Theta"] = np.where(df_clean[previsto] != 0, df_clean[alvo] / df_clean[previsto], np.nan)
    std_resid = float(residuos.std(ddof=1)) if len(residuos) > 1 else 0.0
    mean_theta = float(df_clean["Theta"].mean())
    std_theta = float(df_clean["Theta"].std(ddof=1))
    cov_theta = (std_theta / mean_theta) if mean_theta != 0 else 0.0

    results_bp = check_homoscedasticity(df_clean[alvo], df_clean[previsto])
    p_bp = float(results_bp.get("p-valor (LM)", results_bp.get("p-value", 1.0)))
    _, p_sw = stats.shapiro(residuos)
    _, p_ad = sm.stats.diagnostic.normal_ad(residuos)

    report["residuos"] = {
        "media": float(residuos.mean()),
        "desvio_padrao": std_resid,
        "media_theta": mean_theta,
        "cov_theta_pct": cov_theta * 100,
        "p_breusch_pagan": p_bp,
        "p_shapiro": float(p_sw),
        "p_anderson_darling": float(p_ad),
    }

    # Outliers
    std_out = float(residuos.std())
    if std_out > 0 and len(residuos) > 0:
        z_scores = np.abs((residuos - residuos.mean()) / std_out)
        outliers_z = z_scores > 3
        pct_out = float(outliers_z.sum() / len(residuos) * 100)
        _, grubbs_flag, grubbs_idx = detect_outliers_grubbs(residuos.values)
    else:
        pct_out = 0.0
        grubbs_flag, grubbs_idx = False, 0

    report["outliers"] = {
        "percentagem_z3": pct_out,
        "grubbs_detectado": bool(grubbs_flag),
        "grubbs_indice": int(grubbs_idx),
        "mask_z3": outliers_z.values if std_out > 0 and len(residuos) > 0 else np.zeros(len(residuos), dtype=bool),
    }

    # VIF
    try:
        df_x = df[preditores].dropna()
        if len(df_x) >= 2 and len(preditores) >= 1:
            vif_df = check_multicollinearity(df_x).copy()
            vif_df["Tolerancia"] = 1 / vif_df["VIF"].replace(0, np.nan)
            report["vif"] = vif_df
            criticas = vif_df.loc[vif_df["VIF"] > 10, "Feature"].tolist()
            report["pareceres"]["vif"] = (
                f"Multicolinearidade critica em: {criticas}"
                if criticas
                else "VIF sob controlo (todas <= 10)."
            )
    except Exception as e:
        report["pareceres"]["vif"] = f"Nao foi possivel calcular VIF: {e}"

    n = len(df_clean)
    if p_bp < 0.05:
        report["pareceres"]["homocedasticidade"] = "Heterocedasticidade detectada (Breusch-Pagan p < 0.05)."
    else:
        report["pareceres"]["homocedasticidade"] = "Homocedasticidade confirmada (Breusch-Pagan p >= 0.05)."

    if p_ad >= 0.05 and p_sw >= 0.05:
        report["pareceres"]["normalidade"] = "Normalidade dos residuos confirmada pelos testes formais."
    elif p_ad < 0.05 and p_sw < 0.05:
        report["pareceres"]["normalidade"] = "Residuos nao normais (ambos os testes p < 0.05)."
    else:
        teste_pref = "Shapiro-Wilk" if n < 50 else "Anderson-Darling"
        report["pareceres"]["normalidade"] = (
            f"Resultados divergentes; priorizar {teste_pref} (N={n})."
        )

    if pct_out > 5:
        report["pareceres"]["outliers"] = f"Percentagem elevada de outliers ({pct_out:.2f}% com |Z|>3)."
    else:
        report["pareceres"]["outliers"] = f"Outliers sob controlo ({pct_out:.2f}% com |Z|>3)."

    report["df_clean"] = df_clean
    report["residuos_series"] = residuos
    return report
