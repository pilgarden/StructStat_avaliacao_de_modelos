"""
Módulo de Visualização (visualization.py)
Inclui gráficos avançados: Bland-Altman e Q-Q Plot.
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np
import scipy.stats as stats

def plotar_dispersao_referencia_previsto(df: pd.DataFrame, nome_modelo: str, unidade: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Referencia'], y=df['Previsto'], mode='markers', name='Dados',
                             marker=dict(size=8, color='#1f77b4', opacity=0.7, line=dict(width=1, color='DarkSlateGrey'))))

    min_val = min(df['Referencia'].min(), df['Previsto'].min()) * 0.95
    max_val = max(df['Referencia'].max(), df['Previsto'].max()) * 1.05

    fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='y = x',
                             line=dict(color='red', width=2, dash='dash')))

    fig.update_layout(title=f"Dispersão: Experimental vs Previsto", xaxis_title=f"Referência [{unidade}]",
                      yaxis_title=f"Previsto [{unidade}]", template="simple_white", margin=dict(l=40, r=40, t=40, b=40))
    fig.update_yaxes(scaleanchor="x", scaleratio=1, range=[min_val, max_val])
    fig.update_xaxes(range=[min_val, max_val])
    return fig

def plotar_distribuicao_erros(df: pd.DataFrame, nome_modelo: str, unidade: str) -> go.Figure:
    erros = df['Previsto'] - df['Referencia']
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=erros, name='Distribuição do Erro', marker_color='#2ca02c', opacity=0.75, nbinsx=30))
    fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="red", annotation_text="Erro Zero")
    fig.update_layout(title=f"Distribuição de Resíduos", xaxis_title=f"Erro (Previsto - Referência) [{unidade}]",
                      yaxis_title="Frequência", template="simple_white", bargap=0.05, margin=dict(l=40, r=40, t=40, b=40))
    return fig

def plotar_bland_altman(df: pd.DataFrame, nome_modelo: str, unidade: str) -> go.Figure:
    """
    Gráfico de Bland-Altman: Avalia concordância e vieses proporcionais.
    Ref: Bland, J. M., & Altman, D. (1986). Statistical methods for assessing agreement.
    """
    medias = (df['Referencia'] + df['Previsto']) / 2
    diferencas = df['Previsto'] - df['Referencia']
    
    media_diff = np.mean(diferencas)
    std_diff = np.std(diferencas)
    limite_superior = media_diff + 1.96 * std_diff
    limite_inferior = media_diff - 1.96 * std_diff

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=medias, y=diferencas, mode='markers', name='Diferenças',
                             marker=dict(size=8, color='#9467bd', opacity=0.7, line=dict(width=1, color='DarkSlateGrey'))))

    # Linhas Estatísticas
    fig.add_hline(y=media_diff, line_width=2, line_color="blue", annotation_text=f"Bias Médio: {media_diff:.2f}")
    fig.add_hline(y=limite_superior, line_width=2, line_dash="dash", line_color="red", annotation_text="+1.96 SD")
    fig.add_hline(y=limite_inferior, line_width=2, line_dash="dash", line_color="red", annotation_text="-1.96 SD")

    fig.update_layout(title=f"Gráfico de Bland-Altman", xaxis_title=f"Média (Previsto + Ref) / 2 [{unidade}]",
                      yaxis_title=f"Diferença (Previsto - Ref) [{unidade}]", template="simple_white", margin=dict(l=40, r=40, t=40, b=40))
    return fig

def plotar_qq_residuos(df: pd.DataFrame, nome_modelo: str) -> go.Figure:
    """
    Q-Q Plot: Avalia a normalidade dos resíduos.
    Desvios nas caudas indicam que os erros não seguem distribuição normal teórica.
    """
    erros = df['Previsto'] - df['Referencia']
    qq = stats.probplot(erros, dist="norm")
    quantis_teoricos = qq[0][0]
    quantis_reais = qq[0][1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=quantis_teoricos, y=quantis_reais, mode='markers', name='Resíduos',
                             marker=dict(size=8, color='#ff7f0e', opacity=0.7, line=dict(width=1, color='DarkSlateGrey'))))

    # Linha Teórica (Normal Perfeita)
    x_val = np.array([min(quantis_teoricos), max(quantis_teoricos)])
    y_val = qq[1][0] * x_val + qq[1][1]
    fig.add_trace(go.Scatter(x=x_val, y=y_val, mode='lines', name='Distribuição Normal Teórica',
                             line=dict(color='red', width=2, dash='dash')))

    fig.update_layout(title=f"Q-Q Plot (Normalidade de Resíduos)", xaxis_title="Quantis Teóricos (Normal)",
                      yaxis_title="Quantis Observados (Resíduos)", template="simple_white", margin=dict(l=40, r=40, t=40, b=40))
    return fig