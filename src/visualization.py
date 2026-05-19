"""
Módulo de Visualização (visualization.py)
Gráficos para Análise Exploratória e Avaliação de Modelos. 
Todos integrados ao layout Tufte (Publicação Científica).
"""

import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats
import pandas as pd
import numpy as np
from src.config import MM_TO_PX

def _aplicar_estilo_tufte(fig: go.Figure, kwargs: dict) -> go.Figure:
    """Aplica Data-Ink Ratio e formatação LaTeX/Q1."""
    if not kwargs: return fig
    
    w_px = int(kwargs.get('width_mm', 150) * MM_TO_PX)
    h_px = int(kwargs.get('height_mm', 100) * MM_TO_PX)
    bg_color = 'white' if kwargs.get('fundo_branco', True) else 'rgba(0,0,0,0)'
    font_family = kwargs.get('font_family', 'Arial')
    font_size = kwargs.get('font_size', 12)
    
    fig.update_layout(
        width=w_px, height=h_px, template='simple_white',
        paper_bgcolor=bg_color, plot_bgcolor=bg_color,
        font=dict(family=font_family, size=font_size, color='black'),
        margin=dict(l=50, r=20, t=40, b=50)
    )
    
    # Se os eixos X e Y forem passados explicitamente em kwargs, eles sobrescrevem
    if kwargs.get('title_x'): fig.update_xaxes(title_text=kwargs['title_x'])
    if kwargs.get('title_y'): fig.update_yaxes(title_text=kwargs['title_y'])
    
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor='black')
    fig.update_yaxes(showgrid=False, zeroline=False, linecolor='black')
    return fig

# ==========================================
# GRÁFICOS: AVALIAÇÃO DE MODELOS PREDITIVOS
# ==========================================
def plotar_dispersao_referencia_previsto(df: pd.DataFrame, unidade: str, kwargs: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Referencia'], y=df['Previsto'], mode='markers', name='Dados',
                             marker=dict(size=8, color='#1f77b4', opacity=0.7, line=dict(width=1, color='DarkSlateGrey'))))
    min_val = min(df['Referencia'].min(), df['Previsto'].min()) * 0.95
    max_val = max(df['Referencia'].max(), df['Previsto'].max()) * 1.05
    fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='y = x',
                             line=dict(color='red', width=2, dash='dash')))
    
    # Defaults base
    fig.update_layout(title=f"Dispersão: Real vs Previsto", xaxis_title=f"Referência [{unidade}]", yaxis_title=f"Previsto [{unidade}]")
    fig.update_yaxes(scaleanchor="x", scaleratio=1, range=[min_val, max_val])
    fig.update_xaxes(range=[min_val, max_val])
    return _aplicar_estilo_tufte(fig, kwargs)

def plotar_bland_altman(df: pd.DataFrame, unidade: str, kwargs: dict) -> go.Figure:
    medias = (df['Referencia'] + df['Previsto']) / 2
    diferencas = df['Previsto'] - df['Referencia']
    media_diff, std_diff = np.mean(diferencas), np.std(diferencas)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=medias, y=diferencas, mode='markers', name='Diferenças',
                             marker=dict(size=8, color='#9467bd', opacity=0.7, line=dict(width=1, color='DarkSlateGrey'))))
    fig.add_hline(y=media_diff, line_width=2, line_color="blue", annotation_text=f"Bias: {media_diff:.2f}")
    fig.add_hline(y=media_diff + 1.96 * std_diff, line_width=2, line_dash="dash", line_color="red")
    fig.add_hline(y=media_diff - 1.96 * std_diff, line_width=2, line_dash="dash", line_color="red")
    
    fig.update_layout(title="Bland-Altman", xaxis_title=f"Média [{unidade}]", yaxis_title=f"Erro (Prev - Ref) [{unidade}]")
    return _aplicar_estilo_tufte(fig, kwargs)

def plotar_distribuicao_erros(df: pd.DataFrame, unidade: str, kwargs: dict) -> go.Figure:
    erros = df['Previsto'] - df['Referencia']
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=erros, name='Distribuição', marker_color='#2ca02c', opacity=0.75))
    fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="red")
    fig.update_layout(title="Histograma de Resíduos", xaxis_title=f"Erro [{unidade}]", yaxis_title="Frequência")
    return _aplicar_estilo_tufte(fig, kwargs)

def plotar_qq_residuos(df: pd.DataFrame, kwargs: dict) -> go.Figure:
    erros = df['Previsto'] - df['Referencia']
    qq = stats.probplot(erros, dist="norm")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers', marker=dict(size=8, color='#ff7f0e')))
    fig.add_trace(go.Scatter(x=np.array([min(qq[0][0]), max(qq[0][0])]), y=qq[1][0] * np.array([min(qq[0][0]), max(qq[0][0])]) + qq[1][1], mode='lines', line=dict(color='red', dash='dash')))
    fig.update_layout(title="Q-Q Plot (Resíduos)", xaxis_title="Quantis Teóricos", yaxis_title="Resíduos")
    return _aplicar_estilo_tufte(fig, kwargs)

# ==========================================
# GRÁFICOS: ANÁLISE EXPLORATÓRIA
# ==========================================
def plotar_histograma(amostra: pd.Series, mostrar_normal: bool, kwargs: dict) -> go.Figure:
    amostra = amostra.dropna()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=amostra, histnorm='probability density', marker_color='#1F77B4', opacity=0.7))
    if mostrar_normal and len(amostra) > 2:
        media, desvio = amostra.mean(), amostra.std(ddof=1)
        x_norm = np.linspace(amostra.min(), amostra.max(), 200)
        fig.add_trace(go.Scatter(x=x_norm, y=stats.norm.pdf(x_norm, media, desvio), mode='lines', line=dict(color='red', width=2, dash='dash')))
    return _aplicar_estilo_tufte(fig, kwargs)

def plotar_qq(amostra: pd.Series, kwargs: dict) -> go.Figure:
    amostra = amostra.dropna()
    qq = stats.probplot(amostra, dist='norm')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers', marker=dict(color='#1F77B4', size=6)))
    fig.add_trace(go.Scatter(x=np.array([min(qq[0][0]), max(qq[0][0])]), y=qq[1][0] * np.array([min(qq[0][0]), max(qq[0][0])]) + qq[1][1], mode='lines', line=dict(color='red', dash='dash')))
    return _aplicar_estilo_tufte(fig, kwargs)

def plotar_matriz_calor(df: pd.DataFrame, colunas_x: list, colunas_y: list, metodo: str, kwargs: dict) -> go.Figure:
    todas_cols = list(set(colunas_x + colunas_y))
    matriz = df[todas_cols].corr(method=metodo).loc[colunas_y, colunas_x]
    paleta = kwargs.get('palette', 'viridis')
    fig = px.imshow(matriz, text_auto=".2f", aspect="auto", color_continuous_scale=paleta, origin='lower')
    fig = _aplicar_estilo_tufte(fig, kwargs)
    fig.update_xaxes(showline=False); fig.update_yaxes(showline=False)
    return fig

def plotar_dispersao(df: pd.DataFrame, col_x: str, col_y: str, kwargs: dict) -> go.Figure:
    fig = px.scatter(df, x=col_x, y=col_y, opacity=0.8, color_discrete_sequence=['#1F77B4'])
    return _aplicar_estilo_tufte(fig, kwargs)
