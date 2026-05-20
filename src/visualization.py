import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats
import pandas as pd
import numpy as np

# Tenta importar MM_TO_PX do config, se falhar, define um padrão
try:
    from src.config import MM_TO_PX
except ImportError:
    MM_TO_PX = 3.77953  # Valor padrão (aprox 96 DPI)

def _aplicar_estilo_tufte(fig: go.Figure, kwargs: dict) -> go.Figure:
    """Aplica Data-Ink Ratio e formatação científica."""
    w_px = int(kwargs.get('width_mm', 150) * MM_TO_PX)
    h_px = int(kwargs.get('height_mm', 100) * MM_TO_PX)
    bg_color = 'white' if kwargs.get('fundo_branco', True) else 'rgba(0,0,0,0)'
    
    fig.update_layout(
        width=w_px, height=h_px, 
        template='simple_white',
        paper_bgcolor=bg_color, 
        plot_bgcolor=bg_color,
        font=dict(family=kwargs.get('font_family', 'Arial'), size=kwargs.get('font_size', 12), color='black'),
        margin=dict(l=50, r=20, t=40, b=50)
    )
    
    # Eixos
    if kwargs.get('title_x'): fig.update_xaxes(title_text=kwargs['title_x'])
    if kwargs.get('title_y'): fig.update_yaxes(title_text=kwargs['title_y'])
    
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor='black', ticks="outside")
    fig.update_yaxes(showgrid=False, zeroline=False, linecolor='black', ticks="outside")
    return fig

# ==========================================
# GRÁFICOS DE AVALIAÇÃO DE MODELOS
# ==========================================
def plotar_dispersao_referencia_previsto(df: pd.DataFrame, unidade: str, kwargs: dict) -> go.Figure:
    if df.empty: return go.Figure()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Referencia'], y=df['Previsto'], mode='markers', name='Dados',
                             marker=dict(size=8, color='#1f77b4', opacity=0.7, line=dict(width=1, color='black'))))
    
    # Linha de identidade (y=x)
    min_val = min(df['Referencia'].min(), df['Previsto'].min()) * 0.95
    max_val = max(df['Referencia'].max(), df['Previsto'].max()) * 1.05
    fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Ideal',
                             line=dict(color='red', width=1.5, dash='dash')))
    
    fig.update_layout(title="Real vs Previsto", xaxis_title=f"Referência [{unidade}]", yaxis_title=f"Previsto [{unidade}]")
    return _aplicar_estilo_tufte(fig, kwargs)

# ==========================================
# GRÁFICOS DE ANÁLISE EXPLORATÓRIA
# ==========================================
def plotar_histograma(amostra: pd.Series, mostrar_normal: bool, kwargs: dict) -> go.Figure:
    amostra = amostra.dropna()
    if amostra.empty: return go.Figure()
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=amostra, histnorm='probability density', marker_color='#1F77B4', opacity=0.7, name='Dados'))
    
    if mostrar_normal and len(amostra) > 2:
        media, desvio = amostra.mean(), amostra.std(ddof=1)
        x_norm = np.linspace(amostra.min(), amostra.max(), 200)
        fig.add_trace(go.Scatter(x=x_norm, y=stats.norm.pdf(x_norm, media, desvio), mode='lines', name='Normal', line=dict(color='red', width=2, dash='dash')))
    
    return _aplicar_estilo_tufte(fig, kwargs)

def plotar_matriz_calor(df: pd.DataFrame, colunas_x: list, colunas_y: list, metodo: str, kwargs: dict) -> go.Figure:
    matriz = df[list(set(colunas_x + colunas_y))].corr(method=metodo).loc[colunas_y, colunas_x]
    fig = px.imshow(matriz, text_auto=".2f", color_continuous_scale=kwargs.get('palette', 'RdBu_r'), origin='lower')
    return _aplicar_estilo_tufte(fig, kwargs)
