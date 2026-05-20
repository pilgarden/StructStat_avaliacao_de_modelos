"""
Página: Análise Exploratória (pages/1_Analise_Exploratoria.py)
StructStat: Módulo de diagnóstico estatístico e visualização.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import scipy.stats as stats
from src.data_processing import carregar_dados, aplicar_filtro_dinamico

st.set_page_config(page_title="Análise Exploratória", layout="wide")

st.title("📊 Análise Exploratória e Paramétrica")

# 1. Upload e Persistência
uploaded_file = st.sidebar.file_uploader("Upload de Dados (CSV/Excel)", type=['csv', 'xlsx'])

if uploaded_file:
    df = carregar_dados(uploaded_file, uploaded_file.name)
    
    # Sidebar: Filtros Dinâmicos
    with st.sidebar.expander("🛠️ Filtros de Subconjunto"):
        col_filtro = st.selectbox("Coluna para filtrar:", df.columns)
        tipo_filtro = st.selectbox("Regra:", ["Valores Exatos", "Menor ou igual (<=)", "Maior ou igual (>=)", "Contém (Texto)"])
        val_filtro = st.text_input("Valor:")
        
        if st.button("Aplicar Filtro"):
            df = aplicar_filtro_dinamico(df, col_filtro, tipo_filtro, val_filtro)

    # 2. Tabs de Navegação
    tab1, tab2, tab3 = st.tabs(["Estatísticas Descritivas", "Normalidade e Variância", "Correlação"])

    with tab1:
        st.subheader("Resumo Estatístico")
        st.write(df.describe())
        
        col_hist = st.selectbox("Variável para Histograma:", df.select_dtypes(include=['number']).columns)
        fig = px.histogram(df, x=col_hist, marginal="rug", hover_data=df.columns, nbins=30)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Testes de Normalidade e Q-Q Plot")
        var_norm = st.selectbox("Selecionar variável:", df.select_dtypes(include=['number']).columns)
        
        # Shapiro-Wilk
        stat, p = stats.shapiro(df[var_norm].dropna())
        st.write(f"**Shapiro-Wilk:** p-value = {p:.4f} ({'Normal' if p > 0.05 else 'Não Normal'})")
        
        # Teste de Levene
        st.subheader("Teste de Homocedasticidade (Levene)")
        grupos = st.multiselect("Grupos para comparar (Categorias):", df.select_dtypes(include=['object']).columns)
        # (Lógica simplificada de aplicação do Levene...)

    with tab3:
        st.subheader("Matriz de Correlação")
        x_axis = st.selectbox("Variável X:", df.select_dtypes(include=['number']).columns)
        y_axis = st.selectbox("Variável Y:", df.select_dtypes(include=['number']).columns)
        
        corr, _ = stats.pearsonr(df[x_axis], df[y_axis])
        st.metric("Correlação de Pearson", f"{corr:.3f}")
        
        fig_corr = px.scatter(df, x=x_axis, y=y_axis, trendline="ols")
        st.plotly_chart(fig_corr, use_container_width=True)

else:
    st.info("Por favor, faça o upload de um ficheiro para iniciar a análise.")
