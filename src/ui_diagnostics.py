import streamlit as st
import pandas as pd
import plotly.express as px
from scipy import stats
from src.model_diagnostics import check_multicollinearity, check_homoscedasticity

def render_diagnostics():
    st.title("📊 Diagnóstico Avançado de Modelos")
    
    if 'df_global' not in st.session_state:
        st.warning("⚠️ Carregue um arquivo de dados na barra lateral para ativar este módulo.")
        return
        
    df = st.session_state['df_global']
    
    st.markdown("### ⚙️ Seleção de Variáveis para Análise")
    col1, col2 = st.columns(2)
    with col1:
        alvo = st.selectbox("Variável Real (Referência $y$):", df.columns, key="diag_alvo")
        previsto = st.selectbox("Variável Prevista (Modelo $\hat{y}$):", df.columns, key="diag_prev")
    with col2:
        preditores = st.multiselect("Variáveis Independentes (Inputs $X$ para VIF):", df.columns, key="diag_preds")
        
    if not preditores or alvo == previsto:
        st.info("👈 Selecione as variáveis independentes e garanta que os eixos de Referência e Previsão são distintos.")
        return

    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["Homocedasticidade e Resíduos", "Multicolinearidade (VIF)", "Sensibilidade (Sobol)"])
    
    with tab1:
        st.subheader("Análise de Resíduos")
        try:
            df_clean = df[[alvo, previsto]].dropna()
            residuos = df_clean[alvo] - df_clean[previsto]
            
            fig_res = px.scatter(
                x=df_clean[previsto], y=residuos, 
                labels={'x': 'Valores Previstos ($\hat{y}$)', 'y': 'Resíduos ($y - \hat{y}$)'}
            )
            fig_res.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_res, use_container_width=True)
            
            results_bp = check_homoscedasticity(df_clean[alvo], df_clean[previsto])
            p_val_bp = results_bp.get('p-valor (LM)', results_bp.get('p-value', 0))
            stat_sw, p_val_sw = stats.shapiro(residuos)
            
            c1, c2 = st.columns(2)
            c1.metric("P-Valor (Breusch-Pagan)", f"{p_val_bp:.4f}")
            c2.metric("P-Valor (Shapiro-Wilk)", f"{p_val_sw:.4f}")
        except Exception as e:
            st.error(f"Erro: {e}")
            
    with tab2:
        st.subheader("Multicolinearidade (VIF)")
        try:
            df_X = df[preditores].dropna()
            vif_df = check_multicollinearity(df_X)
            vif_df['Tolerância'] = 1 / vif_df['VIF']
            
            fig_vif = px.bar(vif_df, x='Feature', y='VIF', color='VIF', color_continuous_scale='Reds')
            fig_vif.add_hline(y=10, line_dash="dash", line_color="darkred")
            st.plotly_chart(fig_vif, use_container_width=True)
            
            st.dataframe(vif_df.style.background_gradient(subset=["VIF"], cmap="Reds").format({"VIF": "{:.2f}", "Tolerância": "{:.4f}"}), use_container_width=True)
        except Exception as e:
            st.error(f"Erro: {e}")
            
    with tab3:
        st.info("Módulo de Sensibilidade Sobol em desenvolvimento.")
