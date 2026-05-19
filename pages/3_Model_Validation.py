import streamlit as st
import pandas as pd
import plotly.express as px
from scipy import stats
from src.model_diagnostics import check_multicollinearity, check_homoscedasticity

def interface_diagnostico_avancado():
    st.title("📊 Diagnóstico Avançado de Modelos")
    
    if 'df_global' not in st.session_state:
        st.warning("⚠️ Carregue um arquivo de dados na barra lateral para prosseguir.")
        return
        
    df = st.session_state['df_global']
    
    # 1. Configuração de Variáveis (UI)
    st.markdown("### ⚙️ Seleção de Variáveis para Análise")
    col1, col2 = st.columns(2)
    with col1:
        alvo = st.selectbox("Variável Real (Referência $y$):", df.columns)
        previsto = st.selectbox("Variável Prevista (Modelo $\hat{y}$):", df.columns)
    with col2:
        preditores = st.multiselect("Variáveis Independentes (Inputs $X$ para VIF):", df.columns)
        
    if not preditores or alvo == previsto:
        st.info("👈 Selecione as variáveis independentes e garanta que a Referência e o Previsto são diferentes.")
        return

    st.markdown("---")
    
    # 2. Execução dos Diagnósticos
    tab1, tab2, tab3 = st.tabs(["Homocedasticidade e Resíduos", "Multicolinearidade (VIF)", "Sensibilidade (Sobol)"])
    
    with tab1:
        st.subheader("Análise de Resíduos: Breusch-Pagan e Shapiro-Wilk")
        try:
            df_clean = df[[alvo, previsto]].dropna()
            residuos = df_clean[alvo] - df_clean[previsto]
            
            # Gráfico: Resíduos vs Previstos
            fig_res = px.scatter(
                x=df_clean[previsto], y=residuos, 
                labels={'x': 'Valores Previstos ($\hat{y}$)', 'y': 'Resíduos ($y - \hat{y}$)'},
                title="Dispersão de Resíduos (Análise Visual de Variância)"
            )
            fig_res.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_res, use_container_width=True)
            
            # Cálculos Estatísticos
            results_bp = check_homoscedasticity(df_clean[alvo], df_clean[previsto])
            p_val_bp = results_bp.get('p-valor (LM)', results_bp.get('p-value', 0))
            
            # Novo Método: Teste de Shapiro-Wilk (Normalidade dos Resíduos)
            stat_sw, p_val_sw = stats.shapiro(residuos)
            
            # Exibição de Métricas Lado a Lado
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Teste de Breusch-Pagan (Homocedasticidade)**")
                st.metric("P-Valor (BP)", f"{p_val_bp:.4f}")
                if p_val_bp < 0.05:
                    st.warning("⚠️ **Heterocedasticidade:** Variância dos resíduos não é constante.")
                else:
                    st.success("✅ **Homocedasticidade:** Variância constante confirmada.")
            with c2:
                st.markdown("**Teste de Shapiro-Wilk (Normalidade)**")
                st.metric("P-Valor (SW)", f"{p_val_sw:.4f}")
                if p_val_sw < 0.05:
                    st.warning("⚠️ **Não-Normal:** Os resíduos não seguem distribuição normal.")
                else:
                    st.success("✅ **Normalidade:** Resíduos normalmente distribuídos.")

        except Exception as e:
            st.error(f"Erro na análise de resíduos: {e}")
            
    with tab2:
        st.subheader("Fator de Inflação da Variância (VIF) e Tolerância")
        try:
            df_X = df[preditores].dropna()
            vif_df = check_multicollinearity(df_X)
            
            # Novo Método: Adicionar Tolerância
            vif_df['Tolerância'] = 1 / vif_df['VIF']
            
            # Gráfico de Barras VIF
            fig_vif = px.bar(
                vif_df, x='Feature', y='VIF', 
                color='VIF', color_continuous_scale='Reds',
                title='Nível de Multicolinearidade por Variável'
            )
            # Linha de limite crítico
            fig_vif.add_hline(y=10, line_dash="dash", line_color="darkred", annotation_text="Limite Crítico (VIF=10)")
            st.plotly_chart(fig_vif, use_container_width=True)
            
            # Tabela Formatada
            st.dataframe(vif_df.style.background_gradient(subset=["VIF"], cmap="Reds").format({"VIF": "{:.2f}", "Tolerância": "{:.4f}"}), use_container_width=True)
            
            csv = vif_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Exportar Tabela VIF", data=csv, file_name="vif_report.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Erro ao calcular VIF: {e}")
            
    with tab3:
        st.subheader("Análise de Sensibilidade Global (Sobol)")
        st.info("Módulo em desenvolvimento. Aqui configuraremos os limites (bounds) das variáveis para a simulação de Monte Carlo.")
