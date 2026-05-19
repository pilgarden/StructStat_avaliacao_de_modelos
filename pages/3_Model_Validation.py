import streamlit as st
import pandas as pd
from src.model_diagnostics import check_multicollinearity, check_homoscedasticity

def run_diagnostics_ui(df_X, y_true, y_pred):
    tab1, tab2, tab3 = st.tabs(["Multicolinearidade (VIF)", "Homocedasticidade", "Sensibilidade"])

    with tab1:
        st.subheader("Análise de Multicolinearidade")
        with st.expander("📖 Fundamentação Teórica"):
            st.write("""
            O **Fator de Inflação da Variância (VIF)** quantifica a multicolinearidade.
            * **VIF = 1**: Sem correlação.
            * **1 < VIF < 5**: Correlação moderada.
            * **VIF > 10**: Correlação severa; os coeficientes do modelo tornam-se instáveis.
            """)
        
        vif_df = check_multicollinearity(df_X)
        st.dataframe(vif_df.style.background_gradient(subset=["VIF"], cmap="Reds"), use_container_width=True)
        
        # Correção: CSV precisa de bytes para o download
        csv = vif_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Baixar Relatório VIF", data=csv, file_name="vif_report.csv", mime="text/csv")

    with tab2:
        st.subheader("Teste de Homocedasticidade (Breusch-Pagan)")
        results = check_homoscedasticity(y_true, y_pred)
        
        # Ajuste para as chaves corretas do dicionário criado no módulo
        p_val = results['p-valor (LM)']
        
        col1, col2 = st.columns(2)
        col1.metric("Estatística LM", f"{results['Estatística LM']:.4f}")
        col2.metric("P-Valor", f"{p_val:.4f}")
        
        if p_val < 0.05:
            st.warning("⚠️ **Heterocedasticidade detectada!** (p < 0.05). A variância dos resíduos não é constante.")
        else:
            st.success("✅ **Homocedasticidade confirmada.** (p >= 0.05). O modelo atende à premissa de variância constante.")

    with tab3:
        st.subheader("Análise de Sensibilidade Global (Sobol)")
        st.info("Utilize esta secção para identificar quais variáveis de entrada (ex: propriedades de materiais, geometria) mais influenciam a variância da resposta estrutural.")
        # Aqui você implementará a configuração do dicionário 'problem'
        st.warning("Módulo em desenvolvimento: Configure os limites (bounds) das variáveis para rodar a análise.")
