import streamlit as st
from src.model_diagnostics import check_multicollinearity, check_homoscedasticity

def run_diagnostics_ui(df_X, y_true, y_pred):
    tab1, tab2, tab3 = st.tabs(["Multicolinearidade (VIF)", "Homocedasticidade", "Sensibilidade"])

    with tab1:
        st.subheader("Análise de Multicolinearidade")
        with st.expander("📖 Fundamentação Teórica"):
            st.write("O VIF quantifica o quanto a variância de um coeficiente é inflacionada pela correlação entre preditores. VIF > 10 exige atenção.")
        
        vif_df = check_multicollinearity(df_X)
        st.dataframe(vif_df.style.background_gradient(subset=["VIF"], cmap="Reds"))
        st.download_button("Baixar VIF", vif_df.to_csv(), "vif_report.csv")

    with tab2:
        st.subheader("Teste de Homocedasticidade (Breusch-Pagan)")
        results = check_homoscedasticity(y_true, y_pred)
        st.metric("P-Valor do Teste", f"{results['p-value']:.4f}")
        if results['p-value'] < 0.05:
            st.warning("Heterocedasticidade detectada! Considere usar Erros Padrão Robustos (HC3).")
        else:
            st.success("Homocedasticidade confirmada.")

    with tab3:
        st.subheader("Análise de Sensibilidade")
        # Implementar interface para configurar o 'problem' da SALib
        st.info("A análise global Sobol permite identificar quais variáveis de entrada mais influenciam a variabilidade da resposta estrutural.")
