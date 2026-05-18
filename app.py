# [...] Código anterior (imports e setup)

    # 2. Barra Lateral Dinâmica
    with st.sidebar:
        # [...] Configurações de Unidades omitidas para brevidade
        
        st.markdown("---")
        st.subheader("Guia Científico")
        with st.expander("Métricas de Validação"):
            st.markdown("""
            * **Overfitting ($R^2 - R^2_{ajust}$):** Mede a penalização pela complexidade do modelo. Valores altos indicam que o modelo matemático possui excesso de parâmetros para a quantidade de dados experimentais disponíveis.
            * **Estatística F:** Razão entre a variância explicada pelo modelo e a variância residual. Um valor elevado rejeita a hipótese de que o modelo é inútil.
            * **Valor-p:** Probabilidade do resultado da Estatística F ter ocorrido ao acaso. Na engenharia, aceita-se $p < 0.05$ como estatisticamente significativo.
            """)
        with st.expander("Bland-Altman & Q-Q Plot"):
            st.markdown("* **Bland-Altman:** Identifica vieses proporcionais.\n* **Q-Q Plot:** Avalia normalidade dos resíduos (Ang & Tang, 2007).")

# [...] Código intermediário (Área Principal e leitura de dados)

        if resultados_consolidados:
            df_resultados = pd.DataFrame(resultados_consolidados)
            
            # ATUALIZAÇÃO DA TABELA: Inclusão da formatação científica para o Valor-p
            st.dataframe(
                df_resultados.style.format({
                    'R² (%)': "{:.2f}", 
                    'R² Ajust. (%)': "{:.2f}",
                    'Overfitting (%)': "{:.3f}",
                    'Est. F': "{:.2f}",
                    'Valor-p': "{:.2e}",  # Notação científica obrigatória para p-valores
                    'Pearson (r)': "{:.3f}", 
                    'RMSE': "{:.3f}", 
                    'MAE': "{:.3f}", 
                    'Max Erro': "{:.3f}", 
                    'Bias': "{:.3f}",
                    'MAPE (%)': "{:.2f}", 
                    'CV (%)': "{:.2f}"
                }).background_gradient(subset=['R² (%)', 'CV (%)'], cmap='viridis'),
                use_container_width=True
            )
            
            # [...] Código de exportação (botões CSV/Excel) e gráficos
