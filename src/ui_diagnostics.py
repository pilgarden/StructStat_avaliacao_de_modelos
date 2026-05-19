import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from scipy import stats
from src.model_diagnostics import check_multicollinearity, check_homoscedasticity, run_sobol_sensitivity

def render_diagnostics():
    st.title("📊 Diagnóstico Avançado de Modelos")
    
    if 'df_global' not in st.session_state:
        st.warning("⚠️ Por favor, carregue um arquivo de dados na barra lateral (Hub) para ativar este módulo.")
        return
        
    df = st.session_state['df_global']
    
    # 1. Configuração de Variáveis (UI)
    st.markdown("### ⚙️ Seleção de Variáveis para Análise")
    col1, col2 = st.columns(2)
    with col1:
        alvo = st.selectbox("Variável Real (Referência $y$):", df.columns, key="diag_alvo")
        previsto = st.selectbox("Variável Prevista (Modelo $\hat{y}$):", df.columns, key="diag_prev")
    with col2:
        preditores = st.multiselect("Variáveis Independentes (Inputs $X$):", df.columns, key="diag_preds")
        
    if not preditores or alvo == previsto:
        st.info("👈 Selecione as variáveis independentes físicas e garanta que os eixos de Referência e Previsão são distintos.")
        return

    st.markdown("---")
    
    # 2. Execução dos Separadores de Diagnóstico
    tab1, tab2, tab3 = st.tabs(["Homocedasticidade e Resíduos", "Multicolinearidade (VIF)", "Sensibilidade Global (Sobol)"])
    
    with tab1:
        st.subheader("Análise Avançada e Validação de Resíduos")
        try:
            df_clean = df[[alvo, previsto]].dropna().copy()
            df_clean['Resíduos'] = df_clean[alvo] - df_clean[previsto]
            
            col_grafico, col_texto = st.columns(2)
            with col_grafico:
                fig_res = px.scatter(
                    df_clean, x=previsto, y='Resíduos',
                    labels={previsto: 'Valores Previstos ($\hat{y}$)', 'Resíduos': 'Resíduos ($y - \hat{y}$)'},
                    title="Dispersão de Resíduos vs. Valores Previstos"
                )
                fig_res.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig_res, use_container_width=True)
                
                df_graph_data = df_clean[[previsto, 'Resíduos']].rename(columns={previsto: 'Valores_Previstos'})
                csv_graph = df_graph_data.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Baixar Dados do Gráfico", data=csv_graph, file_name="dados_residuos.csv", mime="text/csv", key="dl_graph_res")
                
            with col_texto:
                results_bp = check_homoscedasticity(df_clean[alvo], df_clean[previsto])
                p_val_bp = results_bp.get('p-valor (LM)', results_bp.get('p-value', 0))
                stat_sw, p_val_sw = stats.shapiro(df_clean['Resíduos'])
                
                st.markdown("#### 🔬 Métricas e Testes Formais")
                c1, c2 = st.columns(2)
                c1.metric("P-Valor (Breusch-Pagan)", f"{p_val_bp:.4f}")
                c2.metric("P-Valor (Shapiro-Wilk)", f"{p_val_sw:.4f}")
                
                st.markdown("#### 📝 Parecer Técnico de Avaliação")
                if p_val_bp < 0.05:
                    st.error("❌ **Heterocedasticidade Detetada:** A variância dos erros não é constante.")
                else:
                    st.success("✅ **Homocedasticidade Confirmada:** Os resíduos distribuem-se de forma homogénea.")
                
                if p_val_sw < 0.05:
                    st.warning("⚠️ **Resíduos Não-Normais:** O teste detetou desvios em relação à curva gaussiana ideal.")
                else:
                    st.success("✅ **Normalidade Confirmada:** Os erros seguem uma distribuição gaussiana ideal.")
        except Exception as e:
            st.error(f"Erro na análise de resíduos: {e}")
            
    with tab2:
        st.subheader("Análise de Multicolinearidade (VIF) e Tolerância")
        try:
            df_X = df[preditores].dropna()
            vif_df = check_multicollinearity(df_X)
            vif_df['Tolerância'] = 1 / vif_df['VIF']
            
            col_grafico_vif, col_tabela_vif = st.columns(2)
            with col_grafico_vif:
                fig_vif = px.bar(
                    vif_df, x='Feature', y='VIF', color='VIF', 
                    color_continuous_scale='Reds', title='Inflação da Variância por Variável'
                )
                fig_vif.add_hline(y=10, line_dash="dash", line_color="darkred", annotation_text="Limite Crítico (VIF=10)")
                st.plotly_chart(fig_vif, use_container_width=True)
                
                csv_vif = vif_df.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Baixar Dados do Relatório VIF", data=csv_vif, file_name="relatorio_vif.csv", mime="text/csv", key="dl_vif")
                
            with col_tabela_vif:
                st.markdown("#### 📋 Matriz de Diagnóstico VIF")
                st.dataframe(vif_df.style.background_gradient(subset=["VIF"], cmap="Reds").format({"VIF": "{:.2f}", "Tolerância": "{:.4f}"}), use_container_width=True)
                
                variaveis_criticas = vif_df[vif_df['VIF'] > 10]['Feature'].tolist()
                st.markdown("#### 📝 Parecer Técnico (VIF)")
                if variaveis_criticas:
                    st.error(f"❌ **Multicolinearidade Crítica:** As variáveis {variaveis_criticas} apresentam VIF > 10. Recomenda-se remoção ou regularização.")
                else:
                    st.success("✅ **Ausência de Multicolinearidade:** Todas as variáveis apresentam VIF sob controlo.")
        except Exception as e:
            st.error(f"Erro ao calcular VIF: {e}")
            
    with tab3:
        st.subheader("Análise de Sensibilidade Global (Método Sobol)")
        st.markdown("""
        Quantifica como a incerteza de cada variável de entrada impacta a resposta estrutural. 
        * **S1 (Ordem Principal):** Efeito isolado da variável.
        * **ST (Ordem Total):** Efeito da variável somado a todas as interações com outras variáveis.
        """)
        
        try:
            # Garante que não há NaN nas colunas usadas
            cols_sobol = preditores + [alvo]
            df_sobol = df[cols_sobol].dropna()
            
            st.markdown("#### 1. Faixas de Incerteza (Limites da Simulação)")
            st.caption("Valores pré-preenchidos com o mínimo e máximo observados na amostra física.")
            
            bounds = []
            col_min, col_max = st.columns(2)
            
            for var in preditores:
                min_val = float(df_sobol[var].min())
                max_val = float(df_sobol[var].max())
                with col_min:
                    lb = st.number_input(f"Mínimo (Lower) para {var}", value=min_val, key=f"min_sob_{var}")
                with col_max:
                    ub = st.number_input(f"Máximo (Upper) para {var}", value=max_val, key=f"max_sob_{var}")
                bounds.append([lb, ub])
                
            if st.button("🚀 Executar Simulação de Monte Carlo (Sobol)", type="primary"):
                with st.spinner("A treinar Superfície de Resposta e a amostrar Matrizes de Saltelli..."):
                    
                    # 1. Definição do Problema (SALib)
                    problem = {
                        'num_vars': len(preditores),
                        'names': preditores,
                        'bounds': bounds
                    }
                    
                    # 2. Treino de Surrogate Model (OLS)
                    X_model = sm.add_constant(df_sobol[preditores])
                    surrogate_model = sm.OLS(df_sobol[alvo], X_model).fit()
                    
                    # 3. Função invólucro (Wrapper) para o SALib avaliar o modelo
                    def surrogate_predict(p):
                        # p é um array (ex: [E, I, L]). Inserimos 1.0 no índice 0 para a constante
                        p_in = np.insert(p, 0, 1.0)
                        return surrogate_model.predict(p_in)[0]
                    
                    # 4. Cálculo via Motor Matemático (src/model_diagnostics.py)
                    Si = run_sobol_sensitivity(surrogate_predict, problem, num_samples=1024)
                    
                    if Si is not None:
                        # Processamento de Dados
                        df_si = pd.DataFrame({
                            'Variável': preditores,
                            'S1 (Isolado)': Si['S1'],
                            'ST (Total)': Si['ST']
                        })
                        
                        col_graf_sob, col_tbl_sob = st.columns(2)
                        with col_graf_sob:
                            # Preparar dados para gráfico de barras agrupadas
                            df_si_melt = df_si.melt(id_vars='Variável', value_vars=['S1 (Isolado)', 'ST (Total)'], var_name='Índice', value_name='Valor')
                            fig_sob = px.bar(
                                df_si_melt, x='Variável', y='Valor', color='Índice', barmode='group',
                                title='Decomposição da Variância (Índices de Sobol)',
                                color_discrete_sequence=['#1f77b4', '#ff7f0e']
                            )
                            st.plotly_chart(fig_sob, use_container_width=True)
                            
                            csv_sob = df_si.to_csv(index=False).encode('utf-8')
                            st.download_button("⬇️ Baixar Índices (CSV)", data=csv_sob, file_name="sobol_indices.csv", mime="text/csv", key="dl_sob")
                            
                        with col_tbl_sob:
                            st.markdown("#### 📋 Matriz Numérica de Sensibilidade")
                            st.dataframe(df_si.style.background_gradient(subset=['ST (Total)'], cmap="Blues").format({"S1 (Isolado)": "{:.4f}", "ST (Total)": "{:.4f}"}), use_container_width=True)
                            
                            # Interpretação Dinâmica
                            var_dominante = df_si.loc[df_si['ST (Total)'].idxmax(), 'Variável']
                            st.markdown("#### 📝 Parecer Técnico (Sensibilidade)")
                            st.info(f"**Variável Dominante:** A variável física `{var_dominante}` apresenta o maior índice de Ordem Total (ST). Isto significa que a maior parte da incerteza na sua resposta estrutural é governada por esta variável (incluindo as suas interações). Para otimizar o projeto, o foco de precisão na medição deve estar em `{var_dominante}`.")

        except Exception as e:
            st.error(f"Erro na configuração do Sobol: Verifique se as variáveis selecionadas não contêm valores infinitos ou texto nulo. Erro: {e}")
