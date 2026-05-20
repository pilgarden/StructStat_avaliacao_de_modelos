import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import r2_score

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
    tab1, tab2, tab3, tab4 = st.tabs([
        "Aderência Visual (Gráficos)", 
        "Homocedasticidade e Resíduos", 
        "Multicolinearidade (VIF)", 
        "Sensibilidade Global (Sobol)"
    ])
    
    with tab1:
        st.subheader("Análise Gráfica de Aderência e Distribuição de Erros")
        try:
            df_clean = df[[alvo, previsto]].dropna().copy()
            y_true = df_clean[alvo]
            y_pred = df_clean[previsto]
            residuos = y_true - y_pred
            
            # Cálculo de métricas e limites teóricos
            r2 = r2_score(y_true, y_pred)
            mean_diff = np.mean(residuos)
            std_diff = np.std(residuos)
            limit_up = mean_diff + 1.96 * std_diff
            limit_down = mean_diff - 1.96 * std_diff
            medias_ba = (y_true + y_pred) / 2
            
            # Layout lado a lado em 3 colunas
            c1, c2, c3 = st.columns(3)
            
            with c1:
                # Gráfico 1: Linearidade (Previsto vs Referência)
                fig_lin = go.Figure()
                fig_lin.add_trace(go.Scatter(x=y_true, y=y_pred, mode='markers', name='Dados', marker=dict(color='#1f77b4', opacity=0.7)))
                min_val = min(y_true.min(), y_pred.min())
                max_val = max(y_true.max(), y_pred.max())
                fig_lin.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val, line=dict(color="red", dash="dash"))
                
                fig_lin.update_layout(
                    title=f"Linearidade: Previsto vs Referência<br><sup>R² = {r2:.3f}</sup>",
                    xaxis_title="Referência ($y$)",
                    yaxis_title="Previsto ($\hat{y}$)",
                    height=400, margin=dict(l=20, r=20, t=60, b=20)
                )
                st.plotly_chart(fig_lin, use_container_width=True)
                
                # REQUISITO EXIGIDO: Botão de Download para Dados de Linearidade
                csv_lin = df_clean[[alvo, previsto]].rename(columns={alvo: 'Referenca_Y', previsto: 'Previsto_Y_Hat'}).to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Baixar Dados de Linearidade", csv_lin, "dados_linearidade.csv", "text/csv", key="dl_lin_tab1")
                
            with c2:
                # Gráfico 2: Bland-Altman
                fig_ba = go.Figure()
                fig_ba.add_trace(go.Scatter(x=medias_ba, y=residuos, mode='markers', name='Diferenças', marker=dict(color='#ff7f0e', opacity=0.7)))
                fig_ba.add_hline(y=mean_diff, line_dash="solid", line_color="blue", annotation_text=f"Média: {mean_diff:.2f}")
                fig_ba.add_hline(y=limit_up, line_dash="dash", line_color="red", annotation_text=f"+1.96 SD: {limit_up:.2f}")
                fig_ba.add_hline(y=limit_down, line_dash="dash", line_color="red", annotation_text=f"-1.96 SD: {limit_down:.2f}", annotation_position="bottom right")
                
                fig_ba.update_layout(
                    title="Bland-Altman: Erro vs Média",
                    xaxis_title="Média (Ref + Prev)/2",
                    yaxis_title="Erro (Ref - Prev)",
                    height=400, margin=dict(l=20, r=20, t=60, b=20)
                )
                st.plotly_chart(fig_ba, use_container_width=True)
                
                # REQUISITO EXIGIDO: Botão de Download para Dados do Bland-Altman
                df_ba_export = pd.DataFrame({'Eixo_X_Medias': medias_ba, 'Eixo_Y_Diferencas': residuos})
                csv_ba = df_ba_export.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Baixar Dados Bland-Altman", csv_ba, "dados_bland_altman.csv", "text/csv", key="dl_ba_tab1")
                
            with c3:
                # Gráfico 3: Distribuição de Resíduos (Histograma)
                fig_hist = px.histogram(
                    x=residuos, nbins=20, 
                    title="Distribuição de Resíduos",
                    labels={'x': 'Erro', 'count': 'Frequência'},
                    color_discrete_sequence=['#2ca02c']
                )
                fig_hist.update_layout(xaxis_title="Erro (Ref - Prev)", yaxis_title="Frequência", height=400, margin=dict(l=20, r=20, t=60, b=20))
                st.plotly_chart(fig_hist, use_container_width=True)
                
                # REQUISITO EXIGIDO: Botão de Download para os Resíduos do Histograma
                df_hist_export = pd.DataFrame({'Residuos_Erros': residuos})
                csv_hist = df_hist_export.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Baixar Dados Histograma", csv_hist, "dados_histograma.csv", "text/csv", key="dl_hist_tab1")
                
            st.markdown("---")
            st.info("📊 **Nota Teórica de Aderência:** O gráfico de linearidade indica se há desvios sistemáticos de proporcionalidade (erros de escala). O Bland-Altman avalia a magnitude do viés constante do modelo físico e se o erro cresce com o tamanho da estrutura. O Histograma ajuda a validar visualmente a premissa de erros gaussianos.")

        except Exception as e:
            st.error(f"Erro ao gerar gráficos de aderência: {e}")

    with tab2:
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
            
    with tab3:
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
            
    with tab4:
        st.subheader("Análise de Sensibilidade Global (Método Sobol)")
        st.markdown("""
        Quantifica como a incerteza de cada variável de entrada impacta a resposta estrutural. 
        * **S1 (Ordem Principal):** Efeito isolado da variável.
        * **ST (Ordem Total):** Efeito da variável somado a todas as interações com outras variáveis.
        """)
        
        try:
            cols_sobol = preditores + [alvo]
            df_sobol = df[cols_sobol].dropna()
            
            st.markdown("#### 1. Faixas de Incerteza (Limites da Simulação)")
            bounds = []
            col_min, col_max = st.columns(2)
            
            for var in preditores:
                min_val = float(df_sobol[var].min())
                max_val = float(df_sobol[var].max())
                with col_min:
                    lb = st.number_input(f"Mínimo para {var}", value=min_val, key=f"min_sob_{var}")
                with col_max:
                    ub = st.number_input(f"Máximo para {var}", value=max_val, key=f"max_sob_{var}")
                bounds.append([lb, ub])
                
            if st.button("🚀 Executar Simulação de Monte Carlo (Sobol)", type="primary"):
                with st.spinner("A treinar Superfície de Resposta e a amostrar Matrizes de Saltelli..."):
                    
                    problem = {'num_vars': len(preditores), 'names': preditores, 'bounds': bounds}
                    X_model = sm.add_constant(df_sobol[preditores])
                    surrogate_model = sm.OLS(df_sobol[alvo], X_model).fit()
                    
                    def surrogate_predict(p):
                        p_in = np.insert(p, 0, 1.0)
                        return surrogate_model.predict(p_in)[0]
                    
                    Si = run_sobol_sensitivity(surrogate_predict, problem, num_samples=1024)
                    
                    if Si is not None:
                        df_si = pd.DataFrame({'Variável': preditores, 'S1 (Isolado)': Si['S1'], 'ST (Total)': Si['ST']})
                        
                        col_graf_sob, col_tbl_sob = st.columns(2)
                        with col_graf_sob:
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
                            var_dominante = df_si.loc[df_si['ST (Total)'].idxmax(), 'Variável']
                            st.info(f"**Variável Dominante:** A incerteza da resposta estrutural é governada por `{var_dominante}`.")

        except Exception as e:
            st.error(f"Erro na configuração do Sobol: {e}")
