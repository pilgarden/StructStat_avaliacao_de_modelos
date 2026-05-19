import streamlit as st
import pandas as pd
import plotly.express as px
from scipy import stats
from src.model_diagnostics import check_multicollinearity, check_homoscedasticity

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
        preditores = st.multiselect("Variáveis Independentes (Inputs $X$ para VIF):", df.columns, key="diag_preds")
        
    if not preditores or alvo == previsto:
        st.info("👈 Selecione as variáveis independentes físicas e garanta que os eixos de Referência e Previsão são distintos.")
        return

    st.markdown("---")
    
    # 2. Execução dos Separadores de Diagnóstico
    tab1, tab2, tab3 = st.tabs(["Homocedasticidade e Resíduos", "Multicolinearidade (VIF)", "Sensibilidade (Sobol)"])
    
    with tab1:
        st.subheader("Análise Avançada e Validação de Resíduos")
        try:
            df_clean = df[[alvo, previsto]].dropna().copy()
            df_clean['Resíduos'] = df_clean[alvo] - df_clean[previsto]
            
            # DIVISÃO EM METADE DA TELA: Gráfico à esquerda, Texto à direita
            col_grafico, col_texto = st.columns(2)
            
            with col_grafico:
                fig_res = px.scatter(
                    df_clean, x=previsto, y='Resíduos',
                    labels={previsto: 'Valores Previstos ($\hat{y}$)', 'Resíduos': 'Resíduos ($y - \hat{y}$)'},
                    title="Dispersão de Resíduos vs. Valores Previstos"
                )
                fig_res.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig_res, use_container_width=True)
                
                # REQUISITO: Exportação dos dados exatos do gráfico para reprodutibilidade
                df_graph_data = df_clean[[previsto, 'Resíduos']].rename(columns={previsto: 'Valores_Previstos'})
                csv_graph = df_graph_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Baixar Dados do Gráfico (CSV para Origin/Excel)",
                    data=csv_graph,
                    file_name="dados_grafico_residuos.csv",
                    mime="text/csv",
                    key="dl_graph_res"
                )
                
            with col_texto:
                # Execução dos cálculos formais do motor matemático
                results_bp = check_homoscedasticity(df_clean[alvo], df_clean[previsto])
                p_val_bp = results_bp.get('p-valor (LM)', results_bp.get('p-value', 0))
                stat_sw, p_val_sw = stats.shapiro(df_clean['Resíduos'])
                
                st.markdown("#### 🔬 Métricas e Testes Formais")
                c1, c2 = st.columns(2)
                c1.metric("P-Valor (Breusch-Pagan)", f"{p_val_bp:.4f}")
                c2.metric("P-Valor (Shapiro-Wilk)", f"{p_val_sw:.4f}")
                
                st.markdown("#### 📝 Parecer Técnico de Avaliação")
                
                # Diagnóstico Teórico: Homocedasticidade
                if p_val_bp < 0.05:
                    st.error("❌ **Heterocedasticidade Detetada:** O teste de Breusch-Pagan rejeitou a hipótese nula ($p < 0.05$). A variância dos erros não é constante, apresentando uma dispersão irregular (geralmente em formato de funil ou cone).")
                    st.caption("**Recomendação para a Tese:** Os intervalos de confiança calculados por Mínimos Quadrados Ordinários (OLS) podem estar subestimados. Sugere-se reportar erros-padrão robustos (ex: **HC3**) ou aplicar uma transformação matemática (Mínimos Quadrados Ponderados - WLS) para blindar a modelagem estrutural contra críticas da banca.")
                else:
                    st.success("✅ **Homocedasticidade Confirmada:** O teste de Breusch-Pagan não rejeitou a hipótese nula ($p \ge 0.05$). Os resíduos distribuem-se de forma homogénea, garantindo que o modelo possui estabilidade de erro em toda a gama de carregamento/respostas.")
                
                # Diagnóstico Teórico: Normalidade dos Erros
                if p_val_sw < 0.05:
                    st.warning("⚠️ **Resíduos Não-Normais:** O teste de Shapiro-Wilk detetou desvios em relação à curva gaussiana ideal ($p < 0.05$).")
                    st.caption("Se a sua amostra for robusta ($N > 30$), o Teorema do Limite Central mitiga o impacto deste desvio sobre as estimativas dos coeficientes. Contudo, evite inferências baseadas em distribuições exatas para subconjuntos muito restritos de dados físicos.")
                else:
                    st.success("✅ **Normalidade Confirmada:** Os erros seguem uma distribuição gaussiana ideal, validando plenamente os testes de hipóteses estatísticas ($t$ e $F$) do modelo.")
                    
        except Exception as e:
            st.error(f"Erro na análise de resíduos: {e}")
            
    with tab2:
        st.subheader("Análise de Multicolinearidade (VIF) e Tolerância")
        try:
            df_X = df[preditores].dropna()
            vif_df = check_multicollinearity(df_X)
            vif_df['Tolerância'] = 1 / vif_df['VIF']
            
            # DIVISÃO EM METADE DA TELA: Gráfico à esquerda, Tabela à direita
            col_grafico_vif, col_tabela_vif = st.columns(2)
            
            with col_grafico_vif:
                fig_vif = px.bar(
                    vif_df, x='Feature', y='VIF', color='VIF', 
                    color_continuous_scale='Reds', title='Nível de Inflação da Variância por Variável'
                )
                fig_vif.add_hline(y=10, line_dash="dash", line_color="darkred", annotation_text="Limite Crítico (VIF=10)")
                st.plotly_chart(fig_vif, use_container_width=True)
                
                # Exportação dos dados de multicolinearidade
                csv_vif = vif_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Baixar Dados do Relatório VIF (CSV)",
                    data=csv_vif,
                    file_name="relatorio_vif.csv",
                    mime="text/csv",
                    key="dl_vif"
                )
                
            with col_tabela_vif:
                st.markdown("#### 📋 Matriz de Diagnóstico VIF")
                st.dataframe(vif_df.style.background_gradient(subset=["VIF"], cmap="Reds").format({"VIF": "{:.2f}", "Tolerância": "{:.4f}"}), use_container_width=True)
                
                # Filtragem de variáveis problemáticas
                variaveis_criticas = vif_df[vif_df['VIF'] > 10]['Feature'].tolist()
                
                st.markdown("#### 📝 Parecer Técnico (VIF)")
                if variaveis_criticas:
                    st.error(f"❌ **Multicolinearidade Crítica Detetada:** As variáveis físicas {variaveis_criticas} apresentam um VIF acima do limite rigoroso de 10.")
                    st.caption("**Impacto no Doutoramento:** Existe dependência linear mútua severa entre estes parâmetros de entrada. Isto inflaciona a variância das previsões e impede o algoritmo de isolar adequadamente a influência de cada propriedade geométrica ou mecânica de forma independente. Recomenda-se a eliminação do parâmetro redundante ou aplicação de técnicas de regularização (Ridge/Lasso).")
                else:
                    st.success("✅ **Ausência de Multicolinearidade:** Todas as variáveis selecionadas apresentam VIF abaixo de 10 (e tolerância adequada). Os coeficientes estruturais do modelo são estáveis e os efeitos de cada variável independente podem ser isolados e discutidos com total segurança científica.")
                    
        except Exception as e:
            st.error(f"Erro ao calcular VIF: {e}")
            
    with tab3:
        st.subheader("Análise de Sensibilidade Global (Sobol)")
        st.info("Módulo de Sensibilidade Global Sobol estruturado. Pronto para a integração das faixas de incerteza física.")
