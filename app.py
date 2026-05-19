"""
StructStat: Plataforma de Análise (app.py)
Unifica a Análise Exploratória e a Avaliação de Modelos Preditivos.
"""

import streamlit as st
import pandas as pd
import io

from src.config import GRANDEZAS_UNIDADES, calcular_fator_conversao, SIDEBAR_FOOTER_HTML, SCIENTIFIC_PALETTES
from src.data_processing import carregar_dados, aplicar_filtro_dinamico, ler_e_limpar_dados
from src.metrics import (
    calcular_metricas, calcular_estatisticas_descritivas, testar_normalidade, 
    detetar_outliers, testar_correlacao, testar_homocedasticidade, comparar_medias
)
from src.visualization import (
    plotar_dispersao_referencia_previsto, plotar_bland_altman, plotar_distribuicao_erros, plotar_qq_residuos,
    plotar_histograma, plotar_qq, plotar_matriz_calor, plotar_dispersao
)

st.set_page_config(page_title="StructStat Unificado", page_icon="🏗️", layout="wide")

def aplicar_menu_edicao_grafico(prefixo_key: str, mostrar_titulos: bool = True) -> dict:
    """Gera o menu lateral para exportação com qualidade de artigo Q1."""
    with st.expander("⚙️ Ajustes de Publicação (Qualidade de Artigo)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            fundo_branco = st.checkbox("Fundo Branco Puro", value=True, key=f"{prefixo_key}_bg")
            font_family = st.selectbox("Família da Fonte:", ["Arial", "Times New Roman", "Courier New", "Verdana"], key=f"{prefixo_key}_font")
            font_size = st.number_input("Tamanho da Fonte:", min_value=8, max_value=32, value=14, key=f"{prefixo_key}_sz")
        with col2:
            width_mm = st.number_input("Largura (mm):", value=150, key=f"{prefixo_key}_w")
            height_mm = st.number_input("Altura (mm):", value=100, key=f"{prefixo_key}_h")
            
        title_x, title_y = None, None
        if mostrar_titulos:
            st.markdown("**(Dica: Utilize LaTeX entre cifrões `$E_0$` para os títulos)**")
            title_x = st.text_input("Título Eixo X:", key=f"{prefixo_key}_tx")
            title_y = st.text_input("Título Eixo Y:", key=f"{prefixo_key}_ty")
            
        return {'fundo_branco': fundo_branco, 'font_family': font_family, 'font_size': font_size,
                'width_mm': width_mm, 'height_mm': height_mm, 'title_x': title_x, 'title_y': title_y}

def gerar_opcoes_plotly(w_mm, h_mm, nome):
    from src.config import MM_TO_PX
    return {'displayModeBar': True, 'toImageButtonOptions': {'format': 'svg', 'filename': nome, 'height': int(h_mm * MM_TO_PX), 'width': int(w_mm * MM_TO_PX), 'scale': 1}}

# ==============================================================================
# FLUXO 1: AVALIAÇÃO DE MODELOS PREDITIVOS (Script Original Integrado)
# ==============================================================================
def interface_avaliacao_modelos():
    st.title("🎯 Avaliação de Modelos Preditivos")
    st.markdown("Dashboard para validação de métricas de erro ($R^2$, RMSE, Bias) e diagnósticos.")
    
    with st.sidebar:
        st.header("⚙️ Configurações Físicas")
        grandeza = st.selectbox("Grandeza Física:", options=list(GRANDEZAS_UNIDADES.keys()))
        opcoes = list(GRANDEZAS_UNIDADES[grandeza].keys())
        u_in = st.selectbox("Unidade Original:", options=opcoes, index=0)
        u_out = st.selectbox("Unidade Desejada:", options=opcoes, index=1 if len(opcoes) > 1 else 0)
        fator_conv = calcular_fator_conversao(grandeza, u_in, u_out)
        sigla = u_out.split(' ')[0]
        
        st.markdown("---")
        with st.expander("📖 Guia Científico"):
            st.markdown("""
            * **Overfitting ($R^2 - R^2_{ajust}$):** Penalização por complexidade.
            * **Est. F e Valor-p:** Rejeita que o modelo é inútil ($p < 0.05$).
            * **Bland-Altman:** Vieses proporcionais.
            * **Q-Q Plot:** Normalidade dos resíduos (Ang & Tang, 2007).
            """)

    arquivos = st.file_uploader("Arraste ficheiros (CSV/XLSX). 1ª Col: Referência | 2ª Col: Previsão.", type=["csv", "xlsx"], accept_multiple_files=True)
    
    if arquivos:
        resultados, dfs = [], {}
        with st.spinner("A processar métricas..."):
            for arquivo in arquivos:
                try:
                    df_limpo = ler_e_limpar_dados(arquivo, arquivo.name, fator_conv)
                    res = calcular_metricas(df_limpo['Referencia'], df_limpo['Previsto'], arquivo.name)
                    resultados.append(res); dfs[arquivo.name] = df_limpo
                except Exception as e: st.error(f"Erro em '{arquivo.name}': {e}")
        
        if resultados:
            df_res = pd.DataFrame(resultados)
            st.dataframe(df_res.style.format({'R² (%)': "{:.2f}", 'R² Ajust. (%)': "{:.2f}", 'Overfitting (%)': "{:.3f}", 'Est. F': "{:.2f}", 'Valor-p': "{:.2e}", 'Pearson (r)': "{:.3f}", 'RMSE': "{:.3f}", 'MAE': "{:.3f}", 'Max Erro': "{:.3f}", 'Bias': "{:.3f}", 'MAPE (%)': "{:.2f}", 'CV (%)': "{:.2f}"}).background_gradient(subset=['R² (%)', 'CV (%)'], cmap='viridis'), use_container_width=True)
            
            c1, c2 = st.columns(2)
            with c1:
                b_csv = io.BytesIO(); df_res.to_csv(b_csv, index=False, sep=';', decimal=','); st.download_button("⬇️ CSV", data=b_csv.getvalue(), file_name="Resultados.csv", mime="text/csv")
            with c2:
                b_xls = io.BytesIO()
                with pd.ExcelWriter(b_xls, engine='openpyxl') as w: df_res.to_excel(w, index=False)
                st.download_button("⬇️ Excel", data=b_xls.getvalue(), file_name="Resultados.xlsx")

            st.markdown("### 📈 Diagnósticos Gráficos")
            # Configuração Global de Gráficos para todos os modelos nesta aba
            kwargs_globais = aplicar_menu_edicao_grafico("model_eval", mostrar_titulos=False)
            cfg = gerar_opcoes_plotly(kwargs_globais['width_mm'], kwargs_globais['height_mm'], "modelo")
            
            abas = st.tabs(list(dfs.keys()))
            for i, (nome, df_g) in enumerate(dfs.items()):
                with abas[i]:
                    col1, col2 = st.columns(2)
                    with col1: st.plotly_chart(plotar_dispersao_referencia_previsto(df_g, sigla, kwargs_globais), use_container_width=True, config=cfg)
                    with col2: st.plotly_chart(plotar_bland_altman(df_g, sigla, kwargs_globais), use_container_width=True, config=cfg)
                    col3, col4 = st.columns(2)
                    with col3: st.plotly_chart(plotar_distribuicao_erros(df_g, sigla, kwargs_globais), use_container_width=True, config=cfg)
                    with col4: st.plotly_chart(plotar_qq_residuos(df_g, kwargs_globais), use_container_width=True, config=cfg)

# ==============================================================================
# FLUXO 2: ANÁLISE EXPLORATÓRIA (Módulo Geral)
# ==============================================================================
def interface_analise_exploratoria():
    st.title("🔍 Análise Exploratória Geral")
    ficheiro = st.file_uploader("Carregue o dataset geral (qualquer nº de colunas)", type=["csv", "xlsx"])
    if not ficheiro: return
    
    if 'df_bruto' not in st.session_state or ficheiro.name != st.session_state.get('n_fich'):
        df = carregar_dados(ficheiro, ficheiro.name)
        st.session_state.update({'df_bruto': df, 'df_filt': df.copy(), 'n_fich': ficheiro.name})
            
    df_at = st.session_state['df_filt']
    colunas_num = df_at.select_dtypes(include=['number']).columns.tolist()
    
    t_filt, t_stat, t_norm, t_corr, t_med = st.tabs(["Filtros", "Estatísticas", "Normalidade/Outliers", "Correlação", "Grupos/Médias"])

    with t_filt:
        c1, c2, c3 = st.columns([1, 1, 2])
        col_alvo = c1.selectbox("Coluna:", df_at.columns)
        regra = c2.selectbox("Regra:", ["Valores Exatos", "<=", ">="] if pd.api.types.is_numeric_dtype(st.session_state['df_bruto'][col_alvo]) else ["Valores Exatos", "Começa com (Prefixo)", "Contém (Texto)"])
        
        v_f = c3.multiselect("Valores:", st.session_state['df_bruto'][col_alvo].dropna().unique()) if regra == "Valores Exatos" else c3.text_input("Padrão:") if "Texto" in regra or "Prefixo" in regra else c3.number_input("Limiar:")
        
        b1, b2 = st.columns(2)
        if b1.button("Aplicar"):
            st.session_state['df_filt'] = aplicar_filtro_dinamico(df_at, col_alvo, regra, v_f); st.rerun()
        if b2.button("Limpar"):
            st.session_state['df_filt'] = st.session_state['df_bruto'].copy(); st.rerun()
        st.dataframe(df_at, use_container_width=True)

    with t_stat:
        if colunas_num:
            st.dataframe(calcular_estatisticas_descritivas(df_at).style.format("{:.3f}"), use_container_width=True)
            c_h1, c_h2 = st.columns([1, 2.5])
            v_hist = c_h1.selectbox("Variável:", colunas_num, key="h_v")
            m_norm = c_h1.checkbox("Curva Normal", True)
            kw_h = aplicar_menu_edicao_grafico("h")
            c_h2.plotly_chart(plotar_histograma(df_at[v_hist], m_norm, kw_h), config=gerar_opcoes_plotly(kw_h['width_mm'], kw_h['height_mm'], "hist"))

    with t_norm:
        if colunas_num:
            c_n1, c_n2 = st.columns([1, 2.5])
            v_norm = c_n1.selectbox("Variável:", colunas_num, key="n_v")
            kw_q = aplicar_menu_edicao_grafico("q")
            c_n2.dataframe(pd.DataFrame(testar_normalidade(df_at[v_norm])).T.style.format(precision=4), use_container_width=True)
            outl = detetar_outliers(df_at[v_norm])
            if outl['Grubbs (Indices)'] or outl['Chauvenet (Indices)']: c_n2.error(f"Outliers Grubbs: {len(outl['Grubbs (Indices)'])} | Chauvenet: {len(outl['Chauvenet (Indices)'])}")
            c_n2.plotly_chart(plotar_qq(df_at[v_norm], kw_q), config=gerar_opcoes_plotly(kw_q['width_mm'], kw_q['height_mm'], "qq"))

    with t_corr:
        if colunas_num:
            c_c1, c_c2 = st.columns([1, 2.5])
            cx = c_c1.multiselect("Eixo X:", colunas_num, default=colunas_num)
            cy = c_c1.multiselect("Eixo Y:", colunas_num, default=colunas_num)
            met = c_c1.radio("Método:", ["pearson", "spearman"])
            kw_c = aplicar_menu_edicao_grafico("c")
            kw_c['palette'] = SCIENTIFIC_PALETTES[c_c1.selectbox("Paleta:", list(SCIENTIFIC_PALETTES.keys()))]
            if cx and cy: c_c2.plotly_chart(plotar_matriz_calor(df_at, cx, cy, met, kw_c), config=gerar_opcoes_plotly(kw_c['width_mm'], kw_c['height_mm'], "heat"))

    with t_med:
        grps = list(set(df_at.select_dtypes(exclude=['number']).columns.tolist() + [c for c in colunas_num if df_at[c].nunique() < 15]))
        if grps:
            c_m1, c_m2 = st.columns([1, 2.5])
            fator, metrica = c_m1.selectbox("Fator:", grps), c_m1.selectbox("Métrica:", colunas_num)
            c_m2.dataframe(pd.DataFrame(testar_homocedasticidade(df_at, fator, metrica)).T.style.format(precision=4))
            c_m2.json(comparar_medias(df_at, fator, metrica))

import streamlit as st
import pandas as pd
from src.model_diagnostics import analisar_homocedasticidade, calcular_vif

def app():
    st.title("📊 Diagnóstico Avançado de Modelos")
    
    # Upload do modelo (ex: via pickle ou fit manual)
    st.subheader("Configuração da Regressão")
    # ... código para seleção de variáveis ...

    tab1, tab2, tab3 = st.tabs(["Homocedasticidade", "Multicolinearidade (VIF)", "Sensibilidade (Sobol)"])

    with tab1:
        st.markdown("### Teste de Breusch-Pagan")
        with st.expander("Fundamentação Teórica"):
            st.write("""
            A homocedasticidade é a premissa de que a variância dos resíduos é constante. 
            Se os resíduos exibirem padrões de dispersão (formato de cone), o modelo 
            está enviesado. O teste de Breusch-Pagan verifica esta constância.
            """)
        # Chamada da função e exibição de gráfico
        
    with tab2:
        st.markdown("### Fator de Inflação da Variância (VIF)")
        with st.expander("Interpretando VIF"):
            st.write("Valores de VIF acima de 5-10 sugerem que a variável é linearmente dependente das outras, dificultando a isolação do efeito individual.")
        # Exibição de tabela com df_vif.style.background_gradient(cmap='Reds')

    with tab3:
        st.markdown("### Análise de Sensibilidade Global")
        st.info("A análise de Sobol decompõe a variância da saída entre os inputs, identificando quais variáveis dominam a incerteza do modelo.")

# ==============================================================================
# ORQUESTRADOR PRINCIPAL
# ==============================================================================
def main():
    st.sidebar.title("🏗️ StructStat")
    modo = st.sidebar.radio("Selecione o Módulo de Operação:", ["🎯 Avaliação de Modelos Preditivos", "🔍 Análise Exploratória Geral"])
    st.sidebar.markdown("---")
    
    if "Avaliação" in modo: interface_avaliacao_modelos()
    else: interface_analise_exploratoria()
    
    st.sidebar.markdown(SIDEBAR_FOOTER_HTML, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
