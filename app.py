import streamlit as st
import pandas as pd
import io

# Importações dos módulos internos (Garanta que a estrutura de pastas src/ está correta)
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
from src.model_diagnostics import analisar_homocedasticidade, calcular_vif
import streamlit as st
import pandas as pd
from src.data_processing import carregar_dados # Certifique-se de que sua função de carga está aqui

# Configuração da Barra Lateral (O Hub de Dados)
st.sidebar.title("🏗️ StructStat Hub")
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader("Upload de Dados (Excel/CSV)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Carregamento e cache dos dados
    if 'df_global' not in st.session_state:
        try:
            df = carregar_dados(uploaded_file) # Sua função customizada
            st.session_state['df_global'] = df
            st.sidebar.success("Dados carregados com sucesso!")
        except Exception as e:
            st.sidebar.error(f"Erro ao carregar: {e}")

# Verificação de persistência
if 'df_global' in st.session_state:
    st.sidebar.info(f"Dataset ativo: {uploaded_file.name}")
    if st.sidebar.button("Limpar Dados"):
        del st.session_state['df_global']
        st.rerun()
# Configuração da Página
st.set_page_config(page_title="StructStat Unificado", page_icon="🏗️", layout="wide")

# --- FUNÇÕES DE INTERFACE ---

def aplicar_menu_edicao_grafico(prefixo_key: str, mostrar_titulos: bool = True) -> dict:
    with st.expander("⚙️ Ajustes de Publicação (Qualidade de Artigo)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            fundo_branco = st.checkbox("Fundo Branco Puro", value=True, key=f"{prefixo_key}_bg")
            font_family = st.selectbox("Família da Fonte:", ["Arial", "Times New Roman", "Courier New"], key=f"{prefixo_key}_font")
            font_size = st.number_input("Tamanho da Fonte:", min_value=8, max_value=32, value=14, key=f"{prefixo_key}_sz")
        with col2:
            width_mm = st.number_input("Largura (mm):", value=150, key=f"{prefixo_key}_w")
            height_mm = st.number_input("Altura (mm):", value=100, key=f"{prefixo_key}_h")
        
        title_x, title_y = None, None
        if mostrar_titulos:
            title_x = st.text_input("Título Eixo X:", key=f"{prefixo_key}_tx")
            title_y = st.text_input("Título Eixo Y:", key=f"{prefixo_key}_ty")
        return {'fundo_branco': fundo_branco, 'font_family': font_family, 'font_size': font_size, 'width_mm': width_mm, 'height_mm': height_mm, 'title_x': title_x, 'title_y': title_y}

def gerar_opcoes_plotly(w_mm, h_mm, nome):
    from src.config import MM_TO_PX
    return {'displayModeBar': True, 'toImageButtonOptions': {'format': 'svg', 'filename': nome, 'height': int(h_mm * MM_TO_PX), 'width': int(w_mm * MM_TO_PX), 'scale': 1}}

# --- MÓDULOS DE FLUXO ---

def interface_avaliacao_modelos():
    st.title("🎯 Avaliação de Modelos Preditivos")
    # [Lógica da avaliação mantida conforme seu script original]
    pass

def interface_analise_exploratoria():
    st.title("🔍 Análise Exploratória Geral")
    # Inicialização do State
    if 'df_filt' not in st.session_state:
        st.session_state['df_filt'] = pd.DataFrame()
    
    ficheiro = st.file_uploader("Carregue o dataset", type=["csv", "xlsx"])
    if ficheiro:
        if 'n_fich' not in st.session_state or ficheiro.name != st.session_state.get('n_fich'):
            df = carregar_dados(ficheiro, ficheiro.name)
            st.session_state.update({'df_bruto': df, 'df_filt': df.copy(), 'n_fich': ficheiro.name})
        
        # [Restante da lógica de abas e filtros usando st.session_state['df_filt']]

def interface_diagnostico_avancado():
    st.title("📊 Diagnóstico Avançado de Modelos")
    # Utilize st.session_state['df_filt'] aqui para cálculos
    tab1, tab2, tab3 = st.tabs(["Homocedasticidade", "VIF", "Sensibilidade"])
    with tab1:
        st.markdown("### Teste de Breusch-Pagan")
        # Implementar chamadas do src.model_diagnostics
    
# --- ORQUESTRADOR ---

def main():
    st.sidebar.title("🏗️ StructStat")
    modo = st.sidebar.radio("Módulo:", ["🎯 Avaliação", "🔍 Exploratória", "📊 Diagnóstico"])
    st.sidebar.markdown("---")
    
    if "Avaliação" in modo: interface_avaliacao_modelos()
    elif "Diagnóstico" in modo: interface_diagnostico_avancado()
    else: interface_analise_exploratoria()
    
    st.sidebar.markdown(SIDEBAR_FOOTER_HTML, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
