import streamlit as st
import pandas as pd
import io

# 1. CONFIGURAÇÃO DE PÁGINA (Deve ser sempre a primeira instrução Streamlit)
st.set_page_config(page_title="StructStat Unificado", page_icon="🏗️", layout="wide")

# 2. IMPORTS DOS MÓDULOS (Após o set_page_config)
from src.config import GRANDEZAS_UNIDADES, calcular_fator_conversao, SIDEBAR_FOOTER_HTML, SCIENTIFIC_PALETTES
from src.data_processing import carregar_dados
from src.model_diagnostics import check_homoscedasticity, check_multicollinearity

# --- HUB DE DADOS (Sidebar) ---
st.sidebar.title("🏗️ StructStat Hub")
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader("Upload de Dados (Excel/CSV)", type=["csv", "xlsx"])

if uploaded_file is not None:
    if 'df_global' not in st.session_state:
        try:
            # Carrega o dado e armazena globalmente
            st.session_state['df_global'] = carregar_dados(uploaded_file, uploaded_file.name)
            st.sidebar.success("Dados carregados!")
        except Exception as e:
            st.sidebar.error(f"Erro: {e}")

if 'df_global' in st.session_state:
    st.sidebar.info(f"Dataset: {uploaded_file.name}")
    if st.sidebar.button("Limpar Dados"):
        del st.session_state['df_global']
        st.rerun()

# --- FUNÇÕES DE INTERFACE ---
def aplicar_menu_edicao_grafico(prefixo_key: str):
    with st.expander("⚙️ Ajustes de Publicação"):
        return {'w': st.number_input("Largura (mm):", value=150, key=f"{prefixo_key}_w")}

# --- MÓDULOS DE FLUXO ---
def interface_diagnostico_avancado():
    st.title("📊 Diagnóstico Avançado")
    
    # Acesso global aos dados
    if 'df_global' not in st.session_state:
        st.warning("Carregue um arquivo na barra lateral para prosseguir.")
        return
        
    df = st.session_state['df_global']
    tab1, tab2 = st.tabs(["Homocedasticidade", "VIF"])
    
    with tab1:
        st.write("Teste de Breusch-Pagan")
        # Exemplo de uso: check_homoscedasticity(y, y_pred)
        
    with tab2:
        st.write("Análise VIF")
        # Exemplo de uso: check_multicollinearity(df)

# --- ORQUESTRADOR ---
def main():
    modo = st.sidebar.radio("Módulo:", ["🔍 Exploratória", "🎯 Avaliação", "📊 Diagnóstico"])
    
    if "Diagnóstico" in modo:
        interface_diagnostico_avancado()
    else:
        st.write("Módulo em desenvolvimento.")

if __name__ == "__main__":
    main()
