import streamlit as st
import pandas as pd

# Import da sua função de carregar dados
from src.data_processing import carregar_dados 

# 1. Configuração Global (Sempre a primeira linha)
st.set_page_config(page_title="StructStat", page_icon="🏗️", layout="wide")

# 2. Sidebar: Hub de Dados Universal
st.sidebar.title("🏗️ StructStat Hub")
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader("Upload de Dados (Excel/CSV)", type=["csv", "xlsx"])

if uploaded_file is not None:
    if 'df_global' not in st.session_state:
        try:
            st.session_state['df_global'] = carregar_dados(uploaded_file, uploaded_file.name)
            st.sidebar.success("Dados carregados com sucesso!")
        except Exception as e:
            st.sidebar.error(f"Erro ao carregar ficheiro: {e}")

if 'df_global' in st.session_state:
    st.sidebar.info(f"Dataset ativo: {uploaded_file.name}")
    if st.sidebar.button("Limpar Dados"):
        del st.session_state['df_global']
        st.rerun()

# 3. Página Inicial
st.title("Bem-vindo ao StructStat 🏗️")
st.markdown("""
Esta é a plataforma central para a sua Tese de Doutoramento.
* Os dados carregados no painel lateral estarão disponíveis em todos os módulos.
* **Por favor, utilize o menu lateral acima (criado automaticamente) para navegar entre os módulos de Análise Exploratória, Avaliação e Diagnóstico.**
""")
