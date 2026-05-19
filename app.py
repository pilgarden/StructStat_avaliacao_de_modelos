import streamlit as st
import pandas as pd

# Import das lógicas visuais e de dados da sua pasta src/
from src.data_processing import carregar_dados 
from src.ui_diagnostics import render_diagnostics # <--- A mágica acontece aqui

st.set_page_config(page_title="StructStat", page_icon="🏗️", layout="wide")

# Menu de Navegação Explícito (À prova de falhas)
st.sidebar.title("Navegação")
modulo_selecionado = st.sidebar.radio("Ir para:", ["🏠 Início", "📊 Diagnóstico Avançado"])

st.sidebar.markdown("---")

# Hub de Dados Universal
st.sidebar.title("🏗️ StructStat Hub (Dados)")
uploaded_file = st.sidebar.file_uploader("Upload de Dados", type=["csv", "xlsx"])

if uploaded_file is not None:
    if 'df_global' not in st.session_state:
        try:
            st.session_state['df_global'] = carregar_dados(uploaded_file, uploaded_file.name)
            st.sidebar.success("Dados carregados!")
        except Exception as e:
            st.sidebar.error(f"Erro ao carregar: {e}")

if 'df_global' in st.session_state:
    st.sidebar.info(f"Dataset ativo: {uploaded_file.name}")
    if st.sidebar.button("Limpar Dados"):
        del st.session_state['df_global']
        st.rerun()

# Roteador de Páginas
if modulo_selecionado == "🏠 Início":
    st.title("Bem-vindo ao StructStat 🏗️")
    st.markdown("Carregue o seu ficheiro no **Hub de Dados** à esquerda e depois selecione o módulo desejado no menu de navegação acima.")
    
elif modulo_selecionado == "📊 Diagnóstico Avançado":
    # Chama a função que desenha toda a interface de diagnóstico
    render_diagnostics()
