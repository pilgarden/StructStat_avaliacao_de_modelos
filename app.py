import streamlit as st
import pandas as pd

# Import das lógicas visuais e de dados da sua pasta src/
from src.data_processing import carregar_dados 
from src.ui_diagnostics import render_diagnostics

# Configuração da página
st.set_page_config(page_title="StructStat", page_icon="🏗️", layout="wide")

# --- Inicialização do Estado da Sessão ---
if 'df_global' not in st.session_state:
    st.session_state['df_global'] = None
if 'filename' not in st.session_state:
    st.session_state['filename'] = None

# --- Menu de Navegação ---
st.sidebar.title("Navegação")
modulo_selecionado = st.sidebar.radio("Ir para:", ["🏠 Início", "📊 Diagnóstico Avançado"])

st.sidebar.markdown("---")

# --- Hub de Dados Universal ---
st.sidebar.title("🏗️ StructStat Hub")
uploaded_file = st.sidebar.file_uploader("Upload de Dados", type=["csv", "xlsx"])

# Lógica de Carregamento
if uploaded_file is not None:
    # Só processa se for um novo arquivo ou se não houver dados carregados
    if st.session_state['df_global'] is None or st.session_state['filename'] != uploaded_file.name:
        try:
            st.session_state['df_global'] = carregar_dados(uploaded_file, uploaded_file.name)
            st.session_state['filename'] = uploaded_file.name
            st.sidebar.success("Dados carregados com sucesso!")
        except Exception as e:
            st.sidebar.error(f"Erro ao carregar: {e}")

# Gerenciamento de Exibição e Limpeza
if st.session_state['df_global'] is not None:
    st.sidebar.info(f"Dataset ativo: {st.session_state['filename']}")
    if st.sidebar.button("Limpar Dados"):
        st.session_state['df_global'] = None
        st.session_state['filename'] = None
        st.rerun()

# --- Roteador de Páginas ---
if modulo_selecionado == "🏠 Início":
    st.title("Bem-vindo ao StructStat 🏗️")
    st.markdown("""
    Este é o seu ambiente de engenharia estrutural. 
    1. Utilize o **Hub de Dados** na barra lateral para carregar seu arquivo.
    2. Navegue pelos módulos para realizar as análises estatísticas necessárias.
    """)
    
elif modulo_selecionado == "📊 Diagnóstico Avançado":
    st.title("Diagnóstico Avançado")
    
    # Verifica se os dados existem antes de chamar a função de diagnóstico
    if st.session_state['df_global'] is not None:
        # Chama a função que renderiza a interface (src/ui_diagnostics.py)
        render_diagnostics()
    else:
        st.warning("⚠️ **Atenção:** Nenhum dado carregado. Por favor, faça o upload de um arquivo no Hub de Dados à esquerda.")
