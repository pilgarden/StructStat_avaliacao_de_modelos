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
    st.title("📊 Diagnóstico Avançado de Modelos")
    
    # Verifica se há dados carregados
    if 'df_global' not in st.session_state:
        st.warning("⚠️ Carregue um arquivo de dados na barra lateral para prosseguir.")
        return
        
    df = st.session_state['df_global']
    
    # 1. Configuração de Variáveis (UI)
    st.markdown("### ⚙️ Seleção de Variáveis para Análise")
    col1, col2 = st.columns(2)
    with col1:
        alvo = st.selectbox("Variável Real (Referência $y$):", df.columns)
        previsto = st.selectbox("Variável Prevista (Modelo $\hat{y}$):", df.columns)
    with col2:
        preditores = st.multiselect("Variáveis Independentes (Inputs $X$ para VIF):", df.columns)
        
    if not preditores or alvo == previsto:
        st.info("👈 Selecione as variáveis independentes e garanta que a Referência e o Previsto são diferentes para gerar os diagnósticos.")
        return

    st.markdown("---")
    
    # 2. Execução dos Diagnósticos
    tab1, tab2, tab3 = st.tabs(["Homocedasticidade", "Multicolinearidade (VIF)", "Sensibilidade (Sobol)"])
    
    with tab1:
        st.subheader("Teste de Breusch-Pagan")
        try:
            # Limpa valores nulos que quebram as fórmulas matemáticas
            df_clean = df[[alvo, previsto]].dropna()
            
            # Chama a função que criámos no src/model_diagnostics.py
            results = check_homoscedasticity(df_clean[alvo], df_clean[previsto])
            
            # Ajuste de chaves caso use a versão original ou atualizada
            p_val = results.get('p-valor (LM)', results.get('p-value', 0))
            est_lm = results.get('Estatística BP', results.get('Lagrange multiplier statistic', 0))
            
            c1, c2 = st.columns(2)
            c1.metric("Estatística LM", f"{est_lm:.4f}")
            c2.metric("P-Valor", f"{p_val:.4f}")
            
            if p_val < 0.05:
                st.warning("⚠️ **Heterocedasticidade detectada!** (p < 0.05). A variância dos resíduos não é constante.")
            else:
                st.success("✅ **Homocedasticidade confirmada.** (p >= 0.05). O modelo atende à premissa de variância constante.")
        except Exception as e:
            st.error(f"Erro ao calcular Homocedasticidade: {e}")
            
    with tab2:
        st.subheader("Fator de Inflação da Variância (VIF)")
        try:
            df_X = df[preditores].dropna()
            vif_df = check_multicollinearity(df_X)
            
            st.dataframe(vif_df.style.background_gradient(subset=["VIF"], cmap="Reds"), use_container_width=True)
            
            # Botão de Exportação
            csv = vif_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Exportar Relatório VIF", data=csv, file_name="vif_report.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Erro ao calcular VIF: {e}")
            
    with tab3:
        st.subheader("Análise de Sensibilidade Global (Sobol)")
        st.info("Módulo em desenvolvimento. Aqui configuraremos os limites (bounds) das variáveis para a simulação de Monte Carlo.")

# --- ORQUESTRADOR ---
def main():
    modo = st.sidebar.radio("Módulo:", ["🔍 Exploratória", "🎯 Avaliação", "📊 Diagnóstico"])
    
    if "Diagnóstico" in modo:
        interface_diagnostico_avancado()
    else:
        st.write("Módulo em desenvolvimento.")

if __name__ == "__main__":
    main()
