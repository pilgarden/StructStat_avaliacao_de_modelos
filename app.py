import streamlit as st

from src.data_processing import carregar_dados
from src.report_generator import generate_diagnostics_pdf
from src.ui_diagnostics import render_diagnostics

st.set_page_config(page_title="StructStat", page_icon="🏗️", layout="wide")

# Fallback: oculta navegação multipágina em versões antigas do Streamlit
st.markdown(
    """
    <style>
        [data-testid="stSidebarNav"] { display: none; }
        section[data-testid="stSidebar"] > div:first-child { padding-top: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "df_global" not in st.session_state:
    st.session_state["df_global"] = None
if "filename" not in st.session_state:
    st.session_state["filename"] = None

# --- Barra lateral ---
st.sidebar.title("🏗️ StructStat")

uploaded_file = st.sidebar.file_uploader("Upload de Dados", type=["csv", "xlsx"])

if uploaded_file is not None:
    if (
        st.session_state["df_global"] is None
        or st.session_state["filename"] != uploaded_file.name
    ):
        try:
            st.session_state["df_global"] = carregar_dados(uploaded_file, uploaded_file.name)
            st.session_state["filename"] = uploaded_file.name
            st.sidebar.success("Dados carregados com sucesso!")
        except Exception as e:
            st.sidebar.error(f"Erro ao carregar: {e}")

df = st.session_state["df_global"]

if df is not None:
    st.sidebar.info(f"Dataset ativo: {st.session_state['filename']}")

    if st.sidebar.button("Limpar Dados"):
        st.session_state["df_global"] = None
        st.session_state["filename"] = None
        for key in ("pdf_report_bytes", "pdf_report_name", "sobol_results"):
            st.session_state.pop(key, None)
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Configurações de Variáveis")

    st.sidebar.selectbox("Variável Real (y):", df.columns, key="diag_alvo")
    st.sidebar.selectbox("Variável Prevista (ŷ):", df.columns, key="diag_prev")
    st.sidebar.multiselect("Variáveis Independentes (X):", df.columns, key="diag_preds")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Relatório PDF")

    alvo = st.session_state.get("diag_alvo")
    previsto = st.session_state.get("diag_prev")
    preditores = st.session_state.get("diag_preds") or []
    vars_ok = bool(preditores) and alvo and previsto and alvo != previsto

    if st.sidebar.button(
        "Gerar Relatório PDF Completo",
        type="primary",
        key="btn_pdf_report",
        disabled=not vars_ok,
    ):
        with st.spinner("A compilar relatório PDF..."):
            try:
                filename = st.session_state.get("filename") or "Dataset"
                pdf_bytes = generate_diagnostics_pdf(
                    df,
                    alvo,
                    previsto,
                    preditores,
                    dataset_name=filename,
                    sobol_df=st.session_state.get("sobol_results"),
                )
                st.session_state["pdf_report_bytes"] = pdf_bytes
                st.session_state["pdf_report_name"] = (
                    f"structstat_{filename.rsplit('.', 1)[0]}.pdf"
                )
                st.sidebar.success("Relatório gerado.")
            except Exception as e:
                st.sidebar.error(f"Erro ao gerar PDF: {e}")

    if st.session_state.get("pdf_report_bytes"):
        st.sidebar.download_button(
            label="Baixar Relatório (PDF)",
            data=st.session_state["pdf_report_bytes"],
            file_name=st.session_state.get("pdf_report_name", "structstat_relatorio.pdf"),
            mime="application/pdf",
            key="dl_pdf_report",
        )
    elif not vars_ok:
        st.sidebar.caption("Selecione as variáveis acima para gerar o PDF.")
st.markdown(
"""
<div style='text-align: center; color: gray; font-size: 0.85em;'>
    <b>StructStat: módulo de avaliação de modelos</b><br>
    Desenvolvido por: Pedro Jardim<br>
    <i>v1.0 - Maio/2026</i>
</div>
""", 
unsafe_allow_html=True
)
# --- Área principal ---
if df is not None:
    render_diagnostics()
else:
    st.title("StructStat")
    st.markdown(
        "Carregue um ficheiro **CSV** ou **Excel** na barra lateral para iniciar "
        "o diagnóstico estatístico do modelo."
    )
