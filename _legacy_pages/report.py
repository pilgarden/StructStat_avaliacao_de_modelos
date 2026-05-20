"""
Página Streamlit: exportação de relatório PDF (usa dados do Hub global).
"""

import streamlit as st
from src.report_generator import generate_diagnostics_pdf, generate_pdf_report
import matplotlib.pyplot as plt
import tempfile
import os

st.set_page_config(page_title="Relatório PDF", layout="wide")
st.title("📄 Exportar Relatório PDF")

st.markdown(
    "Gere um relatório PDF com métricas, testes estatísticos, tabelas VIF e gráficos. "
    "Para o **diagnóstico completo de modelos**, use a página principal em "
    "**Diagnóstico Avançado** (recomendado). Esta página gera um resumo exploratório "
    "quando só existem dados carregados."
)

if st.session_state.get("df_global") is None:
    st.warning("Carregue um ficheiro no Hub de Dados (app principal) antes de exportar.")
    st.stop()

df = st.session_state["df_global"]
filename = st.session_state.get("filename") or "dataset.csv"

# Diagnóstico completo se variáveis já foram escolhidas na outra página
alvo = st.session_state.get("diag_alvo")
previsto = st.session_state.get("diag_prev")
preditores = st.session_state.get("diag_preds")
cols = list(df.columns)

if alvo not in cols:
    alvo = st.selectbox("Variável Real (y):", cols, key="report_alvo")
if previsto not in cols:
    previsto = st.selectbox("Variável Prevista:", cols, key="report_prev")
if not preditores:
    preditores = st.multiselect("Preditores (X):", cols, key="report_preds")

if st.button("Gerar Relatório PDF", type="primary"):
    with st.spinner("A gerar PDF..."):
        try:
            if preditores and alvo and previsto and alvo != previsto:
                pdf_bytes = generate_diagnostics_pdf(
                    df, alvo, previsto, preditores, dataset_name=filename,
                    sobol_df=st.session_state.get("sobol_results"),
                )
            else:
                with tempfile.TemporaryDirectory() as tmp:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    num_cols = df.select_dtypes(include="number").columns
                    col = num_cols[0] if len(num_cols) else None
                    if col is not None:
                        ax.hist(df[col].dropna(), bins=20, color="#1f77b4", edgecolor="white")
                        ax.set_title(f"Distribuicao de {col}")
                    plot_path = os.path.join(tmp, "hist.png")
                    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
                    plt.close(fig)
                    pdf_bytes = generate_pdf_report(
                        df.select_dtypes(include="number"), plot_path, dataset_name=filename
                    )

            st.download_button(
                label="📥 Baixar Relatório (PDF)",
                data=pdf_bytes,
                file_name=f"structstat_{filename.rsplit('.', 1)[0]}.pdf",
                mime="application/pdf",
            )
        except Exception as e:
            st.error(f"Erro ao gerar PDF: {e}")
