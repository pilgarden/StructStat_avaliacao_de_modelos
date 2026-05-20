import streamlit as st
import pandas as pd
from src.report_generator import PDFGenerator
import matplotlib.pyplot as plt

# Exemplo de uso no seu dashboard
def generate_pdf_report(df, plot_path):
    pdf = PDFGenerator()
    pdf.add_page()
    
    # Adicionar estatísticas
    pdf.add_table(df.describe().reset_index(), "Estatísticas Descritivas")
    
    # Adicionar Gráfico
    pdf.add_plot(plot_path, "Distribuição dos Dados")
    
    # Retornar o PDF como bytes para o botão de download
    return pdf.output(dest='S') # Retorna bytes

# Na UI
if st.button("Gerar Relatório PDF"):
    # 1. Salvar gráfico temporário
    fig = plt.figure() # Exemplo
    plt.plot([1, 2, 3], [1, 4, 9])
    temp_plot = "temp_plot.png"
    plt.savefig(temp_plot)
    
    # 2. Gerar PDF
    pdf_bytes = generate_pdf_report(df_analisado, temp_plot)
    
    # 3. Botão de Download
    st.download_button(
        label="📥 Baixar Relatório (PDF)",
        data=pdf_bytes,
        file_name="analise_structstat.pdf",
        mime="application/pdf"
    )
