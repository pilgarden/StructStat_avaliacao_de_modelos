import streamlit as st
from weasyprint import HTML
import matplotlib.pyplot as plt
import base64
import io

def generate_pdf_report(df, alvo, previsto, preditores):
    # 1. Configurar figuras estáticas (Matplotlib) para o PDF
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df[alvo], df[previsto], alpha=0.6)
    ax.plot([df[alvo].min(), df[alvo].max()], [df[alvo].min(), df[alvo].max()], 'r--')
    ax.set_title("Linearidade (Ref vs Prev)")
    
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png')
    img_base64 = base64.b64encode(img_buf.getvalue()).decode('utf-8')
    plt.close(fig)

    # 2. Estrutura HTML do Relatório
    html_content = f"""
    <html>
    <style>
        @page {{ size: A4; margin: 20mm; }}
        body {{ font-family: 'Times New Roman', serif; line-height: 1.6; color: #333; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #2c3e50; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .metric-box {{ background: #f8f9fa; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }}
    </style>
    <body>
        <h1>Relatório de Diagnóstico StructStat</h1>
        <p>Data de Análise: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}</p>
        
        <h2>1. Visão Geral</h2>
        <p>Análise realizada entre <strong>{alvo}</strong> e <strong>{previsto}</strong>.</p>
        
        <h2>2. Aderência Visual</h2>
        <img src="data:image/png;base64,{img_base64}" width="100%">
        
        <h2>3. Tabela de Métricas (VIF)</h2>
        <p>Aqui seriam inseridos os dados das tabelas processadas...</p>
    </body>
    </html>
    """
    
    # 3. Conversão para PDF
    pdf_file = HTML(string=html_content).write_pdf()
    return pdf_file
