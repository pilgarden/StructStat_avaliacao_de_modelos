"""
Módulo Principal (app.py)
StructStat: Avaliação de modelos estruturais.
"""

import streamlit as st
import pandas as pd
import io

# Importações dos módulos internos (Backend)
from src.config import GRANDEZAS_UNIDADES, calcular_fator_conversao
from src.data_processing import ler_e_limpar_dados
from src.metrics import calcular_metricas
from src.visualization import (
    plotar_dispersao_referencia_previsto, 
    plotar_distribuicao_erros,
    plotar_bland_altman,
    plotar_qq_residuos
)

def main():
    st.set_page_config(page_title="StructStat", page_icon="🏗️", layout="wide")

    st.title("🏗️ StructStat: Avaliação de Modelos")
    st.markdown("Dashboard interativo para validação estatística de modelos de engenharia de estruturas.")

    # 2. Barra Lateral Dinâmica (Inputs e Configurações de Unidades)
    with st.sidebar:
        st.header("⚙️ Configurações Físicas")
        
        st.subheader("Seleção de Grandezas")
        
        # 2.1 Seleciona a grandeza física primeiro
        grandeza_selecionada = st.selectbox(
            "Selecione a Grandeza Física:",
            options=list(GRANDEZAS_UNIDADES.keys())
        )
        
        # 2.2 Filtra as opções de unidades com base na grandeza escolhida
        opcoes_unidades = list(GRANDEZAS_UNIDADES[grandeza_selecionada].keys())
        
        unidade_entrada = st.selectbox(
            "1. Unidade Original (Nos arquivos):",
            options=opcoes_unidades,
            index=0
        )
        
        unidade_saida = st.selectbox(
            "2. Unidade Desejada (Para os gráficos e tabela):",
            options=opcoes_unidades,
            # Tenta selecionar uma unidade padrão diferente (ex: kN em vez de N), ou pega a última se a lista for curta
            index=1 if len(opcoes_unidades) > 1 else 0 
        )
        
        st.markdown("---")
        st.subheader("Guia Científico")
        with st.expander("Bland-Altman & Q-Q Plot"):
            st.markdown("""
            * **Bland-Altman:** Identifica heterocedasticidade e vieses proporcionais (se o erro cresce com a carga).
            * **Q-Q Plot:** Avalia normalidade dos resíduos. Pontos fora da diagonal vermelha invalidam o pressuposto de distribuição normal nos erros (Ang & Tang, 2007).
            """)

    # 3. Área Principal
    fator_conv = calcular_fator_conversao(grandeza_selecionada, unidade_entrada, unidade_saida)
    sigla_unidade = unidade_saida.split(' ')[0] # Extrai apenas 'kN', 'mm', 'MPa', etc.

    with st.expander("Carregar base de dados"):
        arquivos_upados = st.file_uploader(
            "Arraste os ficheiros de resultados (CSV ou XLSX). 1ª Coluna: Referência | 2ª Coluna: Previsão.",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True
        )
    
        if arquivos_upados:
            resultados_consolidados = []
            dataframes_processados = {} 
        
        st.markdown("### 📊 Consolidação Estatística")
        
        with st.spinner("A processar tensores matemáticos..."):
            for arquivo in arquivos_upados:
                try:
                    df_limpo = ler_e_limpar_dados(arquivo, arquivo.name, fator_conv)
                    metricas = calcular_metricas(
                        y_true=df_limpo['Referencia'], y_pred=df_limpo['Previsto'], nome_arquivo=arquivo.name
                    )
                    resultados_consolidados.append(metricas)
                    dataframes_processados[arquivo.name] = df_limpo
                except Exception as e:
                    st.error(f"Erro ao processar '{arquivo.name}': {e}")
        
        if resultados_consolidados:
            df_resultados = pd.DataFrame(resultados_consolidados)
            
            # Formatação de Tabela
            st.dataframe(
                df_resultados.style.format({
                    'R² (%)': "{:.2f}", 'R² Ajust. (%)': "{:.2f}", 'Pearson (r)': "{:.3f}",
                    'RMSE': "{:.3f}", 'MAE': "{:.3f}", 'Max Erro': "{:.3f}", 'Bias': "{:.3f}",
                    'MAPE (%)': "{:.2f}", 'CV (%)': "{:.2f}"
                }).background_gradient(subset=['R² (%)', 'CV (%)'], cmap='viridis'),
                use_container_width=True
            )

            # Exportações
            col1, col2 = st.columns(2)
            with col1:
                csv_buffer = io.BytesIO()
                df_resultados.to_csv(csv_buffer, index=False, sep=';', decimal=',')
                st.download_button("⬇️ Exportar Tabela (CSV)", data=csv_buffer.getvalue(), file_name="StructStat_Resultados.csv", mime="text/csv")
            with col2:
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    df_resultados.to_excel(writer, index=False, sheet_name='Métricas')
                st.download_button("⬇️ Exportar Tabela (Excel)", data=excel_buffer.getvalue(), file_name="StructStat_Resultados.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            # Apresentação Gráfica Expandida
            st.markdown("---")
            st.markdown("### 📈 Análise Gráfica Exploratória Avançada")
            st.info("Acesse as abas abaixo para visualizar o diagnóstico completo de cada modelo ensaiado.")
            
            abas_arquivos = st.tabs(list(dataframes_processados.keys()))
            
            for i, (nome_arq, df_grafico) in enumerate(dataframes_processados.items()):
                with abas_arquivos[i]:
                    # Primeira Linha de Gráficos
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(plotar_dispersao_referencia_previsto(df_grafico, nome_arq, sigla_unidade), use_container_width=True)
                    with col2:
                        st.plotly_chart(plotar_bland_altman(df_grafico, nome_arq, sigla_unidade), use_container_width=True)
                        
                    # Segunda Linha de Gráficos
                    col3, col4 = st.columns(2)
                    with col3:
                        st.plotly_chart(plotar_distribuicao_erros(df_grafico, nome_arq, sigla_unidade), use_container_width=True)
                    with col4:
                        st.plotly_chart(plotar_qq_residuos(df_grafico, nome_arq), use_container_width=True)

    else:
        st.info("👈 Aguardando a inserção de dados.")

if __name__ == "__main__":
    main()
