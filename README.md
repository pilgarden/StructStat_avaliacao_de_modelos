# StructStat: Avaliação Estatística de Modelos Estruturais

Dashboard interativo desenvolvido em Python para validação estatística e análise de confiabilidade de modelos preditivos na Engenharia de Estruturas, comparando resultados analíticos/numéricos com dados experimentais de referência.

---

## 🚀 Como Executar Localmente

O projeto utiliza as bibliotecas listadas no ficheiro `requirements.txt`. Para executar a aplicação após um longo período inativo, siga os passos no seu terminal:

1. **Ative o seu ambiente virtual** (opcional, mas recomendado).
2. **Instale as dependências**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Inicie o servidor local do Streamlit**:
   ```bash
   streamlit run app.py
   ```

---

## 🗺️ Mapa de Manutenção (Onde alterar o código)

O software foi construído sob o paradigma da *Clean Architecture* (Martin, 2017). Isto significa que a interface, os gráficos e a matemática estão isolados. Se quiser alterar algo específico, vá direto ao ficheiro correspondente:

### 1. Estética dos Gráficos e Cores (`src/visualization.py`)
Toda a componente gráfica é renderizada pela biblioteca **Plotly**. Se precisar de alterar esquemas de cores para adequar os gráficos às exigências de formatação de um periódico científico:
* **Onde mexer:** Nas funções `plotar_dispersao...`, etc., em `src/visualization.py`.
* **Como alterar cores:** Procure por `marker_color`, `line=dict(color='red')`.
* **Como alterar o fundo:** O parâmetro `template="simple_white"` garante o alto rácio *data-ink* (Tufte, 1983).

### 2. Layout da Página e Interface Web (`app.py`)
A interface do utilizador é controlada pelo **Streamlit**.
* **Formatação da Tabela:** As casas decimais e cores (paleta *Viridis*, recomendada por Wilke, 2019) são definidas em `df_resultados.style.format(...)`.

### 3. Fórmulas Estatísticas (`src/metrics.py`)
Para incluir novas métricas de erro na sua investigação:
* Abra o ficheiro `src/metrics.py`.
* Adicione o cálculo vetorial utilizando o *NumPy* ou *SciPy*.

### 4. Unidades e Fatores de Conversão (`src/config.py`)
* Adicione a nova grandeza e os respetivos fatores de conversão no dicionário `GRANDEZAS_UNIDADES` no ficheiro `src/config.py`.

---

## 📚 Referências Científicas
* **Tufte, E. R. (1983).** *The Visual Display of Quantitative Information.* Graphics Press.
* **Wilke, C. O. (2019).** *Fundamentals of Data Visualization.* O'Reilly Media.
* **Bland, J. M., & Altman, D. (1986).** *Statistical methods for assessing agreement between two methods of clinical measurement.* The Lancet, 327(8476), 307-310.
* **Ang, A. H.-S., & Tang, W. H. (2007).** *Probability Concepts in Engineering.* Wiley.
"""

# Guardar o ficheiro forçando o encoding correto
with open("README.md", "w", encoding="utf-8") as f:
    f.write(conteudo)

print("✅ Ficheiro 'README.md' gerado com sucesso com a formatação correta!")
