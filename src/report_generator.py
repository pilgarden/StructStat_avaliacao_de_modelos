from fpdf import FPDF
import pandas as pd
import os

class PDFGenerator(FPDF):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "StructStat - Relatório de Análise Científica", border=False, ln=True, align="C")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Página {self.page_no()}", align="C")

    def add_table(self, df: pd.DataFrame, title: str):
        self.set_font("Arial", "B", 10)
        self.cell(0, 10, title, ln=True)
        self.set_font("Arial", size=8)
        
        # Cabeçalhos
        col_width = 190 / len(df.columns)
        for col in df.columns:
            self.cell(col_width, 10, str(col), border=1, align="C")
        self.ln()
        
        # Dados
        for _, row in df.iterrows():
            for item in row:
                self.cell(col_width, 10, str(item), border=1, align="C")
            self.ln()
        self.ln(10)

    def add_plot(self, image_path: str, title: str):
        self.set_font("Arial", "B", 10)
        self.cell(0, 10, title, ln=True)
        # O fpdf2 lida bem com caminhos locais
        if os.path.exists(image_path):
            self.image(image_path, w=150)
            self.ln(10)
