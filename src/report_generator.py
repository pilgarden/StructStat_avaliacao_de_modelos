"""
Geração de relatório PDF do diagnóstico avançado (fpdf2).
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fpdf import FPDF
from scipy import stats

from src.diagnostics_analysis import compute_diagnostics_report


def _latin1(text: str) -> str:
    return str(text).encode("latin-1", "replace").decode("latin-1")


class PDFGenerator(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 10, _latin1("StructStat - Relatorio de Diagnostico de Modelos"), ln=True, align="C")
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, _latin1(f"Pagina {self.page_no()}"), align="C")

    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 8, _latin1(title), ln=True)
        self.ln(2)

    def body_text(self, text: str):
        self.set_font("Helvetica", size=9)
        self.multi_cell(0, 5, _latin1(text))
        self.ln(3)

    def add_table(self, df: pd.DataFrame, title: str, max_rows: int = 30):
        self.section_title(title)
        if df is None or df.empty:
            self.body_text("Sem dados disponiveis.")
            return

        df_show = df.head(max_rows).copy()
        cols = [str(c) for c in df_show.columns]
        n_cols = len(cols)
        if n_cols == 0:
            return

        page_w = self.w - self.l_margin - self.r_margin
        col_w = page_w / n_cols
        row_h = 7

        self.set_font("Helvetica", "B", 8)
        for col in cols:
            self.cell(col_w, row_h, _latin1(col)[:28], border=1, align="C")
        self.ln()

        self.set_font("Helvetica", size=7)
        for _, row in df_show.iterrows():
            for val in row:
                cell = _latin1(f"{val:.4g}" if isinstance(val, float) else str(val))[:28]
                self.cell(col_w, row_h, cell, border=1, align="C")
            self.ln()
        self.ln(6)

    def _ensure_space(self, height_mm: float = 75):
        if self.get_y() + height_mm > self.h - 20:
            self.add_page()

    def add_plot(self, image_path: str, title: str, width: float = 170):
        self._ensure_space(85)
        self.section_title(title)
        if os.path.exists(image_path):
            self.image(image_path, w=width)
            self.ln(8)
        else:
            self.body_text("Grafico indisponivel.")

    def add_plots(self, plots: list[tuple[str, str]], section_header: str | None = None):
        if not plots:
            return
        if section_header:
            self._ensure_space(90)
            self.section_title(section_header)
        for title, path in plots:
            self.add_plot(path, title)


def _save_fig(
    fig: plt.Figure,
    output_dir: str,
    filename: str,
    title: str,
    bucket: list[tuple[str, str]],
) -> None:
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    bucket.append((title, path))


def build_report_figures(report: dict[str, Any], output_dir: str) -> dict[str, list[tuple[str, str]]]:
    """Gera todos os graficos das abas de diagnostico, agrupados por secao."""
    groups: dict[str, list[tuple[str, str]]] = {
        "aderencia": [],
        "outliers": [],
        "residuos": [],
        "vif": [],
        "sobol": [],
    }
    if "df_clean" not in report:
        return groups

    meta = report["meta"]
    alvo = meta["alvo"]
    previsto = meta["previsto"]
    df_clean = report["df_clean"]
    residuos = report["residuos_series"]
    y_true = df_clean[alvo]
    y_pred = df_clean[previsto]
    r2 = report["metricas"].get("R2_linear", 0)

    # --- Aba 1: Aderencia visual ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_true, y_pred, alpha=0.7, c="#1f77b4", edgecolors="k", linewidths=0.3)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", label="y=x")
    ax.set_title(f"Linearidade (R2 = {r2:.3f})")
    ax.set_xlabel("Real (y)")
    ax.set_ylabel("Previsto")
    ax.legend()
    _save_fig(fig, output_dir, "01_linearidade.png", "Linearidade: Real vs Previsto", groups["aderencia"])

    fig, ax = plt.subplots(figsize=(6, 4))
    media = (y_true + y_pred) / 2
    ax.scatter(media, residuos, alpha=0.7, c="#ff7f0e", edgecolors="k", linewidths=0.3)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Bland-Altman")
    ax.set_xlabel("Media (Real, Previsto)")
    ax.set_ylabel("Diferenca (Real - Previsto)")
    _save_fig(fig, output_dir, "02_bland_altman.png", "Bland-Altman", groups["aderencia"])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residuos, bins=20, color="#2ca02c", edgecolor="white", alpha=0.85)
    ax.set_title("Distribuicao de Residuos")
    ax.set_xlabel("Residuos")
    ax.set_ylabel("Frequencia")
    _save_fig(fig, output_dir, "03_hist_residuos.png", "Histograma de Residuos (aderencia)", groups["aderencia"])

    # --- Aba 2: Outliers ---
    mask = report.get("outliers", {}).get("mask_z3")
    fig, ax = plt.subplots(figsize=(6, 4))
    if mask is not None and len(mask) == len(residuos):
        normais = ~mask
        ax.scatter(
            y_pred[normais], residuos[normais],
            alpha=0.7, c="#1f77b4", label="Normal", edgecolors="k", linewidths=0.2,
        )
        if mask.any():
            ax.scatter(
                y_pred[mask], residuos[mask],
                alpha=0.9, c="#d62728", label="|Z|>3", edgecolors="k", linewidths=0.3,
            )
        ax.legend()
    else:
        ax.scatter(y_pred, residuos, alpha=0.7, c="#1f77b4", edgecolors="k", linewidths=0.3)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Outliers (|Z| > 3)")
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Residuos")
    _save_fig(fig, output_dir, "04_outliers.png", "Identificacao de Outliers", groups["outliers"])

    # --- Aba 3: Residuos avancados ---
    mean_resid = float(residuos.mean())
    std_resid = float(residuos.std(ddof=1)) if len(residuos) > 1 else 0.0

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_pred, residuos, alpha=0.7, c="#1f77b4", edgecolors="k", linewidths=0.3)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_title("Residuos vs Previsto")
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Residuos")
    _save_fig(fig, output_dir, "05_residuos_disp.png", "Dispersao de Residuos vs Previstos", groups["residuos"])

    if std_resid > 0:
        z_pad = (residuos - mean_resid) / std_resid
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(z_pad, bins=20, density=True, color="#9467bd", alpha=0.7, edgecolor="white")
        x_range = np.linspace(z_pad.min() - 1, z_pad.max() + 1, 100)
        ax.plot(x_range, stats.norm.pdf(x_range, 0, 1), "r--", linewidth=2, label="Normal teorica")
        ax.set_title("Densidade vs Curva Normal")
        ax.set_xlabel("Residuos padronizados (Z)")
        ax.set_ylabel("Densidade")
        ax.legend()
        _save_fig(fig, output_dir, "06_densidade_normal.png", "Densidade vs Curva Normal", groups["residuos"])

        (osm, osr), (slope, intercept, r_qq) = stats.probplot(z_pad, dist="norm")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(osm, osr, alpha=0.7, c="#8c564b", edgecolors="k", linewidths=0.3)
        ax.plot(osm, slope * osm + intercept, "r--", label="Referencia")
        ax.set_title(f"Q-Q Plot (R2 aderencia = {r_qq ** 2:.3f})")
        ax.set_xlabel("Quantis teoricos (Normal)")
        ax.set_ylabel("Quantis observados (Z)")
        ax.legend()
        _save_fig(fig, output_dir, "07_qq_plot.png", "Q-Q Plot", groups["residuos"])

    # --- Aba 4: VIF ---
    vif_df = report.get("vif")
    if vif_df is not None and not vif_df.empty:
        vif_plot = vif_df[vif_df["Feature"].astype(str).str.lower() != "const"].copy()
        if not vif_plot.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            colors = ["#c44e52" if v > 10 else "#4c72b0" for v in vif_plot["VIF"]]
            ax.bar(vif_plot["Feature"].astype(str), vif_plot["VIF"], color=colors)
            ax.axhline(10, color="darkred", linestyle="--", label="VIF=10")
            ax.set_title("Inflacao da Variancia (VIF)")
            ax.tick_params(axis="x", rotation=45)
            ax.legend()
            fig.tight_layout()
            _save_fig(fig, output_dir, "08_vif.png", "Multicolinearidade (VIF)", groups["vif"])

    # --- Aba 5: Sobol ---
    sobol_df = report.get("sobol")
    if sobol_df is not None and not sobol_df.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(sobol_df))
        w = 0.35
        s1_col = "S1 (Isolado)" if "S1 (Isolado)" in sobol_df.columns else sobol_df.columns[1]
        st_col = "ST (Total)" if "ST (Total)" in sobol_df.columns else sobol_df.columns[2]
        var_col = "Variável" if "Variável" in sobol_df.columns else sobol_df.columns[0]
        ax.bar(x - w / 2, sobol_df[s1_col], w, label="S1 (Isolado)", color="#1f77b4")
        ax.bar(x + w / 2, sobol_df[st_col], w, label="ST (Total)", color="#ff7f0e")
        ax.set_xticks(x)
        ax.set_xticklabels(sobol_df[var_col].astype(str), rotation=45, ha="right")
        ax.set_title("Indices de Sobol")
        ax.set_ylabel("Indice")
        ax.legend()
        fig.tight_layout()
        _save_fig(fig, output_dir, "09_sobol.png", "Sensibilidade Global (Sobol)", groups["sobol"])

    return groups


def generate_diagnostics_pdf(
    df: pd.DataFrame,
    alvo: str,
    previsto: str,
    preditores: list[str],
    dataset_name: str = "Dataset",
    sobol_df: pd.DataFrame | None = None,
) -> bytes:
    """Gera PDF completo e devolve bytes para download."""
    report = compute_diagnostics_report(
        df, alvo, previsto, preditores, dataset_name=dataset_name, sobol_df=sobol_df
    )

    pdf = PDFGenerator()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    meta = report["meta"]
    pdf.section_title("1. Identificacao da Analise")
    pdf.body_text(
        f"Dataset: {meta['dataset']}\n"
        f"Gerado em: {meta['gerado_em']}\n"
        f"Variavel real (y): {meta['alvo']}\n"
        f"Variavel prevista: {meta['previsto']}\n"
        f"Preditores (X): {', '.join(meta['preditores'])}\n"
        f"Amostras validas: {meta['n_amostras']}"
    )

    if report.get("erro"):
        pdf.body_text(report["erro"])
        return bytes(pdf.output())

    with tempfile.TemporaryDirectory() as tmpdir:
        plot_groups = build_report_figures(report, tmpdir)

        # Metricas
        pdf.section_title("2. Metricas de Aderencia do Modelo")
        metricas = report["metricas"]
        rows = []
        for k, v in metricas.items():
            if k in ("Arquivo",):
                continue
            if isinstance(v, (int, float, np.floating, np.integer)):
                rows.append({"Metrica": k, "Valor": round(float(v), 4)})
            else:
                rows.append({"Metrica": k, "Valor": str(v)})
        if rows:
            pdf.add_table(pd.DataFrame(rows), "Indicadores de desempenho")
        pdf.add_plots(plot_groups["aderencia"], "2.1 Graficos de Aderencia Visual")

        # Residuos e testes
        pdf.add_page()
        pdf.section_title("3. Residuos e Testes Formais")
        res = report["residuos"]
        testes_df = pd.DataFrame(
            [
                {"Teste / Estatistica": "Media dos residuos", "Valor": f"{res['media']:.4e}"},
                {"Teste / Estatistica": "Desvio padrao (residuos)", "Valor": f"{res['desvio_padrao']:.4f}"},
                {"Teste / Estatistica": "Media fator theta (Real/Prev)", "Valor": f"{res['media_theta']:.4f}"},
                {"Teste / Estatistica": "CoV theta (%)", "Valor": f"{res['cov_theta_pct']:.2f}"},
                {"Teste / Estatistica": "p-valor Breusch-Pagan", "Valor": f"{res['p_breusch_pagan']:.4f}"},
                {"Teste / Estatistica": "p-valor Shapiro-Wilk", "Valor": f"{res['p_shapiro']:.4f}"},
                {"Teste / Estatistica": "p-valor Anderson-Darling", "Valor": f"{res['p_anderson_darling']:.4f}"},
            ]
        )
        pdf.add_table(testes_df, "Resultados estatisticos")
        pdf.add_plots(plot_groups["residuos"], "3.1 Graficos de Validacao de Residuos")

        # Outliers
        pdf.add_page()
        pdf.section_title("4. Analise de Outliers")
        out = report["outliers"]
        pdf.body_text(
            f"Percentagem com |Z|>3: {out['percentagem_z3']:.2f}%\n"
            f"Teste de Grubbs: {'outlier extremo detectado' if out['grubbs_detectado'] else 'sem outlier extremo'} "
            f"(indice {out['grubbs_indice']})."
        )
        pdf.add_plots(plot_groups["outliers"], "4.1 Grafico de Outliers")

        # VIF
        pdf.add_page()
        pdf.section_title("5. Multicolinearidade (VIF)")
        if report.get("vif") is not None:
            vif_show = report["vif"][["Feature", "VIF", "Tolerancia"]].copy()
            vif_show["VIF"] = vif_show["VIF"].round(2)
            vif_show["Tolerancia"] = vif_show["Tolerancia"].round(4)
            pdf.add_table(vif_show, "Tabela VIF")
        else:
            pdf.body_text("VIF nao calculado.")
        pdf.add_plots(plot_groups["vif"], "5.1 Grafico VIF")

        # Sobol
        if report.get("sobol") is not None and not report["sobol"].empty:
            pdf.add_page()
            pdf.section_title("6. Sensibilidade Global (Sobol)")
            sob = report["sobol"].copy()
            for c in sob.select_dtypes(include=[np.number]).columns:
                sob[c] = sob[c].round(4)
            pdf.add_table(sob, "Indices de Sobol")
            pdf.add_plots(plot_groups["sobol"], "6.1 Grafico Sobol")

        # Pareceres
        pdf.add_page()
        pdf.section_title("7. Parecer Tecnico Sintetico")
        for chave, texto in report.get("pareceres", {}).items():
            pdf.body_text(f"- {chave.replace('_', ' ').title()}: {texto}")

    return bytes(pdf.output())


def generate_pdf_report(df: pd.DataFrame, plot_path: str, dataset_name: str = "Dataset") -> bytes:
    """Compatibilidade com API antiga (relatorio exploratorio simples)."""
    pdf = PDFGenerator()
    pdf.add_page()
    pdf.add_table(df.describe().reset_index(), "Estatisticas Descritivas")
    pdf.add_plot(plot_path, "Distribuicao dos Dados")
    return bytes(pdf.output())
