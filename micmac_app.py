import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from matplotlib.backends.backend_pdf import PdfPages
import tempfile

# Configuración general de la app
st.set_page_config(page_title="Análisis MICMAC Interactivo", layout="wide")

# Encabezado personalizado
st.markdown("""
# Análisis MICMAC Interactivo  
by **Martín Pratto**
""")
st.markdown("""
Herramienta visual para el análisis estructural de variables usando el método MICMAC.
- Sube tu matriz MICMAC en formato Excel (cada variable como fila/columna).
- Ajusta los parámetros de propagación de influencias.
- Visualiza los rankings y gráficos.
""")

# Upload del archivo Excel
uploaded_file = st.file_uploader("Sube tu archivo Excel MICMAC:", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file, index_col=0)
    # Elimina última columna si es SUMA
    if 'SUMA' in df.columns: 
        df = df.drop('SUMA', axis=1)
    # Tomar sólo las 40 primeras columnas y filas si hiciera falta
    if df.shape[0] > 40 or df.shape[1] > 40:
        variables = df.columns[:40]
        df = df.loc[variables, variables]
    # Reemplaza cualquier valor no numérico por 0
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    M = df.values.astype(float)
    nombres = df.index.tolist()
    st.success(f"Archivo cargado correctamente. {len(nombres)} variables detectadas.")

    # Parámetros seleccionables por el usuario
    alpha = st.slider("Selecciona el valor de α (atenuación de rutas indirectas):", 0.1, 1.0, 0.5, step=0.05)
    K_max = st.slider("Longitud máxima de rutas (K):", 2, 10, 6)
    
    # Calcula motricidad total
    def motricidad_total(M, alpha, K):
        M_total = np.copy(M).astype(float)
        M_power = np.copy(M).astype(float)
        for k in range(2, K+1):
            M_power = M_power @ M
            M_total += (alpha ** (k - 1)) * M_power
        motricidad = np.sum(M_total, axis=1)
        return motricidad

    motricidad = motricidad_total(M, alpha, K_max)
    dependencia = np.sum(M, axis=0)
    
    # Ranking de variables
    ranking_indices = np.argsort(-motricidad)
    ranking_vars = [nombres[i] for i in ranking_indices]
    
    # Mostrar ranking en tabla
    st.header(f"Ranking de Motricidad Total (α={alpha}, K={K_max})")
    df_rank = pd.DataFrame({
        "Posición": np.arange(1, len(ranking_vars)+1),
        "Variable": ranking_vars,
        "Motricidad": motricidad[ranking_indices]
    })
    st.dataframe(df_rank)

    # Gráfico MICMAC clásico: motricidad vs dependencia
    st.subheader("Mapa Estratégico MICMAC (motricidad vs dependencia)")
    fig4, ax4 = plt.subplots(figsize=(10,8))
    ax4.scatter(motricidad, dependencia)
    for i, var in enumerate(nombres):
        ax4.text(motricidad[i], dependencia[i], var[:12], fontsize=8)
    ax4.set_xlabel("Motricidad (Influencia ejercida)")
    ax4.set_ylabel("Dependencia (Influencia recibida)")
    ax4.set_title("Diagrama estratégico: MICMAC")
    ax4.grid(True)
    st.pyplot(fig4)

    # Botón de descarga para mapa estratégico
    img_mapa = io.BytesIO()
    fig4.savefig(img_mapa, format='png', dpi=300, bbox_inches='tight')
    img_mapa.seek(0)
    st.download_button(
        label="📥 Descargar Mapa Estratégico (PNG)",
        data=img_mapa,
        file_name="micmac_mapa_estrategico.png",
        mime="image/png"
    )

    # Gráfico scatter
    st.subheader("Motricidad Total vs Ranking (Scatter)")
    fig, ax = plt.subplots(figsize=(12,6))
    ax.scatter(range(1, len(motricidad)+1), motricidad[ranking_indices])
    for idx, var in enumerate(ranking_vars):
        ax.text(idx+1, motricidad[ranking_indices][idx], var[:15], fontsize=9, ha='center', va='bottom', rotation=90)
    ax.set_xlabel("Ranking de Variable")
    ax.set_ylabel("Motricidad Total")
    ax.set_title(f"Motricidad Total vs Ranking (α={alpha}, K={K_max})")
    ax.grid(True)
    st.pyplot(fig)
    
    # Botón de descarga para scatter
    img_scatter = io.BytesIO()
    fig.savefig(img_scatter, format='png', dpi=300, bbox_inches='tight')
    img_scatter.seek(0)
    st.download_button(
        label="📥 Descargar Scatter Plot (PNG)",
        data=img_scatter,
        file_name="micmac_scatter_plot.png",
        mime="image/png"
    )

    # Gráfico barplot
    st.subheader("Motricidad de Variables (Barplot)")
    fig2, ax2 = plt.subplots(figsize=(16,6))
    sns.barplot(x="Variable", y="Motricidad", data=df_rank, ax=ax2, palette='Blues_d')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)
    ax2.set_title(f"Motricidad de variables (α={alpha}, K={K_max})")
    st.pyplot(fig2)
    
    # Botón de descarga para barplot
    img_barplot = io.BytesIO()
    fig2.savefig(img_barplot, format='png', dpi=300, bbox_inches='tight')
    img_barplot.seek(0)
    st.download_button(
        label="📥 Descargar Gráfico de Barras (PNG)",
        data=img_barplot,
        file_name="micmac_barplot.png",
        mime="image/png"
    )

    # Gráfico heatmap
    st.subheader("Heatmap de Motricidad y Dependencia (influencias directas)")
    df_heat = pd.DataFrame({
        "Motricidad": np.sum(M, axis=1),
        "Dependencia": np.sum(M, axis=0)
    }, index=nombres)
    fig3, ax3 = plt.subplots(figsize=(14,10))
    sns.heatmap(df_heat, annot=True, fmt=".0f", cmap='YlGnBu', linewidths=0.5, annot_kws={"size": 8}, ax=ax3)
    ax3.set_title("Motricidad vs Dependencia (directa)")
    st.pyplot(fig3)
    
    # Botón de descarga para heatmap
    img_heatmap = io.BytesIO()
    fig3.savefig(img_heatmap, format='png', dpi=300, bbox_inches='tight')
    img_heatmap.seek(0)
    st.download_button(
        label="📥 Descargar Heatmap (PNG)",
        data=img_heatmap,
        file_name="micmac_heatmap.png",
        mime="image/png"
    )
    
    # Archivo Excel de resultados descargable
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_rank.to_excel(writer, index=False)
    output.seek(0)
    st.subheader("Descarga tu ranking en Excel")
    st.download_button(
        label="Descargar Ranking de Motricidad (xlsx)",
        data=output,
        file_name="micmac_ranking.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # Exportar todos los gráficos en un PDF
    st.subheader("Descargar todos los gráficos en PDF")
    if st.button("🔄 Generar PDF con todos los gráficos"):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            with PdfPages(tmp_file.name) as pdf:
                # Recrear mapa estratégico
                fig_pdf1, ax_pdf1 = plt.subplots(figsize=(10,8))
                ax_pdf1.scatter(motricidad, dependencia)
                for i, var in enumerate(nombres):
                    ax_pdf1.text(motricidad[i], dependencia[i], var[:12], fontsize=8)
                ax_pdf1.set_xlabel("Motricidad (Influencia ejercida)")
                ax_pdf1.set_ylabel("Dependencia (Influencia recibida)")
                ax_pdf1.set_title("Diagrama estratégico: MICMAC")
                ax_pdf1.grid(True)
                pdf.savefig(fig_pdf1, bbox_inches='tight')
                plt.close(fig_pdf1)
                
                # Recrear scatter plot
                fig_pdf2, ax_pdf2 = plt.subplots(figsize=(12,6))
                ax_pdf2.scatter(range(1, len(motricidad)+1), motricidad[ranking_indices])
                for idx, var in enumerate(ranking_vars):
                    ax_pdf2.text(idx+1, motricidad[ranking_indices][idx], var[:15], fontsize=9, ha='center', va='bottom', rotation=90)
                ax_pdf2.set_xlabel("Ranking de Variable")
                ax_pdf2.set_ylabel("Motricidad Total")
                ax_pdf2.set_title(f"Motricidad Total vs Ranking (α={alpha}, K={K_max})")
                ax_pdf2.grid(True)
                pdf.savefig(fig_pdf2, bbox_inches='tight')
                plt.close(fig_pdf2)
                
                # Recrear barplot
                fig_pdf3, ax_pdf3 = plt.subplots(figsize=(16,6))
                sns.barplot(x="Variable", y="Motricidad", data=df_rank, ax=ax_pdf3, palette='Blues_d')
                ax_pdf3.set_xticklabels(ax_pdf3.get_xticklabels(), rotation=90)
                ax_pdf3.set_title(f"Motricidad de variables (α={alpha}, K={K_max})")
                pdf.savefig(fig_pdf3, bbox_inches='tight')
                plt.close(fig_pdf3)
                
                # Recrear heatmap
                fig_pdf4, ax_pdf4 = plt.subplots(figsize=(14,10))
                sns.heatmap(df_heat, annot=True, fmt=".0f", cmap='YlGnBu', linewidths=0.5, annot_kws={"size": 8}, ax=ax_pdf4)
                ax_pdf4.set_title("Motricidad vs Dependencia (directa)")
                pdf.savefig(fig_pdf4, bbox_inches='tight')
                plt.close(fig_pdf4)
            
            # Leer el PDF y ofrecer descarga
            with open(tmp_file.name, 'rb') as f:
                pdf_data = f.read()
            
            st.download_button(
                label="📄 Descargar PDF con todos los gráficos",
                data=pdf_data,
                file_name="micmac_graficos_completos.pdf",
                mime="application/pdf"
            )

else:
    st.info("Por favor suba una matriz Excel para comenzar.")

st.caption("Desarrollado para análisis académico. ©2025")

