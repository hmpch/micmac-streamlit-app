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

    # Gráfico MICMAC clásico con cuadrantes definidos
    st.subheader("Mapa MICMAC - Clasificación por Cuadrantes")
    fig_cuadrantes, ax_cuad = plt.subplots(figsize=(12,10))

    # Calcular líneas de referencia (medianas)
    motricidad_media = np.median(motricidad)
    dependencia_media = np.median(dependencia)

    # Definir colores para cada cuadrante
    colors = []
    for i in range(len(nombres)):
        if motricidad[i] >= motricidad_media and dependencia[i] <= dependencia_media:
            colors.append('red')  # Motoras
        elif motricidad[i] >= motricidad_media and dependencia[i] > dependencia_media:
            colors.append('orange')  # Reguladoras  
        elif motricidad[i] < motricidad_media and dependencia[i] > dependencia_media:
            colors.append('green')  # Dependientes
        else:
            colors.append('blue')  # Independientes

    # Crear scatter con colores por cuadrante
    scatter = ax_cuad.scatter(motricidad, dependencia, c=colors, alpha=0.7, s=100)

    # Agregar etiquetas a las variables más importantes (top 10)
    top_indices = ranking_indices[:10]
    for idx in top_indices:
        ax_cuad.annotate(nombres[idx][:20], 
                        (motricidad[idx], dependencia[idx]), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, fontweight='bold')

    # Líneas de referencia que dividen cuadrantes
    ax_cuad.axvline(motricidad_media, color='black', linestyle='--', linewidth=2, alpha=0.7)
    ax_cuad.axhline(dependencia_media, color='black', linestyle='--', linewidth=2, alpha=0.7)

    # Etiquetas de cuadrantes
    max_mot = max(motricidad)
    max_dep = max(dependencia)
    
    ax_cuad.text(motricidad_media + (max_mot - motricidad_media)*0.5, dependencia_media*0.3, 
                'MOTORAS\n(Crítico/inestable)', 
                fontsize=12, fontweight='bold', ha='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3))

    ax_cuad.text(motricidad_media + (max_mot - motricidad_media)*0.5, 
                dependencia_media + (max_dep - dependencia_media)*0.5, 
                'REGULADORAS\n(Crítico/inestable)', 
                fontsize=12, fontweight='bold', ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.3))

    ax_cuad.text(motricidad_media*0.5, dependencia_media + (max_dep - dependencia_media)*0.5, 
                'DEPENDIENTES', 
                fontsize=12, fontweight='bold', ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.3))

    ax_cuad.text(motricidad_media*0.5, dependencia_media*0.3, 
                'INDEPENDIENTES\n(Motriz)', 
                fontsize=12, fontweight='bold', ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="blue", alpha=0.3))

    # Configurar ejes y título
    ax_cuad.set_xlabel("Influencia Total (Motricidad)", fontweight='bold', fontsize=12)
    ax_cuad.set_ylabel("Dependencia Total", fontweight='bold', fontsize=12)
    ax_cuad.set_title("Mapa MICMAC - Clasificación por Cuadrantes", fontweight='bold', fontsize=14)

    # Leyenda
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Motoras'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Reguladoras'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Dependientes'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Independientes')
    ]
    ax_cuad.legend(handles=legend_elements, loc='upper right')

    ax_cuad.grid(True, alpha=0.3)
    st.pyplot(fig_cuadrantes)

    # Botón de descarga para mapa de cuadrantes
    img_cuadrantes = io.BytesIO()
    fig_cuadrantes.savefig(img_cuadrantes, format='png', dpi=300, bbox_inches='tight')
    img_cuadrantes.seek(0)
    st.download_button(
        label="📥 Descargar Mapa MICMAC Cuadrantes (PNG)",
        data=img_cuadrantes,
        file_name="micmac_mapa_cuadrantes.png",
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
                # Recrear mapa de cuadrantes
                fig_pdf1, ax_pdf1 = plt.subplots(figsize=(12,10))
                ax_pdf1.scatter(motricidad, dependencia, c=colors, alpha=0.7, s=100)
                for idx in top_indices:
                    ax_pdf1.annotate(nombres[idx][:20], 
                                    (motricidad[idx], dependencia[idx]), 
                                    xytext=(5, 5), textcoords='offset points',
                                    fontsize=9, fontweight='bold')
                ax_pdf1.axvline(motricidad_media, color='black', linestyle='--', linewidth=2, alpha=0.7)
                ax_pdf1.axhline(dependencia_media, color='black', linestyle='--', linewidth=2, alpha=0.7)
                ax_pdf1.set_xlabel("Influencia Total (Motricidad)", fontweight='bold')
                ax_pdf1.set_ylabel("Dependencia Total", fontweight='bold')
                ax_pdf1.set_title("Mapa MICMAC - Clasificación por Cuadrantes", fontweight='bold')
                ax_pdf1.grid(True, alpha=0.3)
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
