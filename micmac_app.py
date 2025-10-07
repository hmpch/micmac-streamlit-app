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

    # GRÁFICO DE SUBSISTEMAS (con TODAS las variables etiquetadas)
    st.subheader("Gráfico de Subsistemas - Clasificación MICMAC")
    fig_subsistemas, ax_sub = plt.subplots(figsize=(16,12))

    # Calcular líneas de referencia (medianas)
    motricidad_media = np.median(motricidad)
    dependencia_media = np.median(dependencia)

    # Definir colores y tamaños para cada cuadrante
    colors = []
    sizes = []
    for i in range(len(nombres)):
        if motricidad[i] >= motricidad_media and dependencia[i] <= dependencia_media:
            colors.append('red')  # Determinantes
            sizes.append(120)  
        elif motricidad[i] >= motricidad_media and dependencia[i] > dependencia_media:
            colors.append('darkblue')  # Variables clave  
            sizes.append(150)
        elif motricidad[i] < motricidad_media and dependencia[i] > dependencia_media:
            colors.append('lightblue')  # Variables resultado
            sizes.append(80)
        else:
            colors.append('orange')  # Autónomas
            sizes.append(80)

    # Crear scatter con colores y tamaños por cuadrante
    scatter = ax_sub.scatter(dependencia, motricidad, c=colors, alpha=0.7, s=sizes, edgecolors='black', linewidth=0.5)

    # Agregar etiquetas a TODAS las variables
    for i, nombre in enumerate(nombres):
        # Ajustar posición del texto para evitar solapamiento
        offset_x = 8 if dependencia[i] < dependencia_media else -8
        offset_y = 8 if motricidad[i] < motricidad_media else -8
        ha = 'left' if dependencia[i] < dependencia_media else 'right'
        
        ax_sub.annotate(nombre[:20], 
                       (dependencia[i], motricidad[i]), 
                       xytext=(offset_x, offset_y), textcoords='offset points',
                       fontsize=8, fontweight='bold', ha=ha)

    # Líneas de referencia que dividen cuadrantes
    ax_sub.axvline(dependencia_media, color='black', linestyle='--', linewidth=2, alpha=0.7)
    ax_sub.axhline(motricidad_media, color='black', linestyle='--', linewidth=2, alpha=0.7)

    # Etiquetas de cuadrantes con círculos
    max_mot = max(motricidad)
    max_dep = max(dependencia)

    # Círculo Determinantes (superior izquierdo)
    circle1 = plt.Circle((dependencia_media*0.5, max_mot*0.8), max_dep*0.08, 
                        color='red', alpha=0.4)
    ax_sub.add_patch(circle1)
    ax_sub.text(dependencia_media*0.5, max_mot*0.8, 'Determinantes', 
               fontsize=12, fontweight='bold', ha='center', va='center', color='white')

    # Círculo Variables Clave (superior derecho)  
    circle2 = plt.Circle((max_dep*0.8, max_mot*0.8), max_dep*0.12, 
                        color='darkblue', alpha=0.4)
    ax_sub.add_patch(circle2)
    ax_sub.text(max_dep*0.8, max_mot*0.8, 'Variables\nclave', 
               fontsize=12, fontweight='bold', ha='center', va='center', color='white')

    # Círculo Reguladoras (centro)
    circle3 = plt.Circle((dependencia_media, motricidad_media), max_dep*0.08, 
                        color='green', alpha=0.4)
    ax_sub.add_patch(circle3)
    ax_sub.text(dependencia_media, motricidad_media, 'Reguladoras', 
               fontsize=12, fontweight='bold', ha='center', va='center', color='white')

    # Círculo Autónomas (inferior izquierdo)
    circle4 = plt.Circle((dependencia_media*0.5, motricidad_media*0.3), max_dep*0.08, 
                        color='orange', alpha=0.4)
    ax_sub.add_patch(circle4)
    ax_sub.text(dependencia_media*0.5, motricidad_media*0.3, 'Autónomas', 
               fontsize=12, fontweight='bold', ha='center', va='center')

    # Variables resultado (inferior derecho) - sin círculo, solo texto
    ax_sub.text(max_dep*0.8, motricidad_media*0.3, 'Variables\nresultado', 
               fontsize=12, fontweight='bold', ha='center', va='center',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))

    # Configurar ejes y título
    ax_sub.set_xlabel("Dependencia", fontweight='bold', fontsize=14)
    ax_sub.set_ylabel("Motricidad", fontweight='bold', fontsize=14)
    ax_sub.set_title("GRÁFICO DE SUBSISTEMAS", fontweight='bold', fontsize=16)
    ax_sub.grid(True, alpha=0.3)
    st.pyplot(fig_subsistemas)

    # Botón de descarga para gráfico de subsistemas
    img_subsistemas = io.BytesIO()
    fig_subsistemas.savefig(img_subsistemas, format='png', dpi=300, bbox_inches='tight')
    img_subsistemas.seek(0)
    st.download_button(
        label="📥 Descargar Gráfico de Subsistemas (PNG)",
        data=img_subsistemas,
        file_name="micmac_grafico_subsistemas.png",
        mime="image/png"
    )

    # GRÁFICO DEL EJE DE ESTRATEGIA
    st.subheader("Gráfico del Eje de Estrategia")
    fig_estrategia, ax_est = plt.subplots(figsize=(12,10))

    # Calcular el eje de estrategia (diagonal desde origen hacia alta motricidad/alta dependencia)
    estrategia_score = (motricidad / max(motricidad)) + (dependencia / max(dependencia))

    # Crear scatter plot
    scatter_est = ax_est.scatter(dependencia, motricidad, c='steelblue', alpha=0.7, s=100, edgecolors='black')

    # Dibujar línea del eje de estrategia
    max_dep_est = max(dependencia)
    max_mot_est = max(motricidad)
    ax_est.plot([0, max_dep_est], [0, max_mot_est], 'r--', linewidth=3, alpha=0.8, label='Eje de estrategia')

    # Calcular distancia de cada punto al eje de estrategia
    distances_to_axis = []
    for i in range(len(nombres)):
        # Distancia de punto a línea y=x (normalizada)
        x_norm = dependencia[i] / max_dep_est
        y_norm = motricidad[i] / max_mot_est
        dist = abs(y_norm - x_norm) / np.sqrt(2)
        distances_to_axis.append(dist)

    # Mostrar etiquetas para las 15 variables más estratégicas (más cercanas al eje)
    strategic_indices = np.argsort(distances_to_axis)[:15]
    for idx in strategic_indices:
        ax_est.annotate(nombres[idx][:20], 
                       (dependencia[idx], motricidad[idx]), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, fontweight='bold')

    # Configurar ejes y título
    ax_est.set_xlabel("Dependencia", fontweight='bold', fontsize=14)
    ax_est.set_ylabel("Motricidad", fontweight='bold', fontsize=14) 
    ax_est.set_title("GRÁFICO DEL EJE DE ESTRATEGIA", fontweight='bold', fontsize=16)
    ax_est.legend(fontsize=12)
    ax_est.grid(True, alpha=0.3)
    st.pyplot(fig_estrategia)

    # Botón de descarga para eje de estrategia
    img_estrategia = io.BytesIO()
    fig_estrategia.savefig(img_estrategia, format='png', dpi=300, bbox_inches='tight')
    img_estrategia.seek(0)
    st.download_button(
        label="📥 Descargar Gráfico Eje de Estrategia (PNG)",
        data=img_estrategia,
        file_name="micmac_eje_estrategia.png",
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
                # Recrear gráfico de subsistemas
                fig_pdf1, ax_pdf1 = plt.subplots(figsize=(16,12))
                scatter_pdf1 = ax_pdf1.scatter(dependencia, motricidad, c=colors, alpha=0.7, s=sizes, edgecolors='black', linewidth=0.5)
                for i, nombre in enumerate(nombres):
                    offset_x = 8 if dependencia[i] < dependencia_media else -8
                    offset_y = 8 if motricidad[i] < motricidad_media else -8
                    ha = 'left' if dependencia[i] < dependencia_media else 'right'
                    ax_pdf1.annotate(nombre[:20], (dependencia[i], motricidad[i]), xytext=(offset_x, offset_y), 
                                   textcoords='offset points', fontsize=8, fontweight='bold', ha=ha)
                ax_pdf1.axvline(dependencia_media, color='black', linestyle='--', linewidth=2, alpha=0.7)
                ax_pdf1.axhline(motricidad_media, color='black', linestyle='--', linewidth=2, alpha=0.7)
                ax_pdf1.set_xlabel("Dependencia", fontweight='bold')
                ax_pdf1.set_ylabel("Motricidad", fontweight='bold')
                ax_pdf1.set_title("GRÁFICO DE SUBSISTEMAS", fontweight='bold')
                ax_pdf1.grid(True, alpha=0.3)
                pdf.savefig(fig_pdf1, bbox_inches='tight')
                plt.close(fig_pdf1)
                
                # Recrear gráfico eje de estrategia
                fig_pdf2, ax_pdf2 = plt.subplots(figsize=(12,10))
                ax_pdf2.scatter(dependencia, motricidad, c='steelblue', alpha=0.7, s=100, edgecolors='black')
                ax_pdf2.plot([0, max_dep_est], [0, max_mot_est], 'r--', linewidth=3, alpha=0.8, label='Eje de estrategia')
                for idx in strategic_indices:
                    ax_pdf2.annotate(nombres[idx][:20], (dependencia[idx], motricidad[idx]), 
                                   xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
                ax_pdf2.set_xlabel("Dependencia", fontweight='bold')
                ax_pdf2.set_ylabel("Motricidad", fontweight='bold')
                ax_pdf2.set_title("GRÁFICO DEL EJE DE ESTRATEGIA", fontweight='bold')
                ax_pdf2.legend()
                ax_pdf2.grid(True, alpha=0.3)
                pdf.savefig(fig_pdf2, bbox_inches='tight')
                plt.close(fig_pdf2)
                
                # Recrear scatter plot
                fig_pdf3, ax_pdf3 = plt.subplots(figsize=(12,6))
                ax_pdf3.scatter(range(1, len(motricidad)+1), motricidad[ranking_indices])
                for idx, var in enumerate(ranking_vars):
                    ax_pdf3.text(idx+1, motricidad[ranking_indices][idx], var[:15], fontsize=9, ha='center', va='bottom', rotation=90)
                ax_pdf3.set_xlabel("Ranking de Variable")
                ax_pdf3.set_ylabel("Motricidad Total")
                ax_pdf3.set_title(f"Motricidad Total vs Ranking (α={alpha}, K={K_max})")
                ax_pdf3.grid(True)
                pdf.savefig(fig_pdf3, bbox_inches='tight')
                plt.close(fig_pdf3)
                
                # Recrear barplot
                fig_pdf4, ax_pdf4 = plt.subplots(figsize=(16,6))
                sns.barplot(x="Variable", y="Motricidad", data=df_rank, ax=ax_pdf4, palette='Blues_d')
                ax_pdf4.set_xticklabels(ax_pdf4.get_xticklabels(), rotation=90)
                ax_pdf4.set_title(f"Motricidad de variables (α={alpha}, K={K_max})")
                pdf.savefig(fig_pdf4, bbox_inches='tight')
                plt.close(fig_pdf4)
                
                # Recrear heatmap
                fig_pdf5, ax_pdf5 = plt.subplots(figsize=(14,10))
                sns.heatmap(df_heat, annot=True, fmt=".0f", cmap='YlGnBu', linewidths=0.5, annot_kws={"size": 8}, ax=ax_pdf5)
                ax_pdf5.set_title("Motricidad vs Dependencia (directa)")
                pdf.savefig(fig_pdf5, bbox_inches='tight')
                plt.close(fig_pdf5)
            
            # Leer el PDF y ofrecer descarga
            with open(tmp_file.name, 'rb') as f:
                pdf_data = f.read()
            
            st.download_button(
                label="📄 Descargar PDF con todos los gráficos",
                data=pdf_data,
                file_name="micmac_analisis_completo.pdf",
                mime="application/pdf"
            )

else:
    st.info("Por favor suba una matriz Excel para comenzar.")

st.caption("Desarrollado para análisis académico. ©2025")
