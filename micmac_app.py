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

    # GRÁFICO DE SUBSISTEMAS MEJORADO
    st.subheader("Gráfico de Subsistemas - Clasificación MICMAC")
    fig_subsistemas, ax_sub = plt.subplots(figsize=(18,14))

    # Calcular líneas de referencia (medianas)
    motricidad_media = np.median(motricidad)
    dependencia_media = np.median(dependencia)

    # Definir colores y clasificación para cada cuadrante
    colors = []
    labels_cuadrante = []
    sizes = []
    
    for i in range(len(nombres)):
        if motricidad[i] >= motricidad_media and dependencia[i] <= dependencia_media:
            colors.append('#FF4444')  # Rojo - Determinantes
            labels_cuadrante.append('Determinantes')
            sizes.append(120)  
        elif motricidad[i] >= motricidad_media and dependencia[i] > dependencia_media:
            colors.append('#1166CC')  # Azul oscuro - Crítico/inestable
            labels_cuadrante.append('Crítico/inestable')
            sizes.append(150)
        elif motricidad[i] < motricidad_media and dependencia[i] > dependencia_media:
            colors.append('#66BBFF')  # Azul claro - Variables resultado
            labels_cuadrante.append('Variables resultado')
            sizes.append(80)
        else:
            colors.append('#FF9944')  # Naranja - Autónomas
            labels_cuadrante.append('Autónomas')
            sizes.append(80)

    # Crear scatter con colores por cuadrante
    scatter = ax_sub.scatter(dependencia, motricidad, c=colors, alpha=0.8, s=sizes, 
                           edgecolors='black', linewidth=1.5)

    # Líneas de referencia que dividen cuadrantes
    ax_sub.axvline(dependencia_media, color='black', linestyle='--', linewidth=2, alpha=0.7)
    ax_sub.axhline(motricidad_media, color='black', linestyle='--', linewidth=2, alpha=0.7)

    # Etiquetas de cuadrantes más grandes y visibles
    max_mot = max(motricidad)
    max_dep = max(dependencia)
    
    # Rectángulos de fondo para mejor legibilidad
    ax_sub.text(dependencia_media*0.5, max_mot*0.85, 'DETERMINANTES', 
               fontsize=14, fontweight='bold', ha='center', va='center',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="red", alpha=0.7, edgecolor='black'))

    ax_sub.text(max_dep*0.8, max_mot*0.85, 'CRÍTICO/INESTABLE\n(Variables clave)', 
               fontsize=14, fontweight='bold', ha='center', va='center',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="darkblue", alpha=0.7, edgecolor='black'),
               color='white')

    ax_sub.text(dependencia_media*0.5, motricidad_media*0.3, 'AUTÓNOMAS', 
               fontsize=14, fontweight='bold', ha='center', va='center',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="orange", alpha=0.7, edgecolor='black'))

    ax_sub.text(max_dep*0.8, motricidad_media*0.3, 'VARIABLES\nRESULTADO', 
               fontsize=14, fontweight='bold', ha='center', va='center',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7, edgecolor='black'))

    # Sistema de etiquetas espaciadas - solo variables importantes
    importantes = []
    for i, nombre in enumerate(nombres):
        # Criterio para variables importantes
        if (motricidad[i] > np.percentile(motricidad, 80)) or (dependencia[i] > np.percentile(dependencia, 80)) or i in ranking_indices[:12]:
            importantes.append((i, nombre, dependencia[i], motricidad[i]))
    
    # Espaciado inteligente de etiquetas
    for j, (i, nombre, dep, mot) in enumerate(importantes):
        # Calcular offset basado en cuadrante para evitar superposición
        if dep < dependencia_media and mot >= motricidad_media:  # Determinantes
            offset_x = -15 - (j % 3) * 10
            offset_y = 10 + (j % 2) * 15
            ha = 'right'
        elif dep >= dependencia_media and mot >= motricidad_media:  # Crítico
            offset_x = 10 + (j % 3) * 8  
            offset_y = 10 + (j % 2) * 15
            ha = 'left'
        elif dep >= dependencia_media and mot < motricidad_media:  # Variables resultado
            offset_x = 10 + (j % 3) * 8
            offset_y = -15 - (j % 2) * 10
            ha = 'left'
        else:  # Autónomas
            offset_x = -15 - (j % 3) * 10
            offset_y = -15 - (j % 2) * 10
            ha = 'right'
        
        ax_sub.annotate(nombre[:22], 
                       (dep, mot), 
                       xytext=(offset_x, offset_y), textcoords='offset points',
                       fontsize=9, fontweight='bold', ha=ha,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor='gray'),
                       arrowprops=dict(arrowstyle='->', color='gray', alpha=0.6, lw=1))

    # Configurar ejes y título
    ax_sub.set_xlabel("Dependencia", fontweight='bold', fontsize=16)
    ax_sub.set_ylabel("Motricidad", fontweight='bold', fontsize=16)
    ax_sub.set_title("GRÁFICO DE SUBSISTEMAS", fontweight='bold', fontsize=20)
    
    # Leyenda de colores mejorada
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF4444', 
                  markersize=12, label='Determinantes (Palancas de acción)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1166CC', 
                  markersize=15, label='Crítico/inestable (Variables clave)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#66BBFF', 
                  markersize=10, label='Variables resultado (Indicadores)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF9944', 
                  markersize=10, label='Autónomas (Independientes)')
    ]
    ax_sub.legend(handles=legend_elements, loc='upper left', fontsize=12, 
                 frameon=True, fancybox=True, shadow=True)
    
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

    # GRÁFICO DEL EJE DE ESTRATEGIA MEJORADO
    st.subheader("Gráfico del Eje de Estrategia")
    fig_estrategia, ax_est = plt.subplots(figsize=(14,11))

    # Calcular distancia de cada punto al eje de estrategia
    max_dep_norm = max(dependencia)
    max_mot_norm = max(motricidad)
    
    distances_to_axis = []
    strategic_scores = []
    
    for i in range(len(nombres)):
        # Normalizar coordenadas
        x_norm = dependencia[i] / max_dep_norm
        y_norm = motricidad[i] / max_mot_norm
        
        # Distancia al eje de estrategia (línea y=x)
        dist = abs(y_norm - x_norm) / np.sqrt(2)
        distances_to_axis.append(dist)
        
        # Puntuación estratégica (cercanía al eje + valor absoluto)
        strategic_score = (x_norm + y_norm) / 2 - dist
        strategic_scores.append(strategic_score)

    # Colores por nivel estratégico
    colors_estrategia = []
    for score in strategic_scores:
        if score > np.percentile(strategic_scores, 80):
            colors_estrategia.append('#CC0000')  # Rojo - Muy estratégico
        elif score > np.percentile(strategic_scores, 60):
            colors_estrategia.append('#FF6600')  # Naranja - Estratégico
        elif score > np.percentile(strategic_scores, 40):
            colors_estrategia.append('#3388BB')  # Azul - Moderadamente estratégico
        else:
            colors_estrategia.append('#888888')  # Gris - Poco estratégico

    # Tamaños proporcionales al valor estratégico
    sizes_estrategia = [(score - min(strategic_scores)) / (max(strategic_scores) - min(strategic_scores)) * 100 + 50 
                       for score in strategic_scores]

    # Crear scatter plot
    scatter_est = ax_est.scatter(dependencia, motricidad, c=colors_estrategia, alpha=0.8, 
                               s=sizes_estrategia, edgecolors='black', linewidth=1)

    # Dibujar línea del eje de estrategia más gruesa
    ax_est.plot([0, max_dep_norm], [0, max_mot_norm], 'r--', linewidth=4, alpha=0.9, label='Eje de estrategia')

    # Etiquetas solo para variables más estratégicas
    strategic_indices = np.argsort(strategic_scores)[-15:]  # Top 15 más estratégicas
    
    for j, idx in enumerate(strategic_indices):
        # Posicionamiento inteligente para evitar superposición
        if dependencia[idx] < max_dep_norm/2:
            offset_x = 15 + (j % 3) * 10
            ha = 'left'
        else:
            offset_x = -15 - (j % 3) * 10
            ha = 'right'
        
        if motricidad[idx] < max_mot_norm/2:
            offset_y = 15 + (j % 2) * 12
        else:
            offset_y = -15 - (j % 2) * 12
        
        ax_est.annotate(nombres[idx][:22], 
                       (dependencia[idx], motricidad[idx]), 
                       xytext=(offset_x, offset_y), textcoords='offset points',
                       fontsize=10, fontweight='bold', ha=ha,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8, edgecolor='orange'),
                       arrowprops=dict(arrowstyle='->', color='orange', alpha=0.7, lw=1.5))

    # Configurar ejes y título
    ax_est.set_xlabel("Dependencia", fontweight='bold', fontsize=16)
    ax_est.set_ylabel("Motricidad", fontweight='bold', fontsize=16) 
    ax_est.set_title("GRÁFICO DEL EJE DE ESTRATEGIA", fontweight='bold', fontsize=20)
    
    # Leyenda mejorada
    legend_elements_est = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#CC0000', 
                  markersize=12, label='Muy estratégico'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6600', 
                  markersize=10, label='Estratégico'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#3388BB', 
                  markersize=8, label='Moderadamente estratégico'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#888888', 
                  markersize=6, label='Poco estratégico'),
        plt.Line2D([0], [0], color='red', linestyle='--', linewidth=3, label='Eje de estrategia')
    ]
    ax_est.legend(handles=legend_elements_est, loc='upper left', fontsize=12,
                 frameon=True, fancybox=True, shadow=True)
    
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

    # Tabla resumen de variables estratégicas
    st.subheader("Variables Más Estratégicas")
    df_estrategicas = pd.DataFrame({
        'Variable': [nombres[i] for i in strategic_indices],
        'Motricidad': [round(motricidad[i], 2) for i in strategic_indices],
        'Dependencia': [round(dependencia[i], 2) for i in strategic_indices],
        'Puntuación Estratégica': [round(strategic_scores[i], 3) for i in strategic_indices],
        'Clasificación': [labels_cuadrante[i] for i in strategic_indices]
    }).sort_values('Puntuación Estratégica', ascending=False)
    st.dataframe(df_estrategicas)

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
        df_rank.to_excel(writer, sheet_name='Ranking', index=False)
        df_estrategicas.to_excel(writer, sheet_name='Variables_Estrategicas', index=False)
    output.seek(0)
    st.subheader("Descarga tu ranking en Excel")
    st.download_button(
        label="Descargar Ranking de Motricidad (xlsx)",
        data=output,
        file_name="micmac_ranking.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
        # GENERADOR DE INFORME DE INTELIGENCIA (SIN REPORTLAB)
    st.subheader("🎯 Generar Informe de Inteligencia")
    st.markdown("Genera automáticamente un informe ejecutivo completo con análisis estratégico de los resultados MICMAC.")
    
    if st.button("📄 Generar Informe de Inteligencia", type="primary"):
        
        # Análisis automático de resultados
        top_5_motoras = ranking_vars[:5]
        top_3_estrategicas = [nombres[i] for i in np.argsort(strategic_scores)[-3:]][::-1]
        
        # Contar variables por cuadrante
        count_determinantes = sum(1 for label in labels_cuadrante if label == 'Determinantes')
        count_criticas = sum(1 for label in labels_cuadrante if label == 'Crítico/inestable')
        count_resultado = sum(1 for label in labels_cuadrante if label == 'Variables resultado')
        count_autonomas = sum(1 for label in labels_cuadrante if label == 'Autónomas')
        
        # Variables críticas por motricidad
        vars_alta_motricidad = [nombres[i] for i in range(len(nombres)) if motricidad[i] > np.percentile(motricidad, 90)]
        vars_alta_dependencia = [nombres[i] for i in range(len(nombres)) if dependencia[i] > np.percentile(dependencia, 90)]
        
        # Generar contenido del informe
        from datetime import datetime
        fecha_actual = datetime.now().strftime("%d de %B de %Y")
        
        informe_contenido = f"""# INFORME DE INTELIGENCIA ESTRATÉGICA
**Análisis Estructural MICMAC - Sistema Complejo**  
*Generado automáticamente • {fecha_actual}*

---

## RESUMEN EJECUTIVO

El análisis MICMAC realizado sobre **{len(nombres)} variables** del sistema revela patrones estructurales críticos para la toma de decisiones estratégicas. Con parámetros de configuración α={alpha} y K={K_max}, se identificaron **{count_criticas} variables críticas/inestables** y **{count_determinantes} variables determinantes** que requieren atención prioritaria.

**HALLAZGO PRINCIPAL:** Las variables **{top_3_estrategicas[0]}**, **{top_3_estrategicas[1]}** y **{top_3_estrategicas[2]}** emergen como los factores de mayor valor estratégico del sistema.

---

## DIFERENCIAS ENTRE TIPOS DE VARIABLES

### 🔴 VARIABLES DETERMINANTES (Cuadrante Superior Izquierdo)
- **Características:** Alta motricidad + Baja dependencia
- **Interpretación:** Son las **PALANCAS DE CONTROL** del sistema
- **Acción estratégica:** **ACTUAR** - Estas variables son fáciles de controlar y tienen gran impacto
- **Ejemplo:** Políticas, decisiones ejecutivas, inversiones estratégicas
- **Riesgo:** Bajo - Se pueden manejar directamente

### 🔵 VARIABLES CRÍTICAS/INESTABLES (Cuadrante Superior Derecho)  
- **Características:** Alta motricidad + Alta dependencia
- **Interpretación:** Son **AMPLIFICADORES** que magnifican cualquier cambio
- **Acción estratégica:** **MONITOREAR** - Difíciles de controlar pero muy influyentes
- **Ejemplo:** Mercados, tecnologías emergentes, factores regulatorios
- **Riesgo:** Alto - Pueden generar efectos impredecibles

---

## ANÁLISIS DE VARIABLES MOTORAS

### Top 5 Variables con Mayor Influencia Sistémica:

1. **{top_5_motoras[0]}** - Motricidad: {motricidad[ranking_indices[0]]:.0f}
2. **{top_5_motoras[1]}** - Motricidad: {motricidad[ranking_indices[1]]:.0f}  
3. **{top_5_motoras[2]}** - Motricidad: {motricidad[ranking_indices[2]]:.0f}
4. **{top_5_motoras[3]}** - Motricidad: {motricidad[ranking_indices[3]]:.0f}
5. **{top_5_motoras[4]}** - Motricidad: {motricidad[ranking_indices[4]]:.0f}

**IMPLICACIÓN ESTRATÉGICA:** Estas variables constituyen las **palancas de cambio primarias** del sistema. Cualquier modificación en estos factores generará efectos multiplicadores significativos en todo el ecosistema analizado.

---

## CLASIFICACIÓN SISTÉMICA

### Distribución por Cuadrantes MICMAC:

| Categoría | Cantidad | Porcentaje | Interpretación Estratégica |
|-----------|----------|------------|----------------------------|
| **Variables Críticas/Inestables** | {count_criticas} | {count_criticas/len(nombres)*100:.1f}% | Requieren **gestión balanceada** - Alta influencia y alta dependencia |
| **Variables Determinantes** | {count_determinantes} | {count_determinantes/len(nombres)*100:.1f}% | **Palancas de control** - Alta influencia, baja dependencia |
| **Variables Resultado** | {count_resultado} | {count_resultado/len(nombres)*100:.1f}% | **Indicadores de impacto** - Baja influencia, alta dependencia |
| **Variables Autónomas** | {count_autonomas} | {count_autonomas/len(nombres)*100:.1f}% | **Factores independientes** - Baja influencia y dependencia |

---

## VARIABLES DE ALTA CRITICIDAD

### Variables con Motricidad Extrema (Percentil 90+):
{chr(10).join([f"• **{var}**" for var in vars_alta_motricidad[:8]])}

### Variables con Dependencia Extrema (Percentil 90+):
{chr(10).join([f"• **{var}**" for var in vars_alta_dependencia[:8]])}

**ANÁLISIS DE RIESGO:** Las variables con alta dependencia son **vulnerables** a cambios externos y requieren monitoreo continuo como indicadores tempranos de transformaciones sistémicas.

---

## RECOMENDACIONES ESTRATÉGICAS

### PRIORIDAD ALTA - Acción Inmediata
1. **Focalización en Variables Determinantes:** Concentrar recursos en las {count_determinantes} variables determinantes identificadas, especialmente **{top_5_motoras[0]}** como máxima prioridad.

2. **Gestión de Variables Críticas:** Desarrollar planes de contingencia para las {count_criticas} variables crítico/inestables que pueden generar efectos sistémicos impredecibles.

### PRIORIDAD MEDIA - Planificación Táctica  
3. **Monitoreo de Variables Resultado:** Establecer KPIs basados en las {count_resultado} variables resultado como sistema de alerta temprana.

4. **Optimización del Eje Estratégico:** Priorizar inversión en las variables más cercanas al eje estratégico: **{top_3_estrategicas[0]}**, **{top_3_estrategicas[1]}** y **{top_3_estrategicas[2]}**.

### PRIORIDAD BAJA - Gestión Rutinaria
5. **Variables Autónomas:** Las {count_autonomas} variables autónomas pueden gestionarse de forma rutinaria sin impacto sistémico significativo.

---

## ANÁLISIS DE ESCENARIOS

### Escenario Optimista
Si se logra **control efectivo** de las top 5 variables motoras, se proyecta un impacto positivo del **{(sum(motricidad[ranking_indices[:5]])/sum(motricidad)*100):.1f}%** sobre la motricidad total del sistema.

### Escenario de Riesgo  
Las variables con **alta dependencia** ({len(vars_alta_dependencia)} identificadas) son vulnerables a shocks externos. Un impacto negativo simultáneo podría desestabilizar hasta el **{len(vars_alta_dependencia)/len(nombres)*100:.1f}%** del sistema.

### Escenario de Intervención Estratégica
Actuando sobre las **3 variables más estratégicas** identificadas, se puede lograr una influencia controlada y sostenible sobre el **{(sum([motricidad[nombres.index(var)] for var in top_3_estrategicas if var in nombres])/sum(motricidad)*100):.1f}%** de la dinámica sistémica.

---

## INDICADORES CLAVE DE DESEMPEÑO (KPIs)

### KPIs de Control Estratégico:
- **Índice de Motricidad Concentrada:** {(motricidad[ranking_indices[0]]/sum(motricidad)*100):.2f}% (Dominancia de variable líder)
- **Ratio Variables Críticas:** {count_criticas/len(nombres):.3f} (Porcentaje de variables inestables)
- **Coeficiente de Dependencia Media:** {np.mean(dependencia):.2f} (Interconexión sistémica)

### Umbrales de Alerta:
- 🔴 **Crítico:** Si motricidad de variable líder supera 15% del total
- 🟡 **Precaución:** Si más del 30% son variables crítico/inestables  
- 🟢 **Estable:** Distribución equilibrada entre cuadrantes

---

## MATRIZ DE DECISIONES

### Variables para Inversión Prioritaria:
{chr(10).join([f"{i+1}. **{var}** (Motricidad: {motricidad[ranking_indices[i]]:.0f})" for i, var in enumerate(top_5_motoras)])}

### Variables para Monitoreo Especial:
{chr(10).join([f"• **{var}**" for var in vars_alta_dependencia[:5]])}

### Variables de Impacto Estratégico:
{chr(10).join([f"• **{var}**" for var in top_3_estrategicas])}

---

## CONCLUSIONES Y PRÓXIMOS PASOS

**CONCLUSIÓN PRINCIPAL:** El sistema analizado presenta una estructura de **{('alta' if count_criticas > len(nombres)*0.3 else 'media' if count_criticas > len(nombres)*0.15 else 'baja')} complejidad** con {count_criticas} variables críticas que requieren gestión especializada.

**RECOMENDACIÓN OPERATIVA:** Implementar un **sistema de monitoreo continuo** sobre las top 10 variables motoras y desarrollar **planes de intervención específicos** para las variables crítico/inestables identificadas.

**VALIDACIÓN:** Este análisis debe **actualizarse trimestralmente** con nuevos datos para mantener la vigencia de las recomendaciones estratégicas.

---

## METODOLOGÍA APLICADA

- **Algoritmo:** MICMAC extendido con parámetros α={alpha}, K={K_max}
- **Variables analizadas:** {len(nombres)}
- **Criterio de estratégico:** Proximidad al eje estratégico + valor absoluto
- **Umbrales:** Percentiles 80/90 para clasificación crítica
- **Fecha de análisis:** {fecha_actual}

---

*Informe generado automáticamente por Sistema MICMAC Interactivo v2.0*  
*© 2025 - Martín Pratto • Análisis Estructural Avanzado*
"""

        # Crear archivo de texto del informe
        informe_bytes = informe_contenido.encode('utf-8')
        
        st.success("✅ Informe de Inteligencia generado exitosamente!")
        
        # Botones de descarga
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📄 Descargar Informe (TXT)",
                data=informe_bytes,
                file_name=f"informe_inteligencia_micmac_{fecha_actual.replace(' ', '_')}.txt",
                mime="text/plain"
            )
        
        with col2:
            st.download_button(
                label="📋 Descargar Informe (MD)", 
                data=informe_bytes,
                file_name=f"informe_inteligencia_micmac_{fecha_actual.replace(' ', '_')}.md",
                mime="text/markdown"
            )
        
        # Mostrar vista previa del informe en la app
        with st.expander("👁️ Vista Previa del Informe", expanded=True):
            st.markdown(informe_contenido)

else:
    st.info("Por favor suba una matriz Excel para comenzar.")

st.caption("Desarrollado para análisis académico. ©2025")
