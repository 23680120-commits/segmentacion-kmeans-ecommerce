import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ==============================================================================
# CONFIGURACIÓN DE PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Segmentación Inteligente K-Means | E-Commerce",
    page_icon="📊",
    layout="wide"
)

# ==============================================================================
# ENCABEZADO Y CONTEXTO DEL CASO DE ESTUDIO
# ==============================================================================
st.title("📊 Análisis de Agrupamiento y Segmentación Inteligente mediante K-Means")
st.subheader("Caso de Estudio: Segmentación de Clientes en E-Commerce para Campañas de Marketing y Retención")

st.markdown("""
### 🎯 Contexto del Problema de Negocio
Una plataforma de comercio electrónico (*E-Commerce*) desea lanzar una campaña estratégica de incentivos comerciales 
(cupones de descuento, envíos gratuitos y programas de lealtad VIP). Sin embargo, el presupuesto de marketing es limitado 
y otorgar beneficios indiscriminados a toda la base de usuarios reduciría el margen de ganancia de la empresa.

Se dispone de los registros transaccionales de **150 clientes activos durante el último mes**, con dos variables clave:
1. **`frecuencia_compras_mes`**: Número total de pedidos realizados por el cliente al mes (rango: $1$ a $20$ pedidos).
2. **`gasto_promedio_mxn`**: Ticket promedio de compra por pedido en pesos mexicanos (rango: $\\$250$ a $\\$12,000$ MXN).

**Reto de Ingeniería de Datos:** Descubrir cuántos perfiles naturales de compradores existen sin etiquetas previas 
(aprendizaje no supervisado), caracterizar matemáticamente cada segmento y asignar la estrategia comercial óptima.
""")

st.divider()

# ==============================================================================
# PASO 1: GENERACIÓN Y EXPLORACIÓN INICIAL DE LA TELEMETRÍA
# ==============================================================================
st.header("Paso 1: Generación y Exploración Inicial de la Telemetría")
st.markdown("""
**Acción Técnica:** Generamos un conjunto de datos reproducible con semilla fija (`np.random.seed(42)`) 
de **150 clientes** y construimos un diagrama de dispersión inicial sin etiquetas para observar si los datos 
forman agrupaciones o *"islas"* naturales.
""")

@st.cache_data
def generar_datos():
    np.random.seed(42)
    # Grupo latente 1: Compradores esporádicos
    freq_a = np.clip(np.random.normal(loc=2.2, scale=0.7, size=55), 1, 5).round(0)
    gasto_a = np.clip(np.random.normal(loc=650, scale=160, size=55), 250, 1200).round(2)

    # Grupo latente 2: Cazadores de volumen
    freq_b = np.clip(np.random.normal(loc=14.5, scale=1.6, size=50), 10, 20).round(0)
    gasto_b = np.clip(np.random.normal(loc=1150, scale=220, size=50), 600, 1800).round(2)

    # Grupo latente 3: Clientes VIP / Alto valor
    freq_c = np.clip(np.random.normal(loc=5.5, scale=1.2, size=45), 3, 9).round(0)
    gasto_c = np.clip(np.random.normal(loc=8800, scale=950, size=45), 6500, 11500).round(2)

    frecuencia_total = np.concatenate([freq_a, freq_b, freq_c])
    gasto_total = np.concatenate([gasto_a, gasto_b, gasto_c])
    indices = np.random.permutation(len(frecuencia_total))

    df = pd.DataFrame({
        "cliente_id": [f"CLI-{i+1:03d}" for i in range(len(frecuencia_total))],
        "frecuencia_compras_mes": frecuencia_total[indices].astype(int),
        "gasto_promedio_mxn": gasto_total[indices]
    })
    return df

df_clientes = generar_datos()

col1_p1, col2_p1 = st.columns([1, 1.4])
with col1_p1:
    st.markdown("**Muestra de registros crudos (sin etiquetas):**")
    st.dataframe(df_clientes.head(10), use_container_width=True, hide_index=True)
    st.markdown("**Estadísticas descriptivas:**")
    st.dataframe(
        df_clientes[["frecuencia_compras_mes", "gasto_promedio_mxn"]].describe().round(2),
        use_container_width=True
    )

with col2_p1:
    fig_p1 = px.scatter(
        df_clientes,
        x="frecuencia_compras_mes",
        y="gasto_promedio_mxn",
        hover_data=["cliente_id"],
        title="Exploración Visual Inicial (Nube de Puntos sin Etiquetas)",
        labels={
            "frecuencia_compras_mes": "Frecuencia de Compras al Mes (Pedidos)",
            "gasto_promedio_mxn": "Gasto Promedio por Pedido (MXN)"
        },
        color_discrete_sequence=["#34495E"]
    )
    fig_p1.update_traces(marker=dict(size=9, opacity=0.8, line=dict(width=1, color="white")))
    fig_p1.update_layout(height=470)
    st.plotly_chart(fig_p1, use_container_width=True)

st.info("🔍 **Hallazgo Paso 1:** Se distinguen visualmente 3 concentraciones naturales y una diferencia crítica de magnitudes (pedidos de 1 a 18 vs. gasto hasta $10,826 MXN).")

st.divider()

# ==============================================================================
# PASO 2: PREPROCESAMIENTO Y ESCALAMIENTO DE VARIABLES
# ==============================================================================
st.header("Paso 2: Preprocesamiento y Escalamiento de Variables (Z-Score)")
st.markdown("""
**El Problema Técnico:** K-Means calcula distancias geométricas Euclidianas:
$$d(p, q) = \\sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$$
Si no escalamos las variables, `gasto_promedio_mxn` (en miles de pesos) aplastaría matemáticamente a `frecuencia_compras_mes` (en unidades).

**Acción Técnica:** Aplicamos estandarización **Z-Score** (`StandardScaler`) para transformar ambas variables a **media $\\mu = 0$** y **desviación estándar $\\sigma = 1$**:
$$z = \\frac{x - \\mu}{\\sigma}$$
""")

features = ["frecuencia_compras_mes", "gasto_promedio_mxn"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clientes[features])
df_scaled = pd.DataFrame(X_scaled, columns=["frecuencia_z", "gasto_z"])

st.markdown("**Verificación tras aplicar `StandardScaler` (Media = 0, Desviación Estándar = 1):**")
st.dataframe(df_scaled.describe().loc[["mean", "std", "min", "max"]].round(4), use_container_width=True)

st.divider()

# ==============================================================================
# PASO 3: BÚSQUEDA DEL NÚMERO ÓPTIMO DE GRUPOS (k) MEDIANTE SILUETA
# ==============================================================================
st.header("Paso 3: Búsqueda del Número Óptimo de Grupos (k) mediante Coeficiente de Silueta")
st.markdown("""
Evaluamos sistemáticamente desde **$k = 2$ hasta $k = 6$** calculando el **Coeficiente de Silueta promedio** 
(que oscila entre $-1$ y $+1$; valores cercanos a $+1$ indican clústeres densos y bien separados):
$$s(i) = \\frac{b(i) - a(i)}{\\max(a(i), b(i))}$$
""")

resultados_k = []
for k in range(2, 7):
    km_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_temp = km_temp.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels_temp)
    resultados_k.append({
        "k": k,
        "Coeficiente_Silueta": round(sil, 4),
        "Inercia_WCSS": round(km_temp.inertia_, 2)
    })

df_silueta = pd.DataFrame(resultados_k)
k_optimo = int(df_silueta.loc[df_silueta["Coeficiente_Silueta"].idxmax(), "k"])
mejor_silueta = df_silueta["Coeficiente_Silueta"].max()

col1_p3, col2_p3 = st.columns([1, 1.5])
with col1_p3:
    st.markdown("**Evaluación comparativa por candidato $k$:**")
    st.dataframe(df_silueta, use_container_width=True, hide_index=True)
    st.success(f"🏆 **Óptimo seleccionado:** $k = {k_optimo}$ con Coeficiente de Silueta = **{mejor_silueta:.4f}**")

with col2_p3:
    fig_sil = px.line(
        df_silueta,
        x="k",
        y="Coeficiente_Silueta",
        markers=True,
        text="Coeficiente_Silueta",
        title="Evolución del Coeficiente de Silueta (k = 2 hasta k = 6)",
        labels={"k": "Número de Clústeres (k)", "Coeficiente_Silueta": "Coeficiente de Silueta Promedio"}
    )
    fig_sil.update_traces(
        line=dict(color="#2980B9", width=3),
        marker=dict(size=11, color="#E74C3C"),
        textposition="top center"
    )
    fig_sil.update_layout(xaxis=dict(dtick=1), yaxis=dict(range=[0.45, 0.96]), height=360)
    st.plotly_chart(fig_sil, use_container_width=True)

st.divider()

# ==============================================================================
# PASO 4: ENTRENAMIENTO DEL MODELO DEFINITIVO Y ASIGNACIÓN DE ETIQUETAS
# ==============================================================================
st.header("Paso 4: Entrenamiento del Modelo Definitivo y Cálculo de Centroides")
st.markdown(f"""
Entrenamos el modelo definitivo `KMeans(n_clusters={k_optimo}, random_state=42)` sobre los datos estandarizados, 
asignamos la etiqueta de clúster a cada uno de los 150 clientes y aplicamos `scaler.inverse_transform()` 
para obtener los centroides en sus unidades originales.
""")

kmeans_final = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)
df_clientes["cluster_id"] = kmeans_final.fit_predict(X_scaled)

centroides_orig = scaler.inverse_transform(kmeans_final.cluster_centers_)
df_centroides = pd.DataFrame(
    centroides_orig,
    columns=["centroide_frecuencia_mes", "centroide_gasto_mxn"]
)
df_centroides["cluster_id"] = range(k_optimo)

col1_p4, col2_p4 = st.columns(2)
with col1_p4:
    st.markdown("**Coordenadas exactas de los Centroides (unidades reales):**")
    st.dataframe(
        df_centroides[["cluster_id", "centroide_frecuencia_mes", "centroide_gasto_mxn"]].round(2),
        use_container_width=True,
        hide_index=True
    )
with col2_p4:
    st.markdown("**Muestra de clientes etiquetados por el modelo:**")
    st.dataframe(df_clientes.head(6), use_container_width=True, hide_index=True)

st.divider()

# ==============================================================================
# PASO 5: PERFILAMIENTO DE NEGOCIO Y TOMA DE DECISIONES (STORYTELLING)
# ==============================================================================
st.header("Paso 5: Perfilamiento de Negocio y Toma de Decisiones (Storytelling con Datos)")

perfilamiento = df_clientes.groupby("cluster_id").agg(
    cantidad_clientes=("cliente_id", "count"),
    frecuencia_promedio=("frecuencia_compras_mes", "mean"),
    ticket_promedio_mxn=("gasto_promedio_mxn", "mean")
).reset_index()

perfilamiento["porcentaje_clientes"] = (perfilamiento["cantidad_clientes"] / len(df_clientes)) * 100
perfilamiento["ingreso_mensual_cliente_mxn"] = (
    perfilamiento["frecuencia_promedio"] * perfilamiento["ticket_promedio_mxn"]
)

nombres_perfil = {}
estrategias = {}
for _, row in perfilamiento.iterrows():
    cid = int(row["cluster_id"])
    if row["ticket_promedio_mxn"] > 5000:
        nombres_perfil[cid] = "Clientes VIP / Alto Valor"
        estrategias[cid] = "Programa Lealtad VIP, acceso anticipado y atención prioritaria"
    elif row["frecuencia_promedio"] > 10:
        nombres_perfil[cid] = "Cazadores de Volumen"
        estrategias[cid] = "Envío gratis en pedido mínimo >$1,500 MXN (Upselling)"
    else:
        nombres_perfil[cid] = "Compradores Esporádicos"
        estrategias[cid] = "Boletín automatizado sin subsidio promocional"

df_clientes["perfil_cliente"] = df_clientes["cluster_id"].map(nombres_perfil)
df_clientes["estrategia_marketing"] = df_clientes["cluster_id"].map(estrategias)
df_centroides["perfil_cliente"] = df_centroides["cluster_id"].map(nombres_perfil)

perfilamiento["perfil_cliente"] = perfilamiento["cluster_id"].map(nombres_perfil)
perfilamiento["estrategia_marketing"] = perfilamiento["cluster_id"].map(estrategias)

st.dataframe(
    perfilamiento[[
        "cluster_id", "perfil_cliente", "cantidad_clientes", "porcentaje_clientes",
        "frecuencia_promedio", "ticket_promedio_mxn", "ingreso_mensual_cliente_mxn",
        "estrategia_marketing"
    ]].round(2),
    use_container_width=True,
    hide_index=True
)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""
    #### 🔘 Perfil 1: Compradores Esporádicos
    * **Comportamiento:** ~2.0 pedidos/mes | Ticket ~$649 MXN.
    * **Decisión:** **No asignar subsidios.** Mantener en flujo automatizado de *email marketing* sin costo promocional.
    """)
with c2:
    st.markdown("""
    #### 🔵 Perfil 2: Cazadores de Volumen
    * **Comportamiento:** ~14.5 pedidos/mes | Ticket ~$1,185 MXN.
    * **Decisión:** **Membresía de envío gratis** condicionada a compras mayores a $1,500 MXN para elevar su ticket promedio.
    """)
with c3:
    st.markdown("""
    #### 🟢 Perfil 3: Clientes VIP / Alto Valor
    * **Comportamiento:** ~5.7 pedidos/mes | Ticket ~$8,768 MXN.
    * **Decisión:** **Prioridad máxima de retención.** Acceso anticipado a lanzamientos, *cashback* y atención personalizada.
    """)

st.divider()

# ==============================================================================
# PASO 6: VISUALIZACIÓN FINAL Y DASHBOARD EJECUTIVO INTERACTIVO
# ==============================================================================
st.header("Paso 6: Dashboard Ejecutivo Integral de Segmentación")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Clientes Analizados", "150", "100% Base Activa")
k2.metric("Clústeres Óptimos (k)", f"{k_optimo} Perfiles", f"Silueta: {mejor_silueta:.3f}")
k3.metric("Ticket Promedio VIP", "$8,768 MXN", "30.0% de la cartera")
k4.metric("Frecuencia Cazadores", "14.5 ped/mes", "33.3% de la cartera")

paleta_colores = {
    "Compradores Esporádicos": "#7F8C8D",
    "Cazadores de Volumen": "#2980B9",
    "Clientes VIP / Alto Valor": "#27AE60"
}

dash_col1, dash_col2 = st.columns([1.45, 1])

with dash_col1:
    fig_clusters = px.scatter(
        df_clientes,
        x="frecuencia_compras_mes",
        y="gasto_promedio_mxn",
        color="perfil_cliente",
        color_discrete_map=paleta_colores,
        hover_data=["cliente_id", "estrategia_marketing"],
        title="Mapa Final de Clústeres y Ubicación de Centroides (k = 3)",
        labels={
            "frecuencia_compras_mes": "Frecuencia de Compras al Mes (Pedidos)",
            "gasto_promedio_mxn": "Gasto Promedio por Pedido (MXN)",
            "perfil_cliente": "Segmento"
        }
    )
    fig_clusters.update_traces(marker=dict(size=9, opacity=0.88, line=dict(width=0.8, color="white")))

    fig_clusters.add_trace(
        go.Scatter(
            x=df_centroides["centroide_frecuencia_mes"],
            y=df_centroides["centroide_gasto_mxn"],
            mode="markers+text",
            name="Centroides",
            text=[f"★ {p}" for p in df_centroides["perfil_cliente"]],
            textposition="top center",
            marker=dict(
                symbol="star",
                size=18,
                color="#F1C40F",
                line=dict(width=1.5, color="black")
            )
        )
    )
    fig_clusters.update_layout(height=470, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_clusters, use_container_width=True)

with dash_col2:
    orden_perfiles = ["Compradores Esporádicos", "Cazadores de Volumen", "Clientes VIP / Alto Valor"]
    perf_ord = perfilamiento.set_index("perfil_cliente").loc[orden_perfiles].reset_index()

    fig_donut = px.pie(
        perf_ord,
        names="perfil_cliente",
        values="cantidad_clientes",
        color="perfil_cliente",
        color_discrete_map=paleta_colores,
        hole=0.48,
        title="Distribución de la Cartera (150 Clientes)"
    )
    fig_donut.update_traces(textinfo="percent+label+value", textposition="inside")
    fig_donut.update_layout(height=470, showlegend=False)
    st.plotly_chart(fig_donut, use_container_width=True)

dash_col3, dash_col4 = st.columns(2)

with dash_col3:
    fig_ticket = px.bar(
        perf_ord,
        x="perfil_cliente",
        y="ticket_promedio_mxn",
        color="perfil_cliente",
        color_discrete_map=paleta_colores,
        text_auto="$.2s",
        title="Ticket Promedio por Segmento (MXN)",
        labels={"perfil_cliente": "Segmento", "ticket_promedio_mxn": "Ticket Promedio (MXN)"}
    )
    fig_ticket.update_layout(height=390, showlegend=False)
    st.plotly_chart(fig_ticket, use_container_width=True)

with dash_col4:
    fig_freq = px.bar(
        perf_ord,
        x="perfil_cliente",
        y="frecuencia_promedio",
        color="perfil_cliente",
        color_discrete_map=paleta_colores,
        text_auto=".1f",
        title="Frecuencia Media de Compra (Pedidos / Mes)",
        labels={"perfil_cliente": "Segmento", "frecuencia_promedio": "Pedidos por Mes"}
    )
    fig_freq.update_layout(height=390, showlegend=False)
    st.plotly_chart(fig_freq, use_container_width=True)

st.download_button(
    label="📥 Descargar Base de Clientes Segmentada (CSV)",
    data=df_clientes.to_csv(index=False).encode("utf-8"),
    file_name="clientes_segmentados_kmeans.csv",
    mime="text/csv"
)
