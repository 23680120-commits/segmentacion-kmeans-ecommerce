import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ==============================================================================
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILOS VISUALES (AZUL MARINO Y GRIS EJECUTIVO)
# ==============================================================================
st.set_page_config(
    page_title="Segmentación Inteligente K-Means | E-Commerce",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Fondo general gris claro ejecutivo */
    .stApp {
        background-color: #F1F5F9;
        color: #1E293B;
    }

    /* Barra lateral en Azul Marino Profundo */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #09192F 0%, #132C4E 100%);
        color: #F8FAFC;
    }
    section[data-testid="stSidebar"] * {
        color: #F1F5F9 !important;
    }
    section[data-testid="stSidebar"] .stAlert {
        background-color: rgba(255, 255, 255, 0.08);
        border: 1px solid #334155;
    }

    /* Banner Principal (Hero) */
    .hero-banner {
        background: linear-gradient(135deg, #09192F 0%, #163A6B 60%, #1E40AF 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        color: #FFFFFF;
        box-shadow: 0 10px 25px rgba(9, 25, 47, 0.22);
        margin-bottom: 1.5rem;
        border-bottom: 5px solid #64748B;
    }
    .hero-badge {
        display: inline-block;
        background-color: rgba(255, 255, 255, 0.14);
        border: 1px solid rgba(255, 255, 255, 0.28);
        color: #E2E8F0;
        padding: 0.28rem 0.85rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-right: 0.5rem;
        margin-bottom: 0.8rem;
    }
    .hero-title {
        font-size: 2.15rem;
        font-weight: 800;
        margin: 0.2rem 0 0.5rem 0;
        line-height: 1.2;
        color: #FFFFFF;
    }
    .hero-subtitle {
        font-size: 1.08rem;
        color: #CBD5E1;
        margin: 0;
        line-height: 1.5;
    }

    /* Tarjetas de Contexto y Explicación Sencilla */
    .context-card {
        background-color: #FFFFFF;
        border-left: 6px solid #0F294A;
        border-top: 1px solid #CBD5E1;
        border-right: 1px solid #CBD5E1;
        border-bottom: 1px solid #CBD5E1;
        border-radius: 12px;
        padding: 1.35rem 1.6rem;
        box-shadow: 0 4px 12px rgba(15, 41, 74, 0.06);
        margin-bottom: 1.1rem;
    }
    .analogy-box {
        background: linear-gradient(90deg, #E2E8F0 0%, #F8FAFC 100%);
        border-left: 5px solid #475569;
        border-radius: 10px;
        padding: 1rem 1.3rem;
        margin: 0.8rem 0 1.2rem 0;
        color: #1E293B;
    }
    .step-header {
        background: linear-gradient(90deg, #0F294A 0%, #1E3A6E 100%);
        color: #FFFFFF;
        padding: 0.85rem 1.4rem;
        border-radius: 10px;
        font-size: 1.28rem;
        font-weight: 700;
        margin: 1.4rem 0 1rem 0;
        box-shadow: 0 4px 10px rgba(15, 41, 74, 0.14);
        border-left: 6px solid #94A3B8;
    }

    /* Tarjetas de Perfiles de Negocio */
    .profile-card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 1.3rem;
        border: 1px solid #CBD5E1;
        box-shadow: 0 4px 12px rgba(15, 41, 74, 0.06);
        height: 100%;
    }
    .profile-tag {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 0.6rem;
    }

    /* Botones Llamativos en Azul Marino y Gris Metálico */
    div.stButton > button, div.stDownloadButton > button {
        background: linear-gradient(135deg, #0B1F3A 0%, #1D4ED8 100%) !important;
        color: #FFFFFF !important;
        border: 2px solid #93C5FD !important;
        border-radius: 10px !important;
        padding: 0.65rem 1.35rem !important;
        font-weight: 700 !important;
        font-size: 0.96rem !important;
        letter-spacing: 0.3px !important;
        box-shadow: 0 4px 14px rgba(11, 31, 58, 0.28) !important;
        transition: all 0.22s ease-in-out !important;
        width: 100%;
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%) !important;
        border-color: #FFFFFF !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 7px 20px rgba(29, 78, 216, 0.42) !important;
    }
    div.stButton > button:active, div.stDownloadButton > button:active {
        transform: translateY(0px) !important;
    }

    /* Tarjetas de Métricas (KPIs) */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border-top: 4px solid #0F294A;
        border-right: 1px solid #CBD5E1;
        border-bottom: 1px solid #CBD5E1;
        border-left: 1px solid #CBD5E1;
        padding: 1rem 1.2rem;
        border-radius: 12px;
        box-shadow: 0 3px 10px rgba(15, 41, 74, 0.05);
    }
    div[data-testid="stMetricLabel"] {
        color: #475569 !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #0B1F3A !important;
        font-weight: 800 !important;
    }

    /* Pestañas estilizadas */
    button[data-baseweb="tab"] {
        background-color: #E2E8F0 !important;
        color: #1E293B !important;
        border-radius: 8px 8px 0 0 !important;
        font-weight: 700 !important;
        padding: 0.6rem 1.2rem !important;
        margin-right: 4px !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #0F294A !important;
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. BARRA LATERAL DE NAVEGACIÓN Y RESUMEN RÁPIDO
# ==============================================================================
with st.sidebar:
    st.markdown("## 🧭 Guía del Proyecto")
    st.markdown("""
    Esta aplicación demuestra cómo una empresa de **Comercio Electrónico** utiliza **Inteligencia Artificial (K-Means)** 
    para conocer a sus clientes y no desperdiciar presupuesto de publicidad.
    """)
    st.markdown("---")
    st.markdown("### 📑 Mapa de los 6 Pasos")
    st.markdown("""
    - **Paso 1:** Datos crudos (150 clientes)
    - **Paso 2:** Nivelación de escalas (Z-Score)
    - **Paso 3:** Elección de grupos (Silueta)
    - **Paso 4:** Entrenamiento y Centroides
    - **Paso 5:** Estrategias de Negocio
    - **Paso 6:** Dashboard y Simulador
    """)
    st.markdown("---")
    st.info("💡 **Tip de exposición:** Usa los botones azules interactivos en cada sección y prueba el **Simulador de Clientes** al final.")

# ==============================================================================
# 3. BANNER PRINCIPAL E INFOGRAFÍA VISUAL DE CONTEXTO (PARA TODO PÚBLICO)
# ==============================================================================
st.markdown("""
<div class="hero-banner">
    <div>
        <span class="hero-badge">UNIDAD 5 • APRENDIZAJE NO SUPERVISADO</span>
        <span class="hero-badge">ALGORITMO K-MEANS</span>
        <span class="hero-badge">INTELIGENCIA DE NEGOCIOS</span>
    </div>
    <div class="hero-title">📊 Segmentación Inteligente de Clientes en E-Commerce</div>
    <p class="hero-subtitle">
        Cómo transformar datos desordenados de compras en decisiones comerciales rentables mediante agrupamiento matemático automático.
    </p>
</div>
""", unsafe_allow_html=True)

# Infografía SVG integrada: Explica todo el proyecto visualmente en 3 bloques
st.markdown("""
<div class="context-card">
    <h3 style="color:#0B1F3A; margin-top:0;">🎯 ¿De qué trata este proyecto? (Explicación Visual Rápida)</h3>
    <p style="color:#334155; font-size:1.02rem;">
        Imagina una tienda en línea con <b>150 clientes</b> y un <b>presupuesto limitado</b> para promociones. 
        Si regala envíos gratis o cupones a todos por igual, pierde ganancias. En lugar de adivinar, usamos el algoritmo 
        <b>K-Means</b> para analizar cuánto compra cada persona y agruparlas automáticamente según su comportamiento real:
    </p>
    <svg viewBox="0 0 960 195" width="100%" style="max-height:210px; margin-top:0.5rem;" xmlns="http://www.w3.org/2000/svg">
        <rect x="2" y="2" width="956" height="191" rx="14" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="2"/>
        
        <!-- BLOQUE 1: EL PROBLEMA -->
        <rect x="22" y="22" width="265" height="150" rx="12" fill="#FFFFFF" stroke="#64748B" stroke-width="2"/>
        <rect x="22" y="22" width="265" height="36" rx="10" fill="#475569"/>
        <text x="154" y="45" fill="#FFFFFF" font-family="sans-serif" font-size="14" font-weight="bold" text-anchor="middle">1. EL PROBLEMA (DATOS CRUDOS)</text>
        <circle cx="75" cy="95" r="7" fill="#64748B"/>
        <circle cx="110" cy="82" r="7" fill="#94A3B8"/>
        <circle cx="95" cy="125" r="7" fill="#475569"/>
        <circle cx="145" cy="110" r="7" fill="#64748B"/>
        <circle cx="130" cy="145" r="7" fill="#94A3B8"/>
        <text x="215" y="102" fill="#0F294A" font-family="sans-serif" font-size="13" font-weight="bold" text-anchor="middle">150 Clientes</text>
        <text x="215" y="121" fill="#475569" font-family="sans-serif" font-size="11.5" text-anchor="middle">Sin etiquetas previas</text>
        <text x="215" y="140" fill="#475569" font-family="sans-serif" font-size="11.5" text-anchor="middle">(¿Quién es ahorrador</text>
        <text x="215" y="155" fill="#475569" font-family="sans-serif" font-size="11.5" text-anchor="middle">y quién es VIP?)</text>

        <!-- FLECHA 1 -->
        <path d="M 297 97 L 335 97" stroke="#0F294A" stroke-width="4" stroke-linecap="round"/>
        <polygon points="335,89 351,97 335,105" fill="#0F294A"/>

        <!-- BLOQUE 2: INTELIGENCIA ARTIFICIAL -->
        <rect x="360" y="22" width="240" height="150" rx="12" fill="#0B1F3A" stroke="#1D4ED8" stroke-width="2.5"/>
        <rect x="360" y="22" width="240" height="36" rx="10" fill="#1D4ED8"/>
        <text x="480" y="45" fill="#FFFFFF" font-family="sans-serif" font-size="14" font-weight="bold" text-anchor="middle">2. MOTOR K-MEANS (IA)</text>
        <text x="480" y="86" fill="#E2E8F0" font-family="sans-serif" font-size="13" font-weight="bold" text-anchor="middle">📐 Mide 2 Variables Clave:</text>
        <text x="480" y="110" fill="#93C5FD" font-family="sans-serif" font-size="12.5" text-anchor="middle">• Pedidos por mes (1 a 20)</text>
        <text x="480" y="131" fill="#93C5FD" font-family="sans-serif" font-size="12.5" text-anchor="middle">• Ticket promedio ($250 a $12k)</text>
        <text x="480" y="156" fill="#CBD5E1" font-family="sans-serif" font-size="11.5" font-style="italic" text-anchor="middle">Agrupa por cercanía matemática</text>

        <!-- FLECHA 2 -->
        <path d="M 610 97 L 648 97" stroke="#0F294A" stroke-width="4" stroke-linecap="round"/>
        <polygon points="648,89 664,97 648,105" fill="#0F294A"/>

        <!-- BLOQUE 3: LA SOLUCIÓN -->
        <rect x="673" y="22" width="265" height="150" rx="12" fill="#FFFFFF" stroke="#0F294A" stroke-width="2"/>
        <rect x="673" y="22" width="265" height="36" rx="10" fill="#0F294A"/>
        <text x="805" y="45" fill="#FFFFFF" font-family="sans-serif" font-size="14" font-weight="bold" text-anchor="middle">3. DECISIÓN INTELIGENTE</text>
        
        <circle cx="698" cy="82" r="7" fill="#64748B"/>
        <text x="713" y="86" fill="#1E293B" font-family="sans-serif" font-size="12" font-weight="bold">Esporádicos → Sin subsidio</text>
        
        <circle cx="698" cy="113" r="7" fill="#2563EB"/>
        <text x="713" y="117" fill="#1E293B" font-family="sans-serif" font-size="12" font-weight="bold">Volumen → Envío gratis >$1,500</text>
        
        <circle cx="698" cy="144" r="7" fill="#0B1F3A"/>
        <text x="713" y="148" fill="#1E293B" font-family="sans-serif" font-size="12" font-weight="bold">Clientes VIP → Programa Lealtad</text>
    </svg>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# CARGA Y PROCESAMIENTO REPRODUCIBLE DE DATOS
# ==============================================================================
@st.cache_data
def ejecutar_pipeline_kmeans():
    np.random.seed(42)
    # Grupo latente 1: Compradores esporádicos (55 clientes)
    freq_a = np.clip(np.random.normal(loc=2.2, scale=0.7, size=55), 1, 5).round(0)
    gasto_a = np.clip(np.random.normal(loc=650, scale=160, size=55), 250, 1200).round(2)

    # Grupo latente 2: Cazadores de volumen (50 clientes)
    freq_b = np.clip(np.random.normal(loc=14.5, scale=1.6, size=50), 10, 20).round(0)
    gasto_b = np.clip(np.random.normal(loc=1150, scale=220, size=50), 600, 1800).round(2)

    # Grupo latente 3: Clientes VIP / Alto valor (45 clientes)
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

    # Paso 2: Estandarización Z-Score
    features = ["frecuencia_compras_mes", "gasto_promedio_mxn"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    df_scaled = pd.DataFrame(X_scaled, columns=["frecuencia_z", "gasto_z"])

    # Paso 3: Silueta k=2..6
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

    # Paso 4: Modelo final y centroides
    kmeans_final = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)
    df["cluster_id"] = kmeans_final.fit_predict(X_scaled)

    centroides_orig = scaler.inverse_transform(kmeans_final.cluster_centers_)
    df_centroides = pd.DataFrame(
        centroides_orig,
        columns=["centroide_frecuencia_mes", "centroide_gasto_mxn"]
    )
    df_centroides["cluster_id"] = range(k_optimo)

    # Paso 5: Perfilamiento de negocio
    perfilamiento = df.groupby("cluster_id").agg(
        cantidad_clientes=("cliente_id", "count"),
        frecuencia_promedio=("frecuencia_compras_mes", "mean"),
        ticket_promedio_mxn=("gasto_promedio_mxn", "mean")
    ).reset_index()

    perfilamiento["porcentaje_clientes"] = (perfilamiento["cantidad_clientes"] / len(df)) * 100
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

    df["perfil_cliente"] = df["cluster_id"].map(nombres_perfil)
    df["estrategia_marketing"] = df["cluster_id"].map(estrategias)
    df_centroides["perfil_cliente"] = df_centroides["cluster_id"].map(nombres_perfil)
    perfilamiento["perfil_cliente"] = perfilamiento["cluster_id"].map(nombres_perfil)
    perfilamiento["estrategia_marketing"] = perfilamiento["cluster_id"].map(estrategias)

    return df, df_scaled, scaler, kmeans_final, df_silueta, k_optimo, df_centroides, perfilamiento

(
    df_clientes, df_scaled, scaler, kmeans_final,
    df_silueta, k_optimo, df_centroides, perfilamiento
) = ejecutar_pipeline_kmeans()

mejor_silueta = df_silueta["Coeficiente_Silueta"].max()

# ==============================================================================
# PASO 1: GENERACIÓN Y EXPLORACIÓN INICIAL DE LA TELEMETRÍA
# ==============================================================================
st.markdown('<div class="step-header">Paso 1: Exploración Inicial de los Datos (Sin Clasificar)</div>', unsafe_allow_html=True)

st.markdown("""
<div class="analogy-box">
    <b>💡 En palabras sencillas:</b> Antes de usar Inteligencia Artificial, observamos los datos crudos de los <b>150 clientes</b> 
    en un mapa de puntos. Todos los puntos aparecen en <b>gris</b> porque todavía no tienen etiqueta, pero a simple vista 
    ya se notan tres zonas o <i>"islas"</i> donde se concentran los compradores.
</div>
""", unsafe_allow_html=True)

col1_p1, col2_p1 = st.columns([1, 1.45])
with col1_p1:
    st.markdown("**📋 Muestra de registros crudos (primeros 10 clientes):**")
    st.dataframe(df_clientes[["cliente_id", "frecuencia_compras_mes", "gasto_promedio_mxn"]].head(10), use_container_width=True, hide_index=True)
    st.markdown("**📊 Resumen estadístico:**")
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
        title="Mapa Inicial de 150 Clientes (Nube de Puntos sin Etiquetas)",
        labels={
            "frecuencia_compras_mes": "Frecuencia de Compras al Mes (Pedidos)",
            "gasto_promedio_mxn": "Gasto Promedio por Pedido (MXN)"
        },
        color_discrete_sequence=["#475569"]
    )
    fig_p1.update_traces(marker=dict(size=10, opacity=0.82, line=dict(width=1.2, color="#0F294A")))
    fig_p1.update_layout(
        height=475,
        plot_bgcolor="#F8FAFC",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#1E293B")
    )
    st.plotly_chart(fig_p1, use_container_width=True)

# ==============================================================================
# PASO 2: PREPROCESAMIENTO Y ESCALAMIENTO DE VARIABLES
# ==============================================================================
st.markdown('<div class="step-header">Paso 2: Nivelación de Escalas (Estandarización Z-Score)</div>', unsafe_allow_html=True)

col_analogy_p2, col_svg_p2 = st.columns([1.2, 1])
with col_analogy_p2:
    st.markdown("""
    <div class="analogy-box">
        <b>⚖️ ¿Por qué hacemos esto? (Analogía de la balanza):</b><br>
        El algoritmo K-Means mide distancias matemáticas entre puntos. Si comparamos <b>15 pedidos</b> contra <b>$10,000 pesos</b> 
        sin ajustar la escala, el dinero pesaría mil veces más y el algoritmo ignoraría cuántas veces compra el cliente.<br><br>
        Con <b>StandardScaler (Z-Score)</b> convertimos ambas variables a una misma regla justa con <b>media = 0</b> y <b>desviación estándar = 1</b>:
    </div>
    """, unsafe_allow_html=True)
    st.latex(r"z = \frac{x - \mu}{\sigma}")

with col_svg_p2:
    st.markdown("""
    <div class="context-card" style="padding:0.9rem;">
        <svg viewBox="0 0 420 175" width="100%" xmlns="http://www.w3.org/2000/svg">
            <text x="210" y="20" fill="#0B1F3A" font-family="sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Comparación Visual de Escalas</text>
            <!-- Sin Escalar -->
            <rect x="15" y="34" width="390" height="56" rx="8" fill="#F1F5F9" stroke="#94A3B8"/>
            <text x="25" y="54" fill="#475569" font-family="sans-serif" font-size="11.5" font-weight="bold">❌ Sin Escalar (Injusto):</text>
            <rect x="25" y="64" width="18" height="14" rx="3" fill="#64748B"/>
            <text x="48" y="75" fill="#334155" font-family="sans-serif" font-size="10.5">Pedidos (1-18)</text>
            <rect x="155" y="64" width="235" height="14" rx="3" fill="#0F294A"/>
            <text x="272" y="75" fill="#FFFFFF" font-family="sans-serif" font-size="10.5" font-weight="bold" text-anchor="middle">Pesos MXN ($250 - $10,826)</text>
            <!-- Con Z-Score -->
            <rect x="15" y="102" width="390" height="60" rx="8" fill="#EFF6FF" stroke="#1D4ED8"/>
            <text x="25" y="122" fill="#0B1F3A" font-family="sans-serif" font-size="11.5" font-weight="bold">✅ Con Z-Score (Equilibrado):</text>
            <rect x="25" y="133" width="175" height="16" rx="4" fill="#3B82F6"/>
            <text x="112" y="145" fill="#FFFFFF" font-family="sans-serif" font-size="10.5" font-weight="bold" text-anchor="middle">Pedidos (-1.15 a +1.96)</text>
            <rect x="210" y="133" width="180" height="16" rx="4" fill="#0B1F3A"/>
            <text x="300" y="145" fill="#FFFFFF" font-family="sans-serif" font-size="10.5" font-weight="bold" text-anchor="middle">Gasto (-0.83 a +2.07)</text>
        </svg>
    </div>
    """, unsafe_allow_html=True)

st.dataframe(df_scaled.describe().loc[["mean", "std", "min", "max"]].round(4), use_container_width=True)

# ==============================================================================
# PASO 3: BÚSQUEDA DEL NÚMERO ÓPTIMO DE GRUPOS (k) MEDIANTE SILUETA
# ==============================================================================
st.markdown('<div class="step-header">Paso 3: ¿Cuántos Tipos de Clientes Existen? (Coeficiente de Silueta)</div>', unsafe_allow_html=True)

st.markdown("""
<div class="analogy-box">
    <b>🔍 En palabras sencillas:</b> Como nadie nos dice de antemano si hay 2, 3, 4 o 5 tipos de clientes, ponemos a competir 
    distintas cantidades de grupos (<b>k = 2 hasta k = 6</b>) usando el <b>Coeficiente de Silueta</b> (va de -1 a +1). 
    Mientras más cerca esté de <b>+1</b>, significa que los grupos quedaron muy compactos por dentro y bien separados entre sí.
</div>
""", unsafe_allow_html=True)

col1_p3, col2_p3 = st.columns([1, 1.5])
with col1_p3:
    st.markdown("**📊 Competencia de candidatos ($k=2$ a $k=6$):**")
    st.dataframe(df_silueta, use_container_width=True, hide_index=True)
    st.success(f"🏆 **Ganador matemático:** **$k = {k_optimo}$ grupos** con puntaje de **{mejor_silueta:.4f}** (separación sobresaliente).")

with col2_p3:
    fig_sil = px.line(
        df_silueta,
        x="k",
        y="Coeficiente_Silueta",
        markers=True,
        text="Coeficiente_Silueta",
        title="Puntaje de Silueta por Cantidad de Grupos (Pico máximo en k = 3)",
        labels={"k": "Cantidad de Grupos Evaluados (k)", "Coeficiente_Silueta": "Coeficiente de Silueta"}
    )
    fig_sil.update_traces(
        line=dict(color="#0F294A", width=4),
        marker=dict(size=13, color="#2563EB", line=dict(width=2, color="#FFFFFF")),
        textposition="top center",
        textfont=dict(color="#0B1F3A", size=12)
    )
    fig_sil.update_layout(
        xaxis=dict(dtick=1),
        yaxis=dict(range=[0.48, 0.96]),
        height=350,
        plot_bgcolor="#F8FAFC",
        paper_bgcolor="#FFFFFF"
    )
    st.plotly_chart(fig_sil, use_container_width=True)

# ==============================================================================
# PASO 4: ENTRENAMIENTO DEL MODELO DEFINITIVO Y CENTROIDES
# ==============================================================================
st.markdown('<div class="step-header">Paso 4: Entrenamiento Final y Cálculo de Centroides (Cliente Promedio)</div>', unsafe_allow_html=True)

st.markdown("""
<div class="analogy-box">
    <b>📍 ¿Qué es un Centroide?</b> Es el punto exacto que representa al <b>"cliente promedio"</b> del centro de cada grupo. 
    El algoritmo clasifica a cada uno de los 150 clientes asignándolo al centroide que le quede más cerca.
</div>
""", unsafe_allow_html=True)

col1_p4, col2_p4 = st.columns(2)
with col1_p4:
    st.markdown("**⭐ Coordenadas de los 3 Centroides en unidades reales:**")
    st.dataframe(
        df_centroides[["cluster_id", "perfil_cliente", "centroide_frecuencia_mes", "centroide_gasto_mxn"]].round(2),
        use_container_width=True,
        hide_index=True
    )
with col2_p4:
    st.markdown("**🏷️ Muestra de clientes ya clasificados por la IA:**")
    st.dataframe(
        df_clientes[["cliente_id", "frecuencia_compras_mes", "gasto_promedio_mxn", "perfil_cliente"]].head(6),
        use_container_width=True,
        hide_index=True
    )

# ==============================================================================
# PASO 5: PERFILAMIENTO DE NEGOCIO Y ESTRATEGIAS COMERCIALES
# ==============================================================================
st.markdown('<div class="step-header">Paso 5: Perfilamiento de Negocio (De Números a Decisiones)</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""
    <div class="profile-card" style="border-top: 6px solid #64748B;">
        <span class="profile-tag" style="background-color:#64748B;">PERFIL 1 • 36.7% DE LA BASE</span>
        <h4 style="color:#0B1F3A; margin:0.3rem 0;">🔘 Compradores Esporádicos</h4>
        <p style="color:#475569; font-size:0.93rem;">
            <b>Comportamiento:</b> ~2.0 pedidos al mes con ticket bajo (~$649 MXN).<br><br>
            <b>✅ Decisión de Marketing:</b> <b>No gastar presupuesto en cupones ni envíos gratis.</b> 
            Mantenerlos en correos automáticos de novedades sin costo promocional.
        </p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="profile-card" style="border-top: 6px solid #2563EB;">
        <span class="profile-tag" style="background-color:#2563EB;">PERFIL 2 • 33.3% DE LA BASE</span>
        <h4 style="color:#0B1F3A; margin:0.3rem 0;">🔵 Cazadores de Volumen</h4>
        <p style="color:#475569; font-size:0.93rem;">
            <b>Comportamiento:</b> ~14.5 pedidos al mes con ticket moderado (~$1,185 MXN).<br><br>
            <b>✅ Decisión de Marketing:</b> Ofrecer <b>membresía de envío gratis en compras mayores a $1,500 MXN</b> 
            para motivarlos a juntar productos y subir su gasto por pedido.
        </p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="profile-card" style="border-top: 6px solid #0B1F3A;">
        <span class="profile-tag" style="background-color:#0B1F3A;">PERFIL 3 • 30.0% DE LA BASE</span>
        <h4 style="color:#0B1F3A; margin:0.3rem 0;">🔷 Clientes VIP / Alto Valor</h4>
        <p style="color:#475569; font-size:0.93rem;">
            <b>Comportamiento:</b> ~5.7 pedidos al mes con ticket muy alto (~$8,768 MXN).<br><br>
            <b>✅ Decisión de Marketing:</b> <b>Prioridad máxima de retención.</b> 
            Acceso anticipado a lanzamientos exclusivos, <i>cashback</i> y atención VIP personalizada.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.dataframe(
    perfilamiento[[
        "cluster_id", "perfil_cliente", "cantidad_clientes", "porcentaje_clientes",
        "frecuencia_promedio", "ticket_promedio_mxn", "ingreso_mensual_cliente_mxn",
        "estrategia_marketing"
    ]].round(2),
    use_container_width=True,
    hide_index=True
)

# ==============================================================================
# PASO 6: DASHBOARD EJECUTIVO Y SIMULADOR INTERACTIVO EN VIVO
# ==============================================================================
st.markdown('<div class="step-header">Paso 6: Dashboard Ejecutivo y Simulador Interactivo en Vivo</div>', unsafe_allow_html=True)

# KPIs superiores
k1, k2, k3, k4 = st.columns(4)
k1.metric("👥 Clientes Analizados", "150", "100% Cartera Activa")
k2.metric("🎯 Grupos Óptimos (k)", f"{k_optimo} Perfiles", f"Silueta: {mejor_silueta:.3f}")
k3.metric("💎 Ticket Promedio VIP", "$8,768 MXN", "30.0% de clientes")
k4.metric("📦 Frecuencia Cazadores", "14.5 ped/mes", "33.3% de clientes")

paleta_navy_grey = {
    "Compradores Esporádicos": "#64748B",     # Gris pizarra
    "Cazadores de Volumen": "#2563EB",        # Azul rey brillante
    "Clientes VIP / Alto Valor": "#0B1F3A"    # Azul marino profundo
}

tab_dash, tab_sim, tab_data = st.tabs([
    "📊 Dashboard Ejecutivo (Gráficas)",
    "🎮 Simulador Interactivo de Clientes",
    "📁 Explorador y Descarga de Datos"
])

with tab_dash:
    dash_col1, dash_col2 = st.columns([1.45, 1])

    with dash_col1:
        fig_clusters = px.scatter(
            df_clientes,
            x="frecuencia_compras_mes",
            y="gasto_promedio_mxn",
            color="perfil_cliente",
            color_discrete_map=paleta_navy_grey,
            hover_data=["cliente_id", "estrategia_marketing"],
            title="Mapa Final de Segmentación y Centroides (K-Means, k = 3)",
            labels={
                "frecuencia_compras_mes": "Frecuencia de Compras al Mes (Pedidos)",
                "gasto_promedio_mxn": "Gasto Promedio por Pedido (MXN)",
                "perfil_cliente": "Segmento"
            }
        )
        fig_clusters.update_traces(marker=dict(size=10, opacity=0.9, line=dict(width=1, color="#FFFFFF")))

        fig_clusters.add_trace(
            go.Scatter(
                x=df_centroides["centroide_frecuencia_mes"],
                y=df_centroides["centroide_gasto_mxn"],
                mode="markers+text",
                name="Centroides (Centro)",
                text=[f"★ {p}" for p in df_centroides["perfil_cliente"]],
                textposition="top center",
                textfont=dict(color="#0B1F3A", size=11),
                marker=dict(
                    symbol="star",
                    size=20,
                    color="#93C5FD",
                    line=dict(width=2, color="#0B1F3A")
                )
            )
        )
        fig_clusters.update_layout(
            height=470,
            plot_bgcolor="#F8FAFC",
            paper_bgcolor="#FFFFFF",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_clusters, use_container_width=True)

    with dash_col2:
        orden_perfiles = ["Compradores Esporádicos", "Cazadores de Volumen", "Clientes VIP / Alto Valor"]
        perf_ord = perfilamiento.set_index("perfil_cliente").loc[orden_perfiles].reset_index()

        fig_donut = px.pie(
            perf_ord,
            names="perfil_cliente",
            values="cantidad_clientes",
            color="perfil_cliente",
            color_discrete_map=paleta_navy_grey,
            hole=0.5,
            title="Distribución de la Cartera (150 Clientes)"
        )
        fig_donut.update_traces(textinfo="percent+value", textposition="inside", textfont=dict(color="#FFFFFF", size=13))
        fig_donut.update_layout(height=470, paper_bgcolor="#FFFFFF")
        st.plotly_chart(fig_donut, use_container_width=True)

    dash_col3, dash_col4 = st.columns(2)

    with dash_col3:
        fig_ticket = px.bar(
            perf_ord,
            x="perfil_cliente",
            y="ticket_promedio_mxn",
            color="perfil_cliente",
            color_discrete_map=paleta_navy_grey,
            text_auto="$.2s",
            title="Ticket Promedio por Segmento (MXN)",
            labels={"perfil_cliente": "Segmento", "ticket_promedio_mxn": "Ticket Promedio (MXN)"}
        )
        fig_ticket.update_layout(height=380, showlegend=False, plot_bgcolor="#F8FAFC", paper_bgcolor="#FFFFFF")
        st.plotly_chart(fig_ticket, use_container_width=True)

    with dash_col4:
        fig_freq = px.bar(
            perf_ord,
            x="perfil_cliente",
            y="frecuencia_promedio",
            color="perfil_cliente",
            color_discrete_map=paleta_navy_grey,
            text_auto=".1f",
            title="Frecuencia Media de Compra (Pedidos / Mes)",
            labels={"perfil_cliente": "Segmento", "frecuencia_promedio": "Pedidos por Mes"}
        )
        fig_freq.update_layout(height=380, showlegend=False, plot_bgcolor="#F8FAFC", paper_bgcolor="#FFFFFF")
        st.plotly_chart(fig_freq, use_container_width=True)

with tab_sim:
    st.markdown("""
    <div class="context-card">
        <h4 style="color:#0B1F3A; margin-top:0;">🎮 Prueba el Modelo en Tiempo Real</h4>
        <p style="color:#475569;">
            Usa los botones rápidos de ejemplo o mueve los controles para simular un <b>cliente nuevo</b> que acaba de entrar a la tienda. 
            El modelo K-Means calculará a qué grupo pertenece y qué promoción debe recibir.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if "sim_freq" not in st.session_state:
        st.session_state.sim_freq = 6
    if "sim_gasto" not in st.session_state:
        st.session_state.sim_gasto = 8500

    b_col1, b_col2, b_col3 = st.columns(3)
    with b_col1:
        if st.button("🔘 Cargar Ejemplo Esporádico"):
            st.session_state.sim_freq = 2
            st.session_state.sim_gasto = 600
    with b_col2:
        if st.button("🔵 Cargar Ejemplo Cazador"):
            st.session_state.sim_freq = 15
            st.session_state.sim_gasto = 1200
    with b_col3:
        if st.button("🔷 Cargar Ejemplo Cliente VIP"):
            st.session_state.sim_freq = 6
            st.session_state.sim_gasto = 9200

    s_col1, s_col2 = st.columns(2)
    with s_col1:
        freq_input = st.slider("🛒 Frecuencia de compras al mes (pedidos):", 1, 20, key="sim_freq")
    with s_col2:
        gasto_input = st.slider("💵 Gasto promedio por pedido (MXN):", 250, 12000, step=50, key="sim_gasto")

    if st.button("🚀 CLASIFICAR CLIENTE CON INTELIGENCIA ARTIFICIAL"):
        nuevo_scaled = scaler.transform([[freq_input, gasto_input]])
        pred_cluster = int(kmeans_final.predict(nuevo_scaled)[0])
        mapa_nombres = dict(zip(perfilamiento["cluster_id"], perfilamiento["perfil_cliente"]))
        mapa_estrategias = dict(zip(perfilamiento["cluster_id"], perfilamiento["estrategia_marketing"]))

        perfil_pred = mapa_nombres[pred_cluster]
        estrategia_pred = mapa_estrategias[pred_cluster]

        st.markdown(f"""
        <div class="hero-banner" style="padding:1.5rem; margin-top:1rem;">
            <span class="hero-badge">RESULTADO DE CLASIFICACIÓN INSTANTÁNEA</span>
            <h3 style="color:#FFFFFF; margin:0.3rem 0;">Segmento Asignado: {perfil_pred}</h3>
            <p style="color:#E2E8F0; font-size:1.05rem; margin:0;">
                <b>Estrategia Recomendada:</b> {estrategia_pred}
            </p>
        </div>
        """, unsafe_allow_html=True)

with tab_data:
    st.markdown("**🔍 Filtra y explora los 150 clientes segmentados:**")
    filtro_perfil = st.multiselect(
        "Filtrar por perfil de cliente:",
        options=df_clientes["perfil_cliente"].unique().tolist(),
        default=df_clientes["perfil_cliente"].unique().tolist()
    )
    df_filtrado = df_clientes[df_clientes["perfil_cliente"].isin(filtro_perfil)]
    st.dataframe(df_filtrado, use_container_width=True, hide_index=True)

    d_col1, d_col2 = st.columns(2)
    with d_col1:
        st.download_button(
            label="📥 DESCARGAR BASE SEGMENTADA (CSV)",
            data=df_filtrado.to_csv(index=False).encode("utf-8"),
            file_name="clientes_segmentados_kmeans.csv",
            mime="text/csv"
        )
    with d_col2:
        st.download_button(
            label="📊 DESCARGAR TABLA DE PERFILES EJECUTIVOS (CSV)",
            data=perfilamiento.to_csv(index=False).encode("utf-8"),
            file_name="resumen_ejecutivo_segmentos.csv",
            mime="text/csv"
        )
