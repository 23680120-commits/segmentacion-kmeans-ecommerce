import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Segmentación de clientes", layout="wide")

st.markdown("""
<style>
    .stApp { background: #f8f6fb; color: #2d2338; }
    .block-container { max-width: 1200px; padding-top: 2.5rem; }
    h1, h2, h3 { color: #39244d; }
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        background: #ffffff;
        border: 1px solid #e5dcec;
        border-radius: 10px;
        padding: 1.15rem;
    }
    div.stButton > button {
        background: #694487;
        color: white;
        border: 0;
        border-radius: 6px;
    }
    div.stButton > button:hover { background: #52366a; color: white; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def analizar_clientes():
    np.random.seed(42)
    grupos = []
    for n, f_media, f_desv, f_min, f_max, g_media, g_desv, g_min, g_max in [
        (55, 2.2, 0.7, 1, 5, 650, 160, 250, 1200),
        (50, 14.5, 1.6, 10, 20, 1150, 220, 600, 1800),
        (45, 5.5, 1.2, 3, 9, 8800, 950, 6500, 11500),
    ]:
        f = np.clip(np.random.normal(f_media, f_desv, n), f_min, f_max).round().astype(int)
        g = np.clip(np.random.normal(g_media, g_desv, n), g_min, g_max).round(2)
        grupos.append(pd.DataFrame({"Compras al mes": f, "Gasto por compra": g}))
    clientes = pd.concat(grupos, ignore_index=True)
    clientes = clientes.iloc[np.random.permutation(len(clientes))].reset_index(drop=True)
    clientes.insert(0, "Cliente", [f"CLI-{i + 1:03d}" for i in range(len(clientes))])

    variables = ["Compras al mes", "Gasto por compra"]
    escala = StandardScaler()
    datos_escalados = escala.fit_transform(clientes[variables])
    resultados = []
    for k in range(2, 7):
        modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
        etiquetas = modelo.fit_predict(datos_escalados)
        resultados.append((k, silhouette_score(datos_escalados, etiquetas)))
    mejor_k, puntuacion = max(resultados, key=lambda item: item[1])
    modelo = KMeans(n_clusters=mejor_k, random_state=42, n_init=10)
    clientes["Grupo"] = modelo.fit_predict(datos_escalados)

    resumen = clientes.groupby("Grupo").agg(
        clientes=("Cliente", "count"),
        frecuencia=("Compras al mes", "mean"),
        gasto=("Gasto por compra", "mean"),
    ).reset_index()

    def perfil(fila):
        if fila["gasto"] > 5000:
            return "Alto valor", "Ofrecer atención preferente y beneficios de lealtad."
        if fila["frecuencia"] > 10:
            return "Compra frecuente", "Ofrecer envío gratis a partir de $1,500."
        return "Compra ocasional", "Compartir novedades por correo."

    perfiles = {int(fila["Grupo"]): perfil(fila) for _, fila in resumen.iterrows()}
    clientes["Perfil"] = clientes["Grupo"].map(lambda grupo: perfiles[grupo][0])
    return clientes, escala, modelo, mejor_k, puntuacion, resumen, perfiles


clientes, escala, modelo, mejor_k, puntuacion, resumen, perfiles = analizar_clientes()

st.title("Segmentación de clientes")
st.caption("Ejemplo con 150 clientes simulados de una tienda en línea.")

fila_1 = st.columns(3)
fila_2 = st.columns(3)

with fila_1[0]:
    with st.container(border=True):
        st.subheader("1. Datos de clientes")
        st.write("Se registran las compras mensuales y el gasto promedio de cada cliente.")
        st.dataframe(clientes[["Cliente", "Compras al mes", "Gasto por compra"]].head(5), hide_index=True, use_container_width=True)
        st.caption("Se muestran cinco registros de ejemplo.")

with fila_1[1]:
    with st.container(border=True):
        st.subheader("2. Preparación de datos")
        st.write("Se ajustan las escalas para que la frecuencia y el gasto tengan un peso comparable en el análisis.")

with fila_1[2]:
    with st.container(border=True):
        st.subheader("3. Número de grupos")
        st.write("Se comparan de dos a seis grupos y se elige la opción que mejor los separa, según el índice de silueta.")
        st.markdown(f"**Resultado: {mejor_k} grupos**  ")
        st.caption(f"Índice de silueta: {puntuacion:.2f}")

with fila_2[0]:
    with st.container(border=True):
        st.subheader("4. Agrupación")
        st.write("K-Means reúne a los clientes con hábitos de compra similares.")
        grafica = px.scatter(
            clientes, x="Compras al mes", y="Gasto por compra", color="Perfil",
            hover_data=["Cliente"],
            color_discrete_sequence=["#a98bc2", "#744a99", "#482b65", "#cbb8db", "#8e6aaa", "#5c397a"],
            labels={"Gasto por compra": "Gasto por compra (MXN)"},
        )
        grafica.update_traces(marker={"size": 7, "opacity": 0.8})
        grafica.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), legend_title_text="", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#faf8fc")
        st.plotly_chart(grafica, use_container_width=True)

with fila_2[1]:
    with st.container(border=True):
        st.subheader("5. Perfiles y acciones")
        st.write("Se describe cada grupo y se propone una acción comercial.")
        for _, fila in resumen.iterrows():
            nombre, accion = perfiles[int(fila["Grupo"])]
            st.markdown(f"**{nombre}** · {int(fila['clientes'])} clientes  ")
            st.caption(accion)

with fila_2[2]:
    with st.container(border=True):
        st.subheader("6. Clasificación de un cliente")
        st.write("El modelo asigna un grupo según la frecuencia y el gasto indicados.")
        frecuencia = st.number_input("Compras al mes", min_value=1, max_value=20, value=6)
        gasto = st.number_input("Gasto por compra (MXN)", min_value=250, max_value=12000, value=8500, step=50)
        if st.button("Clasificar"):
            nuevo = pd.DataFrame([[frecuencia, gasto]], columns=["Compras al mes", "Gasto por compra"])
            grupo = int(modelo.predict(escala.transform(nuevo))[0])
            nombre, accion = perfiles[grupo]
            st.markdown(f"**Perfil: {nombre}**")
            st.write(accion)
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Segmentación de clientes", layout="wide")

st.markdown("""
<style>
    .stApp { background: #f8f6fb; color: #2d2338; }
    .block-container { max-width: 1200px; padding-top: 2.5rem; }
    h1, h2, h3 { color: #39244d; }
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        background: #fff;
        border: 1px solid #e5dcec;
        border-radius: 10px;
        padding: 1.15rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def analizar_clientes():
    np.random.seed(42)
    grupos = []
    for n, f_media, f_desv, f_min, f_max, g_media, g_desv, g_min, g_max in [
        (55, 2.2, 0.7, 1, 5, 650, 160, 250, 1200),
        (50, 14.5, 1.6, 10, 20, 1150, 220, 600, 1800),
        (45, 5.5, 1.2, 3, 9, 8800, 950, 6500, 11500),
    ]:
        frecuencia = np.clip(np.random.normal(f_media, f_desv, n), f_min, f_max).round().astype(int)
        gasto = np.clip(np.random.normal(g_media, g_desv, n), g_min, g_max).round(2)
        grupos.append(pd.DataFrame({"Compras al mes": frecuencia, "Gasto por compra": gasto}))

    clientes = pd.concat(grupos, ignore_index=True)
    clientes = clientes.iloc[np.random.permutation(len(clientes))].reset_index(drop=True)
    clientes.insert(0, "Cliente", [f"CLI-{i + 1:03d}" for i in range(len(clientes))])
    variables = ["Compras al mes", "Gasto por compra"]
    escala = StandardScaler()
    datos_escalados = escala.fit_transform(clientes[variables])

    resultados = []
    for k in range(2, 7):
        modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
        etiquetas = modelo.fit_predict(datos_escalados)
        resultados.append({"Grupos": k, "Silueta": silhouette_score(datos_escalados, etiquetas)})
    siluetas = pd.DataFrame(resultados)
    mejor_k = int(siluetas.loc[siluetas["Silueta"].idxmax(), "Grupos"])
    modelo = KMeans(n_clusters=mejor_k, random_state=42, n_init=10)
    clientes["Grupo"] = modelo.fit_predict(datos_escalados)

    resumen = clientes.groupby("Grupo").agg(
        clientes=("Cliente", "count"),
        frecuencia=("Compras al mes", "mean"),
        gasto=("Gasto por compra", "mean"),
    ).reset_index()

    def perfil(fila):
        if fila["gasto"] > 5000:
            return "Alto valor", "Se propone atención preferente y beneficios de lealtad."
        if fila["frecuencia"] > 10:
            return "Compra frecuente", "Se propone envío gratis en compras desde $1,500."
        return "Compra ocasional", "Se propone compartir novedades por correo."

    perfiles = {int(fila["Grupo"]): perfil(fila) for _, fila in resumen.iterrows()}
    clientes["Perfil"] = clientes["Grupo"].map(lambda grupo: perfiles[grupo][0])
    centroides = pd.DataFrame(escala.inverse_transform(modelo.cluster_centers_), columns=variables)
    centroides["Grupo"] = range(mejor_k)
    centroides["Perfil"] = centroides["Grupo"].map(lambda grupo: perfiles[grupo][0])
    return clientes, siluetas, mejor_k, resumen, perfiles, centroides, escala, modelo


def formato_grafica(figura):
    figura.update_layout(
        height=280, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#faf8fc",
        legend_title_text="",
    )
    return figura


clientes, siluetas, mejor_k, resumen, perfiles, centroides, escala, modelo = analizar_clientes()

st.title("Segmentación de clientes")
st.write(
    "Una tienda en línea cuenta con un presupuesto limitado para promociones. "
    "Para proponer acciones adecuadas a cada tipo de cliente, se analizan las "
    "compras mensuales y el gasto por pedido. K-Means agrupa a quienes tienen "
    "hábitos similares sin usar categorías previas."
)
st.caption("El ejercicio utiliza datos simulados de 150 clientes; las acciones son propuestas de ejemplo.")

fila_1 = st.columns(3)
fila_2 = st.columns(3)

with fila_1[0]:
    with st.container(border=True):
        st.subheader("1. Datos de clientes")
        st.write("Se observan los datos antes de asignar grupos.")
        inicial = px.scatter(
            clientes, x="Compras al mes", y="Gasto por compra", hover_data=["Cliente"],
            labels={"Gasto por compra": "Gasto por compra (MXN)"},
            color_discrete_sequence=["#8d75a0"],
        )
        inicial.update_traces(marker=dict(size=7, opacity=0.8))
        st.plotly_chart(formato_grafica(inicial), use_container_width=True)

with fila_1[1]:
    with st.container(border=True):
        st.subheader("2. Preparación de datos")
        st.write("Se ajustan las escalas para que la frecuencia y el gasto tengan un peso comparable en el análisis.")
        st.caption("La estandarización centra cada variable en cero y ajusta su dispersión.")

with fila_1[2]:
    with st.container(border=True):
        st.subheader("3. Número de grupos")
        st.write("Se comparan de dos a seis grupos y se elige el de mayor índice de silueta.")
        st.markdown(f"**Resultado: {mejor_k} grupos** · Silueta: {siluetas['Silueta'].max():.2f}")
        curva = px.line(siluetas, x="Grupos", y="Silueta", markers=True, color_discrete_sequence=["#744a99"])
        curva.update_xaxes(dtick=1)
        st.plotly_chart(formato_grafica(curva), use_container_width=True)

with fila_2[0]:
    with st.container(border=True):
        st.subheader("4. Agrupación")
        st.write("K-Means asigna a cada cliente un grupo y calcula su centro o cliente promedio.")
        tabla = centroides[["Grupo", "Perfil", "Compras al mes", "Gasto por compra"]].copy()
        tabla = tabla.rename(columns={"Compras al mes": "Compras/mes", "Gasto por compra": "Gasto (MXN)"})
        tabla[["Compras/mes", "Gasto (MXN)"]] = tabla[["Compras/mes", "Gasto (MXN)"]].round(2)
        st.dataframe(tabla, hide_index=True, use_container_width=True)
        st.caption("Cada fila representa el centro de un grupo, en las unidades originales.")
        st.caption("Ejemplo de clientes y grupos asignados:")
        st.dataframe(clientes[["Cliente", "Grupo"]].head(5), hide_index=True, use_container_width=True)

with fila_2[1]:
    with st.container(border=True):
        st.subheader("5. Perfiles y acciones")
        st.write("Se interpreta cada grupo a partir de sus promedios y se propone una acción comercial.")
        for _, fila in resumen.iterrows():
            nombre, accion = perfiles[int(fila["Grupo"])]
            st.markdown(f"**{nombre}** · {int(fila['clientes'])} clientes")
            st.caption(f"{fila['frecuencia']:.1f} compras al mes · ${fila['gasto']:,.0f} por compra. {accion}")

with fila_2[2]:
    with st.container(border=True):
        st.subheader("6. Resultado final")
        st.write("Se muestran los grupos y sus centros, señalados con una estrella.")
        final = px.scatter(
            clientes, x="Compras al mes", y="Gasto por compra", color="Perfil",
            hover_data=["Cliente", "Grupo"],
            color_discrete_sequence=["#a98bc2", "#744a99", "#482b65", "#cbb8db", "#8e6aaa", "#5c397a"],
            labels={"Gasto por compra": "Gasto por compra (MXN)"},
        )
        final.update_traces(marker=dict(size=7, opacity=0.8))
        final.add_trace(go.Scatter(
            x=centroides["Compras al mes"], y=centroides["Gasto por compra"],
            mode="markers", name="Centros", text=centroides["Perfil"],
            hovertemplate="Centro: %{text}<br>Compras al mes: %{x:.1f}<br>Gasto: $%{y:,.2f}<extra></extra>",
            marker=dict(symbol="star", size=17, color="#301b43", line=dict(color="white", width=1)),
        ))
        st.plotly_chart(formato_grafica(final), use_container_width=True)


with st.container(border=True):
    st.subheader("Simulador de un cliente")
    st.write("El modelo asigna un grupo según la frecuencia y el gasto indicados.")
    entrada_1, entrada_2 = st.columns(2)
    with entrada_1:
        frecuencia = st.number_input("Compras al mes", min_value=1, max_value=20, value=6)
    with entrada_2:
        gasto = st.number_input("Gasto por compra (MXN)", min_value=250, max_value=12000, value=8500, step=50)
    if st.button("Clasificar"):
        nuevo = pd.DataFrame([[frecuencia, gasto]], columns=["Compras al mes", "Gasto por compra"])
        grupo = int(modelo.predict(escala.transform(nuevo))[0])
        nombre, accion = perfiles[grupo]
        st.markdown(f"**Perfil: {nombre}**")
        st.write(accion)
