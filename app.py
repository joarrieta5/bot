import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from statsmodels.tsa.arima.model import ARIMA
from streamlit_autorefresh import st_autorefresh
import time

st.set_page_config(page_title="Análisis de Multiplicadores - Auto Update", layout="wide")
st.title("📊 Análisis y Predicción de Multiplicadores (Actualización automática)")

# Frecuencia de actualización en milisegundos
REFRESH_RATE_MS = 5000  # 5 segundos

# Contador de refresco automático
st_autorefresh(interval=REFRESH_RATE_MS, key="datarefresh")

# Función para simular un nuevo multiplicador cercano al último valor
def simular_nuevo_multiplicador(ultimo, scale=0.2):
    nuevo = np.random.normal(loc=ultimo, scale=scale*ultimo)
    return max(1, round(nuevo, 2))

# Subida inicial de CSV y carga a session_state
if "df" not in st.session_state:
    uploaded_file = st.file_uploader("Sube un archivo CSV con columna 'multiplicador'", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if "multiplicador" not in df.columns:
            st.error("El CSV debe tener una columna llamada 'multiplicador'")
            st.stop()
        df["multiplicador"] = pd.to_numeric(df["multiplicador"], errors="coerce")
        df = df.dropna(subset=["multiplicador"]).reset_index(drop=True)
        st.session_state.df = df.copy()
        st.session_state.last_time = time.time()
    else:
        st.info("Sube un CSV para comenzar.")
        st.stop()

df = st.session_state.df.copy()

# Añadir nuevo dato cada REFRESH_RATE_MS segundos
current_time = time.time()
if ("last_time" not in st.session_state) or (current_time - st.session_state.last_time > REFRESH_RATE_MS / 1000):
    ultimo = df["multiplicador"].iloc[-1]
    nuevo = simular_nuevo_multiplicador(ultimo)
    nuevo_fila = pd.DataFrame({"multiplicador": [nuevo]})
    df = pd.concat([df, nuevo_fila], ignore_index=True)
    st.session_state.df = df
    st.session_state.last_time = current_time

# Mostrar cuenta regresiva aproximada (opcional)
tiempo_restante = max(0, REFRESH_RATE_MS / 1000 - (time.time() - st.session_state.last_time))
st.markdown(f"⏳ Próxima actualización en **{int(tiempo_restante)} segundos**")

# Mostrar datos actualizados
st.subheader("Multiplicadores históricos (últimos 20)")
st.dataframe(df.tail(20).reset_index(drop=True))

# Clasificación
def clasificar(x):
    if x < 2: return 0
    elif x <= 10: return 1
    else: return 2

df["clase"] = df["multiplicador"].apply(clasificar)

# Distribución de clases
st.subheader("Distribución de clases")
fig, ax = plt.subplots()
df["clase"].value_counts().sort_index().plot(kind="bar", ax=ax)
ax.set_xticklabels(["Bajo (<2x)", "Medio (2-10x)", "Alto (>10x)"], rotation=0)
st.pyplot(fig)

# Matriz de transición (Cadena de Markov)
st.subheader("Matriz de transición (Cadena de Markov)")
trans_mat = np.zeros((3,3))
for (a, b) in zip(df["clase"][:-1], df["clase"][1:]):
    trans_mat[a, b] += 1
trans_mat = np.nan_to_num(trans_mat / trans_mat.sum(axis=1, keepdims=True))
st.dataframe(pd.DataFrame(trans_mat, columns=["Bajo", "Medio", "Alto"], index=["Bajo", "Medio", "Alto"]))

# Predicción ARIMA
st.subheader("Predicción ARIMA")
ln_series = np.log(df["multiplicador"])
try:
    model = ARIMA(ln_series, order=(1,0,0)).fit()
    forecast = model.get_forecast(steps=1)
    pred_mean = np.exp(forecast.predicted_mean.values[0])
    conf_int = np.exp(forecast.conf_int().values[0])
    st.write(f"📈 Predicción próxima ronda: **{pred_mean:.2f}x**")
    st.write(f"Intervalo de confianza 95%: [{conf_int[0]:.2f}x , {conf_int[1]:.2f}x]")
except Exception as e:
    st.error(f"Error en ARIMA: {e}")

# Predicción Random Forest (ventana=5)
st.subheader("Predicción Random Forest (ventana=5)")
window = 5
X, y = [], []
for i in range(window, len(df)):
    X.append(df["multiplicador"].iloc[i-window:i].values)
    y.append(df["clase"].iloc[i])
X, y = np.array(X), np.array(y)

if len(np.unique(y)) > 1:
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    pred = rf.predict([df["multiplicador"].iloc[-window:].values])[0]
    proba = rf.predict_proba([df["multiplicador"].iloc[-window:].values])[0]
    st.write(f"Predicción próxima clase: **{['Bajo','Medio','Alto'][pred]}**")
    st.write("Probabilidades:", {['Bajo','Medio','Alto'][i]: round(p,3) for i,p in enumerate(proba)})

    # Matriz de confusión
    y_pred = rf.predict(X)
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm, display_labels=["Bajo","Medio","Alto"]).plot(ax=ax)
    st.pyplot(fig)
else:
    st.warning("Datos insuficientes para entrenar Random Forest")

# Simulaciones de estrategias
st.subheader("Simulaciones de estrategias")

def simular_estrategia(df, estrategia="fijo", stake=1, max_martingala=4):
    balance = 0
    balances = []
    consecutivas = 0
    apuesta = stake
    for x in df["multiplicador"]:
        if x >= 2:
            balance += apuesta
            apuesta = stake
            consecutivas = 0
        else:
            balance -= apuesta
            consecutivas += 1
            if estrategia == "martingala":
                apuesta = stake * (2**min(consecutivas, max_martingala))
        balances.append(balance)
    return balances

bal_fijo = simular_estrategia(df, "fijo")
bal_mart = simular_estrategia(df, "martingala")

fig, ax = plt.subplots()
ax.plot(bal_fijo, label="Stake fijo")
ax.plot(bal_mart, label="Martingala parcial")
ax.set_title("Evolución del balance")
ax.legend()
st.pyplot(fig)
