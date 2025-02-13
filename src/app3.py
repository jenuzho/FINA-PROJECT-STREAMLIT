import streamlit as st
import joblib
import gzip
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier

# Configuración de la aplicación
st.set_page_config(page_title="Fraude Bancario", page_icon="🚨", layout="wide")

# Ruta del modelo
MODEL_PATH = "modelo_RandomForest_optimizado.pkl.gz"

# Función para cargar el modelo
@st.cache_resource()
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("⚠️ Error: El modelo no se encuentra en la ruta especificada.")
        return None
    with gzip.open(MODEL_PATH, "rb") as f:
        model = joblib.load(f)
    if not isinstance(model, RandomForestClassifier):
        st.error("⚠️ El archivo cargado no es un modelo RandomForest.")
        return None
    return model

# Cargar el modelo
model = load_model()

if model:
    st.success("✅ Modelo cargado correctamente.")
    
    # Entrada de datos de prueba
    st.sidebar.header("📊 Introducir Datos de Transacción")
    
    input_data = {
        'income': st.sidebar.number_input("Ingresos", min_value=0.0, max_value=1e6, value=5000.0),
        'name_email_similarity': st.sidebar.slider("Similitud Nombre-Email", 0.0, 1.0, 0.5),
        'customer_age': st.sidebar.number_input("Edad del Cliente", min_value=18, max_value=100, value=30),
        'proposed_credit_limit': st.sidebar.number_input("Límite de Crédito Propuesto", min_value=0.0, max_value=1e6, value=10000.0),
        'velocity_6h': st.sidebar.number_input("Velocidad de Transacción (6h)", min_value=0.0, max_value=10000.0, value=100.0),
    }
    
    df_input = pd.DataFrame([input_data])
    
    if st.sidebar.button("🚀 Predecir Fraude"):
        try:
            prediction = model.predict(df_input)[0]
            result = "Fraude" if prediction == 1 else "No Fraude"
            st.subheader(f"🔮 Predicción: {result}")
        except Exception as e:
            st.error(f"Error en la predicción: {str(e)}")
else:
    st.warning("⚠️ No se pudo cargar el modelo. Verifica la ruta del archivo.")
