import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Cargar el modelo y el escalador
model = joblib.load('f1_predictions_model.pkl')
scaler = joblib.load('scaler.pkl')  # Asegúrate de haber guardado el escalador durante el entrenamiento

# Cargar los datos originales para obtener los valores únicos
drivers = pd.read_csv('drivers.csv')
constructors = pd.read_csv('constructors.csv')
races = pd.read_csv('races.csv')

# Extraer valores únicos
driver_names = drivers['surname'].unique()
team_names = constructors['name'].unique()
circuit_names = races['name'].unique()

# Título de la app
st.title("Predicción de Pódiums de F1 - 2025")

# Descripción
st.write("""
Esta aplicación predice si un piloto estará en el pódium basado en las características de la carrera.
Por favor, selecciona los datos necesarios para realizar la predicción.
""")

# Entradas del usuario
surname = st.selectbox("Selecciona el Piloto", driver_names)
team_name = st.selectbox("Selecciona el Equipo", team_names)
circuit_name = st.selectbox("Selecciona el Circuito", circuit_names)
grid_weighted = st.number_input("Posición de Salida Ponderada (grid_weighted)", min_value=0.0, step=0.1)
points_scaled = st.number_input("Puntos Escalados (points_scaled)", min_value=0.0, step=0.1)
avg_position = st.number_input("Promedio de posición del piloto en el circuito", min_value=0.0, step=0.1)
team_avg_position = st.number_input("Promedio de posición del equipo en el circuito", min_value=0.0, step=0.1)
grid_avg_interaction = st.number_input("Interacción Grid-Promedio (grid_avg_interaction)", min_value=0.0, step=0.1)

# Convertir entradas de texto a valores codificados
driver_encoded = drivers[drivers['surname'] == surname].index[0]
team_encoded = constructors[constructors['name'] == team_name].index[0]
circuit_encoded = races[races['name'] == circuit_name].index[0]

# Crear un array con todas las características necesarias para el escalador
features = np.array([[grid_weighted, points_scaled, avg_position, team_avg_position, grid_avg_interaction, driver_encoded, team_encoded, circuit_encoded]])

# Escalar todas las características
scaled_features = scaler.transform(features)

# Asignar las características escaladas
grid_weighted_scaled = scaled_features[0][0]
points_scaled_scaled = scaled_features[0][1]
avg_position_scaled = scaled_features[0][2]
team_avg_position_scaled = scaled_features[0][3]
grid_avg_interaction_scaled = scaled_features[0][4]
driver_encoded_scaled = scaled_features[0][5]
team_encoded_scaled = scaled_features[0][6]
circuit_encoded_scaled = scaled_features[0][7]

# Botón para realizar la predicción
if st.button("Predecir"):
    # Crear el array de entrada con las características escaladas
    input_data = np.array([[grid_weighted_scaled, points_scaled_scaled, avg_position_scaled, team_avg_position_scaled, grid_avg_interaction_scaled, driver_encoded_scaled, team_encoded_scaled, circuit_encoded_scaled]])
    
    # Realizar la predicción
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    # Mostrar el resultado
    if prediction[0] == 1:
        st.success(f"¡El piloto estará en el pódium! (Probabilidad: {probability:.2f})")
    else:
        st.error(f"El piloto no estará en el pódium. (Probabilidad: {probability:.2f})")