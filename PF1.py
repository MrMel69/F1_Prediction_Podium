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
grid_scaled = st.number_input("Posición de Salida Escalada (grid_scaled)", min_value=0.0, step=0.1)
points_scaled = st.number_input("Puntos Escalados (points_scaled)", min_value=0.0, step=0.1)
avg_position = st.number_input("Promedio de posición del piloto en el circuito", min_value=0.0, step=0.1)
team_avg_position = st.number_input("Promedio de posición del equipo en el circuito", min_value=0.0, step=0.1)

# Convertir entradas de texto a valores codificados
driver_encoded = drivers[drivers['surname'] == surname].index[0]
team_encoded = constructors[constructors['name'] == team_name].index[0]
circuit_encoded = races[races['name'] == circuit_name].index[0]

# Crear interacciones adicionales
grid_avg_interaction = grid_scaled * avg_position
grid_weighted_squared = grid_scaled ** 2
grid_points_interaction = grid_scaled * points_scaled
grid_weighted_inverse = 1 / (grid_scaled + 1)

pilot_circuit_experience = avg_position / (team_avg_position +1)
grid_pilot_circuit_interaction = grid_scaled * avg_position * circuit_encoded

# Crear el DataFrame con las características
# Combinar todas las características en una lista
features = [[grid_scaled, avg_position, team_avg_position, 
             grid_avg_interaction, driver_encoded, team_encoded, circuit_encoded, 
             grid_weighted_squared, grid_points_interaction, grid_weighted_inverse,
             pilot_circuit_experience, grid_pilot_circuit_interaction]]

# Crear el DataFrame con las características
feature_names = ['grid_scaled', 'avg_position', 'team_avg_position',
                'grid_avg_interaction', 'driver_encoded','team_encoded', 'circuit_encoded',
                'grid_weighted_squared', 'grid_points_interaction', 'grid_weighted_inverse', 
              'pilot_circuit_experience', 'grid_pilot_circuit_interaction']
features_df = pd.DataFrame(features, columns=feature_names)

# Escalar todas las características
scaled_features = scaler.transform(features_df)

# Botón para realizar la predicción
if st.button("Predecir"):
    # Realizar la predicción
    prediction = model.predict(scaled_features)
    probability = model.predict_proba(scaled_features)[0][1]

    # Mostrar el resultado
    if prediction[0] == 1:
        st.success(f"¡El piloto estará en el pódium! (Probabilidad: {probability:.2f})")
    else:
        st.error(f"El piloto no estará en el pódium. (Probabilidad: {probability:.2f})")