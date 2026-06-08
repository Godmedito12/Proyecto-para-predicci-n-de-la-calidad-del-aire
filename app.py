#Importamos librerías básicas

import pandas as pd # manipulacion dataframes
import numpy as np  # matrices y vectores
import matplotlib.pyplot as plt #gráfica
import streamlit as st
#Cargamos el modelo
import pickle
model_rf, variables = pickle.load(open('Modelo_Proyecto.pkl', 'rb'))
print(model_rf)
#Cargamos los datos futuros
data = pd.read_csv("Datos futuros proyecto.csv")
data.head()
#Interfaz grafica
#Se crea interfaz gráfica con streamlit para captura de los datos

st.title('Prediccion EAQI')

avg_nitrogen_dioxide = st.number_input("Promedio dióxido de nitrógeno",value=10.0)
max_time_nitrogen_dioxide = st.selectbox("Máximo tiempo de dióxido de nitrógeno",["20:00:00", "21:00:00", "22:00:00", "23:00:00"])
min_time_nitrogen_dioxide = st.selectbox("Mínimo tiempo de dioxido de nitrogeno",["20:00:00", "21:00:00", "22:00:00", "23:00:00"])
min_ozone = st.number_input("Ozono mínimo",value=20)
max_time_ozone = st.selectbox("Máximo tiempo de ozono",["20:00:00", "21:00:00", "22:00:00", "23:00:00"])
min_time_ozone = st.selectbox("Mínimo tiempo de ozono",["20:00:00", "21:00:00", "22:00:00", "23:00:00"])
avg_sulphur_dioxide = st.number_input("Promedio azufre",value=5.0) # Corrected name
pm2_5_eaqi = st.number_input("PM2.5",value=30)
min_temperature_2m = st.number_input("Temperatura mínima",value=20)
avg_relative_humidity_2m = st.number_input("Humedad relativa promedio",value=20)
sum_rain = st.number_input("Lluvia acumulada",value=20)
avg_pressure_msl = st.number_input("Presión promedio",value=1000)
avg_cloud_cover = st.number_input("Porcentaje de nubes",value=20)
avg_vapour_pressure_deficit = st.number_input("Deficit de presión de vapor",value=2)
avg_wind_direction_100m = st.number_input("Dirección del viento",value=20)
max_wind_gusts_10m = st.number_input("Velocidad máxima del viento",value=20)
avg_soil_moisture_7_to_28cm = st.number_input("Humedad sobre el suelo",value=0.5) # Corrected name
avg_direct_radiation = st.number_input("Radiación directa promedio",value=20)
avg_terrestrial_radiation = st.number_input("Radiación terrestre promedio",value=300)

#Dataframe
input_dict = {
    'avg_nitrogen_dioxide': avg_nitrogen_dioxide,
    'max_time_nitrogen_dioxide': max_time_nitrogen_dioxide,
    'min_time_nitrogen_dioxide': min_time_nitrogen_dioxide,
    'min_ozone': min_ozone,
    'max_time_ozone': max_time_ozone,
    'min_time_ozone': min_time_ozone,
    'avg_sulphur_dioxide': avg_sulphur_dioxide, # Corrected key
    'pm2_5_eaqi': pm2_5_eaqi,
    'min_temperature_2m': min_temperature_2m,
    'avg_relative_humidity_2m': avg_relative_humidity_2m,
    'sum_rain': sum_rain,
    'avg_pressure_msl': avg_pressure_msl,
    'avg_cloud_cover': avg_cloud_cover,
    'avg_vapour_pressure_deficit': avg_vapour_pressure_deficit,
    'avg_wind_direction_100m': avg_wind_direction_100m,
    'max_wind_gusts_10m': max_wind_gusts_10m,
    'avg_soil_moisture_7_to_28cm': avg_soil_moisture_7_to_28cm, # Corrected key
    'avg_direct_radiation': avg_direct_radiation,
    'avg_terrestrial_radiation': avg_terrestrial_radiation
}

data = pd.DataFrame([input_dict])
# This step should happen after one-hot encoding in subsequent cells.
# data = data[variables]
#Dummies para variable con más de 2 categorías
data = pd.get_dummies(data, columns=['max_time_nitrogen_dioxide','min_time_nitrogen_dioxide','max_time_ozone','min_time_ozone','pm2_5_eaqi'], drop_first=False, dtype=int) #No se borra

data.head()
#Se adicionan las columnas faltantes
data=data.reindex(columns=variables,fill_value=0)
data.head()
#Hacemos la predicción con el random forest
Y_pred = model_rf.predict(data)
print(Y_pred)
data['Prediccion']=Y_pred
data.head()
#Predicciones finales
data
# Recordar medida de error del modelo

st.warning("El modelo tiene un error del 4% (mape: error porcentual)")
