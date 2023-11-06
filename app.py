import streamlit as st
import pandas as pd
from pycaret.regression import load_model, predict_model

model = load_model("ml_airbnb")
st.title("Sistema de predicción de precios Upgrade-Hub")

neighbourhood = st.selectbox('Barrio', options=[
    'Pankow', 'Friedrichshain-Kreuzberg', 'Neukölln',
    'Charlottenburg-Wilm.', 'Tempelhof - Schöneberg', 'Mitte',
    'Treptow - Köpenick', 'Marzahn - Hellersdorf',
    'Steglitz - Zehlendorf', 'Spandau', 'Lichtenberg', 'Reinickendorf'
])

property_type = st.selectbox('Tipo de Propiedad', options=[
    'Apartment', 'Townhouse', 'Houseboat', 'Bed and breakfast', 'Boat',
    'Guest suite', 'Loft', 'Serviced apartment', 'House',
    'Boutique hotel', 'Guesthouse', 'Other', 'Condominium', 'Chalet',
    'Nature lodge', 'Tiny house', 'Hotel', 'Villa', 'Cabin',
    'Lighthouse', 'Bungalow', 'Hostel', 'Cottage', 'Tent',
    'Earth house', 'Campsite', 'Castle', 'Camper/RV', 'Barn',
    'Casa particular (Cuba)', 'Aparthotel'
])

accommodates = st.slider('Número de Personas', min_value=1, max_value=17, value=1)
room_type = st.selectbox('Tipo de Habitación', options=['Private room', 'Entire home/apt', 'Shared room'])
maximum_nights = st.slider('Noches Máximas', min_value=1, max_value=100, value=1)
minimum_nights = st.slider('Noches Mínimas', min_value=1, max_value=10, value=1)

input_data = pd.DataFrame([[
    neighbourhood, property_type, accommodates, room_type,
    maximum_nights, minimum_nights
]], columns=['neighbourhood', 'property_type', 'accommodates', 'room_type', 'maximum_nights', 'minimum_nights'])


if st.button('¡Descubre el precio!'):
    prediction = predict_model(model, data=input_data)
    st.write(str(prediction["prediction_label"].values[0]) + ' euros')