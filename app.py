import streamlit as st
import pandas as pd
from pycaret.regression import load_model, predict_model

model = load_model("ml_airbnb_xgb")
st.title("Vacation rental prediction system based on Airbnb dataset")

neighbourhood = st.selectbox('Neighbourhood', options=[
    'Pankow', 'Friedrichshain-Kreuzberg', 'Neukölln',
    'Charlottenburg-Wilm.', 'Tempelhof - Schöneberg', 'Mitte',
    'Treptow - Köpenick', 'Marzahn - Hellersdorf',
    'Steglitz - Zehlendorf', 'Spandau', 'Lichtenberg', 'Reinickendorf'
])

property_type = st.selectbox('Property type', options=[
    'Entire rental unit', 'Entire loft', 'Private room in rental unit',
    'Entire condo', 'Entire guest suite', 'Entire home',
    'Entire townhouse', 'Private room in condo',
    'Private room in home', 'Private room',
    'Shared room in rental unit', 'Entire place', 'Entire guesthouse',
    'Private room in loft', 'Entire serviced apartment',
    'Room in boutique hotel', 'Tiny home',
    'Private room in bed and breakfast', 'Shared room in boat',
    'Private room in villa', 'Private room in tipi', 'Room in hostel',
    'Private room in townhouse', 'Entire bungalow', 'Room in hotel',
    'Room in serviced apartment', 'Private room in pension',
    'Room in aparthotel', 'Private room in serviced apartment',
    'Entire vacation home', 'Private room in hostel',
    'Shared room in condo', 'Private room in guesthouse',
    'Shared room in hostel', 'Private room in guest suite', 'Cave',
    'Entire villa', 'Camper/RV', 'Private room in casa particular',
    'Shared room in hotel', 'Houseboat',
    'Shared room in boutique hotel', 'Private room in boat',
    'Private room in houseboat', 'Private room in vacation home',
    'Private room in castle', 'Boat', 'Shared room in casa particular',
    'Private room in shipping container', 'Private room in bungalow',
    'Shared room in serviced apartment', 'Room in bed and breakfast',
    'Entire cottage', 'Private room in cave', 'Dome', 'Bus',
    'Shared room in bed and breakfast', 'Shared room in guesthouse',
    'Entire chalet', 'Shared room in loft', 'Entire cabin',
    'Private room in tiny home', 'Treehouse', 'Shared room', 'Island',
    'Shared room in tiny home'
])

accommodates = st.slider('Accommodates', min_value=1, max_value=16, value=1)
room_type = st.selectbox('Room type', options=['Entire home/apt', 'Private room', 'Shared room', 'Hotel room'])
maximum_nights = st.slider('Maximum nights', min_value=1, max_value=100, value=1)
minimum_nights = st.slider('Minimum nights', min_value=1, max_value=10, value=1)

input_data = pd.DataFrame([[
    neighbourhood, property_type, accommodates, room_type,
    maximum_nights, minimum_nights
]], columns=['neighbourhood_group', 'property_type', 'accommodates', 'room_type', 'maximum_nights', 'minimum_nights'])


if st.button('Discover the price!'):
    prediction = predict_model(model, data=input_data)
    st.write(str(prediction["prediction_label"].values[0]) + '€')