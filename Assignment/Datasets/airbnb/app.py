import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

st.title("Airbnb Price Prediction Model")
st.write("This model helps Airbnb owners estimate their listing prices.")

# Load the saved components with error handling
@st.cache_resource
def load_models():
    try:
        # Try to load the clean versions first
        model = joblib.load('model_clean.pkl')
        scaler = joblib.load('scaler_clean.pkl')
        label_encoders = joblib.load('label_encoders_clean.pkl')
        
        # Verify they loaded correctly
        st.success("All models loaded successfully!")
        st.write(f"Model type: {type(model)}")
        st.write(f"Scaler type: {type(scaler)}")
        st.write(f"Encoders loaded: {list(label_encoders.keys())}")
        
        return model, scaler, label_encoders
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.error("Please make sure you've run the model saving code in your notebook first.")
        st.stop()

# Load models
model, scaler, label_encoders = load_models()

# Alternative: Load encoder mappings from JSON if needed
@st.cache_data
def load_encoder_mappings():
    try:
        with open('encoder_mappings.json', 'r') as f:
            return json.load(f)
    except:
        return None

encoder_mappings = load_encoder_mappings()
if encoder_mappings:
    st.write("Encoder mappings loaded from JSON backup")

# Neighborhood options (you might want to get these from the actual encoder)
try:
    neighborhood_options = list(label_encoders['neighbourhood'].classes_)
except:
    # Fallback to your hardcoded list
    neighborhood_options = ['Kensington', 'Midtown', 'Harlem', 'Clinton Hill', 'East Harlem',
                           'Murray Hill', 'Bedford-Stuyvesant', "Hell's Kitchen",
                           'Upper West Side', 'Chinatown', 'South Slope', 'West Village',
                           'Williamsburg', 'Fort Greene', 'Chelsea', 'Crown Heights',
                           'Park Slope', 'Windsor Terrace', 'Inwood', 'East Village',
                           'Greenpoint', 'Bushwick', 'Flatbush', 'Lower East Side']

# User inputs
selected_neighborhood = st.selectbox('Neighborhood', neighborhood_options)

# For neighbourhood_group (boroughs)
try:
    neighbourhood_group_options = list(label_encoders['neighbourhood_group'].classes_)
except:
    neighbourhood_group_options = ['Brooklyn', 'Manhattan', 'Queens', 'Staten Island', 'Bronx']

neighbourhood_group = st.selectbox('Neighbourhood Group', neighbourhood_group_options)

# For room_type
try:
    room_type_options = list(label_encoders['room_type'].classes_)
except:
    room_type_options = ['Private room', 'Entire home/apt', 'Shared room']

room_type = st.selectbox('Room Type', room_type_options)

# Numeric inputs
longitude = st.slider('Longitude', -74.95, -73.70, value=-74.0)
latitude = st.slider('Latitude', 40.5, 40.9, value=40.7)
minimum_nights = st.slider('Minimum Nights', 1, 1250, value=1)
number_of_reviews = st.slider('Number of Reviews', 0, 629, value=0)
calculated_host_listings_count = st.slider('Host Listings Count', 1, 327, value=1)
availability_365 = st.slider('Availability (days)', 0, 365, value=365)

# Make prediction
if st.button('Predict Airbnb Price'):
    try:
        # Encode categorical features using the loaded encoders
        neighbourhood_group_encoded = label_encoders['neighbourhood_group'].transform([neighbourhood_group])[0]
        neighborhood_encoded = label_encoders['neighbourhood'].transform([selected_neighborhood])[0]
        room_type_encoded = label_encoders['room_type'].transform([room_type])[0]
        
        st.write("✓ Categorical features encoded successfully")
        
        # Create numerical features array for scaling
        # Order should match what was used in training: 
        # minimum_nights, number_of_reviews, calculated_host_listings_count, availability_365, longitude, latitude
        numerical_features = np.array([[minimum_nights, number_of_reviews, calculated_host_listings_count, 
                                       availability_365, longitude, latitude]])
        
        st.write(f"Raw numerical features shape: {numerical_features.shape}")
        
        # Scale the numerical features
        scaled_numerical = scaler.transform(numerical_features)
        st.write("✓ Numerical features scaled successfully")
        
        # Create final feature array in the order expected by the model
        # Order: neighbourhood_group, neighbourhood, room_type, latitude, longitude, minimum_nights, 
        # number_of_reviews, calculated_host_listings_count, availability_365
        features_for_prediction = np.array([[
            neighbourhood_group_encoded,    # neighbourhood_group
            neighborhood_encoded,           # neighbourhood  
            room_type_encoded,             # room_type
            scaled_numerical[0][5],        # latitude (scaled)
            scaled_numerical[0][4],        # longitude (scaled)
            scaled_numerical[0][0],        # minimum_nights (scaled)
            scaled_numerical[0][1],        # number_of_reviews (scaled)
            scaled_numerical[0][2],        # calculated_host_listings_count (scaled)
            scaled_numerical[0][3]         # availability_365 (scaled)
        ]])
        
        st.write(f"Final features shape: {features_for_prediction.shape}")
        st.write(f"Final features: {features_for_prediction}")
        
        # Verify model type before prediction
        st.write(f"Model type before prediction: {type(model)}")
        
        # Make prediction
        prediction = model.predict(features_for_prediction)
        predicted_price = prediction[0]
        
        st.success(f'🎉 Predicted price: ${predicted_price:.2f}')
        
        # Show input summary
        st.write("### Input Summary:")
        st.write(f"- Neighborhood: {selected_neighborhood}")
        st.write(f"- Borough: {neighbourhood_group}")
        st.write(f"- Room Type: {room_type}")
        st.write(f"- Location: ({latitude:.3f}, {longitude:.3f})")
        st.write(f"- Minimum Nights: {minimum_nights}")
        st.write(f"- Reviews: {number_of_reviews}")
        st.write(f"- Host Listings: {calculated_host_listings_count}")
        st.write(f"- Availability: {availability_365} days/year")
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        import traceback
        st.code(traceback.format_exc())