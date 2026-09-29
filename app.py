import streamlit as st
import pandas as pd
import joblib

# 1. Load the saved model and column names
model = joblib.load('car_price_model.pkl')
model_columns = joblib.load('model_columns.pkl')

st.title("🚗 Car Price Predictor")
st.write("Enter the car's details below to get an estimated price.")

# 2. Create the user interface (Sliders and Dropdowns)
col1, col2 = st.columns(2)

with col1:
    year = st.slider("Manufacturing Year", min_value=2010, max_value=2024, value=2018)
    mileage = st.number_input("Mileage", min_value=0, value=30000)
    engine_size = st.number_input("Engine Size (cc)", min_value=500, value=2000)
    mpg = st.number_input("Miles Per Gallon (MPG)", min_value=5, value=25)

with col2:
    brand = st.selectbox("Brand", ["Honda", "Toyota", "BMW", "Ford", "Audi"]) 
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid", "Electric"])
    transmission = st.selectbox("Transmission", ["Automatic", "Manual"])
    owner = st.selectbox("Owner History", ["First", "Second", "Third"])

# 3. Prediction Button
if st.button("Predict Price"):
    # Create a dictionary of the user's input
    user_input = {
        'Year': year,
        'Mileage': mileage,
        'Engine_Size': engine_size,
        'Mpg': mpg,
        f'Brand_{brand}': 1,
        f'Fuel_Type_{fuel_type}': 1,
        f'Transmission_{transmission}': 1,
        f'Owner_{owner}': 1
    }

    # Convert to a DataFrame with exactly one row
    input_df = pd.DataFrame([user_input])

    # Reindex to match the columns the model was trained on, filling missing ones with 0
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # Run the prediction
    prediction = model.predict(input_df)[0]
    
    # Display the result
    st.success(f"Estimated Car Price: ${prediction:,.2f}")