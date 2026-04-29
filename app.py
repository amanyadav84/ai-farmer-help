import streamlit as st
import joblib
import numpy as np
import requests

import requests

def get_crop_price(crop):
    data = {
        "rice": 2200,
        "wheat": 2100,
        "maize": 1800,
        "cotton": 5500,
        "tomato": 1200,
        "potato": 1000
    }

    return data.get(crop.lower(), None)
def analyze_soil(N, P, K, ph):
    suggestions = []

    # Nitrogen
    if N < 50:
        suggestions.append("Add Nitrogen-rich fertilizer (Urea)")
    
    # Phosphorus
    if P < 40:
        suggestions.append("Add Phosphorus fertilizer (DAP)")
    
    # Potassium
    if K < 40:
        suggestions.append("Add Potassium fertilizer (MOP/Potash)")

    # pH check
    if ph < 6:
        suggestions.append("Soil is acidic → add lime")
    elif ph > 7.5:
        suggestions.append("Soil is alkaline → add gypsum")

    return suggestions
model = joblib.load("crop_model.pkl")
le = joblib.load("label_encoder.pkl")

API_KEY = "146ba1b8da44386da50ee40a1ae33dd0"

def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    
    response = requests.get(url)
    data = response.json()

    if data["cod"] != 200:
        return None

    temp = data["main"]["temp"]
    humidity = data["main"]["humidity"]

    return temp, humidity
st.set_page_config(page_title="Crop Recommendation", layout="wide")

st.title("🌾 Smart Crop Recommendation System")
# Sidebar Title
st.sidebar.header("📊 Enter Soil & Weather Details")
use_api = st.sidebar.checkbox("Use Live Weather (Auto)", value=True)
city = st.sidebar.text_input("Enter City Name (e.g., Jaipur)")
if city:
    weather = get_weather(city)

    if weather:
        temp, humidity = weather
        st.sidebar.success(f"🌡 Temperature: {temp} °C")
        st.sidebar.success(f"💧 Humidity: {humidity} %")
    else:
        st.sidebar.error("City not found")

# -------- Soil Nutrients --------
st.sidebar.subheader("Soil Nutrients (kg/ha)")

N = st.sidebar.number_input(
    "Nitrogen (N) [0 - 140]",
    min_value=0,
    max_value=140,
    value=50,
    help="Typical nitrogen range in soil"
)

P = st.sidebar.number_input(
    "Phosphorus (P) [5 - 145]",
    min_value=5,
    max_value=145,
    value=50,
    help="Important for root growth"
)

K = st.sidebar.number_input(
    "Potassium (K) [5 - 205]",
    min_value=5,
    max_value=205,
    value=50,
    help="Helps plant resistance"
)

# -------- Weather --------
st.sidebar.subheader("🌦 Weather Conditions")

temp = st.sidebar.slider(
    "Temperature (°C)",
    min_value=0,
    max_value=50,
    value=25,
    help="Average temperature",
    disabled=use_api
)

humidity = st.sidebar.slider(
    "Humidity (%)",
    min_value=10,
    max_value=100,
    value=60,
    disabled=use_api
)
if use_api and city and weather:
    temperature, humidity = weather

rainfall = st.sidebar.slider(
    "Rainfall (mm)",
    min_value=0,
    max_value=300,
    value=100
)

# -------- Soil Condition --------
st.sidebar.subheader("🌍 Soil Condition")

ph = st.sidebar.slider(
    "Soil pH [0 - 14]",
    min_value=0.0,
    max_value=14.0,
    value=6.5,
    help="Ideal range is 6 - 7.5"
)



# Main area preview
st.subheader(" SELECTED VALUES")
st.write({
    "Nitrogen": N,
    "Phosphorus": P,
    "Potassium": K,
    "Temperature": temp,
    "Humidity": humidity,
    "pH": ph,
    "Rainfall": rainfall
})

# =========================
# Prediction Button
# =========================
if st.button("🌱 Predict Crop"):

    # Prepare input
    data = np.array([[N, P, K, temp, humidity, ph, rainfall]])

    # =========================
    # Get probabilities
    # =========================
    probs = model.predict_proba(data)[0]

    # =========================
    # Top 3 predictions
    # =========================
    top3_idx = np.argsort(probs)[-3:][::-1]

    st.subheader("🌾 Top Crop Recommendations")

    for i in top3_idx:
        crop_name = le.inverse_transform([i])[0]   # FIX: convert number → crop name
        confidence = probs[i] * 100

        # Display result
        st.write(f"{crop_name.upper()} → {confidence:.2f}%")

        if confidence > 80:
            st.success(f"{crop_name} is highly recommended ")
        elif confidence > 50:
            st.warning(f"{crop_name} is moderately suitable ")
        else:
            st.error(f"{crop_name} may not be ideal ")

    
    best_crop_idx = top3_idx[0]
    best_crop = le.inverse_transform([best_crop_idx])[0]

    st.markdown("---")
    st.success(f"🌟 Best Crop for Your Land: **{best_crop.upper()}**")
st.markdown("---")
st.subheader("🧪 Soil Health Analysis")

soil_tips = analyze_soil(N, P, K, ph)

if soil_tips:
    for t in soil_tips:
        st.write(f"{t}")
else:
    st.success("Your soil is healthy!")   

st.markdown("---")

st.markdown("---")

st.header("LIVE CROP PRICE")

crop_selected = st.selectbox(
    "Select Crop",
    ["rice", "wheat", "maize", "cotton", "tomato", "potato"]
)

if st.button("Get Price"):
    price = get_crop_price(crop_selected)

    if price:
        st.success(f" Price: ₹{price}")
   
st.markdown("---")
st.header("   How to Use App ")

st.write("""
1. Enter your soil values (Nitrogen, Phosphorus, Potassium)
2. Enter your city name (for live weather)
3. Enable **"Use Live Weather"** if you want automatic weather data
4. Or use sliders for manual input
5. Click **Predict Crop**
6. View top crop recommendations with confidence %
7. click show price and select crop
""")
st.markdown("---")
st.header(" About This Project")

st.write("""
This Smart Krishi AI system helps farmers by recommending the best crops 
based on soil and weather conditions.

This system uses Machine Learning to recommend the best crops based on:
 - Soil nutrients (N, P, K)
 - Weather conditions
 - pH level and rainfall

💡The goal is to improve farming productivity and support farmers with smart technology.     
                               
📝NOTE-its also include live weather 🌦️ Detect option(Temperature and Humidity )
         

Developed by: Aman  
Email: amanyadav843233@gmail.com
""")    