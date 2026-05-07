import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image

# Setup Page Configuration
st.set_page_config(page_title="Real Estate Insights Pro", layout="wide", page_icon="🏢")

# Custom CSS for Premium Aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        background: linear-gradient(135deg, #6e8efb, #a777e3);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(167, 119, 227, 0.4);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
    }
    h1, h2, h3 {
        color: #e6edf3;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Load Models and Encoders
@st.cache_resource
def load_assets():
    clf = joblib.load('models/best_classifier.pkl')
    reg = joblib.load('models/best_regressor.pkl')
    scaler = joblib.load('models/scaler.pkl')
    features = joblib.load('models/feature_columns.pkl')
    
    state_enc = joblib.load('models/State_encoding.pkl')
    city_enc = joblib.load('models/City_encoding.pkl')
    locality_enc = joblib.load('models/Locality_encoding.pkl')
    
    return clf, reg, scaler, features, state_enc, city_enc, locality_enc

try:
    clf, reg, scaler, feature_cols, state_enc, city_enc, locality_enc = load_assets()
except Exception as e:
    st.error(f"Error loading models. Have you run the training script? ({str(e)})")
    st.stop()

# Header
st.title("🏢 Real Estate Insights Pro")
st.markdown("Predict property profitability & forecast future values using Advanced Machine Learning.")

# Tabs
tab_predict, tab_dashboard = st.tabs(["🔮 Predictive Analytics", "📊 Market Dashboard"])

with tab_predict:
    st.subheader("Property Assessment Form")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            state = st.selectbox("State", list(state_enc.keys()))
            city = st.selectbox("City", [c for c in city_enc.keys()]) # Ideally filter by state
            locality = st.selectbox("Locality", [l for l in locality_enc.keys()]) # Ideally filter by city
            prop_type = st.selectbox("Property Type", ['Apartment', 'Villa', 'House', 'Penthouse'])
            bhk = st.number_input("BHK", min_value=1, max_value=10, value=2)
            
        with col2:
            size = st.number_input("Size in SqFt", min_value=100, max_value=20000, value=1200)
            price = st.number_input("Current Price (Lakhs)", min_value=1.0, max_value=10000.0, value=50.0)
            year_built = st.number_input("Year Built", min_value=1950, max_value=2030, value=2015)
            floor_no = st.number_input("Floor Number", min_value=0, max_value=100, value=2)
            total_floors = st.number_input("Total Floors", min_value=1, max_value=150, value=10)
            
        with col3:
            schools = st.number_input("Nearby Schools", min_value=0, value=2)
            hospitals = st.number_input("Nearby Hospitals", min_value=0, value=1)
            amenities = st.text_input("Amenities (comma separated)", value="Gym, Pool, Security")
            furnishing = st.selectbox("Furnished Status", ['Unfurnished', 'Semi-Furnished', 'Fully Furnished'])
            availability = st.selectbox("Availability", ['Ready_to_Move', 'Under_Construction'])
            
        submitted = st.form_submit_button("Analyze Investment")

    if submitted:
        # Preprocessing Input
        amenity_score = len(amenities.split(',')) if amenities else 0
        price_per_sqft = (price * 100000) / size if size > 0 else 0
        age = 2026 - year_built # Assuming 2026 is current year as per constraints
        
        # Mean encoded values
        state_val = state_enc.get(state, np.mean(list(state_enc.values())))
        city_val = city_enc.get(city, np.mean(list(city_enc.values())))
        loc_val = locality_enc.get(locality, np.mean(list(locality_enc.values())))
        
        # Prepare dataframe
        input_dict = {
            'BHK': bhk, 'Size_in_SqFt': size, 'Price_in_Lakhs': price,
            'Price_per_SqFt': price_per_sqft, 'Year_Built': year_built,
            'Floor_No': floor_no, 'Total_Floors': total_floors, 'Age_of_Property': age,
            'Nearby_Schools': schools, 'Nearby_Hospitals': hospitals,
            'Amenity_Score': amenity_score, 'State_Encoded': state_val,
            'City_Encoded': city_val, 'Locality_Encoded': loc_val
        }
        
        df_input = pd.DataFrame([input_dict])
        
        # OHE Mapping
        cat_mappings = {
            f'Property_Type_{prop_type}': 1,
            f'Furnished_Status_{furnishing}': 1,
            f'Availability_Status_{availability}': 1
        }
        
        # Ensure all feature columns exist
        for col in feature_cols:
            if col not in df_input.columns:
                df_input[col] = cat_mappings.get(col, 0)
                
        # Reorder to match training
        df_input = df_input[feature_cols]
        
        # Scale
        num_cols = ['BHK', 'Size_in_SqFt', 'Price_in_Lakhs', 'Price_per_SqFt', 'Year_Built', 
                    'Floor_No', 'Total_Floors', 'Age_of_Property', 'Nearby_Schools', 
                    'Nearby_Hospitals', 'Amenity_Score']
        df_input[num_cols] = scaler.transform(df_input[num_cols])
        
        # Predict
        is_good = clf.predict(df_input)[0]
        future_price = reg.predict(df_input)[0]
        
        st.markdown("---")
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            if is_good == 1:
                st.markdown(f"""
                <div class="metric-card" style="border-left: 5px solid #2ea043;">
                    <h3 style="color: #2ea043;">🌟 Excellent Investment</h3>
                    <p>Based on current market trends and locality medians, this property is undervalued.</p>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f"""
                <div class="metric-card" style="border-left: 5px solid #f85149;">
                    <h3 style="color: #f85149;">⚠️ High Risk Investment</h3>
                    <p>Property price is above locality average. Proceed with caution.</p>
                </div>
                """, unsafe_allow_html=True)
                
        with res_col2:
            growth = ((future_price - price) / price) * 100
            color = "#2ea043" if growth > 0 else "#f85149"
            arrow = "↗" if growth > 0 else "↘"
            st.markdown(f"""
            <div class="metric-card">
                <h3>5-Year Value Forecast</h3>
                <h2 style="color: {color};">₹ {future_price:.2f} Lakhs</h2>
                <p>Projected ROI: <span style="color: {color}; font-weight: bold;">{arrow} {growth:.2f}%</span></p>
            </div>
            """, unsafe_allow_html=True)

with tab_dashboard:
    st.subheader("Market Insights & EDA")
    
    plots = [
        ("Distribution of Property Prices", "plots/price_distribution.png"),
        ("Price per SqFt by Property Type", "plots/price_by_type.png"),
        ("Correlation Heatmap", "plots/correlation_heatmap.png"),
        ("Top 5 Expensive Localities", "plots/top_localities.png")
    ]
    
    col1, col2 = st.columns(2)
    for i, (title, path) in enumerate(plots):
        if os.path.exists(path):
            img = Image.open(path)
            if i % 2 == 0:
                with col1:
                    st.image(img, caption=title, use_column_width=True)
            else:
                with col2:
                    st.image(img, caption=title, use_column_width=True)
        else:
            st.warning(f"Plot not found: {path}")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #8b949e;'>Powered by MLflow & Streamlit | Intelligent Real Estate Advisor</p>", unsafe_allow_html=True)
