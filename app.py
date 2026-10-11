import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json
import os
import sys
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from inference import build_input_row, load_artifacts  # noqa: E402

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
    return load_artifacts()


@st.cache_data
def load_options():
    """Dropdown options come from the data, so they always match what the models were trained on."""
    df = pd.read_csv('data/clean_data_for_eda.csv', usecols=[
        'State', 'City', 'Locality', 'Property_Type', 'Furnished_Status',
        'Public_Transport_Accessibility', 'Parking_Space', 'Security', 'Facing',
        'Owner_Type', 'Availability_Status', 'Amenities'])
    opts = {col: sorted(df[col].unique()) for col in df.columns if col not in ('City', 'Amenities')}
    opts['cities_by_state'] = df.groupby('State')['City'].unique().apply(sorted).to_dict()
    opts['Amenities'] = sorted({a.strip() for s in df['Amenities'].unique() for a in s.split(',')})
    return opts


def top_importances(model, feature_names, n=10):
    if hasattr(model, 'feature_importances_'):
        values = model.feature_importances_
    elif hasattr(model, 'coef_'):
        values = np.abs(np.ravel(model.coef_))
    else:
        return None
    return pd.Series(values, index=feature_names).sort_values(ascending=False).head(n)[::-1]


try:
    assets = load_assets()
    opts = load_options()
except Exception as e:
    st.error(f"Error loading models. Have you run the training script? ({str(e)})")
    st.stop()

clf, reg, features = assets['clf'], assets['reg'], assets['features']

# Header
st.title("🏢 Real Estate Insights Pro")
st.markdown("Predict property profitability & forecast future values using Advanced Machine Learning.")

# Tabs
tab_predict, tab_dashboard, tab_explorer, tab_models = st.tabs(
    ["🔮 Predictive Analytics", "📊 Market Dashboard", "🔎 Property Explorer", "🧪 Model Performance"])

with tab_predict:
    st.subheader("Property Assessment Form")

    # State sits outside the form so the City list can react to it.
    state = st.selectbox("State", opts['State'])

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            city = st.selectbox("City", opts['cities_by_state'][state])
            locality = st.selectbox("Locality", opts['Locality'])
            prop_type = st.selectbox("Property Type", opts['Property_Type'])
            bhk = st.number_input("BHK", min_value=1, max_value=5, value=2)
            furnishing = st.selectbox("Furnished Status", opts['Furnished_Status'])
            availability = st.selectbox("Availability", opts['Availability_Status'])

        with col2:
            size = st.number_input("Size in SqFt", min_value=500, max_value=5000, value=1200)
            price = st.number_input("Current Price (Lakhs)", min_value=10.0, max_value=500.0, value=50.0)
            year_built = st.number_input("Year Built", min_value=1990, max_value=2025, value=2015)
            floor_no = st.number_input("Floor Number", min_value=0, max_value=30, value=2)
            total_floors = st.number_input("Total Floors", min_value=1, max_value=30, value=10)
            facing = st.selectbox("Facing", opts['Facing'])

        with col3:
            schools = st.number_input("Nearby Schools", min_value=0, max_value=10, value=2)
            hospitals = st.number_input("Nearby Hospitals", min_value=0, max_value=10, value=1)
            transport = st.selectbox("Public Transport Accessibility", opts['Public_Transport_Accessibility'])
            parking = st.selectbox("Parking Space", opts['Parking_Space'])
            security = st.selectbox("Security", opts['Security'])
            owner = st.selectbox("Owner Type", opts['Owner_Type'])

        amenities = st.multiselect("Amenities", opts['Amenities'], default=['Gym', 'Pool'])

        submitted = st.form_submit_button("Analyze Investment")

    if submitted:
        df_input = build_input_row({
            'State': state, 'City': city, 'Locality': locality,
            'Property_Type': prop_type, 'Furnished_Status': furnishing,
            'Availability_Status': availability, 'Public_Transport_Accessibility': transport,
            'Parking_Space': parking, 'Security': security, 'Facing': facing, 'Owner_Type': owner,
            'BHK': bhk, 'Size_in_SqFt': size, 'Price_in_Lakhs': price, 'Year_Built': year_built,
            'Floor_No': floor_no, 'Total_Floors': total_floors,
            'Nearby_Schools': schools, 'Nearby_Hospitals': hospitals,
            'Amenities': ', '.join(amenities),
        }, features, assets['encodings'], assets['scaler'])

        # Predict
        is_good = clf.predict(df_input)[0]
        confidence = clf.predict_proba(df_input)[0][int(is_good)] * 100
        future_price = reg.predict(df_input)[0]

        st.markdown("---")
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            if is_good == 1:
                st.markdown(f"""
                <div class="metric-card" style="border-left: 5px solid #2ea043;">
                    <h3 style="color: #2ea043;">🌟 Good Investment</h3>
                    <p>Priced below the locality median per sq ft. Model confidence: <b>{confidence:.1f}%</b></p>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f"""
                <div class="metric-card" style="border-left: 5px solid #f85149;">
                    <h3 style="color: #f85149;">⚠️ Not a Good Investment</h3>
                    <p>Price is not below the locality norm. Model confidence: <b>{confidence:.1f}%</b></p>
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

        with st.expander("What drives these predictions? (feature importance)"):
            imp_col1, imp_col2 = st.columns(2)
            for col, model, title in [(imp_col1, clf, "Investment classifier"),
                                      (imp_col2, reg, "Price forecast regressor")]:
                imp = top_importances(model, features)
                with col:
                    st.markdown(f"**{title}** ({type(model).__name__})")
                    if imp is not None:
                        st.bar_chart(imp, horizontal=True)

@st.cache_data
def load_market_data():
    return pd.read_csv('data/clean_data_for_eda.csv', usecols=[
        'State', 'City', 'Locality', 'Property_Type', 'BHK', 'Size_in_SqFt', 'Price_in_Lakhs',
        'Price_per_SqFt', 'Year_Built', 'Availability_Status', 'Good_Investment'])


EDA_PLOTS = [
    ("Q1 Distribution of property prices", "price_distribution"),
    ("Q2 Distribution of property sizes", "size_distribution"),
    ("Q3 Price per SqFt by property type", "price_by_type"),
    ("Q4 Size vs price", "q04_size_vs_price"),
    ("Q5 Outliers in price per SqFt and size", "q05_outliers"),
    ("Q6 Average price per SqFt by state", "q06_pps_by_state"),
    ("Q7 Average price by city", "q07_price_by_city"),
    ("Q8 Median property age by locality", "q08_median_age_by_locality"),
    ("Q9 BHK distribution across cities", "q09_bhk_by_city"),
    ("Q10 Top 5 most expensive localities", "top_localities"),
    ("Q10 Price trend of those localities", "q10_top5_price_trend"),
    ("Q11 Correlation heatmap", "correlation_heatmap"),
    ("Q12 Nearby schools vs price per SqFt", "q12_schools_vs_pps"),
    ("Q13 Nearby hospitals vs price per SqFt", "q13_hospitals_vs_pps"),
    ("Q14 Price by furnished status", "q14_price_by_furnishing"),
    ("Q15 Price per SqFt by facing", "q15_pps_by_facing"),
    ("Q16 Properties per owner type", "q16_owner_type"),
    ("Q17 Properties per availability status", "q17_availability"),
    ("Q18 Parking space vs price", "q18_parking_vs_price"),
    ("Q19 Amenities vs price per SqFt", "q19_amenities_vs_pps"),
    ("Q20 Public transport vs price and investment potential", "q20_transport"),
]

with tab_dashboard:
    st.subheader("Market Insights & EDA")
    market = load_market_data()

    st.markdown("**Average price (Lakhs) by state and BHK**")
    heat = market.pivot_table(index='State', columns='BHK', values='Price_in_Lakhs', aggfunc='mean')
    st.plotly_chart(px.imshow(heat, aspect='auto', color_continuous_scale='Viridis',
                              labels=dict(color='Avg Price (Lakhs)')), width='stretch')

    left, right = st.columns(2)
    with left:
        top_n = st.slider("Cities to show", 5, 42, 15)
        city_price = (market.groupby('City')['Price_in_Lakhs'].mean()
                      .sort_values(ascending=False).head(top_n).reset_index())
        st.plotly_chart(px.bar(city_price, x='Price_in_Lakhs', y='City', orientation='h',
                               title='Average price by city (Lakhs)')
                        .update_yaxes(autorange='reversed'), width='stretch')
    with right:
        trend_state = st.selectbox("Price trend for", ['All states'] + opts['State'])
        trend_df = market if trend_state == 'All states' else market[market['State'] == trend_state]
        trend = trend_df.groupby('Year_Built')['Price_in_Lakhs'].mean().reset_index()
        st.plotly_chart(px.line(trend, x='Year_Built', y='Price_in_Lakhs', markers=True,
                                title='Average price by year built (Lakhs)'), width='stretch')

    with st.expander("EDA gallery: answers to the 20 analysis questions"):
        st.caption("Crime-rate and infrastructure-score analyses are not possible: the dataset has no such columns.")
        gal1, gal2 = st.columns(2)
        for i, (title, name) in enumerate(EDA_PLOTS):
            path = f"plots/{name}.png"
            with (gal1 if i % 2 == 0 else gal2):
                if os.path.exists(path):
                    st.image(Image.open(path), caption=title, width='stretch')
                else:
                    st.warning(f"Plot not found: {path}")

with tab_explorer:
    st.subheader("Property Explorer")
    market = load_market_data()
    f1, f2, f3 = st.columns(3)
    with f1:
        sel_states = st.multiselect("State", opts['State'])
        city_choices = sorted(c for s in (sel_states or opts['State']) for c in opts['cities_by_state'][s])
        sel_cities = st.multiselect("City", city_choices)
    with f2:
        price_range = st.slider("Price (Lakhs)", 10.0, 500.0, (10.0, 500.0))
        sel_bhk = st.multiselect("BHK", sorted(market['BHK'].unique()))
    with f3:
        sel_types = st.multiselect("Property Type", opts['Property_Type'])
        only_good = st.checkbox("Only properties labelled Good Investment")

    mask = market['Price_in_Lakhs'].between(*price_range)
    if sel_states:
        mask &= market['State'].isin(sel_states)
    if sel_cities:
        mask &= market['City'].isin(sel_cities)
    if sel_bhk:
        mask &= market['BHK'].isin(sel_bhk)
    if sel_types:
        mask &= market['Property_Type'].isin(sel_types)
    if only_good:
        mask &= market['Good_Investment'] == 1
    found = market[mask]

    m1, m2, m3 = st.columns(3)
    m1.metric("Matching properties", f"{len(found):,}")
    m2.metric("Average price (Lakhs)", f"{found['Price_in_Lakhs'].mean():.1f}" if len(found) else "-")
    m3.metric("Average price per SqFt (Lakhs)", f"{found['Price_per_SqFt'].mean():.3f}" if len(found) else "-")
    st.caption("Showing the 200 cheapest matches.")
    st.dataframe(found.sort_values('Price_in_Lakhs').head(200), width='stretch', hide_index=True)

with tab_models:
    st.subheader("Model Performance")
    metrics_path = 'models/metrics.json'
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        st.markdown(f"Best classifier: **{metrics['best']['classifier']}** (highest F1). "
                    f"Best regressor: **{metrics['best']['regressor']}** (highest R²). "
                    "All runs are tracked in MLflow.")
        cls_df = pd.DataFrame({k: {m: v for m, v in r.items() if m != 'confusion_matrix'}
                               for k, r in metrics['classification'].items()}).T
        st.markdown("**Classification models**")
        st.dataframe(cls_df.style.format("{:.4f}"), width='stretch')
        best_cm = np.array(metrics['classification'][metrics['best']['classifier']]['confusion_matrix'])
        st.plotly_chart(px.imshow(best_cm, text_auto=True, color_continuous_scale='Blues',
                                  x=['Predicted: Not Good', 'Predicted: Good'],
                                  y=['Actual: Not Good', 'Actual: Good'],
                                  title=f"Confusion matrix - {metrics['best']['classifier']}"),
                        width='stretch')
        st.markdown("**Regression models**")
        st.dataframe(pd.DataFrame(metrics['regression']).T.style.format("{:.4f}"), width='stretch')
    else:
        st.info("Run `python src/train.py` to generate model metrics.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #8b949e;'>Powered by MLflow & Streamlit | Intelligent Real Estate Advisor</p>", unsafe_allow_html=True)
