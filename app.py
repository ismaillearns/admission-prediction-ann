import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

# Page Config
st.set_page_config(
    page_title="Admission Probability Predictor",
    page_icon="🎓",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main {background-color: #0f0f1a;}
    .stApp {background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%);}
    h1 {color: #00d4ff !important; text-align: center;}
    h3 {color: #ffffff !important;}
    .result-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 2px solid #00d4ff;
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        margin-top: 20px;
    }
    .stSlider > div > div {color: white;}
    label {color: #cccccc !important;}
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1>🎓 Admission Probability Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#aaaaaa;'>Using Artificial Neural Network (ANN) | BS(AI) Project</p>", unsafe_allow_html=True)
st.markdown("---")

# Load model
@st.cache_resource
def load_ann_model():
    model = load_model('admission_model.h5')
    return model

model = load_ann_model()

# Fit scaler on sample data ranges (same as training)
@st.cache_resource
def get_scaler():
    scaler = MinMaxScaler()
    sample = np.array([
        [260, 60, 1, 1.0, 1.0, 4.0, 0],
        [340, 120, 5, 5.0, 5.0, 10.0, 1]
    ])
    scaler.fit(sample)
    return scaler

scaler = get_scaler()

# Input Section
st.markdown("### 📋 Enter Your Profile")
st.markdown("")

col1, col2 = st.columns(2)

with col1:
    gat = st.slider("🎯 GAT Score", min_value=260, max_value=340, value=300, step=1)
    ielts = st.slider("🌐 IELTS Score", min_value=60.0, max_value=120.0, value=90.0, step=0.5)
    uni_rating = st.selectbox("🏫 University Rating", options=[1, 2, 3, 4, 5], index=2)
    research = st.selectbox("🔬 Research Experience", options=["No", "Yes"], index=0)

with col2:
    cgpa = st.slider("📊 CGPA", min_value=4.0, max_value=10.0, value=7.5, step=0.1)
    sop = st.selectbox("📝 Statement of Purpose (SOP)", options=[1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0], index=4)
    lor = st.selectbox("📄 Letter of Recommendation (LOR)", options=[1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0], index=4)

st.markdown("")

# Predict Button
if st.button("🚀 Predict My Admission Chance", use_container_width=True):
    
    # Prepare input
    research_val = 1 if research == "Yes" else 0
    input_data = np.array([[gat, ielts, uni_rating, sop, lor, cgpa, research_val]])
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)[0][0]
    percentage = round(prediction * 100, 2)
    
    # Color based on result
    if percentage >= 75:
        color = "#00ff88"
        emoji = "🟢"
        verdict = "Strong Profile! High chance of admission."
    elif percentage >= 55:
        color = "#ffaa00"
        emoji = "🟡"
        verdict = "Good Profile! Moderate chance of admission."
    else:
        color = "#ff4444"
        emoji = "🔴"
        verdict = "Needs Improvement. Work on your profile."

    # Result Box
    st.markdown(f"""
        <div class='result-box'>
            <h2 style='color:{color}; font-size:50px;'>{emoji} {percentage}%</h2>
            <h3 style='color:white;'>Chance of Admission</h3>
            <p style='color:{color}; font-size:18px;'>{verdict}</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # Gauge Chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=percentage,
        title={'text': "Admission Probability", 'font': {'color': 'white', 'size': 18}},
        number={'suffix': "%", 'font': {'color': color, 'size': 40}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': "white"},
            'bar': {'color': color},
            'bgcolor': "#1a1a2e",
            'bordercolor': "#00d4ff",
            'steps': [
                {'range': [0, 50], 'color': '#2d0000'},
                {'range': [50, 75], 'color': '#2d2000'},
                {'range': [75, 100], 'color': '#002d00'}
            ],
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "white"},
        height=300
    )
    st.plotly_chart(fig, use_container_width=True)

    # Profile Summary
    st.markdown("### 📊 Your Profile Summary")
    profile_df = pd.DataFrame({
        'Feature': ['GAT Score', 'IELTS Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research'],
        'Your Value': [gat, ielts, uni_rating, sop, lor, cgpa, research],
        'Max Possible': [340, 120, 5, 5.0, 5.0, 10.0, 'Yes']
    })
    st.dataframe(profile_df, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center; color:#555555;'>Admission Probability Predictor | ANN Project | Mohammad Ismail</p>", unsafe_allow_html=True)
