import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

st.set_page_config(page_title="Admission Probability Predictor", page_icon="🎓", layout="centered")

st.markdown("""
    <style>
    .stApp {background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%);}
    h1 {color: #00d4ff !important; text-align: center;}
    .result-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 2px solid #00d4ff;
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        margin-top: 20px;
    }
    label {color: #cccccc !important;}
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>🎓 Admission Probability Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#aaaaaa;'>Using Artificial Neural Network (ANN) | BS(AI) Project</p>", unsafe_allow_html=True)
st.markdown("---")

class SimpleANN:
    def __init__(self):
        np.random.seed(42)
        self.W1 = np.random.randn(7, 32) * np.sqrt(2/7)
        self.b1 = np.zeros((1, 32))
        self.W2 = np.random.randn(32, 16) * np.sqrt(2/32)
        self.b2 = np.zeros((1, 16))
        self.W3 = np.random.randn(16, 8) * np.sqrt(2/16)
        self.b3 = np.zeros((1, 8))
        self.W4 = np.random.randn(8, 1) * np.sqrt(2/8)
        self.b4 = np.zeros((1, 1))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_deriv(self, x):
        return (x > 0).astype(float)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1; self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2; self.a2 = self.relu(self.z2)
        self.z3 = self.a2 @ self.W3 + self.b3; self.a3 = self.relu(self.z3)
        self.z4 = self.a3 @ self.W4 + self.b4
        return self.z4

    def train(self, X, y, epochs=300, lr=0.005):
        y = y.reshape(-1, 1)
        for _ in range(epochs):
            out = self.forward(X)
            dL = 2*(out-y)/len(y)
            dW4 = self.a3.T @ dL; db4 = dL.sum(axis=0, keepdims=True)
            d3 = (dL @ self.W4.T) * self.relu_deriv(self.z3)
            dW3 = self.a2.T @ d3; db3 = d3.sum(axis=0, keepdims=True)
            d2 = (d3 @ self.W3.T) * self.relu_deriv(self.z2)
            dW2 = self.a1.T @ d2; db2 = d2.sum(axis=0, keepdims=True)
            d1 = (d2 @ self.W2.T) * self.relu_deriv(self.z1)
            dW1 = X.T @ d1; db1 = d1.sum(axis=0, keepdims=True)
            self.W4 -= lr*dW4; self.b4 -= lr*db4
            self.W3 -= lr*dW3; self.b3 -= lr*db3
            self.W2 -= lr*dW2; self.b2 -= lr*db2
            self.W1 -= lr*dW1; self.b1 -= lr*db1

    def predict(self, X):
        return self.forward(X).flatten()

@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 5000
    gat   = np.random.randint(260, 341, n).astype(float)
    ielts = np.random.uniform(60, 120, n)
    uni   = np.random.randint(1, 6, n).astype(float)
    sop   = np.random.choice([1,1.5,2,2.5,3,3.5,4,4.5,5], n).astype(float)
    lor   = np.random.choice([1,1.5,2,2.5,3,3.5,4,4.5,5], n).astype(float)
    cgpa  = np.random.uniform(4, 10, n)
    res   = np.random.randint(0, 2, n).astype(float)
    y = (0.25*(gat-260)/80 + 0.15*(ielts-60)/60 + 0.10*(uni-1)/4 +
         0.10*(sop-1)/4 + 0.10*(lor-1)/4 + 0.25*(cgpa-4)/6 + 0.05*res)
    y += np.random.normal(0, 0.03, n)
    y = np.clip(y, 0.30, 0.97)
    X = np.column_stack([gat, ielts, uni, sop, lor, cgpa, res])
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    model = SimpleANN()
    model.train(X_scaled, y, epochs=300, lr=0.005)
    return model, scaler

with st.spinner("Loading AI Model... Please wait ⏳"):
    model, scaler = train_model()

st.success("✅ Model Ready!")
st.markdown("### 📋 Enter Your Profile")

col1, col2 = st.columns(2)
with col1:
    gat   = st.slider("🎯 GAT Score", 260, 340, 300)
    ielts = st.slider("🌐 IELTS Score", 60.0, 120.0, 90.0, step=0.5)
    uni   = st.selectbox("🏫 University Rating", [1,2,3,4,5], index=2)
    res   = st.selectbox("🔬 Research Experience", ["No","Yes"], index=0)
with col2:
    cgpa  = st.slider("📊 CGPA", 4.0, 10.0, 7.5, step=0.1)
    sop   = st.selectbox("📝 SOP Strength", [1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0], index=4)
    lor   = st.selectbox("📄 LOR Strength", [1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0], index=4)

if st.button("🚀 Predict My Admission Chance", use_container_width=True):
    res_val = 1.0 if res == "Yes" else 0.0
    inp = np.array([[float(gat), float(ielts), float(uni), float(sop), float(lor), float(cgpa), res_val]])
    inp_scaled = scaler.transform(inp)
    pred = float(np.clip(model.predict(inp_scaled)[0], 0.30, 0.97))
    pct  = round(pred * 100, 2)

    if pct >= 75:
        color = "#00ff88"; emoji = "🟢"; verdict = "Strong Profile! High chance of admission."
    elif pct >= 55:
        color = "#ffaa00"; emoji = "🟡"; verdict = "Good Profile! Moderate chance of admission."
    else:
        color = "#ff4444"; emoji = "🔴"; verdict = "Needs Improvement. Work on your profile."

    st.markdown(f"""
        <div class='result-box'>
            <h2 style='color:{color}; font-size:50px;'>{emoji} {pct}%</h2>
            <h3 style='color:white;'>Chance of Admission</h3>
            <p style='color:{color}; font-size:18px;'>{verdict}</p>
        </div>
    """, unsafe_allow_html=True)

    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=pct,
        title={'text': "Admission Probability", 'font': {'color': 'white', 'size': 18}},
        number={'suffix': "%", 'font': {'color': color, 'size': 40}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': "white"},
            'bar': {'color': color}, 'bgcolor': "#1a1a2e", 'bordercolor': "#00d4ff",
            'steps': [{'range': [0,50],'color':'#2d0000'},{'range': [50,75],'color':'#2d2000'},{'range': [75,100],'color':'#002d00'}],
        }
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, height=300)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📊 Your Profile Summary")
    st.dataframe(pd.DataFrame({
        'Feature': ['GAT Score','IELTS Score','University Rating','SOP','LOR','CGPA','Research'],
        'Your Value': [gat, ielts, uni, sop, lor, cgpa, res],
        'Max Possible': [340, 120, 5, 5.0, 5.0, 10.0, 'Yes']
    }), use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("<p style='text-align:center; color:#555555;'>Admission Probability Predictor | ANN Project | Mohammad Ismail</p>", unsafe_allow_html=True)
