# -*- coding: utf-8 -*-
"""
🏠 California Housing Price Predictor
📱 Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os

# --- Configuration ---
st.set_page_config(
    page_title="California Housing Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        margin: 1rem 0;
    }
    .result-value {
        font-size: 3rem;
        font-weight: bold;
        margin: 1rem 0;
        color: #ffffff;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Model ---
@st.cache_resource
def load_model():
    try:
        model_path = Path('model_files/rf_model.pkl')
        scaler_path = Path('model_files/scaler.pkl')
        features_path = Path('model_files/feature_names.pkl')
        
        if not all([model_path.exists(), scaler_path.exists(), features_path.exists()]):
            st.error("❌ ไม่พบไฟล์โมเดล กรุณาตรวจสอบโฟลเดอร์ model_files/")
            st.stop()
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        feature_names = joblib.load(features_path)
        return model, scaler, feature_names
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
        st.stop()

model, scaler, feature_names = load_model()

# --- Sidebar ---
st.sidebar.title("📖 เกี่ยวกับแอป")
st.sidebar.markdown("""
### 🎯 วัตถุประสงค์
ทำนายราคาบ้านในแคลิฟอร์เนียด้วย Machine Learning (Random Forest)

### 📊 ข้อมูล Dataset
- **Samples:** 20,640
- **Features:** 8 ตัวแปร
- **Model:** Random Forest Regressor
- **R² Score:** ~0.80
""")

# --- Main Content ---
st.title("🏠 California Housing Price Predictor")
st.markdown("### ทำนายราคาบ้านในแคลิฟอร์เนียด้วย Machine Learning")
st.markdown("---")

st.markdown("""
<div class='info-box'>
    <b>💡 คำแนะนำ:</b> ปรับค่า slider ด้านล่างให้ตรงกับข้อมูลบ้านที่ต้องการทำนาย แล้วกดปุ่ม "ทำนายราคาบ้าน"
</div>
""", unsafe_allow_html=True)

# --- Input Section ---
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📍 ตำแหน่งที่ตั้งและประชากร")
    med_inc = st.slider("💰 รายได้เฉลี่ย (×$10,000)", 0.5, 15.0, 5.0, 0.1)
    house_age = st.slider("🏚️ อายุบ้าน (ปี)", 1.0, 52.0, 20.0, 1.0)
    latitude = st.slider("🌎 Latitude", 32.0, 42.0, 35.0, 0.01)
    longitude = st.slider("🌎 Longitude", -124.0, -114.0, -119.0, 0.01)

with col2:
    st.markdown("### 🏡 ลักษณะบ้าน")
    ave_rooms = st.slider("🛏️ จำนวนห้องเฉลี่ย", 1.0, 15.0, 5.0, 0.1)
    ave_bedrms = st.slider("🛌 จำนวนห้องนอนเฉลี่ย", 0.5, 5.0, 1.1, 0.1)
    population = st.slider("👥 ประชากรในพื้นที่", 3.0, 35000.0, 1500.0, 10.0)
    ave_occup = st.slider("👨‍👩‍👧‍👦 จำนวนคนอยู่อาศัยเฉลี่ย", 1.0, 10.0, 3.0, 0.1)

st.markdown("---")

# --- Prediction ---
if st.button("🔮 ทำนายราคาบ้าน", use_container_width=True, type="primary"):
    # สร้าง Dictionary ให้ตรงกับชื่อ Feature เป๊ะๆ (ป้องกันลำดับผิด)
    input_data = {
        'MedInc': med_inc,
        'HouseAge': house_age,
        'AveRooms': ave_rooms,
        'AveBedrms': ave_bedrms,
        'Population': population,
        'AveOccup': ave_occup,
        'Latitude': latitude,
        'Longitude': longitude
    }
    
    # แปลงเป็น DataFrame เพื่อให้มี column names ตรงกับตอนเทรน
    input_df = pd.DataFrame([input_data])
    
    # Scale และ Predict
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    prediction_usd = prediction * 100000
    
    st.markdown("---")
    st.markdown(f"""
    <div class='result-card'>
        <h2>💰 ราคาบ้านที่ทำนายได้</h2>
        <div class='result-value'>${prediction_usd:,.0f}</div>
        <p style='font-size: 1.2rem;'>หรือประมาณ <b>{prediction:,.2f}</b> × $100,000</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("## 📊 การวิเคราะห์เพิ่มเติม")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        # Feature Importance
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig = go.Figure(go.Bar(
            x=importance_df['Importance'],
            y=importance_df['Feature'],
            orientation='h',
            marker_color='#667eea'
        ))
        fig.update_layout(
            title='🔍 Feature Importance',
            xaxis_title='Importance Score',
            yaxis_title='Feature',
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col_b:
        st.markdown("### 📋 สรุปข้อมูลที่ป้อน")
        input_summary = pd.DataFrame({
            'Feature': ['รายได้เฉลี่ย', 'อายุบ้าน', 'จำนวนห้อง', 'จำนวนห้องนอน', 'ประชากร', 'คนอยู่อาศัย', 'ละติจูด', 'ลองจิจูด'],
            'Value': [f'${med_inc*10000:,.0f}', f'{house_age:.0f} ปี', f'{ave_rooms:.1f} ห้อง', f'{ave_bedrms:.1f} ห้อง', f'{population:,.0f} คน', f'{ave_occup:.1f} คน', f'{latitude:.2f}°', f'{longitude:.2f}°']
        })
        st.dataframe(input_summary, hide_index=True, use_container_width=True)

# --- Footer ---
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'><p>🎓 วิชา: การโปรแกรมสำหรับการเรียนรู้ด้วยเครื่องด้วยภาษาไพทอน</p></div>", unsafe_allow_html=True)
