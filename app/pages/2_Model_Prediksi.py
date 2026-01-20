import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import os
import time
import random

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Industrial Predictor", page_icon="🏭", layout="wide")

# --- CUSTOM CSS (Styling) ---
st.markdown("""
<style>
    /* Mengubah style Container agar terlihat seperti Card */
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        border: 1px solid #f0f2f6;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    /* Tombol Utama */
    .stButton>button {
        width: 100%;
        background-color: #00ADB5;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        height: 50px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #007E85;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- FUNGSI LOAD ASSETS ---
def load_assets():
    try:
        model = joblib.load('models/best_model.pkl')
        scaler = joblib.load('models/preprocessing.pkl')
        return model, scaler
    except:
        return None, None

model, scaler = load_assets()

# --- HEADER JUDUL ---
st.title(" Predictive Maintenance Control Room")
st.markdown("Dashboard kontrol untuk simulasi dan deteksi dini kegagalan mesin industri.")
st.markdown("---")

# --- SIDEBAR (HANYA UNTUK REFERENSI) ---
with st.sidebar:
    st.header("📘 Referensi Operator")
    st.info("""
    **Panduan Nilai Normal:**
    
     **Suhu Udara:** 295 - 300 K
            
     **Suhu Proses:** 305 - 310 K
            
     **RPM:** 1400 - 1600 RPM
            
     **Torsi:** 30 - 45 Nm
     
     **Tool Wear:** < 150 min
    """)
    st.markdown("*Gunakan panel di sebelah kanan untuk input data.*")

# --- LAYOUT UTAMA (SPLIT SCREEN) ---
if model is not None:
    
    # Membagi Layar: Kiri (Input 40%) - Kanan (Output 60%)
    col_control, col_monitor = st.columns([1, 1.5], gap="large")

    # === KOLOM KIRI: PANEL KONTROL ===
    with col_control:
        st.subheader("🎛️ Panel Kontrol Mesin")
        
        with st.container(border=True):
            # 1. Tombol Random Generator
            col_rand1, col_rand2 = st.columns([3, 1])
            with col_rand1:
                st.write("**Simulasi Data Sensor**")
            with col_rand2:
                if st.button("🎲 Random", help="Isi nilai acak untuk demo"):
                    st.session_state['rpm_val'] = int(random.randint(1200, 2800))
                    st.session_state['torque_val'] = round(random.uniform(20.0, 90.0), 1)
                    st.session_state['wear_val'] = int(random.randint(0, 250))
                    st.session_state['air_val'] = round(random.uniform(295, 305), 1)
            
            st.divider()

            # 2. Input Form
            type_input = st.selectbox("Tipe Kualitas Produk", ['L', 'M', 'H'], index=0)
            
            c1, c2 = st.columns(2)
            with c1:
                air_temp = st.number_input("Suhu Udara [K]", 250.0, 350.0, st.session_state.get('air_val', 300.0), step=0.1)
            with c2:
                process_temp = st.number_input("Suhu Proses [K]", 250.0, 400.0, 310.0, step=0.1)

            rpm = st.slider("Rotasi (RPM)", 1000, 3000, st.session_state.get('rpm_val', 1500))
            
            # Fix Slider Torsi (Float)
            torque = st.slider("Torsi (Nm)", 10.0, 100.0, st.session_state.get('torque_val', 40.0), step=0.1)
            
            tool_wear = st.slider("Keausan Alat (Min)", 0, 300, st.session_state.get('wear_val', 10))

            st.markdown("---")
            
            # 3. Tombol Eksekusi Besar
            predict_btn = st.button("🚀 JALANKAN DIAGNOSA")

    # === KOLOM KANAN: MONITOR DIAGNOSA ===
    with col_monitor:
        st.subheader("📊 Monitor Status Real-time")
        
        # Container hasil (default tinggi agar layout stabil)
        with st.container(border=True):
            
            if not predict_btn:
                # Tampilan Standby (Belum ada prediksi)
                st.markdown("""
                <div style="text-align: center; padding: 60px;">
                    <h2 style="color: #cccccc;">SISTEM STANDBY</h2>
                    <p style="color: gray;">Menunggu input data sensor dari Panel Kontrol...</p>
                    <img src="https://cdn-icons-png.flaticon.com/512/2590/2590566.png" width="100" style="opacity: 0.3;">
                </div>
                """, unsafe_allow_html=True)
            
            else:
                # Tampilan Hasil (Setelah klik)
                with st.spinner('Menghubungkan ke Model AI...'):
                    time.sleep(0.5) # Efek loading
                    
                    # Logika Prediksi
                    type_map = {'L': 0, 'M': 1, 'H': 2}
                    input_df = pd.DataFrame([[type_map[type_input], air_temp, process_temp, rpm, torque, tool_wear]],
                                          columns=['Type', 'Air temperature [K]', 'Process temperature [K]', 
                                                   'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'])
                    input_scaled = scaler.transform(input_df)
                    proba = model.predict_proba(input_scaled)[0][1]
                    pred = 1 if proba > 0.5 else 0

                    # 1. GAUGE CHART (Spedometer)
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = proba * 100,
                        number = {'suffix': "%", 'font': {'size': 40, 'color': "black"}},
                        title = {'text': "Probabilitas Kegagalan", 'font': {'size': 18}},
                        gauge = {
                            'axis': {'range': [None, 100], 'tickwidth': 1},
                            'bar': {'color': "rgba(0,0,0,0)"}, # Jarum transparan
                            'bgcolor': "white",
                            'steps': [
                                {'range': [0, 40], 'color': '#2ecc71'},   # Hijau
                                {'range': [40, 75], 'color': '#f1c40f'},  # Kuning
                                {'range': [75, 100], 'color': '#e74c3c'}],# Merah
                            'threshold': {
                                'line': {'color': "black", 'width': 6},
                                'thickness': 0.8,
                                'value': proba * 100}}))
                    
                    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
                    st.plotly_chart(fig, use_container_width=True)

                    # 2. ALERT STATUS
                    st.divider()
                    if pred == 1:
                        st.error("🚨 **CRITICAL WARNING**")
                        st.markdown(f"""
                        **Mesin terdeteksi tidak aman!** Risiko kegagalan mencapai **{proba*100:.1f}%**.
                        
                        **Rekomendasi Tindakan:**
                        1. Hentikan mesin segera.
                        2. Periksa pendingin (Suhu saat ini: {process_temp} K).
                        3. Cek kondisi mata bor/alat (Keausan: {tool_wear} min).
                        """)
                    else:
                        st.success("✅ **SYSTEM HEALTHY**")
                        st.markdown(f"Mesin beroperasi dalam parameter optimal. Risiko kegagalan rendah (**{proba*100:.1f}%**).")

                    # 3. METRIK RINGKAS
                    st.markdown("---")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Power Strain", f"{torque*rpm/1000:.1f} kW", help="Estimasi beban daya (Torsi x RPM)")
                    m2.metric("Temp Delta", f"{process_temp - air_temp:.1f} K", help="Selisih Suhu Proses & Udara")
                    m3.metric("Tool Life", f"{200 - tool_wear} min", delta_color="normal", help="Sisa umur alat sebelum batas kritis 200 min")

else:
    st.warning("⚠️ Model belum dimuat. Pastikan file 'models/best_model.pkl' tersedia.")