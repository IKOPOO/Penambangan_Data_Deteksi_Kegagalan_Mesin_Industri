import streamlit as st
import pandas as pd
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🏭",
    layout="wide"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .big-font {
        font-size: 20px !important;
        font-weight: 500;
        color: #333;
    }
    .stCard {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #ddd;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- FUNGSI LOAD DATA (Untuk Preview) ---
@st.cache_data
def load_preview_data():
    path = 'data/processed/data_cleaned.csv'
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df
    return None

df = load_preview_data()

# --- HERO SECTION ---
col_logo, col_title = st.columns([1, 5])

with col_logo:
    # Menampilkan Logo Industri (Emoji Besar atau Gambar URL)
    st.markdown("<h1 style='text-align: center; font-size: 80px;'>🏭</h1>", unsafe_allow_html=True)

with col_title:
    st.title("Predictive Maintenance AI")
    st.markdown("#### Project Data Mining - Deteksi Kegagalan Mesin Industri")
    
    # Tech Stack Badges (Shields.io)
    st.markdown("""
    ![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
    ![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit)
    ![Scikit-Learn](https://img.shields.io/badge/SKLearn-Modeling-F7931E?style=for-the-badge&logo=scikit-learn)
    ![XGBoost](https://img.shields.io/badge/XGBoost-State%20of%20Art-green?style=for-the-badge)
    """)

st.divider()

# --- 1. PROJECT STATS (RINGKASAN CEPAT) ---
# Menampilkan angka penting agar terlihat "Data Driven"
if df is not None:
    st.subheader(" Project Overview")
    m1, m2, m3, m4 = st.columns(4)
    
    with m1:
        st.metric("Total Dataset", f"{len(df):,} Baris", help="Jumlah total data historis mesin")
    with m2:
        st.metric("Fitur Sensor", f"{len(df.columns) - 1} Variabel", help="Termasuk Suhu, RPM, Torsi, dll")
    with m3:
        fail_count = df['Machine failure'].sum()
        st.metric("Total Kegagalan", f"{fail_count} Kasus", help="Jumlah mesin yang rusak dalam dataset")
    with m4:
        st.metric("Model Utama", "XGBoost Tuned", help="Algoritma terbaik berdasarkan evaluasi F1-Score")
else:
    st.warning("Data belum diproses. Silakan jalankan notebook preprocessing.")

st.markdown("---")

# --- 2. LAYOUT KONTEN (KIRI: LATAR BELAKANG, KANAN: DATASET PREVIEW) ---
col_desc, col_data = st.columns([1, 1.3], gap="large")

with col_desc:
    st.header("📌 Latar Belakang")
    st.markdown("""
    <div style="text-align: justify; font-size: 16px;">
    Kegagalan mesin yang tidak terduga menyebabkan <b>Downtime</b> yang mahal di industri manufaktur. 
    Proyek ini bertujuan membangun sistem <b>Predictive Maintenance</b> yang mampu mendeteksi potensi kerusakan 
    sebelum terjadi berdasarkan data sensor real-time.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("🎯 Tujuan Bisnis")
    st.success(" **Reduksi Biaya:** Mengurangi biaya perbaikan darurat.")
    st.info(" **Efisiensi:** Mengoptimalkan jadwal maintenance teknisi.")
    st.warning(" **Keselamatan:** Mencegah kecelakaan kerja akibat mesin meledak.")

with col_data:
    st.header("📂 Intip Data (Dataset Sample)")
    st.markdown("Berikut adalah 5 baris pertama dari data sensor yang telah dibersihkan:")
    if df is not None:
        st.dataframe(df.head(), use_container_width=True, hide_index=True)
        st.caption("*Data bersumber dari: UCI Machine Learning Repository (AI4I 2020)*")

st.markdown("---")

# --- 3. ALUR KERJA (VISUAL WORKFLOW) ---
st.header("⚙️ Arsitektur Sistem")
st.markdown("Bagaimana sistem ini bekerja dari data mentah hingga prediksi:")

# Menggunakan Kolom sebagai "Kartu Langkah"
step1, step2, step3, step4 = st.columns(4)

with step1:
    st.container(border=True).markdown("""
    ### 1️⃣ Ingestion
    **Data Collection**
    
    Pengambilan data sensor mentah (CSV) dari mesin produksi.
    """)

with step2:
    st.container(border=True).markdown("""
    ### 2️⃣ Preprocessing
    **Cleaning & Scaling**
    
    - Hapus ID tidak relevan
    - Encoding Kategori
    - Standard Scaler
    """)

with step3:
    st.container(border=True).markdown("""
    ### 3️⃣ Modeling
    **AI Training**
    
    Melatih model **XGBoost** dan **Random Forest** untuk mengenali pola kerusakan.
    """)

with step4:
    st.container(border=True).markdown("""
    ### 4️⃣ Deployment
    **Streamlit App**
    
    Interface web interaktif untuk Engineer memantau kondisi mesin.
    """)

st.markdown("---")

# --- 4. QUICK ACCESS MENU ---
st.header("🚀 Mulai Eksplorasi")
st.markdown("Pilih menu di sidebar atau klik pintasan di bawah:")

qa1, qa2, qa3 = st.columns(3)

with qa1:
    with st.expander("📊 Lihat Dashboard EDA", expanded=True):
        st.write("Visualisasi sebaran data dan korelasi antar sensor.")
        st.markdown("**Buka Menu: 1_Dashboard_EDA**")

with qa2:
    with st.expander("🤖 Coba Model Prediksi", expanded=True):
        st.write("Simulasikan input sensor dan cek kesehatan mesin.")
        st.markdown("**Buka Menu: 2_Model_Prediksi**")

with qa3:
    with st.expander("📈 Evaluasi Performa", expanded=True):
        st.write("Audit akurasi model dan dampak bisnis.")
        st.markdown("**Buka Menu: 3_Evaluasi_Insight**")