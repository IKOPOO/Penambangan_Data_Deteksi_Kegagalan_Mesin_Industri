import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Konfigurasi Halaman
st.set_page_config(page_title="Model Evaluation", page_icon="📈", layout="wide")

# --- LOAD DATA ---
@st.cache_resource
def load_eval_data():
    try:
        # Load data test yang sudah displit di awal project
        data = joblib.load('data/processed/split_data.pkl')
        model = joblib.load('models/best_model.pkl')
        return data['X_test'], data['y_test'], data['feature_names'], model
    except Exception as e:
        return None, None, None, None

X_test, y_test, feature_names, model = load_eval_data()

st.title("📈 Evaluasi Kinerja Model")
st.markdown("Halaman ini mengaudit seberapa akurat model memprediksi kegagalan pada data uji (Unseen Data).")
st.markdown("---")

if model is not None:
    # --- 1. HITUNG METRIK UTAMA ---
    y_pred = model.predict(X_test)
    
    # Hitung skor
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)

    # TAMPILKAN KPI CARDS (Executive Summary)
    st.subheader("🏆 Rapor Nilai Model")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Akurasi (Overall)", f"{acc:.1%}", help="Seberapa sering model benar secara keseluruhan.")
    with col2:
        st.metric("F1-Score (Keseimbangan)", f"{f1:.1%}", delta_color="normal", help="Metrik paling penting untuk data tidak seimbang. Gabungan Precision & Recall.")
    with col3:
        st.metric("Recall (Sensitivitas)", f"{recall:.1%}", help="Kemampuan mendeteksi kegagalan. Recall rendah = Bahaya (Gagal tak terdeteksi).")
    with col4:
        st.metric("Precision (Ketepatan)", f"{prec:.1%}", help="Seberapa tepat saat bilang 'Gagal'. Precision rendah = Banyak alarm palsu.")

    st.markdown("---")

    # --- 2. DETAIL ANALISIS (TABS) ---
    tab1, tab2, tab3 = st.tabs(["🧩 Confusion Matrix", "💰 Dampak Bisnis", "🧠 Logika Model"])

    # TAB 1: CONFUSION MATRIX (Visualisasi Kebenaran)
    with tab1:
        st.subheader("Detail Prediksi Benar vs Salah")
        
        cm = confusion_matrix(y_test, y_pred)
        # Struktur CM: [[TN, FP], [FN, TP]]
        tn, fp, fn, tp = cm.ravel()

        # Visualisasi Custom dengan Plotly Heatmap
        z = [[fn, tp], [tn, fp]] # Dibalik agar TP/FN di atas (sesuai standar industri)
        x = ['Prediksi: Aman', 'Prediksi: Gagal']
        y = ['Aktual: Gagal (Rusak)', 'Aktual: Aman (Normal)']

        # Catatan: Kita sesuaikan urutan agar visualnya enak dibaca
        fig_cm = go.Figure(data=go.Heatmap(
            z=[[tp, fn], [fp, tn]], # Re-order untuk heatmap: Baris=Aktual, Kolom=Prediksi
            x=['Prediksi: Gagal', 'Prediksi: Aman'],
            y=['Aktual: Gagal', 'Aktual: Aman'],
            text=[[f"<b>BENAR GAGAL (TP)</b><br>{tp}", f"<b>GAGAL MELESET (FN)</b><br>{fn}"],
                  [f"<b>ALARM PALSU (FP)</b><br>{fp}", f"<b>BENAR AMAN (TN)</b><br>{tn}"]],
            texttemplate="%{text}",
            colorscale='Blues',
            showscale=False
        ))
        
        fig_cm.update_layout(title="Confusion Matrix Interaktif", height=400)
        st.plotly_chart(fig_cm, use_container_width=True)
        
        st.info("""
        **Cara Membaca:**
        - **Benar Gagal (True Positive):** Sukses! Mesin rusak berhasil dideteksi.
        - **Gagal Meleset (False Negative):** BAHAYA! Mesin rusak tapi dibilang aman. (Harus diminimalkan).
        - **Alarm Palsu (False Positive):** Mesin aman tapi dibilang rusak. (Bikin teknisi capek, tapi tidak fatal).
        """)

    # TAB 2: ANALISIS BISNIS (Konteks Uang)
    with tab2:
        st.subheader("Simulasi Dampak Finansial")
        st.markdown("Berapa uang yang diselamatkan model ini dibandingkan tidak pakai AI?")

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            st.markdown("#### ⚙️ Asumsi Biaya")
            cost_maintenance = st.number_input("Biaya Inspeksi (Maintenance)", value=50, help="Biaya kirim teknisi cek mesin (USD)")
            cost_failure = st.number_input("Biaya Kerusakan Fatal", value=5000, help="Kerugian jika mesin meledak/stop produksi (USD)")
        
        with col_b2:
            st.markdown("#### 💵 Kalkulasi Penghematan")
            
            # Skenario Tanpa AI (Semua rusak lolos)
            loss_without_ai = (tp + fn) * cost_failure 
            
            # Skenario Dengan AI
            # Kita bayar inspeksi untuk TP dan FP (karena diprediksi rusak)
            cost_inspection = (tp + fp) * cost_maintenance
            # Kita bayar kerusakan fatal hanya untuk FN (yang lolos)
            cost_missed_failure = fn * cost_failure
            
            total_cost_with_ai = cost_inspection + cost_missed_failure
            
            saved = loss_without_ai - total_cost_with_ai
            
            st.metric("Total Biaya (Tanpa AI)", f"${loss_without_ai:,.0f}")
            st.metric("Total Biaya (Dengan AI)", f"${total_cost_with_ai:,.0f}")

    # TAB 3: FEATURE IMPORTANCE (Kenapa model memilih itu?)
    with tab3:
        st.subheader("Faktor Penentu Kegagalan")
        st.markdown("Fitur mana yang paling dilihat oleh Model saat mengambil keputusan?")

        if hasattr(model, 'feature_importances_'):
            # Buat DataFrame untuk plotting
            fi_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=True)

            # Plot Horizontal Bar Chart
            fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h', 
                            title="Ranking Fitur Terpenting",
                            color='Importance', color_continuous_scale='Teal')
            
            fig_fi.update_layout(height=500)
            st.plotly_chart(fig_fi, use_container_width=True)
            
            st.write("""
            **Interpretasi Insight:**
            1. **Fitur Teratas (Paling Panjang):** Adalah penyebab utama kerusakan. Biasanya Torsi (Torque) atau RPM.
            2. **Strategi Bisnis:** Fokuskan sensor monitoring yang lebih canggih pada fitur-fitur teratas ini, karena mereka adalah indikator paling sensitif.
            """)
        else:
            st.warning("Model yang digunakan tidak mendukung Feature Importance bawaan.")

else:
    st.error("Model belum dimuat. Pastikan Anda sudah menjalankan notebook modeling.")