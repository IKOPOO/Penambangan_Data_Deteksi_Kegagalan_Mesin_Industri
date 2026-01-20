import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="EDA Dashboard", page_icon="📊", layout="wide")

@st.cache_data
def load_data():
    path = 'data/processed/data_cleaned.csv'
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

df = load_data()

st.title("📊 Analisis Data Eksploratif")

if df is not None:
    # --- SIDEBAR FILTERS ---
    st.sidebar.header("🔍 Filter Data")
    
    # Filter Tipe Produk
    kategori_pilihan = st.sidebar.multiselect(
        "Pilih Kualitas Produk:",
        options=df['Type'].unique(),
        default=df['Type'].unique()
    )
    
    # Filter Status Mesin
    status_pilihan = st.sidebar.radio(
        "Status Mesin:",
        options=["Semua", "Hanya Gagal (1)", "Hanya Normal (0)"]
    )

    # Terapkan Filter
    df_filtered = df[df['Type'].isin(kategori_pilihan)]
    
    if status_pilihan == "Hanya Gagal (1)":
        df_filtered = df_filtered[df_filtered['Machine failure'] == 1]
    elif status_pilihan == "Hanya Normal (0)":
        df_filtered = df_filtered[df_filtered['Machine failure'] == 0]

    # Tampilkan Jumlah Data setelah Filter
    st.markdown(f"**Menampilkan {len(df_filtered)} data dari total {len(df)} data.**")
    st.divider()

    # --- TABS LAYOUT (Agar rapi) ---
    tab1, tab2, tab3 = st.tabs(["📈 Distribusi Sensor", "🔥 Korelasi Panas", "🧊 Scatter 3D"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Distribusi Suhu Udara")
            fig1 = px.histogram(df_filtered, x='Air temperature [K]', color='Machine failure', 
                                barmode='overlay', color_discrete_map={0:'blue', 1:'red'})
            st.plotly_chart(fig1, use_container_width=True)
        with c2:
            st.subheader("Distribusi Torsi")
            fig2 = px.box(df_filtered, x='Machine failure', y='Torque [Nm]', color='Machine failure')
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.subheader("Hubungan Suhu Proses vs Rotasi")
        fig3 = px.density_heatmap(df_filtered, x='Rotational speed [rpm]', y='Process temperature [K]', 
                                  marginal_x="histogram", marginal_y="histogram")
        st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        st.subheader("Analisis 3D: RPM, Torsi, dan Keausan")
        fig4 = px.scatter_3d(df_filtered, x='Rotational speed [rpm]', y='Torque [Nm]', z='Tool wear [min]',
                             color='Machine failure', opacity=0.7, size_max=10)
        st.plotly_chart(fig4, use_container_width=True)

else:
    st.error("Data tidak ditemukan!")