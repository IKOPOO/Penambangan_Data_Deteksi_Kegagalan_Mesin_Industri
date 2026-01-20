# 🏭 Predictive Maintenance Capstone Project

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green?style=flat)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview

Proyek ini bertujuan untuk membangun sistem **Predictive Maintenance** berbasis Machine Learning untuk memprediksi kegagalan mesin industri sebelum terjadi. Menggunakan dataset **AI4I 2020 Predictive Maintenance** dari UCI Machine Learning Repository, proyek ini mengembangkan model klasifikasi untuk meminimalkan _downtime_ dan biaya perbaikan.

### 🎯 Tujuan Bisnis

1.  **Reduksi Biaya:** Mencegah kerusakan fatal yang memakan biaya besar.
2.  **Efisiensi:** Mengoptimalkan jadwal perawatan teknisi.
3.  **Keselamatan:** Mengurangi risiko kecelakaan kerja akibat kegagalan mesin.

---

## 📂 Struktur Repository

```text
capstone-project-data-mining/
├── app/
│   ├── app.py                  # Main Streamlit App
│   └── pages/                  # Halaman Tambahan (EDA, Prediksi, Evaluasi)
├── data/
│   ├── raw/                    # Data mentah (ai4i2020.csv)
│   └── processed/              # Data bersih & split (pickle files)
├── models/
│   ├── best_model.pkl          # Model XGBoost Tuned
│   └── preprocessing.pkl       # Scaler Pipeline
├── notebooks/
│   ├── 01_eda.ipynb            # Eksplorasi Data & Cleaning
│   ├── 02_modeling.ipynb       # Training & Tuning Model
│   └── 03_interpretation.ipynb # SHAP Analysis
├── src/                        # Script modular (helper functions)
├── requirements.txt            # Daftar library
└── README.md                   # Dokumentasi Proyek
```
