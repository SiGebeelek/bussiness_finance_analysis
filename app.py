import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import streamlit as st

st.set_page_config(page_title="Business Finance Analytics", layout="centered")

st.title("📈 Business Finance Predictor")
st.write("Aplikasi prediksi performa keuangan (Tahun 2023) berdasarkan data historis 3 tahun sebelumnya menggunakan Random Forest.")

@st.cache_resource
def load_model():
    try:
        model = joblib.load('financial_model.pkl')
        return model
    except FileNotFoundError:
        st.error("File 'financial_model.pkl' tidak ditemukan. Harap jalankan train_model.py terlebih dahulu.")
        return None

model = load_model()

st.sidebar.header("Input Data Keuangan")
st.sidebar.info("Masukkan nilai dalam format angka penuh (contoh: 1000000000)")

val_2020 = st.sidebar.number_input("Nilai Tahun 2020", value=0.0)
val_2021 = st.sidebar.number_input("Nilai Tahun 2021", value=0.0)
val_2022 = st.sidebar.number_input("Nilai Tahun 2022", value=0.0)

if st.button("Prediksi Tahun 2023"):
    if model is not None:
        input_data = pd.DataFrame([[val_2020, val_2021, val_2022]], columns=['2020', '2021', '2022'])
        prediction = model.predict(input_data)[0]
        st.success(f"Prediksi Nilai Tahun 2023:")
        st.metric(label="Projected Value", value=f"Rp {prediction:,.2f}")
        chart_data = pd.DataFrame({
            'Tahun': ['2020', '2021', '2022', '2023 (Prediksi)'],
            'Nilai': [val_2020, val_2021, val_2022, prediction]
        })
        st.line_chart(chart_data, x='Tahun', y='Nilai')

    else:
        st.error("Model belum dimuat.")

st.markdown("---")
st.caption("Dikembangkan oleh Kelompok 8 - Business Finance Analytics")
