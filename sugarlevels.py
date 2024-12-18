import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

#konfigurasi halaman streamlit
st.set_page_config(page_title="Prediksi Kadar Gula Darah", layout="wide")

#judul
st.title("Prediksi Kadar Gula Darah dengan Machine Learning")

uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

if uploaded_file is not None:
    # Membaca data dari file yang diunggah
    data = pd.read_csv(uploaded_file)
    st.write("**Unggah Dataset:**")
    st.dataframe(data)

    # Menambahkan kolom kategori kadar gula darah
    def categorize_gula(x):
        if x < 100:
            return 'Normal'
        elif 100 <= x < 126:
            return 'Pre-diabetes'
        else:
            return 'Diabetes'

    # Menambahkan kolom kategori
    if 'Kadar_Gula' in data.columns:
        data['Kategori'] = data['Kadar_Gula'].apply(categorize_gula)

        # Membuat DataFrame untuk visualisasi
        df_plot = data[['id', 'Kadar_Gula']].copy()
        if 'id' not in data.columns:
            df_plot['id'] = range(1, len(data) + 1)
        df_plot = df_plot.set_index('id')

        # Membuat line plot
        st.write("**Visualisasi Line Plot (ID vs Kadar Gula):**")
        fig, ax = plt.subplots(figsize=(10, 6))
        df_plot.plot(kind='line', ax=ax)

        plt.title('Perbandingan Kadar Gula Darah\n', size=16)
        plt.ylabel('Kadar Gula Darah', size=14)
        plt.xlabel('ID (Individu)', size=14)
        plt.grid(True)
        st.pyplot(fig)
    else:
        st.error("Dataset Anda tidak memiliki kolom 'Kadar_Gula'.")
else:
    st.info("Silakan unggah file CSV untuk menganalisa.")
