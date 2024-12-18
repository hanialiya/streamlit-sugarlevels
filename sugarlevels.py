import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# konfigurasi halaman streamlit
st.set_page_config(page_title="Prediksi Kadar Gula Darah", layout="wide")

# judul
st.title("Prediksi Kadar Gula Darah dengan Machine Learning")

# file uploader
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

        # Tampilkan data setelah penambahan kolom kategori
        st.write("**Dataset setelah menambahkan kategori:**")
        st.dataframe(data)

        # Membuat line plot sederhana
        if 'id' not in data.columns:
            data['id'] = range(1, len(data) + 1)

        st.write("**Visualisasi Line Plot (ID vs Kategori Gula):**")
        fig, ax = plt.subplots()
        plt.plot(data['id'], data['Kategori'], marker='o', linestyle='-', color='blue')
        plt.xlabel('ID')
        plt.ylabel('Kategori Gula')
        plt.title('Line Plot ID vs Kategori Gula')
        plt.grid(True)
        st.pyplot(fig)
    else:
        st.error("Dataset Anda tidak memiliki kolom 'Kadar_Gula'.")
else:
    st.info("Silakan unggah file CSV untuk menganalisa.")
