import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="Prediksi Kadar Gula Darah", layout="wide")

# 1. Judul Aplikasi
st.title("Prediksi Kadar Gula Darah dengan Machine Learning")
st.write("Unggah dataset Anda, lakukan analisis, dan lihat hasil prediksi!")

# 2. Mengunggah Dataset
uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

if uploaded_file is not None:
    # Membaca data dari file yang diunggah
    data = pd.read_csv(uploaded_file)
    st.write("**Dataset yang diunggah:**")
    st.dataframe(data)

    # 3. Menambahkan kolom kategori kadar gula darah
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

        # 4. Preprocessing
        st.write("**Melakukan preprocessing data...**")
        if 'Kategori' in data.columns:
            data['Kategori'] = data['Kategori'].map({'Normal': 0, 'Pre-diabetes': 1, 'Diabetes': 2})
        
        # Pilih fitur yang akan digunakan
        features = ['Usia', 'Jenis_Kelamin', 'Aktivitas_Fisik', 'Pola_Makan']
        if all(col in data.columns for col in features):
            X = data[features]
            y = data['Kategori']

            # 5. Membagi data menjadi training dan testing
            st.write("**Membagi data menjadi training dan testing...**")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            st.write(f"Ukuran data training: {X_train.shape}, data testing: {X_test.shape}")

            # 6. Melatih Model
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            st.write("**Model telah dilatih!**")

            # 7. Evaluasi Model
            y_pred = model.predict(X_test)
            st.write("**Confusion Matrix:**")
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=['Normal', 'Pre-diabetes', 'Diabetes'], yticklabels=['Normal', 'Pre-diabetes', 'Diabetes'])
            st.pyplot(fig)

            st.write("**Classification Report:**")
            st.text(classification_report(y_test, y_pred))

            # 8. Visualisasi pentingnya fitur
            st.write("**Pentingnya Fitur:**")
            importances = model.feature_importances_
            fig, ax = plt.subplots()
            plt.bar(features, importances, color='skyblue')
            plt.title('Feature Importances')
            plt.xlabel('Fitur')
            plt.ylabel('Importance')
            st.pyplot(fig)
        else:
            st.error("Dataset Anda tidak memiliki semua kolom fitur yang dibutuhkan: Usia, Jenis_Kelamin, Aktivitas_Fisik, Pola_Makan.")
    else:
        st.error("Dataset Anda tidak memiliki kolom 'Kadar_Gula'.")
else:
    st.info("Silakan unggah file CSV untuk memulai.")
