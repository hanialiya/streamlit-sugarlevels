import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Membaca data dari file CSV
data = pd.read_csv('kadar_gula.csv')

# 2. Menambahkan kolom kategori kadar gula darah
def categorize_gula(x):
    if x < 100:
        return 'Normal'
    elif 100 <= x < 126:
        return 'Pre-diabetes'
    else:
        return 'Diabetes'

data['Kategori'] = data['Kadar_Gula'].apply(categorize_gula)

# 3. Preprocessing (Encode kategori menjadi angka)
data['Kategori'] = data['Kategori'].map({'Normal': 0, 'Pre-diabetes': 1, 'Diabetes': 2})

# 4. Memisahkan fitur dan target
X = data[['Usia', 'Jenis_Kelamin', 'Aktivitas_Fisik', 'Pola_Makan']]
y = data['Kategori']

# 5. Membagi data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 6. Membuat model machine learning
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 7. Evaluasi model
y_pred = model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 8. Visualisasi pentingnya fitur
importances = model.feature_importances_
plt.bar(X.columns, importances, color='skyblue')
plt.title('Feature Importances')
plt.show()
