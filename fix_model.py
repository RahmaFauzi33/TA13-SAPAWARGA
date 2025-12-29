"""
Script untuk memperbaiki model yang tidak ter-fit dengan benar.
Script ini akan melatih ulang dan menyimpan model dengan parameter terbaik.
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Path file
CLEAN_PATH = "sapawarga_reviews_clean.csv"
MODEL_PATH = "svm_sentiment_model_sapawarga.joblib"

print("=" * 60)
print("MEMPERBAIKI MODEL SVM SENTIMENT")
print("=" * 60)
print()

# 1. Load data
print(f"1. Memuat data dari {CLEAN_PATH}...")
try:
    df = pd.read_csv(CLEAN_PATH)
    print(f"   ✓ Data berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
except FileNotFoundError:
    print(f"   ✗ Error: File {CLEAN_PATH} tidak ditemukan!")
    exit(1)

# 2. Cek kolom
if "text" not in df.columns or "label" not in df.columns:
    print("   ✗ Error: CSV harus memiliki kolom 'text' dan 'label'")
    exit(1)

# 3. Prepare data
print("\n2. Menyiapkan data...")
df["text"] = df["text"].astype(str)
df["label"] = df["label"].astype(int)

X = df["text"]
y = df["label"]

print(f"   ✓ Data siap: {len(X)} sampel")
print(f"   Distribusi label: {y.value_counts().to_dict()}")

# 4. Train-test split
print("\n3. Membagi data (train/test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(f"   ✓ Train: {len(X_train)} sampel, Test: {len(X_test)} sampel")

# 5. Buat model dengan parameter terbaik (dari notebook)
print("\n4. Membuat model dengan parameter terbaik (RBF, C=3, gamma=0.1)...")
model = make_pipeline(
    TfidfVectorizer(),
    SVC(kernel="rbf", C=3, gamma=0.1)
)

# 6. Train model
print("\n5. Melatih model...")
model.fit(X_train, y_train)
print("   ✓ Model berhasil dilatih")

# 7. Test prediksi
print("\n6. Menguji model...")
try:
    test_pred = model.predict(X_test[:5])
    print(f"   ✓ Prediksi test berhasil: {test_pred}")
except Exception as e:
    print(f"   ✗ Error saat prediksi: {e}")
    exit(1)

# 8. Simpan model
print(f"\n7. Menyimpan model ke {MODEL_PATH}...")
try:
    joblib.dump(model, MODEL_PATH)
    print(f"   ✓ Model berhasil disimpan!")
except Exception as e:
    print(f"   ✗ Error saat menyimpan: {e}")
    exit(1)

print("\n" + "=" * 60)
print("SELESAI! Model telah diperbaiki dan disimpan.")
print("=" * 60)
print("\nAnda sekarang dapat menjalankan aplikasi Streamlit:")
print("  streamlit run app.py")

