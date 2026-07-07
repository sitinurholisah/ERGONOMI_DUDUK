import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

print("=== MEMULAI TRAINING MODEL DEEP LEARNING ===")

dataset_dir = 'dataset'
# Urutan label harus alfabetis biar konsisten
labels = ['bungkuk', 'miring', 'tegak'] 

X = []
y = []

# 1. Load Data dari Folder Dataset
for label in labels:
    folder_path = os.path.join(dataset_dir, label)
    if not os.path.exists(folder_path):
        print(f"⚠️ Folder {folder_path} tidak ada. Melewatkan...")
        continue
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.npy'):
            file_path = os.path.join(folder_path, file_name)
            data = np.load(file_path)
            X.append(data)
            y.append(labels.index(label))

if len(X) == 0:
    print("❌ Tidak ada data ditemukan di folder dataset! Jalankan collect_data.py dulu.")
    exit()

X = np.array(X)
y = np.array(y)

print(f"✅ Total data terkumpul: {len(X)}")
print(f"📊 Bentuk data (X): {X.shape}")

# 2. Preprocessing Data
y_categorical = to_categorical(y, num_classes=len(labels))
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# 3. Bangun Arsitektur Deep Learning (Neural Network)
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3), # Mencegah overfitting
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(len(labels), activation='softmax') # Output: probabilitas 3 kelas
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. Proses Training
print("\n🚀 Memulai Training... (Tunggu beberapa saat)")
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), verbose=1)

# 5. Evaluasi Model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\n Akurasi Testing Model: {test_acc * 100:.2f}%")

# 6. Simpan Model
model_dir = 'model'
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'posture_model.keras')
model.save(model_path)
print(f"💾 Model berhasil disimpan di: {model_path}")
print("=== TRAINING SELESAI ===")