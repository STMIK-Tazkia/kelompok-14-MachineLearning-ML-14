import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Mengatur seed agar hasil bisa direproduksi (opsional tapi disarankan)
np.random.seed(42)
tf.random.set_seed(42)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf

# Membaca data
# Pastikan file csv ada di folder yang sama dengan file .ipynb ini
df = pd.read_csv('heart_failure_clinical_records_dataset.csv')

# Menampilkan 5 baris data pertama
df.head()
# Membaca dataset
# Pastikan file csv sudah diupload jika menggunakan Google Colab
data = pd.read_csv("heart_failure_clinical_records_dataset.csv")

# Menampilkan 5 baris pertama
print("Data Head:")
print(data.head()) # Menggunakan display() agar lebih rapi di notebook

# Menampilkan informasi dataset
print("\nInformasi Dataset:")
data.info()
# Memisahkan Fitur (X) dan Target (y)
X = data.drop("DEATH_EVENT", axis=1)
y = data["DEATH_EVENT"]

# Membagi data menjadi Train dan Test set (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standarisasi fitur (penting untuk Neural Networks)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Shape X_train: {X_train.shape}")
print(f"Shape X_test: {X_test.shape}")
model = Sequential()

# Input Layer & Hidden Layer 1
model.add(Dense(32, activation='relu', input_shape=(X_train.shape[1],)))

# Hidden Layer 2
model.add(Dense(16, activation='relu'))

# Output Layer (Sigmoid untuk klasifikasi biner)
model.add(Dense(1, activation='sigmoid'))

# Compile Model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Menampilkan ringkasan model
model.summary()
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.1, # Menggunakan sebagian data train untuk validasi
    verbose=1
)
# Prediksi probabilitas
y_pred_prob = model.predict(X_test)

# Konversi probabilitas ke kelas biner (Threshold 0.5)
y_pred = (y_pred_prob > 0.5).astype(int)

# Menghitung Metrik
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)

# Menampilkan Hasil
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")
print(f"AUC-ROC  : {roc_auc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
# Plot Loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Accuracy Curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()