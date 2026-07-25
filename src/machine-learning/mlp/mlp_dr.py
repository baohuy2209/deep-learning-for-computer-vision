import pickle
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ==========================================
# 1. ĐỌC DỮ LIỆU
# ==========================================
with open("../../../data/processed/aptos/sample_50/sample_dataset.pkl", "rb") as f:
    data = pickle.load(f)

X           = data['data']        
y           = data['labels']      
class_names = data['class_names'] 

# ==========================================
# 2. TIỀN XỬ LÝ
# ==========================================
print("Đang tiến hành tiền xử lý dữ liệu cho MLP...")

N = X.shape[0]
X_flatten = X.reshape(N, -1) 

X_train, X_test, y_train, y_test = train_test_split(
    X_flatten, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Đang chạy thuật toán nén PCA...")
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# ==========================================
# 3. KHỞI TẠO VÀ HUẤN LUYỆN MLP
# ==========================================
print("\nBắt đầu huấn luyện mô hình MLP (Multi-Layer Perceptron)...")
start_time = time.time()

# Cấu hình kiến trúc mạng nơ-ron cơ bản
mlp_model = MLPClassifier(
    hidden_layer_sizes=(512, 256), # Tạo 2 lớp ẩn, lớp đầu 512 nơ-ron, lớp sau 256 nơ-ron
    activation='relu',             # Hàm kích hoạt chuẩn
    solver='adam',                 # Thuật toán tối ưu giống hệt Keras
    alpha=0.0001,                  # Tham số phạt L2 để chống Overfitting
    batch_size=32,                 # Đẩy dữ liệu vào theo từng cụm 32 ảnh
    max_iter=500,                  # Số vòng lặp tối đa
    early_stopping=True,           # Tự dừng nếu không tiến bộ
    random_state=42
)

# Train model
mlp_model.fit(X_train_pca, y_train)

end_time = time.time()
print(f"✅ Đã huấn luyện xong! Thời gian: {end_time - start_time:.2f} giây")

# ==========================================
# 4. ĐÁNH GIÁ MÔ HÌNH
# ==========================================
print("\nĐang dự đoán tập Test...")
y_pred = mlp_model.predict(X_test_pca)

acc = accuracy_score(y_test, y_pred)
print(f"\n🏆 ĐỘ CHÍNH XÁC TỔNG THỂ (Accuracy): {acc * 100:.2f}%\n")
print(classification_report(y_test, y_pred, target_names=class_names))