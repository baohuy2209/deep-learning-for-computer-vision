import pickle
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Cài đặt thư viện: pip install xgboost
from xgboost import XGBClassifier

# ==========================================
# 1. ĐỌC DỮ LIỆU
# ==========================================
with open("../../../data/processed/aptos/sample_50/sample_dataset.pkl", "rb") as f:
    data = pickle.load(f)

X           = data['data']        
y           = data['labels']      
class_names = data['class_names'] 

# ==========================================
# 2. TIỀN XỬ LÝ (GIỐNG VỚI SVM)
# ==========================================
print("Đang tiến hành tiền xử lý dữ liệu cho XGBoost...")

# Duỗi phẳng ảnh (Flatten)
N = X.shape[0]
X_flatten = X.reshape(N, -1) 
print(f"Kích thước sau khi duỗi: {X_flatten.shape}")

# Chia tập Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X_flatten, y, test_size=0.2, random_state=42, stratify=y
)

# Chuẩn hóa thang đo (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Giảm chiều dữ liệu bằng PCA (Giữ lại 95% thông tin)
print("Đang chạy thuật toán nén PCA...")
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"Số lượng đặc trưng (features) đã giảm từ {X_flatten.shape[1]} xuống còn {X_train_pca.shape[1]}!")

# ==========================================
# 3. KHỞI TẠO VÀ HUẤN LUYỆN XGBOOST
# ==========================================
print("\nBắt đầu huấn luyện mô hình XGBoost...")
start_time = time.time()

# Cấu hình XGBoost cho bài toán phân loại đa lớp (Multi-class classification)
xgb_model = XGBClassifier(
    objective='multi:softprob', # Dùng cho đa lớp, trả về xác suất
    num_class=5,                # Khai báo đúng 5 nhãn (0, 1, 2, 3, 4)
    tree_method='hist',         # Dùng thuật toán histogram giúp tăng tốc độ train cực nhanh
    learning_rate=0.1,          # Tốc độ học (có thể tinh chỉnh sau)
    max_depth=6,                # Độ sâu tối đa của mỗi cây
    n_estimators=100,           # Số lượng cây quyết định
    random_state=42
)

# Train model
xgb_model.fit(X_train_pca, y_train)

end_time = time.time()
print(f"✅ Đã huấn luyện xong! Thời gian: {end_time - start_time:.2f} giây")

# ==========================================
# 4. ĐÁNH GIÁ MÔ HÌNH
# ==========================================
print("\nĐang dự đoán tập Test...")
y_pred = xgb_model.predict(X_test_pca)

acc = accuracy_score(y_test, y_pred)
print(f"\n🏆 ĐỘ CHÍNH XÁC TỔNG THỂ (Accuracy): {acc * 100:.2f}%\n")

print("--- BÁO CÁO PHÂN LOẠI CHI TIẾT ---")
print(classification_report(y_test, y_pred, target_names=class_names))

print("--- MA TRẬN NHẦM LẪN (CONFUSION MATRIX) ---")
print(confusion_matrix(y_test, y_pred))