import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    ConfusionMatrixDisplay,
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from tensorflow.keras.datasets import mnist
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 60)
print("1. LOAD DỮ LIỆU MNIST")
print("=" * 60)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f"Train : {x_train.shape} | Label: {y_train.shape}")
print(f"Test  : {x_test.shape}  | Label: {y_test.shape}")
print(f"Classes: {np.unique(y_train)}")
print(f"Pixel range: [{x_train.min()}, {x_train.max()}]")

# Phân phối class
print("\nPhân phối class (train):")
for cls in range(10):
    count = (y_train == cls).sum()
    bar = "█" * (count // 200)
    print(f"  Digit {cls}: {count} ảnh  {bar}")

# ============================================================
# 2. VISUALIZE MẪU ẢNH
# ============================================================
print("\n" + "=" * 60)
print("2. VISUALIZE MẪU ẢNH")
print("=" * 60)

fig, axes = plt.subplots(2, 10, figsize=(20, 4))
for cls in range(10):
    idx = np.where(y_train == cls)[0][0]
    axes[0, cls].imshow(x_train[idx], cmap="gray")
    axes[0, cls].set_title(f"Digit {cls}", fontsize=10)
    axes[0, cls].axis("off")

    idx2 = np.where(y_train == cls)[0][1]
    axes[1, cls].imshow(x_train[idx2], cmap="gray")
    axes[1, cls].axis("off")

plt.suptitle("Mẫu ảnh MNIST — 2 ảnh mỗi class", fontsize=13)
plt.tight_layout()
plt.savefig("1_sample_images.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# 3. TIỀN XỬ LÝ
# ============================================================
print("\n" + "=" * 60)
print("3. TIỀN XỬ LÝ")
print("=" * 60)

# 3.1 Flatten 28x28 → 784
x_train_flat = x_train.reshape(x_train.shape[0], -1).astype("float32")
x_test_flat = x_test.reshape(x_test.shape[0], -1).astype("float32")
print(f"Sau flatten — Train: {x_train_flat.shape} | Test: {x_test_flat.shape}")

# 3.2 Normalize về [0, 1]
x_train_norm = x_train_flat / 255.0
x_test_norm = x_test_flat / 255.0
print(f"Sau normalize — range: [{x_train_norm.min()}, {x_train_norm.max()}]")

# 3.3 PCA — giảm chiều 784 → 100 (tăng tốc KNN đáng kể)
print("\nÁp dụng PCA giảm chiều 784 → 100...")
pca = PCA(n_components=100, random_state=42)
x_train_pca = pca.fit_transform(x_train_norm)
x_test_pca = pca.transform(x_test_norm)

explained = pca.explained_variance_ratio_.cumsum()[-1] * 100
print(f"Variance giữ lại: {explained:.2f}%")
print(f"Sau PCA — Train: {x_train_pca.shape} | Test: {x_test_pca.shape}")

# Plot explained variance
plt.figure(figsize=(8, 4))
plt.plot(np.cumsum(pca.explained_variance_ratio_) * 100, color="steelblue")
plt.axhline(
    y=explained,
    color="red",
    linestyle="--",
    label=f"{explained:.1f}% với 100 components",
)
plt.xlabel("Số components")
plt.ylabel("Cumulative Explained Variance (%)")
plt.title("PCA — Explained Variance")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("2_pca_variance.png", dpi=150, bbox_inches="tight")
plt.show()

# Dùng subset để train nhanh hơn (KNN chậm với 60k samples)
N_TRAIN = 10000
N_TEST = 2000
idx_train = np.random.choice(len(x_train_pca), N_TRAIN, replace=False)
idx_test = np.random.choice(len(x_test_pca), N_TEST, replace=False)

x_tr = x_train_pca[idx_train]
y_tr = y_train[idx_train]
x_te = x_test_pca[idx_test]
y_te = y_test[idx_test]

print(
    f"\nDùng subset: Train={N_TRAIN}, Test={N_TEST} (KNN chậm với full dataset)"
)

# ============================================================
# 4. TÌM K TỐI ƯU
# ============================================================
print("\n" + "=" * 60)
print("4. TÌM K TỐI ƯU")
print("=" * 60)

k_values = [1, 3, 5, 7, 9, 11, 15]
k_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean", n_jobs=-1)
    score = cross_val_score(knn, x_tr, y_tr, cv=3, scoring="accuracy").mean()
    k_scores.append(score)
    print(f"  k={k:2d} → CV Accuracy: {score:.4f}")

best_k = k_values[np.argmax(k_scores)]
print(f"\n✅ K tốt nhất: {best_k} (accuracy={max(k_scores):.4f})")

plt.figure(figsize=(8, 4))
plt.plot(k_values, k_scores, "o-", color="steelblue", linewidth=2)
plt.axvline(x=best_k, color="red", linestyle="--", label=f"Best k={best_k}")
plt.xlabel("Giá trị K")
plt.ylabel("CV Accuracy")
plt.title("KNN — Tìm K tối ưu (3-fold CV)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("3_best_k.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# 5. TRAIN & PREDICT
# ============================================================
print("\n" + "=" * 60)
print("5. TRAIN & PREDICT")
print("=" * 60)

knn = KNeighborsClassifier(n_neighbors=best_k, metric="euclidean", n_jobs=-1)
knn.fit(x_tr, y_tr)
print(f"✅ Train xong với k={best_k}")

y_pred = knn.predict(x_te)
acc = accuracy_score(y_te, y_pred)
print(f"Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")

# ============================================================
# 6. KẾT QUẢ
# ============================================================
print("\n" + "=" * 60)
print("6. KẾT QUẢ PHÂN LOẠI")
print("=" * 60)

print("\nClassification Report:")
print(
    classification_report(
        y_te, y_pred, target_names=[f"Digit {i}" for i in range(10)]
    )
)

# Confusion Matrix
cm = confusion_matrix(y_te, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=range(10),
    yticklabels=range(10),
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(
    f"Confusion Matrix — KNN (k={best_k}) trên MNIST\nAccuracy: {acc*100:.2f}%"
)
plt.tight_layout()
plt.savefig("4_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# 7. VISUALIZE ẢNH SAI
# ============================================================
print("\n" + "=" * 60)
print("7. ẢNH PHÂN LOẠI SAI")
print("=" * 60)

wrong_idx = np.where(y_pred != y_te)[0]
print(
    f"Số ảnh sai: {len(wrong_idx)} / {N_TEST} ({len(wrong_idx)/N_TEST*100:.1f}%)"
)

fig, axes = plt.subplots(3, 8, figsize=(20, 8))
for i, ax in enumerate(axes.flat):
    if i >= len(wrong_idx):
        ax.axis("off")
        continue
    wi = wrong_idx[i]
    true_id = idx_test[wi]
    ax.imshow(x_test[true_id], cmap="gray")
    ax.set_title(f"True:{y_te[wi]}\nPred:{y_pred[wi]}", fontsize=9, color="red")
    ax.axis("off")

plt.suptitle("Ảnh phân loại SAI — KNN MNIST", fontsize=13)
plt.tight_layout()
plt.savefig("5_wrong_predictions.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# 8. PHÂN TÍCH KẾT QUẢ
# ============================================================
print("\n" + "=" * 60)
print("8. PHÂN TÍCH KẾT QUẢ")
print("=" * 60)

report = classification_report(y_te, y_pred, output_dict=True)

# Accuracy từng class
per_class_acc = {}
for cls in range(10):
    per_class_acc[cls] = report[str(cls)]["f1-score"]

worst_cls = min(per_class_acc, key=per_class_acc.get)
best_cls = max(per_class_acc, key=per_class_acc.get)

print(
    f"\nClass dễ nhận nhất  : Digit {best_cls}  (F1={per_class_acc[best_cls]:.4f})"
)
print(
    f"Class khó nhận nhất : Digit {worst_cls} (F1={per_class_acc[worst_cls]:.4f})"
)

# F1 từng class
plt.figure(figsize=(10, 5))
colors = ["red" if i == worst_cls else "steelblue" for i in range(10)]
plt.bar(range(10), [per_class_acc[i] for i in range(10)], color=colors)
plt.axhline(
    y=acc, color="orange", linestyle="--", label=f"Overall Accuracy={acc:.3f}"
)
plt.xlabel("Digit")
plt.ylabel("F1-Score")
plt.title("F1-Score từng class — KNN MNIST")
plt.xticks(range(10))
plt.legend()
plt.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("6_f1_per_class.png", dpi=150, bbox_inches="tight")
plt.show()

# Cặp class hay nhầm nhau
print("\nCặp class hay nhầm nhất:")
np.fill_diagonal(cm, 0)
for _ in range(5):
    idx_max = np.unravel_index(cm.argmax(), cm.shape)
    true_cls, pred_cls = idx_max
    print(
        f"  Digit {true_cls} → bị nhầm thành Digit {pred_cls}: {cm[idx_max]} lần"
    )
    cm[idx_max] = 0

# ============================================================
# 9. TỔNG KẾT
# ============================================================
print("\n" + "=" * 60)
print("9. TỔNG KẾT")
print("=" * 60)
print(f"""
  Thuật toán   : KNN (K-Nearest Neighbors)
  Dataset      : MNIST ({N_TRAIN} train / {N_TEST} test)
  Tiền xử lý  : Flatten → Normalize → PCA (784 → 100 dims)
  K tối ưu     : {best_k} (chọn bằng 3-fold Cross Validation)
  Metric       : Euclidean Distance

  ┌─────────────────────────────────┐
  │  Test Accuracy : {acc*100:.2f}%          │
  │  Ảnh đúng     : {int(acc*N_TEST):4d} / {N_TEST}         │
  │  Ảnh sai      : {N_TEST - int(acc*N_TEST):4d} / {N_TEST}         │
  └─────────────────────────────────┘

  Nhận xét:
  - KNN đạt ~{acc*100:.0f}% accuracy trên MNIST — khá tốt cho thuật toán đơn giản
  - Digit {best_cls} dễ nhận nhất (F1={per_class_acc[best_cls]:.3f})
  - Digit {worst_cls} khó nhận nhất (F1={per_class_acc[worst_cls]:.3f})
  - PCA giúp giảm chiều từ 784 → 100 nhưng vẫn giữ {explained:.1f}% thông tin
  - Hạn chế: KNN chậm với dataset lớn, không scale tốt
""")
