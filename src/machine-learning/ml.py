import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2gray
from tensorflow.keras.datasets import cifar10
import time
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 60)
print("1. LOAD DỮ LIỆU CIFAR-10")
print("=" * 60)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = y_train.ravel()
y_test = y_test.ravel()

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

print(f"Train : {x_train.shape} | Label: {y_train.shape}")
print(f"Test  : {x_test.shape}  | Label: {y_test.shape}")

# Subset để chạy nhanh
N_TRAIN, N_TEST = 10000, 2000
idx_tr = np.random.choice(len(x_train), N_TRAIN, replace=False)
idx_te = np.random.choice(len(x_test), N_TEST, replace=False)

x_tr_raw = x_train[idx_tr]
y_tr = y_train[idx_tr]
x_te_raw = x_test[idx_te]
y_te = y_test[idx_te]

print(f"Subset: Train={N_TRAIN}, Test={N_TEST}")

# Visualize mẫu
fig, axes = plt.subplots(2, 10, figsize=(20, 4))
for cls in range(10):
    idx = np.where(y_tr == cls)[0][0]
    axes[0, cls].imshow(x_tr_raw[idx])
    axes[0, cls].set_title(class_names[cls], fontsize=8)
    axes[0, cls].axis("off")
    axes[1, cls].imshow(x_tr_raw[np.where(y_tr == cls)[0][1]])
    axes[1, cls].axis("off")
plt.suptitle("Mẫu ảnh CIFAR-10", fontsize=13)
plt.tight_layout()
plt.savefig("1_sample_images.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# 2. TRÍCH XUẤT ĐẶC TRƯNG
# ============================================================
print("\n" + "=" * 60)
print("2. TRÍCH XUẤT ĐẶC TRƯNG (Feature Extraction)")
print("=" * 60)


# ----------------------------------------------------------
# 2.1 Flatten + Normalize (baseline đơn giản nhất)
# ----------------------------------------------------------
def extract_flatten(images):
    return images.reshape(len(images), -1).astype("float32") / 255.0


# ----------------------------------------------------------
# 2.2 HOG — Histogram of Oriented Gradients
#     Nắm bắt hình dạng (shape) và cạnh (edge) của vật thể
# ----------------------------------------------------------
def extract_hog(images):
    features = []
    for img in images:
        gray = rgb2gray(img)
        feat = hog(
            gray,
            orientations=9,  # 9 hướng gradient
            pixels_per_cell=(4, 4),  # cell nhỏ vì ảnh 32x32
            cells_per_block=(2, 2),
            feature_vector=True,
        )
        features.append(feat)
    return np.array(features)


# ----------------------------------------------------------
# 2.3 LBP — Local Binary Pattern
#     Nắm bắt texture (kết cấu bề mặt)
# ----------------------------------------------------------
def extract_lbp(images, P=8, R=1):
    features = []
    for img in images:
        gray = rgb2gray(img)
        lbp = local_binary_pattern(gray, P=P, R=R, method="uniform")
        hist, _ = np.histogram(
            lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2)
        )
        hist = hist.astype("float") / (hist.sum() + 1e-6)
        features.append(hist)
    return np.array(features)


# ----------------------------------------------------------
# 2.4 Color Histogram
#     Nắm bắt màu sắc của ảnh
# ----------------------------------------------------------
def extract_color_hist(images, bins=32):
    features = []
    for img in images:
        hist_r, _ = np.histogram(img[:, :, 0], bins=bins, range=(0, 256))
        hist_g, _ = np.histogram(img[:, :, 1], bins=bins, range=(0, 256))
        hist_b, _ = np.histogram(img[:, :, 2], bins=bins, range=(0, 256))
        feat = np.concatenate([hist_r, hist_g, hist_b]).astype("float32")
        feat /= feat.sum() + 1e-6
        features.append(feat)
    return np.array(features)


# ----------------------------------------------------------
# 2.5 HOG + Color Hist kết hợp (thường tốt nhất)
# ----------------------------------------------------------
def extract_combined(images):
    hog_feat = extract_hog(images)
    color_feat = extract_color_hist(images)
    return np.concatenate([hog_feat, color_feat], axis=1)


# Trích xuất tất cả
print("Đang trích xuất features...")

print("  [1/5] Flatten...")
x_tr_flat = extract_flatten(x_tr_raw)
x_te_flat = extract_flatten(x_te_raw)

print("  [2/5] HOG...")
x_tr_hog = extract_hog(x_tr_raw)
x_te_hog = extract_hog(x_te_raw)

print("  [3/5] LBP...")
x_tr_lbp = extract_lbp(x_tr_raw)
x_te_lbp = extract_lbp(x_te_raw)

print("  [4/5] Color Histogram...")
x_tr_color = extract_color_hist(x_tr_raw)
x_te_color = extract_color_hist(x_te_raw)

print("  [5/5] HOG + Color (Combined)...")
x_tr_comb = extract_combined(x_tr_raw)
x_te_comb = extract_combined(x_te_raw)

print(f"\nKích thước feature vectors:")
print(f"  Flatten  : {x_tr_flat.shape[1]}")
print(f"  HOG      : {x_tr_hog.shape[1]}")
print(f"  LBP      : {x_tr_lbp.shape[1]}")
print(f"  Color    : {x_tr_color.shape[1]}")
print(f"  Combined : {x_tr_comb.shape[1]}")

# ============================================================
# 3. PCA VISUALIZATION
# ============================================================
print("\n" + "=" * 60)
print("3. PCA VISUALIZATION")
print("=" * 60)

pca_2d = PCA(n_components=2)
x_2d = pca_2d.fit_transform(x_tr_hog)

plt.figure(figsize=(10, 8))
colors = plt.cm.tab10(np.linspace(0, 1, 10))
for cls in range(10):
    mask = y_tr == cls
    plt.scatter(
        x_2d[mask, 0],
        x_2d[mask, 1],
        c=[colors[cls]],
        label=class_names[cls],
        alpha=0.5,
        s=15,
    )
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.title("PCA 2D — HOG features (CIFAR-10)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig("2_pca_visualization.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# 4. SO SÁNH CÁC PHƯƠNG PHÁP ML
# ============================================================
print("\n" + "=" * 60)
print("4. SO SÁNH CÁC PHƯƠNG PHÁP ML TRUYỀN THỐNG")
print("=" * 60)

# Dùng HOG + Color (Combined) cho tất cả model
scaler = StandardScaler()
x_tr_sc = scaler.fit_transform(x_tr_comb)
x_te_sc = scaler.transform(x_te_comb)

# PCA giảm chiều
pca = PCA(n_components=200, random_state=42)
x_tr_pca = pca.fit_transform(x_tr_sc)
x_te_pca = pca.transform(x_te_sc)

models = {
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "SVM (RBF)": SVC(kernel="rbf", C=10, gamma="scale", random_state=42),
    "SVM (Linear)": SVC(kernel="linear", C=1, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, n_jobs=-1, random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100, random_state=42
    ),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42, n_jobs=-1
    ),
}

results = {}

for name, model in models.items():
    print(f"\n  Đang train: {name}...")
    t0 = time.time()
    model.fit(x_tr_pca, y_tr)
    t_train = time.time() - t0

    t0 = time.time()
    y_pred = model.predict(x_te_pca)
    t_pred = time.time() - t0

    acc = accuracy_score(y_te, y_pred)
    results[name] = {
        "model": model,
        "y_pred": y_pred,
        "acc": acc,
        "t_train": t_train,
        "t_pred": t_pred,
    }
    print(f"    Accuracy : {acc*100:.2f}%")
    print(f"    Train    : {t_train:.1f}s | Predict: {t_pred:.2f}s")

# ============================================================
# 5. SO SÁNH KẾT QUẢ
# ============================================================
print("\n" + "=" * 60)
print("5. SO SÁNH KẾT QUẢ")
print("=" * 60)

names = list(results.keys())
accs = [results[n]["acc"] * 100 for n in names]
t_trains = [results[n]["t_train"] for n in names]

# Bar chart accuracy
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

colors_bar = ["gold" if a == max(accs) else "steelblue" for a in accs]
bars = axes[0].bar(
    names, accs, color=colors_bar, edgecolor="black", linewidth=0.5
)
axes[0].set_ylim(0, 100)
axes[0].set_ylabel("Accuracy (%)")
axes[0].set_title(
    "Accuracy — ML Methods trên CIFAR-10\n(HOG + Color features + PCA)"
)
axes[0].tick_params(axis="x", rotation=30)
for bar, acc in zip(bars, accs):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{acc:.1f}%",
        ha="center",
        va="bottom",
        fontsize=9,
    )

# Train time
axes[1].bar(names, t_trains, color="salmon", edgecolor="black", linewidth=0.5)
axes[1].set_ylabel("Thời gian train (giây)")
axes[1].set_title("Thời gian Train")
axes[1].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig("3_model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# 6. PHÂN TÍCH CHI TIẾT MODEL TỐT NHẤT
# ============================================================
print("\n" + "=" * 60)
print("6. PHÂN TÍCH CHI TIẾT MODEL TỐT NHẤT")
print("=" * 60)

best_name = max(results, key=lambda n: results[n]["acc"])
best = results[best_name]
y_pred = best["y_pred"]

print(f"Model tốt nhất: {best_name}")
print(f"Accuracy: {best['acc']*100:.2f}%")
print(f"\nClassification Report:")
print(classification_report(y_te, y_pred, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(y_te, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix — {best_name}\nAccuracy: {best['acc']*100:.2f}%")
plt.tight_layout()
plt.savefig("4_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# 7. SO SÁNH FEATURE EXTRACTION
# ============================================================
print("\n" + "=" * 60)
print("7. SO SÁNH FEATURE EXTRACTION với SVM")
print("=" * 60)

feature_sets = {
    "Flatten": (x_tr_flat, x_te_flat),
    "HOG": (x_tr_hog, x_te_hog),
    "LBP": (x_tr_lbp, x_te_lbp),
    "Color": (x_tr_color, x_te_color),
    "Combined": (x_tr_comb, x_te_comb),
}

feat_accs = {}
for feat_name, (x_tr_f, x_te_f) in feature_sets.items():
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "pca",
                PCA(n_components=min(100, x_tr_f.shape[1]), random_state=42),
            ),
            ("svm", SVC(kernel="rbf", C=10, gamma="scale", random_state=42)),
        ]
    )
    pipe.fit(x_tr_f, y_tr)
    acc = accuracy_score(y_te, pipe.predict(x_te_f))
    feat_accs[feat_name] = acc * 100
    print(f"  {feat_name:10s}: {acc*100:.2f}%")

plt.figure(figsize=(8, 5))
colors_feat = [
    "gold" if v == max(feat_accs.values()) else "steelblue"
    for v in feat_accs.values()
]
bars = plt.bar(
    feat_accs.keys(),
    feat_accs.values(),
    color=colors_feat,
    edgecolor="black",
    linewidth=0.5,
)
for bar, acc in zip(bars, feat_accs.values()):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.2,
        f"{acc:.1f}%",
        ha="center",
        va="bottom",
        fontsize=10,
    )
plt.ylabel("Accuracy (%)")
plt.title("So sánh Feature Extraction — SVM (RBF)")
plt.ylim(0, 80)
plt.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("5_feature_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# 8. VISUALIZE ẢNH SAI
# ============================================================
print("\n" + "=" * 60)
print("8. VISUALIZE ẢNH PHÂN LOẠI SAI")
print("=" * 60)

wrong_idx = np.where(y_pred != y_te)[0]
print(
    f"Sai: {len(wrong_idx)} / {N_TEST} ảnh ({len(wrong_idx)/N_TEST*100:.1f}%)"
)

fig, axes = plt.subplots(3, 8, figsize=(20, 8))
for i, ax in enumerate(axes.flat):
    if i >= len(wrong_idx):
        ax.axis("off")
        continue
    wi = wrong_idx[i]
    ax.imshow(x_te_raw[wi])
    ax.set_title(
        f"T:{class_names[y_te[wi]]}\nP:{class_names[y_pred[wi]]}",
        fontsize=7,
        color="red",
    )
    ax.axis("off")
plt.suptitle(f"Ảnh phân loại SAI — {best_name}", fontsize=13)
plt.tight_layout()
plt.savefig("6_wrong_predictions.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# 9. TỔNG KẾT
# ============================================================
print("\n" + "=" * 60)
print("9. TỔNG KẾT")
print("=" * 60)

print(f"""
┌─────────────────────────────────────────────────────┐
│           KẾT QUẢ PHÂN LOẠI CIFAR-10               │
│         Phương pháp ML Truyền Thống                 │
├──────────────────────────┬──────────────────────────┤
│ Model                    │ Accuracy                 │""")
for name in sorted(results, key=lambda n: results[n]["acc"], reverse=True):
    marker = " ✅" if name == best_name else "   "
    print(f"│ {name:24s} │ {results[name]['acc']*100:6.2f}%{marker:>16s}│")
print(f"""├──────────────────────────┴──────────────────────────┤
│ Feature tốt nhất : Combined (HOG + Color Hist)      │
│ Model tốt nhất   : {best_name:33s}│
│ Accuracy tốt nhất: {best['acc']*100:.2f}%                             │
└─────────────────────────────────────────────────────┘

  Nhận xét:
  ✅ SVM (RBF) thường là model mạnh nhất ML truyền thống
  ✅ HOG nắm bắt hình dạng tốt — quan trọng nhất
  ✅ Color Histogram bổ sung thông tin màu sắc
  ✅ Kết hợp HOG + Color cho kết quả tốt nhất

  Hạn chế so với Deep Learning:
  ❌ Accuracy ~50-60% vs CNN ~90-95%
  ❌ Feature extraction thủ công, không tự học
  ❌ Không nắm bắt được đặc trưng phức tạp
  ❌ Không scale tốt với dataset lớn
""")
