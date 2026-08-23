import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from pathlib import Path

# ============================================================
# CẤU HÌNH
# ============================================================
SOURCE_DIR  = "../data/raw/samples/aptos/sample_200"   # thư mục chứa các folder class
OUTPUT_DIR  = "../data/processed"    # nơi lưu file .npy
IMG_SIZE    = (300, 300)             # resize về kích thước này
BATCH_SIZE  = 32
TEST_SIZE   = 0.2                    # 20% test
VAL_SIZE    = 0.1                    # 10% val (tính trên phần train còn lại)
RANDOM_SEED = 42
# ============================================================

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# BƯỚC 1: Load ảnh từ thư mục
# ------------------------------------------------------------
print("📂 Đang load ảnh...")

full_ds = tf.keras.utils.image_dataset_from_directory(
    SOURCE_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int',   # integer label giống CIFAR (0, 1, 2, ...)
    shuffle=False       # không shuffle để giữ đúng thứ tự khi convert
)

class_names = full_ds.class_names
print(f"✅ Classes ({len(class_names)}): {class_names}")

# ------------------------------------------------------------
# BƯỚC 2: Convert sang numpy
# ------------------------------------------------------------
print("\n🔄 Đang convert sang numpy...")

X_list, y_list = [], []
for images, labels in full_ds:
    X_list.append(images.numpy())
    y_list.append(labels.numpy())

X = np.concatenate(X_list, axis=0)   # (N, 300, 300, 3) uint8
y = np.concatenate(y_list, axis=0)   # (N,) int

print(f"   X shape : {X.shape}  | dtype: {X.dtype}")
print(f"   y shape : {y.shape}  | dtype: {y.dtype}")
print(f"   Pixel range: [{X.min()}, {X.max()}]")

# Giữ uint8 (0-255) như CIFAR gốc — normalize khi train
# Lý do: tốn ít RAM hơn float32 (~4x), normalize sau khi load

# ------------------------------------------------------------
# BƯỚC 3: Chia train / test / val
# ------------------------------------------------------------
print("\n✂️  Đang chia dataset...")

# Chia train+val | test
x_trainval, x_test, y_trainval, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=y   # giữ tỉ lệ class đều nhau
)

# Chia train | val
x_train, x_val, y_train, y_val = train_test_split(
    x_trainval, y_trainval,
    test_size=VAL_SIZE / (1 - TEST_SIZE),  # tính lại tỉ lệ
    random_state=RANDOM_SEED,
    stratify=y_trainval
)

print(f"   Train : x={x_train.shape}, y={y_train.shape}")
print(f"   Val   : x={x_val.shape},   y={y_val.shape}")
print(f"   Test  : x={x_test.shape},  y={y_test.shape}")

# Kiểm tra phân phối class
print("\n📊 Phân phối class trong từng split:")
for split_name, y_split in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
    print(f"   {split_name}:", {class_names[i]: int((y_split == i).sum()) for i in range(len(class_names))})

# ------------------------------------------------------------
# BƯỚC 4: Lưu file .npy
# ------------------------------------------------------------
print(f"\n💾 Đang lưu vào '{OUTPUT_DIR}'...")

np.save(f"{OUTPUT_DIR}/x_train.npy", x_train)
np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
np.save(f"{OUTPUT_DIR}/x_val.npy",   x_val)
np.save(f"{OUTPUT_DIR}/y_val.npy",   y_val)
np.save(f"{OUTPUT_DIR}/x_test.npy",  x_test)
np.save(f"{OUTPUT_DIR}/y_test.npy",  y_test)
np.save(f"{OUTPUT_DIR}/class_names.npy", np.array(class_names))

print("✅ Lưu xong!")

# ------------------------------------------------------------
# BƯỚC 5: Verify — load lại kiểm tra
# ------------------------------------------------------------
print("\n🔍 Verify bằng cách load lại...")

x_train_ = np.load(f"{OUTPUT_DIR}/x_train.npy")
y_train_ = np.load(f"{OUTPUT_DIR}/y_train.npy")
class_names_ = np.load(f"{OUTPUT_DIR}/class_names.npy", allow_pickle=True)

print(f"   x_train : {x_train_.shape} | dtype: {x_train_.dtype}")
print(f"   y_train : {y_train_.shape} | dtype: {y_train_.dtype}")
print(f"   classes : {list(class_names_)}")
print("\n✅ Tất cả OK!")

x_train = np.load("../data/processed/x_train.npy")
y_train = np.load("../data/processed/y_train.npy")
x_val   = np.load("../data/processed/x_val.npy")
y_val   = np.load("../data/processed/y_val.npy")
x_test  = np.load("../data/processed/x_test.npy")
y_test  = np.load("../data/processed/y_test.npy")
class_names = np.load("../data/processed/class_names.npy", allow_pickle=True)

# Normalize khi dùng (giống CIFAR workflow)
x_train = x_train.astype('float32') / 255.0
x_val   = x_val.astype('float32')   / 255.0
x_test  = x_test.astype('float32')  / 255.0

print(x_train.shape)  # (N, 300, 300, 3)
print(y_train.shape)  # (N,) — integer label