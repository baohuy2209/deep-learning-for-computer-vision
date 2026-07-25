import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import pickle
import numpy as np

# 1. IMPORT KIẾN TRÚC VIT TỪ VIT-KERAS
from vit_keras import vit

# Xóa rác bộ nhớ GPU
tf.keras.backend.clear_session()

# ==========================================
# 2. ĐỌC VÀ CHIA DỮ LIỆU
# ==========================================
with open("../../../data/processed/aptos/sample_50/sample_dataset.pkl", "rb") as f:
    data = pickle.load(f)

X           = data['data']        
y           = data['labels']      
class_names = data['class_names'] 
NUM_CLASSES = len(class_names)

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Chuyển đổi nhãn sang One-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test  = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

# ==========================================
# 3. HÀM TẠO DATASET PIPELINE (CÓ RESIZE TỰ ĐỘNG)
# ==========================================
BATCH_SIZE = 16 
VIT_IMAGE_SIZE = 224 # Kích thước chuẩn của ViT, bắt buộc chia hết cho 16

def create_vit_pipeline(x_data, y_data):
    def preprocess_wrapper(image, label):
        image = tf.cast(image, tf.float32)
        # Resize ảnh từ 300x300 về 224x224 để chia hết cho patch_size=16
        image = tf.image.resize(image, (VIT_IMAGE_SIZE, VIT_IMAGE_SIZE))
        # Hàm tiền xử lý của ViT (Scale về [-1, 1])
        image = vit.preprocess_inputs(image)
        return image, label

    with tf.device('/CPU:0'):
        dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
        dataset = dataset.map(preprocess_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = create_vit_pipeline(x_train, y_train)
val_dataset   = create_vit_pipeline(x_test, y_test)

# ==========================================
# 4. HÀM XÂY DỰNG MÔ HÌNH VIT
# ==========================================
def build_vit_model(image_size, num_classes):
    print("🚀 Đang khởi tạo kiến trúc: ViT-Base (Patch 16)...")
    
    # Khởi tạo Base Model ViT_B16
    base_model = vit.vit_b16(
        image_size=image_size,
        activation='softmax',
        pretrained=True,
        include_top=False, # Bỏ lớp phân loại của ImageNet
        pretrained_top=False
    )
    
    # Đóng băng (Freeze) phần thân Transformer
    base_model.trainable = False
    
    # Xây dựng Classification Head
    # Kích thước đầu ra của include_top=False là (None, 768) do lấy token [CLS]
    x = base_model.output
    x = BatchNormalization()(x)
    x = Dense(256, activation='gelu')(x) # GELU thường kết hợp tốt hơn với Transformer
    x = Dropout(0.4)(x) 
    
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Khuyến nghị dùng learning_rate nhỏ hơn so với CNN khi train ViT
    model.compile(optimizer=Adam(learning_rate=5e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ==========================================
# 5. KHỞI TẠO VÀ HUẤN LUYỆN
# ==========================================
model = build_vit_model(VIT_IMAGE_SIZE, NUM_CLASSES)
model.summary()

# Định nghĩa Callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        '../../../data/models/vit_b16_aptos_best.keras',
        monitor='val_accuracy', save_best_only=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=12, restore_best_weights=True, verbose=1
    )
]

# Bắt đầu huấn luyện
history = model.fit(
    train_dataset,
    epochs=100, 
    validation_data=val_dataset,
    callbacks=callbacks,
    verbose=1
)