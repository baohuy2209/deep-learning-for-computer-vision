import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import pickle
import numpy as np

# 1. IMPORT KIẾN TRÚC VÀ HÀM TIỀN XỬ LÝ CỦA CONVNEXT
# Bạn có thể dùng ConvNeXtTiny, ConvNeXtSmall, ConvNeXtBase, ConvNeXtLarge, ConvNeXtXLarge
from tensorflow.keras.applications import ConvNeXtTiny
from tensorflow.keras.applications.convnext import preprocess_input as preprocess_convnext

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
IMG_SIZE    = X.shape[1]          # Đang là 300

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Chuyển đổi nhãn sang One-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test  = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

# ==========================================
# 3. HÀM TẠO DATASET PIPELINE
# ==========================================
BATCH_SIZE = 16 # Bắt đầu với 16. Nếu báo lỗi OOM (Hết VRAM), hãy giảm xuống 8 hoặc 4.

def create_convnext_pipeline(x_data, y_data):
    def preprocess_wrapper(image, label):
        image = tf.cast(image, tf.float32)
        # Hàm tiền xử lý chuẩn của ConvNeXt
        image = preprocess_convnext(image)
        return image, label

    with tf.device('/CPU:0'): # Đẩy việc chuẩn bị Dataset lên CPU
        dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
        dataset = dataset.map(preprocess_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = create_convnext_pipeline(x_train, y_train)
val_dataset   = create_convnext_pipeline(x_test, y_test)

# ==========================================
# 4. HÀM XÂY DỰNG MÔ HÌNH CONVNEXT
# ==========================================
def build_convnext_model(input_shape, num_classes):
    print("🚀 Đang khởi tạo kiến trúc: ConvNeXtTiny...")
    
    # Khởi tạo Base Model ConvNeXtTiny
    base_model = ConvNeXtTiny(weights='imagenet', 
                              include_top=False, 
                              input_shape=input_shape)
    
    # Đóng băng (Freeze) phần thân để Transfer Learning
    base_model.trainable = False
    
    # Xây dựng Classification Head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # ConvNeXt vốn đã học đặc trưng cực kỳ xịn, 
    # ta không cần nhồi nhét quá nhiều lớp Dense làm gì để tránh học vẹt.
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x) 
    
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ==========================================
# 5. KHỞI TẠO VÀ HUẤN LUYỆN
# ==========================================
model = build_convnext_model((IMG_SIZE, IMG_SIZE, 3), NUM_CLASSES)
model.summary()

# Định nghĩa Callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        '../../../data/models/convnext_tiny_aptos_best.keras',
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