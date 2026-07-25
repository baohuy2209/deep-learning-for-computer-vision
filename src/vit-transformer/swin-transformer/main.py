import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import pickle
import numpy as np

# 1. IMPORT HUGGING FACE TRANSFORMERS
from transformers import TFSwinModel

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

y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test  = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

# ==========================================
# 3. HÀM TẠO DATASET PIPELINE CHO SWIN
# ==========================================
BATCH_SIZE = 16 
SWIN_IMAGE_SIZE = 224 # Swin-Tiny được train mặc định trên ảnh 224x224

# Các hằng số chuẩn hóa của ImageNet (Bắt buộc với model của Hugging Face)
IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406])
IMAGENET_STD  = tf.constant([0.229, 0.224, 0.225])

def create_swin_pipeline(x_data, y_data):
    def preprocess_wrapper(image, label):
        # 1. Ép kiểu và chuẩn hóa pixel về dải [0, 1]
        image = tf.cast(image, tf.float32) / 255.0
        
        # 2. Resize về 224x224
        image = tf.image.resize(image, (SWIN_IMAGE_SIZE, SWIN_IMAGE_SIZE))
        
        # 3. Chuẩn hóa theo Mean và Std của ImageNet
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        return image, label

    with tf.device('/CPU:0'):
        dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
        dataset = dataset.map(preprocess_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = create_swin_pipeline(x_train, y_train)
val_dataset   = create_swin_pipeline(x_test, y_test)

# ==========================================
# 4. TẠO MODEL BẰNG KERAS SUBCLASSING (CHỐNG LỖI HUGGING FACE)
# ==========================================
class SwinClassifier(tf.keras.Model):
    def __init__(self, num_classes):
        super(SwinClassifier, self).__init__()
        print("🚀 Đang tải trọng số Swin-Tiny từ Hugging Face...")
        # Khởi tạo phần thân (Backbone)
        self.backbone = TFSwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        # Đóng băng Backbone để Transfer Learning
        self.backbone.trainable = False
        
        # Khởi tạo phần đầu phân loại (Classification Head)
        self.dense = Dense(256, activation='gelu')
        self.dropout = Dropout(0.4)
        self.classifier = Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        # Đưa ảnh qua Swin Transformer
        outputs = self.backbone(inputs, training=training)
        
        # pooler_output là vector đặc trưng đại diện cho toàn bộ bức ảnh (shape: Batch, 768)
        x = outputs.pooler_output 
        
        # Đưa qua các lớp Dense
        x = self.dense(x)
        x = self.dropout(x, training=training)
        return self.classifier(x)

# ==========================================
# 5. KHỞI TẠO VÀ HUẤN LUYỆN
# ==========================================
# Build model bằng cách truyền một batch ảo vào để Keras xác định kích thước
model = SwinClassifier(NUM_CLASSES)
model.build((None, SWIN_IMAGE_SIZE, SWIN_IMAGE_SIZE, 3)) 
model.summary()

# Compile model
# Tương tự ViT, Swin thích learning rate nhỏ
model.compile(optimizer=Adam(learning_rate=5e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Định nghĩa Callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        '../../../data/models/swin_tiny_aptos_best.keras',
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