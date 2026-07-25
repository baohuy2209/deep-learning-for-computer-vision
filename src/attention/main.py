import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D,
    Reshape, multiply, add, concatenate, Activation, Input,
    MaxPooling2D, Flatten, Dropout, BatchNormalization, MultiHeadAttention
)
from tensorflow.keras.models import Model

# ==========================================
# 1. CHÚ Ý THEO KÊNH (CHANNEL ATTENTION - Cấu trúc SE-Net)
# ==========================================
def channel_attention_block(input_tensor, reduction_ratio=8):
    """
    Kỹ thuật Squeeze-and-Excitation: Tìm xem kênh (feature map) nào quan trọng.
    """
    channels = input_tensor.shape[-1]
    
    # Squeeze: Nén không gian (H, W) thành 1 điểm
    x = GlobalAveragePooling2D()(input_tensor)
    
    # Excitation: Dùng 2 lớp Dense để học trọng số
    x = Dense(channels // reduction_ratio, activation='relu')(x)
    x = Dense(channels, activation='sigmoid')(x)
    x = Reshape((1, 1, channels))(x)
    
    # Nhân trọng số trở lại ảnh gốc
    return multiply([input_tensor, x])

# ==========================================
# 2. CHÚ Ý KHÔNG GIAN (SPATIAL ATTENTION - SAM)
# ==========================================
def spatial_attention_block(input_tensor):
    """
    Tìm xem vị trí nào trên bức ảnh (pixel) là quan trọng.
    """
    # Lấy giá trị trung bình và lớn nhất dọc theo trục Kênh
    avg_pool = tf.reduce_mean(input_tensor, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(input_tensor, axis=-1, keepdims=True)
    
    # Nối 2 bản đồ lại
    concat = concatenate([avg_pool, max_pool], axis=-1)
    
    # Dùng lớp tích chập 7x7 để tạo Mặt nạ không gian (Spatial Mask)
    attention = Conv2D(1, (7, 7), padding='same', activation='sigmoid')(concat)
    
    return multiply([input_tensor, attention])

# ==========================================
# 3. CHÚ Ý LAI (HYBRID ATTENTION - CBAM)
# ==========================================
def cbam_block(input_tensor, reduction_ratio=8):
    """
    Kết hợp nối tiếp: Channel Attention -> Spatial Attention
    """
    # 3.1. Nhánh Channel (CBAM chuẩn dùng cả Avg và Max pooling)
    channels = input_tensor.shape[-1]
    shared_dense_1 = Dense(channels // reduction_ratio, activation='relu')
    shared_dense_2 = Dense(channels)

    # Đường Avg
    avg_pool = GlobalAveragePooling2D()(input_tensor)
    avg_pool = Reshape((1, 1, channels))(avg_pool)
    avg_pool = shared_dense_2(shared_dense_1(avg_pool))

    # Đường Max
    max_pool = GlobalMaxPooling2D()(input_tensor)
    max_pool = Reshape((1, 1, channels))(max_pool)
    max_pool = shared_dense_2(shared_dense_1(max_pool))

    channel_attention = Activation('sigmoid')(add([avg_pool, max_pool]))
    x = multiply([input_tensor, channel_attention])

    # 3.2. Nhánh Spatial
    x = spatial_attention_block(x)
    return x

# ==========================================
# 4. TỰ CHÚ Ý (SELF-ATTENTION CHO ẢNH)
# ==========================================
def self_attention_block(input_tensor, num_heads=4):
    """
    Biến đổi ảnh thành dạng chuỗi (sequence) giống văn bản để tính Multi-Head Attention.
    """
    _, h, w, c = input_tensor.shape
    
    # Reshape ảnh từ (Batch, H, W, Channels) thành (Batch, H*W, Channels)
    # H*W ở đây đóng vai trò như độ dài câu (sequence length)
    x = Reshape((h * w, c))(input_tensor)
    
    # Multi-Head Attention (Self-Attention: Q, K, V đều là x)
    attention_out = MultiHeadAttention(num_heads=num_heads, key_dim=c)(x, x)
    
    # Reshape ngược lại thành ảnh 2D
    attention_out = Reshape((h, w, c))(attention_out)
    
    # Residual Connection (Kết nối tắt)
    return add([input_tensor, attention_out])

def build_attention_model(input_shape, num_classes, attention_type=None):
    """
    attention_type: 'channel', 'spatial', 'cbam', 'self', hoặc None
    """
    inputs = Input(shape=input_shape)
    
    # -----------------------------------
    # Block 1: CNN Trích xuất cơ bản
    # -----------------------------------
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    # -----------------------------------
    # CẤY MODULE ATTENTION VÀO ĐÂY
    # (Thường cắm ở các lớp sâu, nơi feature map đã mang tính ngữ nghĩa)
    # -----------------------------------
    if attention_type == 'channel':
        print("💡 Đang tích hợp Squeeze-and-Excitation (Channel Attention)...")
        x = channel_attention_block(x)
    elif attention_type == 'spatial':
        print("💡 Đang tích hợp Spatial Attention...")
        x = spatial_attention_block(x)
    elif attention_type == 'cbam':
        print("💡 Đang tích hợp CBAM (Hybrid Attention)...")
        x = cbam_block(x)
    elif attention_type == 'self':
        print("💡 Đang tích hợp Multi-Head Self-Attention...")
        x = self_attention_block(x)
    else:
        print("⚠️ Không dùng Attention (CNN thuần).")

    # -----------------------------------
    # Classification Head (Đầu phân loại)
    # -----------------------------------
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ==========================================
# VÍ DỤ KHỞI TẠO MÔ HÌNH VỚI CBAM (Dùng cho ảnh 300x300, 5 nhãn)
# ==========================================
model_cbam = build_attention_model(
    input_shape=(300, 300, 3), 
    num_classes=5, 
    attention_type='cbam' # THAY ĐỔI TÊN Ở ĐÂY ĐỂ ĐỔI CƠ CHẾ
)

model_cbam.summary()

# (Sau bước này, bạn gọi hàm model_cbam.fit(train_dataset, ...) y hệt như các code trước đó)