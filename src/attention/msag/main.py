from tensorflow.keras.layers import concatenate, UpSampling2D

def multi_scale_block(x, filters):
    """
    Trích xuất đặc trưng đa tỷ lệ (như Inception)
    """
    branch1x1 = Conv2D(filters, (1, 1), padding='same', activation='relu')(x)
    
    branch3x3 = Conv2D(filters, (1, 1), padding='same', activation='relu')(x)
    branch3x3 = Conv2D(filters, (3, 3), padding='same', activation='relu')(branch3x3)
    
    branch5x5 = Conv2D(filters, (1, 1), padding='same', activation='relu')(x)
    branch5x5 = Conv2D(filters, (5, 5), padding='same', activation='relu')(branch5x5)
    
    return concatenate([branch1x1, branch3x3, branch5x5], axis=-1)

def attention_gate(x, g, inter_shape):
    """
    Cổng chú ý không gian.
    x: Feature map ở lớp nông (Nhiều chi tiết).
    g: Gating signal từ lớp sâu (Nhiều ngữ nghĩa).
    """
    # Ép x và g về cùng số kênh inter_shape
    theta_x = Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)
    phi_g   = Conv2D(inter_shape, (1, 1), padding='same')(g)
    
    # Cộng gộp và tính toán Attention Mask
    concat_xg = add([theta_x, phi_g])
    act_xg    = Activation('relu')(concat_xg)
    psi       = Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    
    # Phóng to Mask về bằng kích thước của x
    alpha = UpSampling2D(size=(2, 2))(sigmoid_xg)
    
    # Nhân Mask vào x gốc
    return multiply([x, alpha])

# XÂY DỰNG MÔ HÌNH MSAG
def build_msag_model(input_shape=(300, 300, 3), num_classes=5):
    inputs = Input(shape=input_shape)
    
    # Trích xuất đặc trưng nông
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    
    # Lớp nông (Multi-scale)
    ms_feat = multi_scale_block(x, filters=32) # Kích thước 150x150
    
    # Lớp sâu (Tạo Gating Signal)
    g = MaxPooling2D((2, 2))(ms_feat) # Kích thước 75x75
    g = Conv2D(128, (3, 3), padding='same', activation='relu')(g)
    
    # Đưa qua Attention Gate
    # Cổng sẽ dùng g (hiểu bức tranh lớn) để lọc ms_feat (giữ chi tiết nhỏ)
    attended_features = attention_gate(x=ms_feat, g=g, inter_shape=64)
    
    # Xử lý phần còn lại
    out = GlobalAveragePooling2D()(attended_features)
    out = Dense(256, activation='relu')(out)
    outputs = Dense(num_classes, activation='softmax')(out)
    
    return Model(inputs, outputs, name="MSAG_DR")