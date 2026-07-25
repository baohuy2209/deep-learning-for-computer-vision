import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, GlobalAveragePooling2D, Dense, Reshape, multiply, 
    add, BatchNormalization, Activation, MaxPooling2D, Flatten, Dropout
)
from tensorflow.keras.models import Model

def se_resnet_block(x, filters, reduction_ratio=16):
    """
    Tạo một khối SE-ResNet hoàn chỉnh.
    """
    # 1. Nhánh chính (Main path - giống ResNet)
    shortcut = x
    
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    
    # Đồng bộ số kênh cho nhánh shortcut nếu cần
    if shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    # 2. Khối Squeeze-and-Excitation (SE)
    # Squeeze
    se = GlobalAveragePooling2D()(x)
    # Excitation
    se = Dense(filters // reduction_ratio, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    se = Reshape((1, 1, filters))(se)
    
    # 3. Kích hoạt trọng số SE lên nhánh chính
    x = multiply([x, se])
    
    # 4. Cộng nhánh shortcut (Residual Connection)
    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x

# XÂY DỰNG MÔ HÌNH SENET MINI
def build_senet_model(input_shape=(300, 300, 3), num_classes=5):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(inputs)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    x = se_resnet_block(x, filters=64)
    x = se_resnet_block(x, filters=128)
    
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, outputs, name="SENet_DR")