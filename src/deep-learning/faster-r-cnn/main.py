import tensorflow as tf
import keras_cv
from tensorflow.keras.optimizers import SGD

# 1. ĐỊNH NGHĨA SỐ LƯỢNG LỚP VÀ ĐỊNH DẠNG HỘP (BOUNDING BOX)
NUM_CLASSES = 5
# Bounding box trong KerasCV thường dùng định dạng "xyxy" (xmin, ymin, xmax, ymax)
BBOX_FORMAT = "xyxy"


def build_faster_rcnn_model(num_classes, bbox_format):
    # 2. KHỞI TẠO BACKBONE (Phần thân trích xuất đặc trưng)
    # Tương tự như bài toán phân loại, ta dùng ResNet50 làm xương sống
    backbone = keras_cv.models.ResNet50V2Backbone.from_preset(
        "resnet50_v2_imagenet"
    )

    # 3. XÂY DỰNG KIẾN TRÚC FASTER R-CNN
    model = keras_cv.models.FasterRCNN(
        num_classes=num_classes,
        bounding_box_format=bbox_format,
        backbone=backbone,
        # Kích thước anchor boxes do RPN sinh ra để quét vật thể
        anchor_generator=keras_cv.models.FeaturePyramidNetwork(
            bottom_resolution=256, top_resolution=256
        ),
    )

    # 4. THIẾT LẬP OPTIMIZER VÀ BỘ MẤT MÁT (LOSS)
    # Khác với phân loại chỉ có 1 loss, Faster R-CNN có 2 loss:
    # Box Loss (sai số tọa độ hộp) và Classification Loss (sai số nhãn)
    optimizer = SGD(learning_rate=0.005, momentum=0.9, global_clipnorm=10.0)

    model.compile(
        optimizer=optimizer,
        box_loss="Huber",  # Loss dùng cho tính toán tọa độ
        classification_loss="FocalCrossEntropy",  # Loss dùng cho phân loại vật thể
    )

    return model


# Khởi tạo mô hình
faster_rcnn_model = build_faster_rcnn_model(NUM_CLASSES, BBOX_FORMAT)

# In ra cấu trúc của mô hình
faster_rcnn_model.summary()
