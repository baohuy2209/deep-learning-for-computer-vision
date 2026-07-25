from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

# 1. HÀM SQUASH TOÁN HỌC (Ép chiều dài vector về dải 0 -> 1)
def squash(vectors, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors

# 2. LỚP CAPSULE CHÍNH (Định tuyến động - Dynamic Routing)
class CapsuleLayer(Layer):
    def __init__(self, num_capsules, dim_capsules, routings=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules
        self.routings = routings

    def build(self, input_shape):
        # input_shape: (batch, input_num_capsules, input_dim_capsules)
        self.input_num_capsules = input_shape[1]
        self.input_dim_capsules = input_shape[2]
        
        # Trọng số W biến đổi vector từ lớp trước sang lớp sau
        self.W = self.add_weight(shape=[self.num_capsules, self.input_num_capsules, 
                                        self.dim_capsules, self.input_dim_capsules],
                                 initializer='glorot_uniform',
                                 name='W')
        self.built = True

    def call(self, inputs, training=None):
        # Mở rộng chiều để nhân ma trận
        inputs_expand = tf.expand_dims(tf.expand_dims(inputs, 1), -1)
        
        # Nhân inputs với trọng số W -> u_hat
        inputs_tiled = tf.tile(inputs_expand, [1, self.num_capsules, 1, 1, 1])
        u_hat = tf.map_fn(lambda x: tf.matmul(self.W, x), elems=inputs_tiled)
        u_hat = tf.squeeze(u_hat, axis=-1)

        # THUẬT TOÁN ĐỊNH TUYẾN ĐỘNG (Dynamic Routing)
        b = tf.zeros(shape=[tf.shape(inputs)[0], self.num_capsules, self.input_num_capsules])
        
        for i in range(self.routings):
            # Softmax các hệ số ghép nối
            c = tf.nn.softmax(b, axis=1)
            # Tính tổng có trọng số
            s = tf.einsum('bij,bijk->bik', c, u_hat)
            # Ép vector (Squash)
            v = squash(s)
            
            # Cập nhật hệ số b cho vòng lặp tiếp theo
            if i < self.routings - 1:
                b += tf.einsum('bik,bijk->bij', v, u_hat)
                
        return v # Output shape: (Batch, num_capsules, dim_capsules)

# 3. LỚP CHUYỂN ĐỔI TỪ CNN SANG CAPSULE
def PrimaryCap(inputs, dim_capsule, n_channels, kernel_size, strides, padding):
    output = Conv2D(filters=dim_capsule*n_channels, kernel_size=kernel_size, strides=strides, padding=padding)(inputs)
    # Reshape về dạng danh sách các vector
    outputs = Reshape(target_shape=[-1, dim_capsule])(output)
    return Activation(squash)(outputs)

# 4. TÍCH HỢP ATTENTION VÀO CAPSULENET
def build_am_capsulenet(input_shape=(300, 300, 3), num_classes=5):
    inputs = Input(shape=input_shape)
    
    # Khối CNN trích xuất cơ sở
    x = Conv2D(64, (5, 5), strides=(2, 2), padding='valid', activation='relu')(inputs)
    x = Conv2D(128, (3, 3), strides=(2, 2), padding='valid', activation='relu')(x)
    
    # GẮN CƠ CHẾ ATTENTION (Spatial Attention) TRƯỚC KHI VÀO CAPSULE
    avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
    concat   = concatenate([avg_pool, max_pool], axis=-1)
    attention_mask = Conv2D(1, (7, 7), padding='same', activation='sigmoid')(concat)
    x = multiply([x, attention_mask]) # Chỉ vùng có tổn thương mới được cho qua
    
    # Tạo Primary Capsules
    # Ép Feature Map thành các vector độ dài 8
    primarycaps = PrimaryCap(x, dim_capsule=8, n_channels=16, kernel_size=9, strides=2, padding='valid')
    
    # Lớp Routing Capsules (Tương đương lớp Classification Head)
    # 5 capsules đại diện cho 5 mức độ DR, mỗi capsule là 1 vector độ dài 16
    digitcaps = CapsuleLayer(num_capsules=num_classes, dim_capsules=16, routings=3)(primarycaps)
    
    # Tính độ dài của vector để ra xác suất dự đoán
    # Công thức: L2-norm của vector cuối cùng
    out_caps = tf.sqrt(tf.reduce_sum(tf.square(digitcaps), axis=2) + K.epsilon())
    
    return Model(inputs, out_caps, name="AM_CapsNet_DR")