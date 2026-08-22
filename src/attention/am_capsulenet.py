import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K


def squash(vectors, axis=-1, epsilon=1e-7):
    """Squash activation function for vector capsules.

    Scales vector magnitudes into the range [0, 1) while preserving direction.

    Args:
        vectors (tf.Tensor): Tensor of capsule vectors.
        axis (int): Axis along which to compute vector length.
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        tf.Tensor: Squashed vector tensor.
    """
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis=axis, keepdims=True)
    scale = s_squared_norm / (1.0 + s_squared_norm) / (tf.sqrt(s_squared_norm + epsilon))
    return scale * vectors


@tf.keras.utils.register_keras_serializable(package="src.attention")
class PrimaryCapsule(layers.Layer):
    """Primary Capsule Layer.

    Converts standard 2D feature maps from convolutional backbones into initial 1D vector capsules.

    Args:
        num_channels (int): Number of capsule channels. Default: 8.
        dim_capsules (int): Dimension of each capsule vector. Default: 8.
        kernel_size (int or tuple): Conv2D kernel size. Default: 3.
        strides (int or tuple): Conv2D stride. Default: 1.
        padding (str): Conv2D padding mode. Default: 'same'.
        **kwargs: Additional keyword arguments for tf.keras.layers.Layer.
    """

    def __init__(self, num_channels=8, dim_capsules=8, kernel_size=3, strides=1, padding="same", **kwargs):
        super(PrimaryCapsule, self).__init__(**kwargs)
        self.num_channels = num_channels
        self.dim_capsules = dim_capsules
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

    def build(self, input_shape):
        total_filters = self.num_channels * self.dim_capsules
        self.conv = layers.Conv2D(
            filters=total_filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            name="primary_conv"
        )
        super(PrimaryCapsule, self).build(input_shape)

    def call(self, inputs):
        # 1. Apply Convolution: (B, H, W, num_channels * dim_capsules)
        x = self.conv(inputs)
        # 2. Reshape to vector capsules: (B, N_primary, dim_capsules)
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, (batch_size, -1, self.dim_capsules))
        # 3. Apply Squash non-linear activation
        return squash(x)

    def get_config(self):
        config = super(PrimaryCapsule, self).get_config()
        config.update({
            "num_channels": self.num_channels,
            "dim_capsules": self.dim_capsules,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="src.attention")
class AttentionRoutingCapsule(layers.Layer):
    """Attention-based Routing Capsule Layer.

    Replaces computationally intensive iterative dynamic routing with an attention mechanism
    that calculates routing weights (coupling coefficients) directly between primary capsules
    and high-level target capsules via query-key attention projection.

    Clinical Significance in Diabetic Retinopathy (DR):
        Traditional CNNs suffer from spatial invariance, losing relative positional context of ocular
        anatomical features. Capsule Networks preserve spatial hierarchies (e.g., the distance and orientation
        of hard exudates relative to the fovea/macula, crucial for detecting Diabetic Macular Edema).
        The attention routing mechanism efficiently dynamically routes salient feature vectors to target
        capsules representing DR severity stages with fast execution.

    Args:
        num_capsules (int): Number of output target capsules (e.g., 5 for DR grade 0 to 4). Default: 5.
        dim_capsules (int): Dimensionality of each output target capsule vector. Default: 16.
        **kwargs: Additional keyword arguments for tf.keras.layers.Layer.
    """

    def __init__(self, num_capsules=5, dim_capsules=16, **kwargs):
        super(AttentionRoutingCapsule, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules

    def build(self, input_shape):
        # input_shape: (B, input_num_capsules, input_dim_capsules)
        self.input_num_capsules = input_shape[1]
        self.input_dim_capsules = input_shape[2]

        if self.input_num_capsules is None or self.input_dim_capsules is None:
            raise ValueError("Input capsule dimensions must be fully defined.")

        # Transformation weight tensor W: [num_capsules, input_num_capsules, dim_capsules, input_dim_capsules]
        self.W = self.add_weight(
            shape=(self.num_capsules, self.input_num_capsules, self.dim_capsules, self.input_dim_capsules),
            initializer="glorot_uniform",
            name="W"
        )

        # Query vector key for attention routing per target capsule: [num_capsules, dim_capsules]
        self.query_vectors = self.add_weight(
            shape=(self.num_capsules, self.dim_capsules),
            initializer="glorot_uniform",
            name="query_vectors"
        )
        super(AttentionRoutingCapsule, self).build(input_shape)

    def call(self, inputs):
        # inputs: (B, N_in, D_in)
        # Compute votes u_hat: (B, N_out, N_in, D_out)
        # u_hat_j|i = W_ji * u_i
        # Using einsum for vectorized execution:
        # 'o i d_out d_in', 'b i d_in' -> 'b o i d_out'
        u_hat = tf.einsum("oidn,bin->boid", self.W, inputs)

        # Attention Routing Mechanism:
        # Calculate alignment between predicted votes u_hat (b, o, i, d) and learnable query vector (o, d)
        attention_scores = tf.einsum("boid,od->boi", u_hat, self.query_vectors)

        # Softmax over input capsules (dim i) to compute coupling coefficients c_ij
        coupling_coeffs = tf.nn.softmax(attention_scores, axis=-1)  # (b, o, i)

        # Weighted aggregation s_j = sum_i (c_ij * u_hat_j|i) -> (b, o, d)
        s = tf.einsum("boi,boid->bod", coupling_coeffs, u_hat)

        # Squash activation to get output capsule vectors v: (b, o, d)
        v = squash(s, axis=-1)

        # Compute capsule vector magnitudes (probabilities) for classification output: (b, o)
        v_probs = tf.sqrt(tf.reduce_sum(tf.square(v), axis=-1) + K.epsilon())

        return v, v_probs

    def get_config(self):
        config = super(AttentionRoutingCapsule, self).get_config()
        config.update({
            "num_capsules": self.num_capsules,
            "dim_capsules": self.dim_capsules,
        })
        return config
