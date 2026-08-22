import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="src.attention")
class MSAGBlock(layers.Layer):
    """Multi-Scale Spatial Attention Gate (MSAG) Block.

    Captures multi-scale contextual features using parallel receptive fields
    (1x1, 3x3, and dilated 3x3 convolutions) to construct a spatial attention
    map that highlights lesion locations across various scales.

    Clinical Significance in Diabetic Retinopathy (DR):
        DR lesions exhibit extreme spatial variance in retinal images. Tiny microaneurysms
        occupy only a few pixels (small receptive field required), whereas large hemorrhages
        and soft exudates (cotton wool spots) span wider regions (larger/dilated receptive fields).
        The MSAG block allows the model to simultaneously focus on subtle dot hemorrhages and
        larger patch-like lesions while preserving fine-grained spatial topography.

    Args:
        filters_per_scale (int): Number of filters per multi-scale convolutional branch. Default: 32.
        dilation_rate (int): Dilation rate for the dilated 3x3 convolution branch. Default: 2.
        **kwargs: Additional keyword arguments for tf.keras.layers.Layer.
    """

    def __init__(self, filters_per_scale=32, dilation_rate=2, **kwargs):
        super(MSAGBlock, self).__init__(**kwargs)
        self.filters_per_scale = filters_per_scale
        self.dilation_rate = dilation_rate

    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError("Channel dimension of input tensor must be defined.")

        # Multi-scale feature branches
        self.branch_1x1 = layers.Conv2D(
            filters=self.filters_per_scale,
            kernel_size=(1, 1),
            padding="same",
            activation="relu",
            name="branch_1x1"
        )
        self.branch_3x3 = layers.Conv2D(
            filters=self.filters_per_scale,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
            name="branch_3x3"
        )
        self.branch_dilated = layers.Conv2D(
            filters=self.filters_per_scale,
            kernel_size=(3, 3),
            dilation_rate=self.dilation_rate,
            padding="same",
            activation="relu",
            name="branch_dilated"
        )

        self.concat = layers.Concatenate(axis=-1)

        # Spatial attention map generator (collapses multi-scale channels into 1 spatial map)
        self.spatial_conv = layers.Conv2D(
            filters=1,
            kernel_size=(3, 3),
            padding="same",
            activation="sigmoid",
            name="spatial_attention_map"
        )
        self.multiply = layers.Multiply()
        self.add = layers.Add()

        super(MSAGBlock, self).build(input_shape)

    def call(self, inputs):
        # 1. Multi-scale feature extraction
        f_1x1 = self.branch_1x1(inputs)
        f_3x3 = self.branch_3x3(inputs)
        f_dilated = self.branch_dilated(inputs)

        # 2. Concatenate multi-scale representations
        ms_features = self.concat([f_1x1, f_3x3, f_dilated])

        # 3. Compute 2D spatial attention weight map (B, H, W, 1)
        attention_map = self.spatial_conv(ms_features)

        # 4. Modulate input feature map spatially
        attended_inputs = self.multiply([inputs, attention_map])

        # 5. Residual shortcut connection
        return self.add([inputs, attended_inputs])

    def get_config(self):
        config = super(MSAGBlock, self).get_config()
        config.update({
            "filters_per_scale": self.filters_per_scale,
            "dilation_rate": self.dilation_rate,
        })
        return config
