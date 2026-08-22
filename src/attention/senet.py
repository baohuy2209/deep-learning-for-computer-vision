import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="src.attention")
class SEBlock(layers.Layer):
    """Squeeze-and-Excitation (SE) Block for Channel-wise Attention.

    This block recalibrates feature channels dynamically by explicitly
    modeling interdependencies between channels.

    Clinical Significance in Diabetic Retinopathy (DR):
        Different feature channels in a CNN activate for distinct pathology patterns.
        For instance, red lesions (microaneurysms, hemorrhages) and bright lesions
        (hard/soft exudates, cotton wool spots) present contrasting spectral and structural
        characteristics. The SE mechanism allows the network to adaptively boost channels
        sensitive to subtler DR lesions while suppressing irrelevant background retinal tissue.

    Args:
        reduction_ratio (int): Reduction factor for the channel bottleneck. Default: 16.
        **kwargs: Additional keyword arguments for tf.keras.layers.Layer.
    """

    def __init__(self, reduction_ratio=16, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        if reduction_ratio <= 0:
            raise ValueError(f"reduction_ratio must be positive, got {reduction_ratio}")
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError("Channel dimension of input tensor must be defined.")

        reduced_channels = max(1, channels // self.reduction_ratio)

        self.global_avg_pool = layers.GlobalAveragePooling2D(keepdims=True)
        self.fc1 = layers.Dense(
            units=reduced_channels,
            activation="relu",
            use_bias=True,
            name="squeeze"
        )
        self.fc2 = layers.Dense(
            units=channels,
            activation="sigmoid",
            use_bias=True,
            name="excitation"
        )
        self.multiply = layers.Multiply()
        super(SEBlock, self).build(input_shape)

    def call(self, inputs):
        # Squeeze phase: Global Average Pooling (B, H, W, C) -> (B, 1, 1, C)
        squeeze = self.global_avg_pool(inputs)

        # Excitation phase: Bottleneck MLP (B, 1, 1, C) -> (B, 1, 1, C)
        excitation = self.fc1(squeeze)
        excitation = self.fc2(excitation)

        # Scale phase: Element-wise channel multiplication
        return self.multiply([inputs, excitation])

    def get_config(self):
        config = super(SEBlock, self).get_config()
        config.update({"reduction_ratio": self.reduction_ratio})
        return config
