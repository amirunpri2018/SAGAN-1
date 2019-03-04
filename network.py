import tensorflow as tf
import numpy as np
from ops import *


class SAGAN(object):

    def __init__(self, min_resolution, max_resolution, min_channels, max_channels):

        self.min_resolution = np.asanyarray(min_resolution)
        self.max_resolution = np.asanyarray(max_resolution)
        self.min_channels = min_channels
        self.max_channels = max_channels

        def log2(x): return 0 if (x == 1).all() else 1 + log2(x >> 1)

        self.min_depth = log2(self.min_resolution // self.min_resolution)
        self.max_depth = log2(self.max_resolution // self.min_resolution)

    def generator(self, latents, labels, training, name="ganerator", reuse=None):

        def resolution(depth): return self.min_resolution << depth

        def channels(depth): return min(self.max_channels, self.min_channels << (self.max_depth - depth))

        def residual_block(inputs, depth):
            ''' A single block for ResNet v2, without a bottleneck.
                Batch normalization then ReLu then convolution as described by:
                [Identity Mappings in Deep Residual Networks]
                (https://arxiv.org/pdf/1603.05027.pdf)
            '''
            with tf.variable_scope("conditional_batch_norm_1st"):
                inputs = conditional_batch_norm(
                    inputs=inputs,
                    labels=labels,
                    training=training,
                    center_initializer=tf.initializers.zeros(),
                    scale_initializer=tf.initializers.ones(),
                    apply_spectral_norm=True
                )
            inputs = tf.nn.relu(inputs)
            # projection shortcut should come after batch norm and relu
            # since it performs a 1x1 convolution
            with tf.variable_scope("projection_shortcut"):
                shortcut = upscale2d(inputs)
                shortcut = conv2d(
                    inputs=shortcut,
                    filters=channels(depth),
                    kernel_size=[1, 1],
                    use_bias=False,
                    weight_initializer=tf.initializers.he_normal(),
                    bias_initializer=tf.initializers.zeros(),
                    apply_spectral_norm=True
                )
            with tf.variable_scope("conv_1st"):
                inputs = upscale2d(inputs)
                inputs = conv2d(
                    inputs=inputs,
                    filters=channels(depth),
                    kernel_size=[3, 3],
                    use_bias=False,
                    weight_initializer=tf.initializers.he_normal(),
                    bias_initializer=tf.initializers.zeros(),
                    apply_spectral_norm=True
                )
            with tf.variable_scope("conditional_batch_norm_2nd"):
                inputs = conditional_batch_norm(
                    inputs=inputs,
                    labels=labels,
                    training=training,
                    center_initializer=tf.initializers.zeros(),
                    scale_initializer=tf.initializers.ones(),
                    apply_spectral_norm=True
                )
            inputs = tf.nn.relu(inputs)
            with tf.variable_scope("conv_2nd"):
                inputs = conv2d(
                    inputs=inputs,
                    filters=channels(depth),
                    kernel_size=[3, 3],
                    use_bias=False,
                    weight_initializer=tf.initializers.he_normal(),
                    bias_initializer=tf.initializers.zeros(),
                    apply_spectral_norm=True
                )
            inputs += shortcut
            return inputs

        with tf.variable_scope(name, reuse=reuse):

            with tf.variable_scope("dense"):
                inputs = dense(
                    inputs=latents,
                    units=channels(self.min_depth) * resolution(self.min_depth).prod(),
                    use_bias=False,
                    weight_initializer=tf.initializers.he_normal(),
                    bias_initializer=tf.initializers.zeros(),
                    apply_spectral_norm=True
                )
            inputs = tf.reshape(
                tensor=inputs,
                shape=[-1, channels(self.min_depth), *resolution(self.min_depth)]
            )

            for depth in range(self.min_depth + 1, (self.min_depth + self.max_depth) // 2):
                with tf.variable_scope("residual_block_{}x{}".format(*resolution(depth))):
                    inputs = residual_block(inputs, depth)

            with tf.variable_scope("self_attention"):
                inputs = self_attention(
                    inputs=inputs,
                    filters=inputs.shape[1] // 8,
                    weight_initializer=tf.initializers.glorot_normal,
                    apply_spectral_norm=True
                )

            for depth in range((self.min_depth + self.max_depth) // 2, self.max_depth + 1):
                with tf.variable_scope("residual_block_{}x{}".format(*resolution(depth))):
                    inputs = residual_block(inputs, depth)

            # standard batch norm
            with tf.variable_scope("batch_norm"):
                inputs = batch_norm(
                    inputs=inputs,
                    training=training
                )
            inputs = tf.nn.relu(inputs)
            with tf.variable_scope("conv"):
                inputs = conv2d(
                    inputs=inputs,
                    filters=3,
                    kernel_size=[1, 1],
                    use_bias=True,
                    weight_initializer=tf.initializers.glorot_normal(),
                    bias_initializer=tf.initializers.zeros(),
                    apply_spectral_norm=True
                )
            inputs = tf.nn.tanh(inputs)
            return inputs

    def discriminator(self, images, labels, training, name="dicriminator", reuse=None):

        def resolution(depth): return self.min_resolution << depth

        def channels(depth): return min(self.max_channels, self.min_channels << (self.max_depth - depth))

        def residual_block(inputs, depth):
            inputs = tf.nn.relu(inputs)
            # projection shortcut should come after batch norm and relu
            # since it performs a 1x1 convolution
            with tf.variable_scope("projection_shortcut"):
                shortcut = conv2d(
                    inputs=inputs,
                    filters=channels(depth - 1),
                    kernel_size=[1, 1],
                    use_bias=False,
                    weight_initializer=tf.initializers.he_normal(),
                    bias_initializer=tf.initializers.zeros(),
                    apply_spectral_norm=True
                )
                shortcut = downscale2d(shortcut)
            with tf.variable_scope("conv_1st"):
                inputs = conv2d(
                    inputs=inputs,
                    filters=channels(depth),
                    kernel_size=[3, 3],
                    use_bias=True,
                    weight_initializer=tf.initializers.he_normal(),
                    bias_initializer=tf.initializers.zeros(),
                    apply_spectral_norm=True
                )
            inputs = tf.nn.relu(inputs)
            with tf.variable_scope("conv_2nd"):
                inputs = conv2d(
                    inputs=inputs,
                    filters=channels(depth - 1),
                    kernel_size=[3, 3],
                    use_bias=True,
                    weight_initializer=tf.initializers.he_normal(),
                    bias_initializer=tf.initializers.zeros(),
                    apply_spectral_norm=True
                )
                inputs = downscale2d(inputs)
            inputs += shortcut
            return inputs

        with tf.variable_scope(name, reuse=reuse):

            with tf.variable_scope("conv"):
                inputs = conv2d(
                    inputs=images,
                    filters=channels(self.max_depth),
                    kernel_size=[1, 1],
                    use_bias=True,
                    weight_initializer=tf.initializers.he_normal(),
                    bias_initializer=tf.initializers.zeros(),
                    apply_spectral_norm=True
                )

            for depth in range(self.max_depth, (self.min_depth + self.max_depth) // 2, -1):
                with tf.variable_scope("residual_block_{}x{}".format(*resolution(depth))):
                    inputs = residual_block(inputs, depth)

            with tf.variable_scope("self_attention"):
                inputs = self_attention(
                    inputs=inputs,
                    filters=inputs.shape[1] // 8,
                    weight_initializer=tf.initializers.glorot_normal,
                    apply_spectral_norm=True
                )

            for depth in range((self.min_depth + self.max_depth) // 2, self.min_depth, -1):
                with tf.variable_scope("residual_block_{}x{}".format(*resolution(depth))):
                    inputs = residual_block(inputs, depth)

            inputs = tf.nn.relu(inputs)
            inputs = tf.reduce_sum(inputs, axis=[2, 3])

            with tf.variable_scope("logits"):
                logits = dense(
                    inputs=inputs,
                    units=1,
                    use_bias=True,
                    weight_initializer=tf.initializers.glorot_normal(),
                    bias_initializer=tf.initializers.zeros(),
                    apply_spectral_norm=True
                )
            with tf.variable_scope("projections"):
                embeddings = embed(
                    inputs=labels,
                    units=inputs.shape[1],
                    weight_initializer=tf.initializers.glorot_normal(),
                    apply_spectral_norm=True
                )
                projections = tf.reduce_sum(
                    input_tensor=inputs * embeddings,
                    axis=1,
                    keepdims=True
                )
            inputs = logits + projections

            return inputs
