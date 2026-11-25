from typing import Tuple

import keras
from keras import layers
import tensorflow as tf


def dice_coef(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    """Computes differentiable Dice Coefficient (pixel-level F1 score)."""
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Computes Dice Loss: 1 - Dice Coefficient."""
    return 1.0 - dice_coef(y_true, y_pred)


def iou_metric(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    """Computes Intersection over Union (Jaccard Index) for binary crack masks."""
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred > 0.5, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def combined_bce_dice_loss(bce_weight: float = 0.5, dice_weight: float = 0.5):
    """
    Returns combined loss:
      L_total = bce_weight * BCE + dice_weight * DiceLoss
    Combines pixel cross-entropy with overlap penalty for thin crack boundaries.
    """
    bce = keras.losses.BinaryCrossentropy(from_logits=False)

    def loss(y_true, y_pred):
        bce_val = bce(y_true, y_pred)
        dice_val = dice_loss(y_true, y_pred)
        return bce_weight * bce_val + dice_weight * dice_val

    loss.__name__ = "combined_bce_dice_loss"
    return loss


def _conv_block(inputs: keras.KerasTensor, filters: int, name: str) -> keras.KerasTensor:
    """Double 3x3 convolution block with BatchNormalization and ReLU activation."""
    x = layers.Conv2D(
        filters,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal",
        name=f"{name}_conv1"
    )(inputs)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_relu1")(x)

    x = layers.Conv2D(
        filters,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal",
        name=f"{name}_conv2"
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.Activation("relu", name=f"{name}_relu2")(x)
    return x


def build_unet(
    input_shape: Tuple[int, int, int] = (256, 256, 3),
    initial_filters: int = 16,
    dropout_rate: float = 0.2,
    learning_rate: float = 1e-3
) -> keras.Model:
    """
    Builds a high-efficiency U-Net architecture for concrete crack segmentation.
    Designed for fast CPU convergence and sharp pixel-level boundary delineation.

    Args:
        input_shape: Input image dimensions (H, W, C), default (256, 256, 3).
        initial_filters: Base number of filters (default: 16).
        dropout_rate: Dropout rate in the bottleneck.
        learning_rate: Adam optimizer learning rate.

    Returns:
        Compiled Keras Model with Combined BCE + Dice loss and segmentation metrics.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # ENCODER (Contracting Path)
    # Level 1: 256x256 -> 128x128
    c1 = _conv_block(inputs, initial_filters, "enc1")
    p1 = layers.MaxPooling2D((2, 2), name="enc1_pool")(c1)

    # Level 2: 128x128 -> 64x64
    c2 = _conv_block(p1, initial_filters * 2, "enc2")
    p2 = layers.MaxPooling2D((2, 2), name="enc2_pool")(c2)

    # Level 3: 64x64 -> 32x32
    c3 = _conv_block(p2, initial_filters * 4, "enc3")
    p3 = layers.MaxPooling2D((2, 2), name="enc3_pool")(c3)

    # BOTTLENECK: 32x32
    bn = _conv_block(p3, initial_filters * 8, "bottleneck")
    if dropout_rate > 0:
        bn = layers.Dropout(dropout_rate, name="bottleneck_dropout")(bn)

    # DECODER (Expansive Path with Skip Connections)
    # Level 3 Up: 32x32 -> 64x64
    u3 = layers.Conv2DTranspose(
        initial_filters * 4, (2, 2), strides=(2, 2), padding="same", name="dec3_up"
    )(bn)
    u3 = layers.concatenate([u3, c3], name="dec3_concat")
    d3 = _conv_block(u3, initial_filters * 4, "dec3")

    # Level 2 Up: 64x64 -> 128x128
    u2 = layers.Conv2DTranspose(
        initial_filters * 2, (2, 2), strides=(2, 2), padding="same", name="dec2_up"
    )(d3)
    u2 = layers.concatenate([u2, c2], name="dec2_concat")
    d2 = _conv_block(u2, initial_filters * 2, "dec2")

    # Level 1 Up: 128x128 -> 256x256
    u1 = layers.Conv2DTranspose(
        initial_filters, (2, 2), strides=(2, 2), padding="same", name="dec1_up"
    )(d2)
    u1 = layers.concatenate([u1, c1], name="dec1_concat")
    d1 = _conv_block(u1, initial_filters, "dec1")

    # OUTPUT: 1x1 Conv with Sigmoid activation (Crack Probability Map)
    outputs = layers.Conv2D(
        1, (1, 1), activation="sigmoid", name="crack_mask"
    )(d1)

    model = keras.Model(inputs=inputs, outputs=outputs, name="InfraScan_UNet")

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=combined_bce_dice_loss(bce_weight=0.5, dice_weight=0.5),
        metrics=[
            dice_coef,
            iou_metric,
            keras.metrics.BinaryAccuracy(name="pixel_accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall")
        ]
    )

    return model
