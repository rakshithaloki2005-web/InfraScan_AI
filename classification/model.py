
from typing import Tuple

import keras
from keras import layers, regularizers
from keras.applications import MobileNetV2


def build_crack_classifier(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    freeze_base: bool = True,
    dropout_rate: float = 0.3,
    l2_reg: float = 1e-4,
    learning_rate: float = 1e-3
) -> keras.Model:
    """
    Constructs a transfer learning binary classification model for crack detection.

    Architecture:
      1. Input: (224, 224, 3), values normalized in [0, 1].
      2. Backbone: MobileNetV2 (ImageNet pre-trained weights, frozen or partially fine-tuned).
      3. GlobalAveragePooling2D.
      4. BatchNormalization & Dropout for regularization.
      5. Fully-connected Dense layer (128 units, ReLU activation, L2 regularization).
      6. Output Dense layer (1 unit, Sigmoid activation) outputting P(Cracked).

    Args:
        input_shape: Input image dimensions (H, W, C).
        freeze_base: Whether to freeze feature extraction layers of the backbone.
        dropout_rate: Dropout rate to prevent overfitting.
        l2_reg: L2 kernel regularization coefficient.
        learning_rate: Initial learning rate for Adam optimizer.

    Returns:
        Compiled Keras Model.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

    # MobileNetV2 expects [-1, 1] input range; we normalize from [0, 1]
    # x = (inputs * 2.0) - 1.0
    rescaled = layers.Rescaling(scale=2.0, offset=-1.0, name="mobilenet_rescaling")(inputs)

    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_tensor=rescaled
    )
    base_model.trainable = not freeze_base

    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="head_bn1")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout1")(x)

    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="dense_features"
    )(x)
    x = layers.BatchNormalization(name="head_bn2")(x)
    x = layers.Dropout(dropout_rate * 0.7, name="head_dropout2")(x)

    outputs = layers.Dense(
        1,
        activation="sigmoid",
        name="crack_probability"
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="InfraScan_CrackClassifier")

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc")
        ]
    )

    return model

