"""
v2.0.0 - 2026-06-01 - Augmentation + fine-tuning das últimas camadas do MobileNetV2
v1.0.0 - 2026-06-01 - Modelo base com backbone congelado
"""
import tensorflow as tf
from tensorflow import keras

# Camadas do MobileNetV2 a descongelar no fine-tuning (últimas 30)
FINE_TUNE_AT = 124


def _bloco_augmentation(img_size: int) -> keras.Sequential:
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.1),
        keras.layers.RandomBrightness(0.1),
    ], name="augmentation")


def construir_modelo(n_tabular: int, n_classes: int = 5, img_size: int = 224) -> keras.Model:
    entrada_img = keras.Input(shape=(img_size, img_size, 3), name="imagem")
    entrada_tab = keras.Input(shape=(n_tabular,), name="tabular")

    # Augmentation (ativo só no treino)
    x_img = _bloco_augmentation(img_size)(entrada_img)

    # Backbone MobileNetV2
    backbone = keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = True
    for layer in backbone.layers[:FINE_TUNE_AT]:
        layer.trainable = False

    x_img = backbone(x_img, training=False)
    x_img = keras.layers.GlobalAveragePooling2D()(x_img)
    x_img = keras.layers.Dense(128, activation="relu")(x_img)
    x_img = keras.layers.Dropout(0.3)(x_img)

    # Ramo tabular
    x_tab = keras.layers.Dense(64, activation="relu")(entrada_tab)
    x_tab = keras.layers.BatchNormalization()(x_tab)
    x_tab = keras.layers.Dense(32, activation="relu")(x_tab)

    # Fusão
    fusao = keras.layers.Concatenate()([x_img, x_tab])
    x = keras.layers.Dense(64, activation="relu")(fusao)
    x = keras.layers.Dropout(0.3)(x)
    saida = keras.layers.Dense(n_classes, activation="softmax", name="saida")(x)

    return keras.Model(
        inputs={"imagem": entrada_img, "tabular": entrada_tab},
        outputs=saida,
        name="petfinder_multimodal_v2",
    )
