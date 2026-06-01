"""
v1.0.0 - 2026-06-01 - Treino no Vertex AI Custom Training Job (GPU T4)
Lê dados do Cloud Storage, salva modelo de volta no bucket.
"""
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras

# Vertex AI injeta AIP_MODEL_DIR como destino do modelo salvo
MODEL_DIR = os.environ.get("AIP_MODEL_DIR", "gs://lucid-parsec-290001-mm-data/models")
DATA_CSV  = os.environ.get("DATA_CSV",  "gs://lucid-parsec-290001-mm-data/data/processed/train_14k.csv")
IMG_BASE  = os.environ.get("IMG_BASE",  "gs://lucid-parsec-290001-mm-data/data/images")

BATCH_SIZE = 32
EPOCHS     = 15
IMG_SIZE   = 224

COLS_TABULAR = [
    "Type", "Age", "Breed1", "Breed2", "Gender",
    "Color1", "Color2", "Color3", "MaturitySize", "FurLength",
    "Vaccinated", "Dewormed", "Sterilized", "Health",
    "Quantity", "Fee", "State", "VideoAmt", "PhotoAmt",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

FINE_TUNE_AT = 124


def construir_modelo(n_tabular: int, n_classes: int = 5) -> keras.Model:
    augmentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.1),
        keras.layers.RandomBrightness(0.1),
    ], name="augmentation")

    entrada_img = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="imagem")
    entrada_tab = keras.Input(shape=(n_tabular,), name="tabular")

    x_img = augmentation(entrada_img)
    backbone = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    backbone.trainable = True
    for layer in backbone.layers[:FINE_TUNE_AT]:
        layer.trainable = False

    x_img = backbone(x_img, training=False)
    x_img = keras.layers.GlobalAveragePooling2D()(x_img)
    x_img = keras.layers.Dense(128, activation="relu")(x_img)
    x_img = keras.layers.Dropout(0.3)(x_img)

    x_tab = keras.layers.Dense(64, activation="relu")(entrada_tab)
    x_tab = keras.layers.BatchNormalization()(x_tab)
    x_tab = keras.layers.Dense(32, activation="relu")(x_tab)

    fusao = keras.layers.Concatenate()([x_img, x_tab])
    x = keras.layers.Dense(64, activation="relu")(fusao)
    x = keras.layers.Dropout(0.3)(x)
    saida = keras.layers.Dense(n_classes, activation="softmax", name="saida")(x)

    return keras.Model(
        inputs={"imagem": entrada_img, "tabular": entrada_tab},
        outputs=saida,
        name="petfinder_multimodal_v2",
    )


def carregar_imagem(path: str) -> tf.Tensor:
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return keras.applications.mobilenet_v2.preprocess_input(img)


def criar_dataset(df: pd.DataFrame, shuffle: bool = False) -> tf.data.Dataset:
    # Converte caminhos locais para GCS
    imgs = [
        p.replace("\\", "/").replace(
            str(Path("data/petfinder/cat_images/train_images").as_posix()),
            IMG_BASE.rstrip("/"),
        ) if not p.startswith("gs://") else p
        for p in df["image_path"].astype(str).tolist()
    ]
    tabs   = df[COLS_TABULAR].values.astype("float32")
    labels = df["AdoptionSpeed"].values.astype("int32")

    def carregar(path, tab, label):
        img = tf.py_function(
            lambda p: carregar_imagem(p.numpy().decode()), [path], tf.float32
        )
        img.set_shape([IMG_SIZE, IMG_SIZE, 3])
        return {"imagem": img, "tabular": tab}, label

    ds = tf.data.Dataset.from_tensor_slices((imgs, tabs, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=42)
    return ds.map(carregar, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


if __name__ == "__main__":
    logger.info("=== Vertex AI Training Job ===")
    logger.info(f"TF version: {tf.__version__} | GPUs: {tf.config.list_physical_devices('GPU')}")
    logger.info(f"DATA_CSV={DATA_CSV} | MODEL_DIR={MODEL_DIR}")

    df = pd.read_csv(DATA_CSV)
    df = df[df["image_path"].notna() & (df["image_path"] != "")].reset_index(drop=True)
    logger.info(f"Registros: {len(df)}")

    df_train, df_val = train_test_split(
        df, test_size=0.2, stratify=df["AdoptionSpeed"], random_state=42
    )
    logger.info(f"Treino: {len(df_train)} | Validação: {len(df_val)}")

    modelo = construir_modelo(n_tabular=len(COLS_TABULAR))
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = modelo.fit(
        criar_dataset(df_train, shuffle=True),
        validation_data=criar_dataset(df_val),
        epochs=EPOCHS,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2),
        ],
    )

    val_acc = max(history.history["val_accuracy"])
    logger.info(f"Melhor val_accuracy: {val_acc:.4f}")

    modelo.save(MODEL_DIR)
    logger.info(f"Modelo salvo em: {MODEL_DIR}")
    logger.info("=== Concluído ===")
