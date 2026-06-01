"""
v2.0.0 - 2026-06-01 - 14k dados + augmentation + fine-tuning
v1.0.0 - 2026-06-01 - Treino base com 1k registros
"""
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras

from models.multimodal_model import construir_modelo

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/processed")
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 5

COLS_TABULAR = [
    "Type", "Age", "Breed1", "Breed2", "Gender",
    "Color1", "Color2", "Color3", "MaturitySize", "FurLength",
    "Vaccinated", "Dewormed", "Sterilized", "Health",
    "Quantity", "Fee", "State", "VideoAmt", "PhotoAmt",
]


def carregar_imagem(path: str) -> np.ndarray:
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = keras.applications.mobilenet_v2.preprocess_input(img)
    return img


def criar_dataset(df: pd.DataFrame, shuffle: bool = False) -> tf.data.Dataset:
    imgs = df["image_path"].astype(str).tolist()
    tabs = df[COLS_TABULAR].values.astype("float32")
    labels = df["AdoptionSpeed"].values.astype("int32")

    def carregar(path, tab, label):
        img = tf.py_function(
            lambda p: carregar_imagem(p.numpy().decode()),
            [path], tf.float32
        )
        img.set_shape([IMG_SIZE, IMG_SIZE, 3])
        return {"imagem": img, "tabular": tab}, label

    ds = tf.data.Dataset.from_tensor_slices((imgs, tabs, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=42)
    ds = ds.map(carregar, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


if __name__ == "__main__":
    logger.info("=== Fase 3: Treino Multimodal ===")

    df = pd.read_csv(DATA_DIR / "train_subset.csv")
    df = df[df["image_path"].notna() & (df["image_path"] != "")].reset_index(drop=True)
    logger.info(f"Registros com imagem: {len(df)}")

    df_train, df_val = train_test_split(df, test_size=0.2, stratify=df["AdoptionSpeed"], random_state=42)
    logger.info(f"Treino: {len(df_train)} | Validação: {len(df_val)}")

    ds_train = criar_dataset(df_train, shuffle=True)
    ds_val = criar_dataset(df_val)

    modelo = construir_modelo(n_tabular=len(COLS_TABULAR))
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    modelo.summary(print_fn=logger.info)

    logger.info("Iniciando treino...")
    history = modelo.fit(
        ds_train,
        validation_data=ds_val,
        epochs=EPOCHS,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True),
        ],
    )

    val_acc = max(history.history["val_accuracy"])
    logger.info(f"Melhor val_accuracy: {val_acc:.4f}")

    out_path = Path("models/petfinder_multimodal_portfolio.keras")
    modelo.save(out_path)
    logger.info(f"Modelo salvo em: {out_path}")
    logger.info("=== Concluído ===")
