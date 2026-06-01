"""
v0.2.0 - 2026-06-01 - Download via Kaggle API e inspeção do PetFinder (subconjunto Fase 1)
v0.1.0 - 2026-06-01 - Versão inicial com download direto S3
"""
import logging
import zipfile
import subprocess
from pathlib import Path
import pandas as pd
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent
EXTRACT_DIR = DATA_DIR / "petfinder"
SUBSET_SIZE = 1000
KAGGLE_DATASET = "gl2024/petfinder-adoption-prediction"
ZIP_NAME = "petfinder-adoption-prediction.zip"


def download() -> None:
    zip_path = DATA_DIR / ZIP_NAME
    if zip_path.exists():
        logger.info("ZIP já existe — pulando download")
        return

    logger.info(f"Baixando dataset '{KAGGLE_DATASET}' via Kaggle API...")
    result = subprocess.run(
        ["kaggle", "datasets", "download", KAGGLE_DATASET, "-p", str(DATA_DIR)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Kaggle download falhou: {result.stderr}")
    logger.info("Download concluído")


def extract() -> None:
    zip_path = DATA_DIR / ZIP_NAME
    if EXTRACT_DIR.exists() and any(EXTRACT_DIR.iterdir()):
        logger.info("Dados já extraídos — pulando extração")
        return

    logger.info(f"Extraindo para {EXTRACT_DIR}")
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(EXTRACT_DIR)
    logger.info("Extração concluída")


def inspecionar(n: int = SUBSET_SIZE) -> pd.DataFrame:
    csv_path = next(EXTRACT_DIR.rglob("train.csv"))
    logger.info(f"CSV encontrado: {csv_path}")

    df = pd.read_csv(csv_path).head(n)
    logger.info(f"Subconjunto: {df.shape[0]} linhas x {df.shape[1]} colunas")
    logger.info(f"Colunas: {list(df.columns)}")
    logger.info(f"Nulos por coluna:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    img_dir = next(EXTRACT_DIR.rglob("train_images"), None)
    if img_dir:
        jpgs = [p for p in img_dir.glob("*.jpg") if not p.name.startswith("._")]
        logger.info(f"Imagens disponíveis: {len(jpgs)} .jpg")
        for img_path in jpgs[:3]:
            with Image.open(img_path) as img:
                logger.info(f"  {img_path.name} — {img.size} {img.mode}")

    return df


if __name__ == "__main__":
    logger.info("=== Fase 1: PetFinder — Download e Inspeção ===")
    download()
    extract()
    df = inspecionar()
    print(df.head(3).to_string())
    logger.info("=== Concluído ===")
