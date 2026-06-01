"""
v1.1.0 - 2026-06-01 - Escala subset para 14k registros
v1.0.0 - 2026-06-01 - Pré-processamento tabular e vínculo imagem↔registro (Fase 2)
"""
import logging
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent
RAW_DIR = DATA_DIR / "petfinder"
OUT_DIR = DATA_DIR / "processed"
IMG_DIR = RAW_DIR / "cat_images" / "train_images"
SUBSET_SIZE = 14000

COLS_CATEGORICAS = [
    "Type", "Gender", "Color1", "Color2", "Color3",
    "MaturitySize", "FurLength", "Vaccinated",
    "Dewormed", "Sterilized", "Health", "State",
]
COLS_NUMERICAS = ["Age", "Quantity", "Fee", "VideoAmt", "PhotoAmt"]
COL_TARGET = "AdoptionSpeed"
COL_ID = "PetID"


def carregar(n: int = SUBSET_SIZE) -> pd.DataFrame:
    csv_path = RAW_DIR / "train.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"train.csv não encontrado em {RAW_DIR}")
    df = pd.read_csv(csv_path).head(n)
    logger.info(f"Carregado: {df.shape[0]} linhas x {df.shape[1]} colunas")
    return df


def tratar_nulos(df: pd.DataFrame) -> pd.DataFrame:
    df["Name"] = df["Name"].fillna("Unknown")
    df["Description"] = df["Description"].fillna("")
    logger.info("Nulos tratados: Name → 'Unknown', Description → ''")
    return df


def vincular_imagens(df: pd.DataFrame) -> pd.DataFrame:
    """Lê o diretório uma única vez e faz lookup O(1) por PetID."""
    logger.info("Indexando diretório de imagens (leitura única)...")
    indice: dict[str, str] = {}
    for p in IMG_DIR.glob("*.jpg"):
        if p.name.startswith("._"):
            continue
        pet_id = p.stem.rsplit("-", 1)[0]
        if pet_id not in indice:
            indice[pet_id] = str(p)
    logger.info(f"Índice criado: {len(indice)} pets com imagem")

    df["image_path"] = df[COL_ID].map(indice).fillna("")
    com_imagem = (df["image_path"] != "").sum()
    logger.info(f"Imagens vinculadas: {com_imagem}/{len(df)} registros com foto")
    return df


def codificar_categoricas(df: pd.DataFrame) -> pd.DataFrame:
    for col in COLS_CATEGORICAS:
        df[col] = df[col].astype("category").cat.codes
    logger.info(f"Categóricas codificadas: {COLS_CATEGORICAS}")
    return df


def escalar_numericas(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    df[COLS_NUMERICAS] = scaler.fit_transform(df[COLS_NUMERICAS])
    logger.info(f"Numéricas normalizadas (StandardScaler): {COLS_NUMERICAS}")
    return df, scaler


def salvar(df: pd.DataFrame) -> Path:
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "train_14k.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Salvo em: {out_path} ({len(df)} linhas)")
    return out_path


if __name__ == "__main__":
    logger.info("=== Fase 2: Pré-processamento ===")

    df = carregar()
    df = tratar_nulos(df)
    df = vincular_imagens(df)
    df = codificar_categoricas(df)
    df, scaler = escalar_numericas(df)

    # Remove colunas não usadas no modelo
    df = df.drop(columns=["RescuerID"], errors="ignore")

    salvar(df)

    logger.info(f"Colunas finais: {list(df.columns)}")
    logger.info(f"Registros sem imagem: {(df['image_path'] == '').sum()}")
    logger.info(f"Distribuição do target (AdoptionSpeed):\n{df[COL_TARGET].value_counts().sort_index()}")
    logger.info("=== Concluído ===")
