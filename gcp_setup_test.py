"""
v1.0.0 - 2026-06-01 - Valida conexão com GCP: Vertex AI + Cloud Storage + BigQuery
"""
import logging
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "credentials.json")

import vertexai
from google.cloud import storage, bigquery

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT = os.environ.get("GCP_PROJECT")
REGION  = os.environ.get("GCP_REGION")
BUCKET  = os.environ.get("GCP_BUCKET")


def testar_vertex() -> None:
    vertexai.init(project=PROJECT, location=REGION)
    logger.info(f"Vertex AI OK — project={PROJECT}, region={REGION}")


def testar_storage() -> None:
    client = storage.Client(project=PROJECT)
    bucket_name = BUCKET.replace("gs://", "")
    bucket = client.get_bucket(bucket_name)
    logger.info(f"Cloud Storage OK — bucket={bucket.name}, location={bucket.location}")


def testar_bigquery() -> None:
    client = bigquery.Client(project=PROJECT)
    datasets = list(client.list_datasets())
    logger.info(f"BigQuery OK — {len(datasets)} dataset(s) no projeto")


if __name__ == "__main__":
    logger.info("=== Testando conexão GCP ===")
    testar_vertex()
    testar_storage()
    testar_bigquery()
    logger.info("=== Tudo OK — ambiente pronto para Vertex AI ===")
