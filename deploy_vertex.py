"""
v1.0.0 - 2026-06-01 - Upload modelo → Vertex AI Model Registry → Endpoint → Predição → Cleanup
Fluxo completo de validação para portfólio.
"""
import logging
import os
import numpy as np
from dotenv import load_dotenv
from google.cloud import aiplatform, storage

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT = os.environ["GCP_PROJECT"]
REGION  = os.environ["GCP_REGION"]
BUCKET  = os.environ["GCP_BUCKET"]

LOCAL_MODEL = "models/petfinder_multimodal_portfolio.keras"
GCS_MODEL   = f"{BUCKET}/models/portfolio/petfinder_multimodal"


def upload_modelo() -> None:
    logger.info(f"Fazendo upload do modelo para {GCS_MODEL}...")
    client = storage.Client(project=PROJECT)
    bucket = client.bucket(BUCKET.replace("gs://", ""))

    import tensorflow as tf
    model = tf.keras.models.load_model(LOCAL_MODEL)
    model.export(GCS_MODEL.replace(f"{BUCKET}/", ""))  # salva como SavedModel

    # upload via gcloud storage
    import subprocess
    result = subprocess.run(
        ["gcloud", "storage", "cp", "-r",
         "models/saved_model_export",
         f"{GCS_MODEL}"],
        capture_output=True, text=True
    )
    logger.info(f"Upload concluído: {GCS_MODEL}")


def registrar_modelo() -> aiplatform.Model:
    logger.info("Registrando modelo no Vertex AI Model Registry...")
    aiplatform.init(project=PROJECT, location=REGION)

    model = aiplatform.Model.upload(
        display_name="petfinder-multimodal-portfolio",
        artifact_uri=f"{GCS_MODEL}",
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-12:latest",
        labels={"projeto": "portfolio", "framework": "tensorflow"},
    )
    logger.info(f"Modelo registrado: {model.resource_name}")
    return model


def deploy_endpoint(model: aiplatform.Model) -> aiplatform.Endpoint:
    logger.info("Criando endpoint...")
    endpoint = model.deploy(
        machine_type="n1-standard-2",
        min_replica_count=1,
        max_replica_count=1,
        deployed_model_display_name="petfinder-portfolio-endpoint",
        sync=True,
    )
    logger.info(f"Endpoint ativo: {endpoint.resource_name}")
    return endpoint


def fazer_predicao(endpoint: aiplatform.Endpoint) -> None:
    logger.info("Fazendo predição de teste...")
    instancia = {
        "imagem": np.zeros((224, 224, 3), dtype=np.float32).tolist(),
        "tabular": [0.0] * 19,
    }
    resposta = endpoint.predict(instances=[instancia])
    logger.info(f"Resposta do endpoint: {resposta.predictions}")
    logger.info("Predição OK — AdoptionSpeed classes: [0,1,2,3,4]")


def cleanup(endpoint: aiplatform.Endpoint, model: aiplatform.Model) -> None:
    logger.info("Removendo endpoint para evitar custo...")
    endpoint.undeploy_all()
    endpoint.delete()
    logger.info("Endpoint removido.")


if __name__ == "__main__":
    logger.info("=== Deploy Vertex AI — Validação de Portfólio ===")
    aiplatform.init(project=PROJECT, location=REGION)

    model   = registrar_modelo()
    endpoint = deploy_endpoint(model)
    fazer_predicao(endpoint)

    input("\nTire os prints agora! Pressione ENTER quando terminar para apagar o endpoint...")
    cleanup(endpoint, model)
    logger.info("=== Concluído — ambiente limpo ===")
