"""
v1.0.0 - 2026-06-01 - Submete Custom Training Job no Vertex AI com GPU T4
"""
import logging
import os
from dotenv import load_dotenv
import vertexai
from google.cloud import aiplatform
from google.cloud.aiplatform import CustomTrainingJob

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT = os.environ["GCP_PROJECT"]
REGION  = os.environ["GCP_REGION"]
BUCKET  = os.environ["GCP_BUCKET"]

MODEL_DIR = f"{BUCKET}/models/petfinder_v2"
DATA_CSV  = f"{BUCKET}/data/processed/train_14k.csv"
IMG_BASE  = f"{BUCKET}/data/images"

if __name__ == "__main__":
    aiplatform.init(project=PROJECT, location=REGION, staging_bucket=BUCKET)
    logger.info(f"Submetendo job — project={PROJECT}, region={REGION}")

    job = CustomTrainingJob(
        display_name="petfinder-multimodal-v2",
        script_path="trainer/train_vertex.py",
        container_uri="us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-12.py310:latest",
        requirements=["scikit-learn", "pandas"],
        model_serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-12:latest",
    )

    model = job.run(
        replica_count=1,
        machine_type="n1-standard-4",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        environment_variables={
            "DATA_CSV": DATA_CSV,
            "IMG_BASE": IMG_BASE,
        },
        model_display_name="petfinder-multimodal-v2",
        base_output_dir=MODEL_DIR,
        sync=False,  # não bloqueia — job roda async
    )

    logger.info("Job submetido com sucesso (async)")
    logger.info(f"Acompanhe em: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT}")
