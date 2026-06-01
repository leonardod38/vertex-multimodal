# Multimodal ML Pipeline — Vertex AI + BigQuery

> Pipeline completa de Machine Learning multimodal (imagem + tabular) com deploy em produção no Google Cloud Vertex AI.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange?logo=tensorflow)
![Google Cloud](https://img.shields.io/badge/Google_Cloud-Vertex_AI-4285F4?logo=googlecloud)
![BigQuery](https://img.shields.io/badge/BigQuery-Data_Engineering-4285F4?logo=googlebigquery)
![Status](https://img.shields.io/badge/Status-Em_desenvolvimento-green)

---

## Objetivo

Demonstrar domínio de uma pipeline de ML completa — da ingestão de dados ao deploy em produção — utilizando o ecossistema Google Cloud, com foco em **Vertex AI** e **BigQuery**.

O modelo prevê a velocidade de adoção de animais (dataset PetFinder) combinando dados tabulares (idade, raça, saúde) com imagens dos pets em uma arquitetura multimodal.

---

## Stack Tecnológica

| Camada | Tecnologia |
|---|---|
| Modelagem | TensorFlow 2.x, MobileNetV2, Keras |
| Pipeline de dados | Python, Pandas, Scikit-learn |
| Download de dados | Kaggle API |
| Armazenamento | Google Cloud Storage |
| Registro de modelos | Vertex AI Model Registry |
| Inferência em produção | Vertex AI Endpoints |
| Versionamento | Git / GitHub |

---

## Arquitetura do Modelo

```
Entrada: Imagem (224x224x3)          Entrada: Features Tabulares (19)
         ↓                                    ↓
  [Data Augmentation]               [Dense(64) + BatchNorm]
  [MobileNetV2 - fine-tuning]       [Dense(32)]
  [GlobalAveragePooling]
  [Dense(128) + Dropout]
         ↓                                    ↓
              [Concatenate]
              [Dense(64) + Dropout]
              [Softmax(5 classes)]
                    ↓
         AdoptionSpeed: 0-4
```

---

## Pipeline Completa

```
1. Ingestão       → Kaggle API → 14.000 registros + 58k imagens
2. Pré-proc.      → Normalização, encoding, vínculo imagem↔tabular
3. Treino local   → MobileNetV2 + augmentation + fine-tuning
4. Cloud Storage  → Upload dataset + modelo para gs://bucket
5. Model Registry → Registro no Vertex AI com labels e versionamento
6. Endpoint       → Deploy em n1-standard-2, inferência online
7. Monitoramento  → Cloud Logging + métricas de inferência
```

---

## Evidências de Execução em Produção

### Cloud Storage — Dataset e Modelo na nuvem
![Cloud Storage](docs/screenshots/01_cloud_storage.png)

### Vertex AI Model Registry — Modelo registrado com versionamento
![Model Registry](docs/screenshots/02_model_registry.png)

### Vertex AI Endpoint — Deploy em produção
![Endpoint Criando](docs/screenshots/03_endpoint_criando.png)

### Vertex AI Endpoint — Detalhes e métricas de inferência
![Endpoint Detalhes](docs/screenshots/04_endpoint_detalhes.png)

### Cloud Logging — Observabilidade do endpoint
![Cloud Logging](docs/screenshots/05_cloud_logging.png)

---

## Estrutura do Repositório

```
vertex-multimodal/
├── data/
│   ├── download_petfinder.py   # Download via Kaggle API
│   └── preprocess.py           # Pré-processamento + vínculo imagem↔tabular
├── models/
│   └── multimodal_model.py     # Arquitetura MobileNetV2 + tabular
├── trainer/
│   └── train_vertex.py         # Script de treino para Vertex AI (GPU)
├── train.py                    # Treino local
├── deploy_vertex.py            # Deploy → Model Registry → Endpoint
├── submit_job.py               # Submissão de Custom Training Job
├── gcp_setup_test.py           # Validação da conexão GCP
└── CLAUDE.md                   # Contexto do projeto
```

---

## Competências Demonstradas

- **ML Engineering:** design, treino e deploy de modelo multimodal com TensorFlow
- **MLOps:** pipeline end-to-end com versionamento, registry e endpoint gerenciado
- **Google Cloud:** Vertex AI, Cloud Storage, IAM, Service Accounts
- **Engenharia de Dados:** ingestão via API, pré-processamento em escala, vínculo tabular-imagem
- **Boas práticas:** sem hardcode, logging estruturado, separação de configuração via `.env`

---

## Como Executar

```bash
# 1. Ambiente
python -m venv .venv && .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Configurar credenciais
cp .env.example .env  # preencher com suas chaves GCP e Kaggle

# 3. Download dos dados
py data/download_petfinder.py

# 4. Pré-processamento
py data/preprocess.py

# 5. Treino local
py train.py

# 6. Deploy no Vertex AI
py deploy_vertex.py
```

---

## Autor

**Leonardo** — ML Engineer  
[LinkedIn](https://linkedin.com/in/leonardod38) · [GitHub](https://github.com/leonardod38)
