# Meesho Visual Quality & Seller Optimization Engine

## Overview
This project builds a two-layer engine for **Image Quality & Seller Optimization** to improve Meesho’s marketplace efficiency, seller ROI, and customer trust.  
It detects poor-quality product images, provides feedback to sellers, and clusters duplicate listings while ranking sellers fairly within product groups.

## Features
- Automated image & text preprocessing
- Multimodal embeddings (CLIP, SBERT) for similarity comparisons
- Scoring engine combining image quality, text relevance, pricing, and performance
- Duplicate detection via clustering
- Fair seller ranking powered by data-driven models
- Actionable feedback loop for continuous seller improvement

---

## Tech Stack
- **Event Processing:** Kafka, Kinesis, AWS Lambda, GKE  
- **Databases:** Qdrant (vector search), Postgres (metadata), Feast (feature store)  
- **Models:** CLIP, ViT, SBERT, LambdaMART / XGBoost for ranking  
- **Clustering:** HDBSCAN  
- **Infrastructure:** Docker, docker-compose  

---

# Repository Structure

------------
meesho-visual-quality/
├── LICENSE              <- License file
├── README.md            <- Top-level README with project details
├── requirements.txt     <- Python dependencies
├── .env.example         <- Example environment variables
├── docker-compose.yml   <- Docker Compose config for API, DB, and Qdrant
├── Dockerfile           <- Dockerfile for API service
│
├── api                  <- FastAPI application
│   ├── main.py          <- Entry point for FastAPI server
│   ├── routes.py        <- API endpoint definitions
│   └── schemas.py       <- Pydantic models for request/response
│
├── services             <- Core service layer
│   ├── ingest.py        <- Data ingestion logic
│   ├── preprocess.py    <- Image preprocessing pipeline
│   └── embeddings.py    <- Embedding generation using ML models
│
├── db                   <- Database clients
│   ├── qdrant_client.py <- Qdrant (vector DB) client wrapper
│   └── postgres_client.py <- PostgreSQL client wrapper
│
├── clustering           <- Image clustering logic
│   └── cluster.py
│
├── ranking              <- Image ranking and scoring logic
│   └── ranker.py
│
├── utils                <- Utility functions
│   └── image_utils.py   <- Helper functions for image operations
│
├── tests                <- Unit tests
│   └── test_scoring.py  <- Tests for ranking/scoring module

### File/Folder Descriptions
- **README.md** → Project overview, documentation, and setup guide.  
- **requirements.txt** → Python dependencies for the pipeline and APIs.  
- **.env.example** → Environment variable template for DB/API keys.  
- **docker-compose.yml** → Orchestration to run API, Qdrant, and Postgres services together.  
- **Dockerfile** → Container configuration for the engine and services.  

#### API
- **main.py** → FastAPI entrypoint to serve endpoints for scoring, ranking, and feedback.  
- **routes.py** → API route definitions (upload, fetch recommendations, cluster queries).  
- **schemas.py** → Data validation models using Pydantic for requests/responses.  

#### Services
- **ingest.py** → Handles streaming ingestion events (seller uploads, product updates).  
- **preprocess.py** → Image resizing, normalization, text cleanup, and feature standardization.  
- **embeddings.py** → Generates CLIP/SBERT embeddings for product images and text.  

#### Database
- **qdrant_client.py** → Wrapper for Qdrant vector DB operations (store/retrieve embeddings).  
- **postgres_client.py** → Handles metadata storage and canonical product cluster mappings.  

#### Clustering
- **cluster.py** → HDBSCAN-based duplicate detection and canonical node creation.  

#### Ranking
- **ranker.py** → Seller scoring and ranking logic using weighted formula or learning-to-rank model.  

#### Utils
- **image_utils.py** → Helper functions (background removal, cropping, brightness adjustment).  

#### Tests
- **test_scoring.py** → Unit tests for composite scoring logic and similarity checks.  

---

## Roadmap
- [x] Ingestion and preprocessing pipeline  
- [x] Embedding generation (CLIP + SBERT)  
- [ ] Clustering and canonicalization  
- [ ] Ranking with LambdaMART/XGBoost  
- [ ] Seller dashboard integration  
- [ ] Automated feedback loop with predicted uplift  

---

## Getting Started
### 1. Clone the repository:  
```bash
git clone https://github.com/Hrishitcodethis/meesho-visual-quality.git
cd meesho-visual-quality
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### Copy Environment Variables
```bash
cp .env.example .env
```

### 3. Start services with Docker:  
```bash
docker-compose up --build
```

### API usage
```bash
docker-compose up --build
```







