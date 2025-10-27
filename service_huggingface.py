"""
HuggingFace/LangChain embedding service implementation.
This represents the current baseline approach using LangChain's HuggingFace embeddings.
"""

import torch
import structlog
from typing import List
from fastapi import FastAPI, HTTPException
from langchain_community.embeddings import HuggingFaceEmbeddings
from models import EmbeddingRequest, EmbeddingResponse, HealthResponse

logger = structlog.get_logger()

# Initialize FastAPI app
app = FastAPI(title="HuggingFace Embedding Service", version="1.0.0")

# Global model instance
model = None
MODEL_NAME = "answerdotai/ModernBERT-base"  # Modern BERT model for embeddings


@app.on_event("startup")
async def startup_event():
    """Load the model on startup."""
    global model
    logger.info("starting_service", backend="huggingface", model=MODEL_NAME)

    try:
        # Initialize HuggingFace embeddings
        model_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
        encode_kwargs = {"normalize_embeddings": True}

        model = HuggingFaceEmbeddings(
            model_name=MODEL_NAME,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

        logger.info(
            "model_loaded",
            backend="huggingface",
            model=MODEL_NAME,
            device=model_kwargs["device"],
        )
    except Exception as e:
        logger.error("model_load_failed", error=str(e))
        raise


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        backend="huggingface",
        model=MODEL_NAME,
        gpu_available=torch.cuda.is_available(),
    )


@app.post("/embed", response_model=EmbeddingResponse)
async def embed_texts(request: EmbeddingRequest):
    """
    Generate embeddings for the provided texts.

    Args:
        request: EmbeddingRequest containing texts to embed

    Returns:
        EmbeddingResponse with embeddings and metadata
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        logger.info("embedding_request", num_texts=len(request.texts))

        # Generate embeddings
        embeddings = model.embed_documents(request.texts)

        logger.info("embedding_complete", num_embeddings=len(embeddings))

        return EmbeddingResponse(
            embeddings=embeddings,
            model=MODEL_NAME,
            num_embeddings=len(embeddings),
            embedding_dim=len(embeddings[0]) if embeddings else 0,
        )
    except Exception as e:
        logger.error("embedding_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
