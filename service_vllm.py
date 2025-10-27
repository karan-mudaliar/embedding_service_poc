"""
vLLM embedding service implementation.
This demonstrates the optimized approach using vLLM for fast inference.
"""

import torch
import structlog
from typing import List
from fastapi import FastAPI, HTTPException
from vllm import LLM
from models import EmbeddingRequest, EmbeddingResponse, HealthResponse

logger = structlog.get_logger()

# Initialize FastAPI app
app = FastAPI(title="vLLM Embedding Service", version="1.0.0")

# Global model instance
model = None
MODEL_NAME = "answerdotai/ModernBERT-base"  # Modern BERT model for embeddings


@app.on_event("startup")
async def startup_event():
    """Load the model on startup with vLLM optimizations."""
    global model
    logger.info("starting_service", backend="vllm", model=MODEL_NAME)

    try:
        # Initialize vLLM with optimizations
        # vLLM automatically enables:
        # - PagedAttention for memory efficiency
        # - Continuous batching for higher throughput
        # - CUDA graph optimization
        model = LLM(
            model=MODEL_NAME,
            task="embed",  # Specify embedding task
            max_model_len=512,  # Max sequence length
            gpu_memory_utilization=0.9,  # Use 90% of GPU memory
            tensor_parallel_size=1,  # Number of GPUs
        )

        logger.info(
            "model_loaded",
            backend="vllm",
            model=MODEL_NAME,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    except Exception as e:
        logger.error("model_load_failed", error=str(e))
        raise


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        backend="vllm",
        model=MODEL_NAME,
        gpu_available=torch.cuda.is_available(),
    )


@app.post("/embed", response_model=EmbeddingResponse)
async def embed_texts(request: EmbeddingRequest):
    """
    Generate embeddings for the provided texts using vLLM.

    Args:
        request: EmbeddingRequest containing texts to embed

    Returns:
        EmbeddingResponse with embeddings and metadata
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        logger.info("embedding_request", num_texts=len(request.texts))

        # Generate embeddings using vLLM's embed() method
        outputs = model.embed(request.texts)

        # Extract embeddings from vLLM output
        embeddings = [output.outputs.embedding for output in outputs]

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

    uvicorn.run(app, host="0.0.0.0", port=8001)  # Different port from HuggingFace service
