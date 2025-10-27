"""
ONNX Runtime + TensorRT embedding service implementation.
Uses ONNX Runtime with TensorRT execution provider for maximum GPU optimization.
"""

import torch
import structlog
from typing import List
from fastapi import FastAPI, HTTPException
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
import numpy as np
from models import EmbeddingRequest, EmbeddingResponse, HealthResponse

logger = structlog.get_logger()

# Initialize FastAPI app
app = FastAPI(title="ONNX+TensorRT Embedding Service", version="1.0.0")

# Global model instance
model = None
tokenizer = None
MODEL_NAME = "answerdotai/ModernBERT-base"  # Modern BERT model for embeddings


@app.on_event("startup")
async def startup_event():
    """Load the model on startup with ONNX Runtime + TensorRT optimizations."""
    global model, tokenizer
    logger.info("starting_service", backend="onnx_tensorrt", model=MODEL_NAME)

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        # Load model with ONNX Runtime using TensorRT execution provider
        # TensorRT provides additional optimizations on top of ONNX
        model = ORTModelForFeatureExtraction.from_pretrained(
            MODEL_NAME,
            export=True,  # Export to ONNX if needed
            provider="TensorrtExecutionProvider",  # Use TensorRT for max performance
            provider_options={
                "trt_fp16_enable": True,  # Enable FP16 for faster inference
                "trt_engine_cache_enable": True,  # Cache TensorRT engines
                "trt_engine_cache_path": "./trt_cache",
            },
        )

        logger.info(
            "model_loaded",
            backend="onnx_tensorrt",
            model=MODEL_NAME,
            device="cuda" if torch.cuda.is_available() else "cpu",
            fp16_enabled=True,
        )
    except Exception as e:
        logger.error("model_load_failed", error=str(e))
        # Fallback to CUDA execution provider if TensorRT fails
        logger.warning("tensorrt_failed_fallback_to_cuda", error=str(e))
        try:
            model = ORTModelForFeatureExtraction.from_pretrained(
                MODEL_NAME,
                export=True,
                provider="CUDAExecutionProvider",
            )
            logger.info("model_loaded_with_cuda_fallback")
        except Exception as e2:
            logger.error("cuda_fallback_failed", error=str(e2))
            raise


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        backend="onnx_tensorrt",
        model=MODEL_NAME,
        gpu_available=torch.cuda.is_available(),
    )


@app.post("/embed", response_model=EmbeddingResponse)
async def embed_texts(request: EmbeddingRequest):
    """
    Generate embeddings for the provided texts using ONNX Runtime + TensorRT.

    Args:
        request: EmbeddingRequest containing texts to embed

    Returns:
        EmbeddingResponse with embeddings and metadata
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        logger.info("embedding_request", num_texts=len(request.texts))

        # Tokenize inputs
        inputs = tokenizer(
            request.texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )

        # Generate embeddings using ONNX Runtime + TensorRT
        with torch.no_grad():
            outputs = model(**inputs)

        # Use mean pooling on token embeddings
        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = (sum_embeddings / sum_mask).cpu().numpy()

        # Convert to list for JSON serialization
        embeddings_list = embeddings.tolist()

        logger.info("embedding_complete", num_embeddings=len(embeddings_list))

        return EmbeddingResponse(
            embeddings=embeddings_list,
            model=MODEL_NAME,
            num_embeddings=len(embeddings_list),
            embedding_dim=len(embeddings_list[0]) if embeddings_list else 0,
        )
    except Exception as e:
        logger.error("embedding_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)  # Port 8002 for ONNX+TensorRT service
