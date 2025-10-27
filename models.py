"""
Pydantic models for the embedding service API.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class EmbeddingRequest(BaseModel):
    """Request model for embedding generation."""
    texts: List[str] = Field(..., description="List of texts to embed", min_length=1)
    model: Optional[str] = Field(None, description="Model name (optional)")


class EmbeddingResponse(BaseModel):
    """Response model for embedding generation."""
    embeddings: List[List[float]] = Field(..., description="List of embedding vectors")
    model: str = Field(..., description="Model used for embeddings")
    num_embeddings: int = Field(..., description="Number of embeddings returned")
    embedding_dim: int = Field(..., description="Dimension of each embedding")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    backend: str = Field(..., description="Backend type (huggingface/vllm)")
    model: str = Field(..., description="Model name")
    gpu_available: bool = Field(..., description="Whether GPU is available")
