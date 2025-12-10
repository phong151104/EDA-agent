"""
OpenAI Embeddings.

Generates embeddings using OpenAI's embedding API.
"""

from __future__ import annotations

import logging
from typing import List

from openai import OpenAI

from config import config

logger = logging.getLogger(__name__)


class OpenAIEmbedder:
    """
    Generates embeddings using OpenAI's embedding API.
    """
    
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or config.openai.api_key
        self.model = model or config.openai.embedding_model
        self.client = OpenAI(api_key=self.api_key)
        self.dimensions = config.vector_index.dimensions
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding
    
    def embed_texts(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts per API call
            
        Returns:
            List of embedding vectors
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"Embedding batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}")
            
            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
            )
            
            # Sort by index to maintain order
            sorted_data = sorted(response.data, key=lambda x: x.index)
            batch_embeddings = [item.embedding for item in sorted_data]
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
