"""
Episodic Memory module.

Stores and retrieves past analysis episodes to avoid repeating failed approaches.
Uses vector similarity for retrieval.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4

from openai import OpenAI

from config import config

logger = logging.getLogger(__name__)


@dataclass
class Episode:
    """
    An episode of past analysis.
    
    Stores successful approaches and failed attempts
    for future reference.
    """
    
    id: str = field(default_factory=lambda: str(uuid4()))
    question: str = ""
    hypotheses: list[str] = field(default_factory=list)
    approach: str = ""
    outcome: str = ""  # "success", "failure", "partial"
    learnings: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    embedding: list[float] | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "question": self.question,
            "hypotheses": self.hypotheses,
            "approach": self.approach,
            "outcome": self.outcome,
            "learnings": self.learnings,
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Episode":
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid4())),
            question=data.get("question", ""),
            hypotheses=data.get("hypotheses", []),
            approach=data.get("approach", ""),
            outcome=data.get("outcome", ""),
            learnings=data.get("learnings", []),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
        )


class EpisodicMemory:
    """
    Episodic memory for storing and retrieving past analyses.
    
    Uses vector similarity search to find relevant past episodes.
    Can be backed by various vector stores (in-memory, PostgreSQL, etc.)
    """
    
    def __init__(self):
        """Initialize episodic memory."""
        self._episodes: list[Episode] = []
        self._openai_client: OpenAI | None = None
    
    @property
    def openai_client(self) -> OpenAI:
        """Get or create OpenAI client for embeddings."""
        if self._openai_client is None:
            self._openai_client = OpenAI(api_key=config.openai.api_key)
        return self._openai_client
    
    async def add_episode(self, episode: Episode) -> str:
        """
        Add a new episode to memory.
        
        Args:
            episode: Episode to store
            
        Returns:
            Episode ID
        """
        # Generate embedding for the episode
        text = self._episode_to_text(episode)
        episode.embedding = await self._generate_embedding(text)
        
        self._episodes.append(episode)
        logger.info(f"Added episode {episode.id} to memory")
        
        return episode.id
    
    async def search(
        self,
        query: str,
        top_k: int = 5,
        outcome_filter: str | None = None,
    ) -> list[Episode]:
        """
        Search for similar episodes.
        
        Args:
            query: Search query (usually the new question)
            top_k: Number of results to return
            outcome_filter: Filter by outcome ("success", "failure")
            
        Returns:
            List of similar episodes
        """
        if not self._episodes:
            return []
        
        # Generate query embedding
        query_embedding = await self._generate_embedding(query)
        
        # Calculate similarities
        scored_episodes = []
        for episode in self._episodes:
            if episode.embedding is None:
                continue
            
            if outcome_filter and episode.outcome != outcome_filter:
                continue
            
            similarity = self._cosine_similarity(query_embedding, episode.embedding)
            scored_episodes.append((similarity, episode))
        
        # Sort by similarity and return top_k
        scored_episodes.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored_episodes[:top_k]]
    
    async def get_failed_approaches(
        self,
        question: str,
        top_k: int = 3,
    ) -> list[str]:
        """
        Get approaches that failed for similar questions.
        
        Used to avoid repeating mistakes.
        
        Args:
            question: Current question
            top_k: Number of failed approaches to return
            
        Returns:
            List of failed approach descriptions
        """
        failed_episodes = await self.search(
            question,
            top_k=top_k,
            outcome_filter="failure",
        )
        
        return [
            f"Failed approach: {ep.approach}. Learnings: {', '.join(ep.learnings)}"
            for ep in failed_episodes
        ]
    
    async def get_successful_patterns(
        self,
        question: str,
        top_k: int = 3,
    ) -> list[Episode]:
        """
        Get successful episodes for similar questions.
        
        Args:
            question: Current question
            top_k: Number of successful episodes to return
            
        Returns:
            List of successful episodes
        """
        return await self.search(
            question,
            top_k=top_k,
            outcome_filter="success",
        )
    
    async def _generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for text using OpenAI."""
        try:
            response = self.openai_client.embeddings.create(
                model=config.openai.embedding_model,
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return [0.0] * config.vector_index.dimensions
    
    def _episode_to_text(self, episode: Episode) -> str:
        """Convert episode to text for embedding."""
        parts = [
            f"Question: {episode.question}",
            f"Hypotheses: {', '.join(episode.hypotheses)}",
            f"Approach: {episode.approach}",
            f"Outcome: {episode.outcome}",
            f"Learnings: {', '.join(episode.learnings)}",
        ]
        return "\n".join(parts)
    
    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def clear(self) -> None:
        """Clear all episodes from memory."""
        self._episodes.clear()
        logger.info("Cleared episodic memory")
