"""
Context Builder.

Combines Query Rewriter + Schema Retriever to produce enriched context
for the rest of the EDA system.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from .models import AnalyzedQuery, SubGraph
from .query_rewriter import QueryRewriter, quick_analyze
from .schema_retriever import SchemaRetriever
from .session_context import EDASession, cache_session

logger = logging.getLogger(__name__)


@dataclass
class EnrichedContext:
    """
    Final enriched context for EDA.
    
    This is what gets passed to Planner, Critic, and other agents.
    """
    
    original_query: str
    analyzed_query: AnalyzedQuery
    sub_graph: SubGraph
    prompt_context: str = ""  # Text representation for LLM prompts
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "analyzed_query": {
                "intent": self.analyzed_query.intent.value,
                "keywords": self.analyzed_query.keywords,
                "entities": [
                    {"text": e.text, "type": e.entity_type, "normalized": e.normalized_name}
                    for e in self.analyzed_query.entities
                ],
            },
            "sub_graph": self.sub_graph.to_dict(),
        }
    
    def to_session(self, domain: str = "vnfilm_ticketing") -> EDASession:
        """Convert to EDASession for pipeline usage."""
        return EDASession.create(
            query=self.original_query,
            analyzed_query=self.analyzed_query,
            sub_graph=self.sub_graph,
            prompt_context=self.prompt_context,
            domain=domain,
        )


class ContextBuilder:
    """
    Builds enriched context from user query.
    
    Pipeline:
    1. Query Rewriter analyzes the query
    2. Schema Retriever gets relevant sub-graph from Neo4j
    3. Combines into EnrichedContext for agents
    """
    
    def __init__(
        self,
        use_llm: bool = True,
        domain: str = "vnfilm_ticketing",
    ):
        """
        Initialize ContextBuilder.
        
        Args:
            use_llm: Use LLM for query analysis
            domain: Default domain for schema retrieval
        """
        self.query_rewriter = QueryRewriter(use_llm=use_llm)
        self.schema_retriever = SchemaRetriever()
        self.domain = domain
    
    async def build(
        self,
        query: str,
        domain: str | None = None,
        top_k: int = 10,
    ) -> EnrichedContext:
        """
        Build enriched context from user query.
        
        Args:
            query: Raw user query
            domain: Optional domain override
            top_k: Number of vector search results
            
        Returns:
            EnrichedContext with all information for EDA
        """
        domain = domain or self.domain
        logger.info(f"Building context for: {query}")
        
        # Step 1: Analyze query
        analyzed_query = await self.query_rewriter.analyze(query)
        logger.info(f"Intent: {analyzed_query.intent.value}")
        logger.info(f"Keywords: {analyzed_query.keywords}")
        
        # Step 2: Retrieve sub-graph
        sub_graph = await self.schema_retriever.retrieve(
            analyzed_query,
            domain=domain,
            top_k=top_k,
        )
        
        # Step 3: Generate prompt context
        prompt_context = self._generate_prompt_context(analyzed_query, sub_graph)
        
        return EnrichedContext(
            original_query=query,
            analyzed_query=analyzed_query,
            sub_graph=sub_graph,
            prompt_context=prompt_context,
        )
    
    async def build_session(
        self,
        query: str,
        domain: str | None = None,
        top_k: int = 10,
        cache: bool = True,
    ) -> EDASession:
        """
        Build EDASession from user query.
        
        This is the recommended method for pipeline usage.
        Returns an EDASession that can be passed to all downstream agents.
        
        Args:
            query: Raw user query
            domain: Optional domain override
            top_k: Number of vector search results
            cache: Whether to cache the session
            
        Returns:
            EDASession with SubGraph and metadata
        """
        domain = domain or self.domain
        
        # Build context
        context = await self.build(query, domain, top_k)
        
        # Convert to session
        session = context.to_session(domain=domain)
        
        # Cache if requested
        if cache:
            cache_session(session)
        
        logger.info(f"Session created: {session.summary()}")
        return session
    
    def _generate_prompt_context(
        self,
        analyzed_query: AnalyzedQuery,
        sub_graph: SubGraph,
    ) -> str:
        """Generate text context for LLM prompts."""
        lines = [
            "# Context for Analysis",
            "",
            f"## Query Information",
            f"- **Original Query**: {analyzed_query.original_query}",
            f"- **Intent**: {analyzed_query.intent.value}",
            f"- **Keywords**: {', '.join(analyzed_query.keywords)}",
        ]
        
        if analyzed_query.time_range:
            lines.append(f"- **Time Range**: {analyzed_query.time_range}")
        
        lines.append("")
        lines.append(sub_graph.to_prompt_context())
        
        return "\n".join(lines)
    
    def close(self) -> None:
        """Close connections."""
        self.schema_retriever.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Convenience functions
async def build_context(query: str, domain: str = "vnfilm_ticketing") -> EnrichedContext:
    """Quick function to build context from a query."""
    builder = ContextBuilder(domain=domain)
    try:
        return await builder.build(query)
    finally:
        builder.close()


async def build_session(query: str, domain: str = "vnfilm_ticketing") -> EDASession:
    """
    Quick function to build session from a query.
    
    This is the recommended entry point for the EDA pipeline.
    
    Example:
        session = await build_session("Doanh thu theo vendor tháng này")
        
        # SubGraph is stored in session, pass to agents:
        planner.process(session.sub_graph)
        sql_gen.process(session.sub_graph)
        
        # Track state across agents:
        session.set_state("plan", planner_result)
    """
    builder = ContextBuilder(domain=domain)
    try:
        return await builder.build_session(query)
    finally:
        builder.close()
