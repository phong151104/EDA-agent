#!/usr/bin/env python3
"""
Index embeddings for Neo4j Vector Search.

Generates OpenAI embeddings for all graph nodes and stores them in Neo4j.

Usage:
    python scripts/index_embeddings.py
"""

import asyncio
import sys
import os
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("Neo4j Vector Index Builder")
    logger.info("=" * 60)
    
    from src.context_fusion.vector_index import Neo4jVectorIndex
    
    # Initialize vector index
    vector_index = Neo4jVectorIndex()
    
    try:
        # Index all nodes
        logger.info("\nCreating vector indexes and generating embeddings...")
        counts = vector_index.index_all_nodes()
        
        logger.info("\n" + "=" * 60)
        logger.info("INDEXING COMPLETE!")
        logger.info("=" * 60)
        
        for label, count in counts.items():
            logger.info(f"  {label}: {count} nodes indexed")
        
        logger.info("\nYou can now use vector search for semantic schema retrieval.")
        logger.info("Test with: python scripts/test_context_fusion.py \"your query\"")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        vector_index.close()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
