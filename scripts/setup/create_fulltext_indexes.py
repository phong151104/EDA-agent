"""
Create Neo4j Fulltext Indexes.

Run this ONCE before using the text-to-sql verbose test.

Usage:
    python scripts/setup/create_fulltext_indexes.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def main():
    from src.context_fusion import Neo4jVectorIndex
    
    print("=" * 50)
    print("  Creating Neo4j Fulltext Indexes")
    print("=" * 50)
    
    try:
        index = Neo4jVectorIndex()
        
        print("\n🔧 Creating fulltext indexes...")
        index.create_fulltext_index()
        
        print("\n✅ Done! Fulltext indexes created successfully.")
        print("   You can now run the verbose test script.")
        
        index.close()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("  1. Neo4j is running")
        print("  2. Your config has correct Neo4j credentials")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
