#!/usr/bin/env python3
"""
Test Redis cache functionality.

Usage:
    # Start Redis first:
    docker run -d --name redis -p 6379:6379 redis
    
    # Then run test:
    python scripts/test/test_redis_cache.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(override=True)


def test_redis_connection():
    """Test basic Redis connection."""
    from src.cache import RedisCache
    
    print("=" * 50)
    print("Testing Redis Connection")
    print("=" * 50)
    
    if RedisCache.ping():
        print("✅ Redis is connected!")
        return True
    else:
        print("❌ Redis connection failed!")
        print("   Make sure Redis is running:")
        print("   docker run -d --name redis -p 6379:6379 redis")
        return False


def test_session_cache():
    """Test SessionCache functionality."""
    from src.cache import SessionCache
    
    print("\n" + "=" * 50)
    print("Testing SessionCache")
    print("=" * 50)
    
    # Create session
    cache = SessionCache("test-session-123")
    
    # Test generic set/get
    print("\n1. Testing generic set/get...")
    cache.set("custom_key", {"foo": "bar", "count": 42})
    result = cache.get("custom_key")
    assert result == {"foo": "bar", "count": 42}
    print("   ✅ Generic set/get works")
    
    # Test step result
    print("\n2. Testing step result storage...")
    cache.save_step_result(
        step_id="s1",
        sql="SELECT * FROM orders",
        data=[
            {"id": 1, "total": 100000},
            {"id": 2, "total": 200000},
        ],
        code=None,
        output=None,
    )
    
    step_result = cache.get_step_result("s1")
    assert step_result["sql"] == "SELECT * FROM orders"
    assert len(step_result["data"]) == 2
    print("   ✅ Step result storage works")
    
    # Test get_step_data
    print("\n3. Testing get_step_data...")
    data = cache.get_step_data("s1")
    assert len(data) == 2
    print(f"   ✅ Got {len(data)} rows")
    
    # Test plan storage
    print("\n4. Testing plan storage...")
    cache.save_plan({
        "version": 1,
        "hypotheses": [{"id": "h1", "statement": "Test hypothesis"}],
        "steps": [{"id": "s1", "action_type": "query"}],
    })
    plan = cache.get_plan()
    assert plan["version"] == 1
    print("   ✅ Plan storage works")
    
    # Test subgraph storage
    print("\n5. Testing subgraph storage...")
    cache.save_subgraph({
        "tables": ["orders", "users"],
        "columns": ["id", "total"],
    })
    subgraph = cache.get_subgraph()
    assert "orders" in subgraph["tables"]
    print("   ✅ Subgraph storage works")
    
    # Test session summary
    print("\n6. Testing session summary...")
    summary = cache.get_session_summary()
    print(f"   Session: {summary['session_id']}")
    print(f"   Keys: {summary['key_count']}")
    print(f"   Has plan: {summary['has_plan']}")
    print(f"   Steps: {summary['steps']}")
    
    # Cleanup
    print("\n7. Cleaning up session...")
    deleted = cache.clear_session()
    print(f"   ✅ Deleted {deleted} keys")
    
    return True


def test_sql_cache():
    """Test SQLCache functionality."""
    from src.cache import SQLCache
    
    print("\n" + "=" * 50)
    print("Testing SQLCache")
    print("=" * 50)
    
    sql_cache = SQLCache()
    
    # Test cache miss
    print("\n1. Testing cache miss...")
    result = sql_cache.get("SELECT * FROM orders WHERE id = 999")
    assert result is None
    print("   ✅ Cache miss returned None")
    
    # Test cache set
    print("\n2. Testing cache set...")
    test_sql = "SELECT * FROM orders WHERE status = 'payment'"
    test_rows = [{"id": 1, "status": "payment"}, {"id": 2, "status": "payment"}]
    sql_cache.set(test_sql, test_rows)
    print("   ✅ Cache set completed")
    
    # Test cache hit
    print("\n3. Testing cache hit...")
    cached = sql_cache.get(test_sql)
    assert cached == test_rows
    print(f"   ✅ Cache hit: {len(cached)} rows")
    
    # Test normalized query (different whitespace, same query)
    print("\n4. Testing normalized query matching...")
    same_query = "SELECT   *   FROM   orders   WHERE   status = 'payment'"
    cached2 = sql_cache.get(same_query)
    assert cached2 == test_rows
    print("   ✅ Normalized query matched")
    
    return True


def test_multiple_step_data():
    """Test getting data from multiple steps."""
    from src.cache import SessionCache
    
    print("\n" + "=" * 50)
    print("Testing Multiple Step Data Retrieval")
    print("=" * 50)
    
    cache = SessionCache("test-multi-step")
    
    # Save multiple steps
    cache.save_step_result("s1", data=[{"month": 11, "revenue": 7772000}])
    cache.save_step_result("s2", data=[{"month": 12, "revenue": 4759000}])
    
    # Get multiple at once
    dataframes = cache.get_multiple_step_data(["s1", "s2"])
    
    print(f"   Retrieved {len(dataframes)} DataFrames:")
    for step_id, data in dataframes.items():
        print(f"   - {step_id}: {data}")
    
    assert "s1" in dataframes
    assert "s2" in dataframes
    print("   ✅ Multiple step data retrieval works")
    
    # Cleanup
    cache.clear_session()
    
    return True


if __name__ == "__main__":
    print("\n🧪 Redis Cache Test Suite\n")
    
    # Check connection first
    if not test_redis_connection():
        sys.exit(1)
    
    # Run tests
    try:
        test_session_cache()
        test_sql_cache()
        test_multiple_step_data()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed!")
        print("=" * 50)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
