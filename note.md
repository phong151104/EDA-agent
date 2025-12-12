# Command to delete all nodes in the graph database
MATCH (n) DETACH DELETE n

# Query to find all tables and their relationships in the graph database
MATCH p = (t:Table)-[:HAS_COLUMN|JOIN|FK*1..2]-(x)
RETURN p
LIMIT 5000;

# Command to show indexes
SHOW INDEXES YIELD name, type WHERE name CONTAINS 'concept'

# Command to build the graph database
python scripts/build_neo4j_graph.py --domain vnfilm_ticketing

# Command to index embeddings
python scripts/index_embeddings.py

# Command to test context fusion
python scripts/test/test_context_fusion.py "Tại sao doanh thu giảm?"

# Command to test planner integration
python scripts/test/test_planner_integration.py "Tại sao doanh thu giảm?"

# Command to test session and subgraph
python scripts/test/test_session_subgraph.py

# Command to test planner-critic loop
python scripts/test/test_planner_critic_loop.py "Tại sao doanh thu giảm?"

# Command to test text to sql
python scripts/test/test_text_to_sql.py
