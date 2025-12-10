# Query to find all tables and their relationships in the graph database
MATCH p = (t:Table)-[:HAS_COLUMN|JOIN|FK*1..2]-(x)
RETURN p
LIMIT 1000;

# Command to build the graph database
python scripts\build_neo4j_graph.py --domain vnfilm_ticketing --metadata-root metadata\domains