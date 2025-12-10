# Query to find all tables and their relationships in the graph database
MATCH p = (t:Table)-[:HAS_COLUMN|JOIN|FK*1..2]-(x)
RETURN p
LIMIT 1000;