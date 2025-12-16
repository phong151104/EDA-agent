"""
Script to load CSV mock data into PostgreSQL database.
This script:
1. Reads all CSV files from mock_data folder
2. Reads column types from metadata YAML definitions
3. Creates schemas (lh_vnfilm_v2, cdp_mart)
4. Creates tables with EXACT column types from metadata
5. Loads data into tables
"""
import os
import sys
import yaml
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text, inspect
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
# NOTE: Loading into 'lakehouse' database to match Trino catalog structure
DB_HOST = os.getenv("POSTGRES_HOST")
DB_PORT = os.getenv("POSTGRES_PORT")
DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_NAME = os.getenv("POSTGRES_DB") 

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Directories
MOCK_DATA_DIR = Path(__file__).parent.parent / "mock_data"
METADATA_DIR = Path(__file__).parent.parent / "metadata/domains/vnfilm_ticketing/tables"


def load_metadata(table_name: str) -> dict:
    """Load metadata YAML for a table."""
    yml_path = METADATA_DIR / f"{table_name}.yml"
    if not yml_path.exists():
        return None
    
    with open(yml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def map_metadata_type_to_pg(data_type: str) -> str:
    """Convert metadata data_type to PostgreSQL type."""
    dt = data_type.lower()
    
    # Direct mappings
    if dt == 'bigint':
        return 'BIGINT'
    elif dt == 'integer' or dt == 'int':
        return 'INTEGER'
    elif dt in ('varchar', 'text', 'string'):
        return 'TEXT'
    elif dt.startswith('decimal') or dt.startswith('numeric'):
        # Extract precision from decimal(38,18) -> DECIMAL(38,18)
        return dt.upper()
    elif dt.startswith('timestamp'):
        return 'TIMESTAMP'
    elif dt == 'date':
        return 'DATE'
    elif dt == 'boolean' or dt == 'bool':
        return 'BOOLEAN'
    elif dt == 'json' or dt == 'jsonb':
        return 'TEXT'  # Store as TEXT for compatibility
    else:
        return 'TEXT'


def get_column_types_from_metadata(table_name: str) -> dict:
    """Get column -> PostgreSQL type mapping from metadata."""
    metadata = load_metadata(table_name)
    if not metadata or 'columns' not in metadata:
        return None
    
    column_types = {}
    for col_name, col_info in metadata['columns'].items():
        metadata_type = col_info.get('data_type', 'varchar')
        pg_type = map_metadata_type_to_pg(metadata_type)
        column_types[col_name] = pg_type
    
    return column_types


def get_postgres_type_fallback(dtype, column_name: str, series: pd.Series = None) -> str:
    """Fallback: Map pandas dtype to PostgreSQL type when no metadata available."""
    dtype_str = str(dtype)
    col_lower = column_name.lower()
    
    # ID columns - check if actually numeric before using BIGINT
    if col_lower.endswith('_id') or col_lower == 'id':
        if series is not None:
            sample = series.dropna().head(10).tolist()
            if sample:
                are_numeric = True
                for val in sample:
                    val_str = str(val)
                    if val_str.endswith('.0'):
                        val_str = val_str[:-2]
                    if not val_str.lstrip('-').isdigit():
                        are_numeric = False
                        break
                if not are_numeric:
                    return 'TEXT'
        return 'BIGINT'
    
    # Date/time columns
    if 'date' in col_lower or 'time' in col_lower or col_lower.endswith('_at'):
        return 'TIMESTAMP'
    
    # Price/amount columns
    if any(word in col_lower for word in ['price', 'amount', 'total', 'fee', 'cost', 'discount', 'profit']):
        return 'DECIMAL(38,18)'
    
    # Standard type mapping
    if 'int' in dtype_str:
        return 'BIGINT'
    elif 'float' in dtype_str:
        return 'DECIMAL(38,18)'
    elif 'datetime' in dtype_str:
        return 'TIMESTAMP'
    elif 'bool' in dtype_str:
        return 'BOOLEAN'
    else:
        return 'TEXT'


def parse_csv_filename(filename: str) -> tuple[str, str, str]:
    """
    Parse CSV filename to extract schema and table name.
    Format: lakehouse.schema_name.table_name.csv
    Returns: (full_schema, table_name, qualified_name)
    """
    # Remove .csv extension
    base_name = filename.replace('.csv', '')
    parts = base_name.split('.')
    
    if len(parts) >= 3:
        # lakehouse.lh_vnfilm_v2.table_name -> schema: lh_vnfilm_v2, table: table_name
        schema = parts[1]  # lh_vnfilm_v2 or cdp_mart
        table = parts[2]
        return schema, table, f"{schema}.{table}"
    else:
        # Fallback: use public schema
        return "public", base_name, f"public.{base_name}"


def create_schemas(engine, schemas: set):
    """Create schemas if they don't exist."""
    with engine.connect() as conn:
        for schema in schemas:
            conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
            print(f"✓ Schema '{schema}' ready")
        conn.commit()


def create_table_from_df(engine, schema: str, table: str, df: pd.DataFrame):
    """Create table with column types from metadata YAML (or fallback to inference)."""
    
    # Try to get types from metadata first
    metadata_types = get_column_types_from_metadata(table)
    
    columns = []
    for col in df.columns:
        # Use metadata type if available, otherwise fallback to inference
        if metadata_types and col in metadata_types:
            pg_type = metadata_types[col]
        else:
            pg_type = get_postgres_type_fallback(df[col].dtype, col, df[col])
        
        # Escape column names
        safe_col = f'"{col}"'
        columns.append(f"{safe_col} {pg_type}")
    
    columns_sql = ",\n    ".join(columns)
    create_sql = f'''
    DROP TABLE IF EXISTS "{schema}"."{table}" CASCADE;
    CREATE TABLE "{schema}"."{table}" (
        {columns_sql}
    );
    '''
    
    with engine.connect() as conn:
        conn.execute(text(create_sql))
        conn.commit()


def load_csv_to_table(engine, schema: str, table: str, csv_path: Path):
    """Load CSV data into PostgreSQL table."""
    print(f"\n📁 Loading: {csv_path.name}")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")
    
    # Create table
    create_table_from_df(engine, schema, table, df)
    print(f"   ✓ Created table: {schema}.{table}")
    
    # Load data using pandas to_sql
    df.to_sql(
        name=table,
        con=engine,
        schema=schema,
        if_exists='append',
        index=False,
        method='multi',
        chunksize=1000
    )
    print(f"   ✓ Loaded {len(df)} rows")
    
    return len(df)


def main():
    print("=" * 60)
    print("📊 Loading Mock Data to PostgreSQL")
    print("=" * 60)
    print(f"\n🔗 Database: {DATABASE_URL.replace(DB_PASSWORD, '****')}")
    print(f"📂 Mock data directory: {MOCK_DATA_DIR}")
    
    # Create database engine
    try:
        engine = create_engine(DATABASE_URL)
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✓ Database connection successful\n")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("\nMake sure PostgreSQL is running:")
        print("  docker-compose up -d postgres")
        sys.exit(1)
    
    # Find all CSV files
    csv_files = list(MOCK_DATA_DIR.glob("*.csv"))
    if not csv_files:
        print(f"❌ No CSV files found in {MOCK_DATA_DIR}")
        sys.exit(1)
    
    print(f"Found {len(csv_files)} CSV files\n")
    
    # Parse filenames to get schemas
    schemas = set()
    tables_info = []
    for csv_file in csv_files:
        schema, table, qualified = parse_csv_filename(csv_file.name)
        schemas.add(schema)
        tables_info.append((schema, table, csv_file))
    
    # Create schemas
    print("-" * 40)
    print("Creating schemas...")
    create_schemas(engine, schemas)
    
    # Load each CSV
    print("-" * 40)
    print("Loading tables...")
    total_rows = 0
    loaded_tables = 0
    failed_tables = []
    
    for schema, table, csv_path in tables_info:
        try:
            rows = load_csv_to_table(engine, schema, table, csv_path)
            total_rows += rows
            loaded_tables += 1
        except Exception as e:
            failed_tables.append((table, csv_path.name, str(e)))
            print(f"   ❌ Error loading {csv_path.name}: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ LOAD COMPLETE")
    print("=" * 60)
    print(f"   Tables loaded: {loaded_tables}/{len(csv_files)}")
    print(f"   Total rows: {total_rows:,}")
    print(f"   Schemas: {', '.join(sorted(schemas))}")
    
    # Show failed tables
    if failed_tables:
        print("\n" + "=" * 60)
        print("❌ FAILED TABLES:")
        print("=" * 60)
        for table_name, csv_name, error in failed_tables:
            print(f"   • {table_name}")
            print(f"     File: {csv_name}")
            print(f"     Error: {error[:200]}...")
    
    # Show sample query
    print("\n📝 Sample queries to test:")
    print("-" * 40)
    for schema in sorted(schemas):
        print(f'   SELECT * FROM "{schema}"."{tables_info[0][1]}" LIMIT 5;')
    
    print("\n🎉 Ready for testing!")


if __name__ == "__main__":
    main()
