from sqlalchemy import create_engine, inspect
from classes import Base
from db import DATABASE_URL

def migrate_database():
    engine = create_engine(DATABASE_URL)
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()
    
    # Create new tables only if they don't exist
    Base.metadata.create_all(
        engine,
        tables=[table for table in Base.metadata.tables.values()
                if table.name not in existing_tables]
    )
    
    print("✅ Migration complete - Added new tables while preserving existing data")

if __name__ == "__main__":
    migrate_database()