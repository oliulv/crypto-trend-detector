from sqlalchemy import create_engine, inspect
from classes import Base, Experiment, Results
from db import DATABASE_URL

def migrate_database():
    engine = create_engine(DATABASE_URL)
    inspector = inspect(engine)
    
    # Drop specific tables if they exist
    Base.metadata.drop_all(
        engine,
        tables=[
            Base.metadata.tables['experiments'],
            Base.metadata.tables['results']
        ]
    )
    
    # Recreate the tables
    Base.metadata.create_all(
        engine,
        tables=[
            Base.metadata.tables['experiments'],
            Base.metadata.tables['results']
        ]
    )
    
    print("✅ Migration complete - Experiments and Results tables have been reset")

if __name__ == "__main__":
    migrate_database()