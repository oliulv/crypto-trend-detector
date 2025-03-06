from sqlalchemy import create_engine, Table, MetaData
from classes import Base, FeatureImportance
from db import DATABASE_URL
from alembic.operations import Operations
from alembic.migration import MigrationContext

def migrate_database():
    """Safe migration that preserves existing data while adding new relationships."""
    engine = create_engine(DATABASE_URL)
    
    # Create migration context
    conn = engine.connect()
    ctx = MigrationContext.configure(conn)
    op = Operations(ctx)
    
    try:
        # Check if feature_importance table exists
        if not engine.dialect.has_table(conn, 'feature_importance'):
            # Create new table without dropping existing ones
            FeatureImportance.__table__.create(engine)
            print("✅ Created feature_importance table")
        
        # Add relationships (foreign keys) if they don't exist
        try:
            op.create_foreign_key(
                "fk_feature_importance_experiment",
                "feature_importance",
                "experiments",
                ["experiment_id"],
                ["experiment_id"]
            )
            print("✅ Added experiment relationship")
        except Exception as e:
            print("ℹ️ Experiment relationship already exists or other error:", str(e))
            
        try:
            op.create_foreign_key(
                "fk_feature_importance_results",
                "feature_importance",
                "results",
                ["result_id"],
                ["result_id"]
            )
            print("✅ Added results relationship")
        except Exception as e:
            print("ℹ️ Results relationship already exists or other error:", str(e))
            
        print("✅ Migration complete - existing data preserved")
        
    finally:
        conn.close()

if __name__ == "__main__":
    migrate_database()