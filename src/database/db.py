import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv


# Load environment variables from .env file in the project root
load_dotenv()

# Get your PostgreSQL connection URL from the environment variables
DATABASE_URL = os.getenv("DATABASE_URL")

# Create an engine and a session factory
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)


def get_db():
    """Provide a transactional scope around a series of operations."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()