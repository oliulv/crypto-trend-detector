from sqlalchemy import create_engine
from classes import Base
import os
from dotenv import load_dotenv


load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")


try:
    engine = create_engine(DATABASE_URL, echo=True)
    connection = engine.connect()
    print("✅ Connected to database!")
    connection.close()
except Exception as e:
    print(f"❌ Database connection failed: {e}")


def init_db():
    # Base.metadata.drop_all(bind=engine)  # Delete existing tables
    Base.metadata.create_all(bind=engine)  # Create new ones


if __name__ == "__main__":
    init_db()
