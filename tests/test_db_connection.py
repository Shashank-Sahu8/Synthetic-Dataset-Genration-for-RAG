import os
import sys
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

def test_connection():
    load_dotenv()
    
    db_url = os.getenv("DATABASE_URL")
    
    if not db_url:
        print("Error: DATABASE_URL is not set in the .env file.")
        print("Please add it to your .env in the format:")
        print("DATABASE_URL=postgresql://user:password@ec2-instance-ip:5432/dbname")
        sys.exit(1)
        
    print(f"Attempting to connect to the database...")
    
    try:
        engine = create_engine(db_url, pool_pre_ping=True)
        
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1;"))
            # Fetch the result
            for row in result:
                if row[0] == 1:
                    print("Successfully connected to the PostgreSQL database on EC2!")
                    
    except OperationalError as e:
        print("Database connection failed!")
        print("Please check your EC2 Security Groups (port 5432 must be open), credentials, and DB URL.")
        print(f"\nError details:\n{e}")
    except Exception as e:
        print("An unexpected error occurred!")
        print(f"\nError details:\n{e}")

if __name__ == "__main__":
    test_connection()
