import pymongo
from Autism.constant.database import DATABASE_NAME
from Autism.constant.env_variable import MONGODB_URL_KEY
import certifi
import os
from dotenv import load_dotenv

# Load local .env if exists
load_dotenv()

# ca = certifi.where()

class MongoDBClient:
    client = None

    def __init__(self, database_name=DATABASE_NAME) -> None:
        try:
            if MongoDBClient.client is None:
                mongo_db_url = os.getenv(MONGODB_URL_KEY)

                if not mongo_db_url:
                    raise ValueError("MONGO_URL environment variable is not set.")

                print(f"🔗 Connecting to MongoDB: {mongo_db_url}")  # Remove in prod
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url)

            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name

        except Exception as e:
            raise Exception(f"❌ Failed to connect to MongoDB: {e}")
