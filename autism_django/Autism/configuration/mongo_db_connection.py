import pymongo
from Autism.constant.database import DATABASE_NAME
from Autism.constant.env_variable import MONGODB_URL_KEY
import certifi
import os

# ca = certifi.where()





class MongoDBClient:
    client= None
    def __init__(self, database_name =DATABASE_NAME) -> None:
        try:
            if MongoDBClient.client is None:
                mongo_db_url =os.getenv(MONGODB_URL_KEY)
                print(mongo_db_url)
                MongoDBClient.client = pymongo.MongoClient("mongodb+srv://Bhavesh:Bhavesh123@cluster0.zfyclqm.mongodb.net/")
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name

        except Exception as e:
            raise e
