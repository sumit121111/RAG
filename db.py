# api/db.py (async version)
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
client = AsyncIOMotorClient(MONGO_URI)
db = client["RAG"]
messages = db["messages"]

async def save_message_to_db(message: dict):
    message['timestamp'] = datetime.utcnow()
    await messages.insert_one(message)

async def get_chat_history(user_id: str):
    return await messages.find(
        {"user_id": user_id},
        {"_id": 0, "content": 1, "role": 1, "timestamp": 1}
    ).sort("timestamp", 1).to_list(length=100)
