import os
import json
import redis
import PyPDF2
from io import BytesIO
from datetime import datetime
from typing import List, Optional, Union, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient
from dotenv import load_dotenv

# Import with type ignoring for Pylance issues
try:
    import google.generativeai as genai  # type: ignore
except ImportError:
    genai = None  # type: ignore

load_dotenv()

# Initialize services
app = FastAPI(title="Minimal RAG Chat")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Configure Gemini
model = None
if genai:
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))  # type: ignore
        model = genai.GenerativeModel('gemini-2.0-flash')  # type: ignore
    except Exception as e:
        print(f"Error configuring Gemini: {e}")
        model = None

# MongoDB
mongo_client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
db = mongo_client["ragchat"]
messages = db["messages"]

# Redis with proper typing
redis_client: Optional[redis.Redis] = None
try:
    # Check if REDIS_URL is provided first (for cloud services)
    redis_url = os.getenv("REDIS_URL")
    
    if redis_url:
        # Use Redis URL (works with Redis Cloud, Upstash, AWS ElastiCache, etc.)
        redis_client = redis.from_url(redis_url, decode_responses=True)
    else:
        # Fallback to individual parameters
        redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"), 
            port=int(os.getenv("REDIS_PORT", "6379")),
            password=os.getenv("REDIS_PASSWORD"),  # Add password support
            db=int(os.getenv("REDIS_DB", "0")),    # Add database selection
            ssl=os.getenv("REDIS_SSL", "false").lower() == "true",  # Add SSL support
            decode_responses=True
        )
    
    redis_client.ping()
    print("✅ Redis connected successfully")
except Exception as e:
    print(f"❌ Redis connection failed: {e}")
    redis_client = None

# In-memory storage for PDF content
pdf_storage: dict[str, str] = {}

class ChatMessage(BaseModel):
    user_id: str
    message: str

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes"""
    try:
        reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        return f"Error extracting PDF: {str(e)}"

def save_to_db(user_id: str, question: str, answer: str) -> None:
    """Save chat to MongoDB"""
    try:
        messages.insert_one({
            "user_id": user_id,
            "question": question,
            "answer": answer,
            "timestamp": datetime.utcnow()
        })
    except Exception as e:
        print(f"DB Error: {e}")

def cache_message(user_id: str, message: str) -> None:
    """Cache message in Redis"""
    if redis_client:
        try:
            redis_client.lpush(f"history:{user_id}", message)
            redis_client.ltrim(f"history:{user_id}", 0, 19)  # Keep last 20 messages
            redis_client.expire(f"history:{user_id}", 3600)
        except Exception as e:
            print(f"Redis Error: {e}")

def get_cached_history(user_id: str) -> List[str]:
    """Get cached chat history"""
    if redis_client:
        try:
            # Use a safer approach - get one item at a time
            history = []
            key = f"history:{user_id}"
            
            # Get the length of the list first
            list_length = redis_client.llen(key)
            if list_length and isinstance(list_length, int) and list_length > 0:
                # Get items one by one to avoid iteration issues
                for i in range(min(list_length, 20)):  # Limit to 20 items
                    item = redis_client.lindex(key, i)
                    if item is not None:
                        history.append(str(item))
            
            return history
        except Exception as e:
            print(f"Redis history error: {e}")
            return []
    return []

@app.get("/")
async def health():
    return {
        "status": "healthy", 
        "message": "Minimal RAG Chat API",
        "gemini_available": model is not None,
        "redis_available": redis_client is not None
    }

@app.post("/upload")
async def upload_pdfs(user_id: str = Form(...), files: List[UploadFile] = File(...)):
    """Upload and process PDF files"""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    combined_text = ""
    processed_files = 0
    
    for file in files:
        # Safely check filename
        filename = getattr(file, 'filename', None)
        if not filename or not filename.endswith('.pdf'):
            raise HTTPException(
                status_code=400, 
                detail=f"File {filename or 'unknown'} is not a PDF"
            )
        
        try:
            pdf_bytes = await file.read()
            text = extract_text_from_pdf(pdf_bytes)
            combined_text += f"\n--- {filename} ---\n{text}\n"
            processed_files += 1
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Error processing {filename}: {str(e)}"
            )
    
    # Store PDF content for this user
    pdf_storage[user_id] = combined_text
    
    # Cache upload info
    cache_message(user_id, f"SYSTEM: Uploaded {processed_files} PDF(s)")
    
    return {
        "message": f"Successfully processed {processed_files} PDF(s)",
        "files_processed": processed_files,
        "total_text_length": len(combined_text)
    }

@app.post("/chat")
async def chat(input: ChatMessage):
    """Chat with PDFs or general knowledge"""
    if not model:
        raise HTTPException(
            status_code=503, 
            detail="Gemini AI service is not available"
        )
    
    user_id = input.user_id
    question = input.message.strip()
    
    if not question:
        raise HTTPException(status_code=400, detail="Empty message")
    
    try:
        # Check if user has uploaded PDFs
        if user_id in pdf_storage:
            # RAG with PDF content
            pdf_content = pdf_storage[user_id]
            prompt = f"""
Based on the following document content, answer the user's question. If the answer is not in the documents, say so clearly.

DOCUMENT CONTENT:
{pdf_content[:4000]}  

USER QUESTION: {question}

Please provide a clear, accurate answer based on the document content.
"""
            response = model.generate_content(prompt)  # type: ignore
            answer = response.text if hasattr(response, 'text') else str(response)
            source = "pdf_rag"
        else:
            # General chat with history
            history = get_cached_history(user_id)
            history_context = "\n".join(history[-5:]) if history else "No previous conversation"
            
            prompt = f"""
Previous conversation context:
{history_context}

Current question: {question}

Please provide a helpful response.
"""
            response = model.generate_content(prompt)  # type: ignore
            answer = response.text if hasattr(response, 'text') else str(response)
            source = "general_chat"
        
        # Save to database
        save_to_db(user_id, question, answer)
        
        # Cache the conversation
        cache_message(user_id, f"USER: {question}")
        cache_message(user_id, f"BOT: {answer}")
        
        return {
            "user_id": user_id,
            "question": question,
            "answer": answer,
            "source": source,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        error_msg = f"Sorry, I encountered an error processing your request: {str(e)}"
        save_to_db(user_id, question, error_msg)
        return {
            "user_id": user_id,
            "question": question,
            "answer": error_msg,
            "source": "error",
            "timestamp": datetime.utcnow()
        }

@app.get("/history/{user_id}")
async def get_history(user_id: str):
    """Get chat history"""
    try:
        # Get from MongoDB
        db_history = list(messages.find(
            {"user_id": user_id}, 
            {"_id": 0}
        ).sort("timestamp", -1).limit(10))
        
        # Get from Redis
        redis_history = get_cached_history(user_id)
        
        return {
            "user_id": user_id,
            "database_history": db_history,
            "cached_history": redis_history,
            "has_pdf": user_id in pdf_storage
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/session/{user_id}")
async def clear_session(user_id: str):
    """Clear user session"""
    try:
        # Remove PDF content
        pdf_storage.pop(user_id, None)
        
        # Clear Redis cache
        if redis_client:
            redis_client.delete(f"history:{user_id}")
        
        return {"message": f"Session cleared for {user_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)