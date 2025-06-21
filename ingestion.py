import os
import json
import redis
import tempfile
import hashlib
from io import BytesIO
from datetime import datetime
from typing import List, Optional, Union, Any, Dict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient
from dotenv import load_dotenv

# LangChain imports (minimal, no HuggingFace)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Import with type ignoring for Pylance issues
try:
    import google.generativeai as genai  # type: ignore
except ImportError:
    genai = None  # type: ignore

load_dotenv()

# Initialize services
app = FastAPI(title="Gemini-Powered RAG Chat")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Configure Gemini
model = None
embedding_model = None
if genai:
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))  # type: ignore
        model = genai.GenerativeModel('gemini-2.0-flash')  # type: ignore
        embedding_model = genai.GenerativeModel('gemini-2.0-flash')  # type: ignore
    except Exception as e:
        print(f"Error configuring Gemini: {e}")
        model = None
        embedding_model = None

# MongoDB
mongo_client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
db = mongo_client["ragchat"]
messages = db["messages"]
documents = db["documents"]  # Store document chunks with metadata

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

class PDFProcessor:
    """Gemini-powered PDF processor with text chunking and semantic search"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        
        # Text splitter configuration
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Larger chunks for better context
            chunk_overlap=200,  # Good overlap for continuity
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
    
    def process_pdfs(self, pdf_files: List[bytes]) -> Dict[str, Any]:
        """Process PDF files and store chunks in MongoDB"""
        try:
            # Clear existing documents for this user
            documents.delete_many({"user_id": self.user_id})
            
            # Load documents from bytes
            docs_list = self._load_documents_from_bytes(pdf_files)
            
            if not docs_list:
                return {"success": False, "error": "No documents loaded"}
            
            # Split documents into chunks
            doc_splits = self.text_splitter.split_documents(docs_list)
            
            # Store chunks in MongoDB with metadata
            stored_chunks = 0
            for i, chunk in enumerate(doc_splits):
                chunk_data = {
                    "user_id": self.user_id,
                    "chunk_id": f"{self.user_id}_{i}",
                    "content": chunk.page_content,
                    "metadata": {
                        **chunk.metadata,
                        "chunk_index": i,
                        "word_count": len(chunk.page_content.split()),
                        "char_count": len(chunk.page_content)
                    },
                    "created_at": datetime.utcnow(),
                    "content_hash": hashlib.md5(chunk.page_content.encode()).hexdigest()
                }
                
                documents.insert_one(chunk_data)
                stored_chunks += 1
            
            return {
                "success": True,
                "documents_processed": len(docs_list),
                "chunks_created": stored_chunks,
                "total_text_length": sum(len(doc.page_content) for doc in docs_list)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _load_documents_from_bytes(self, pdf_files: List[bytes]) -> List:
        """Load documents from PDF bytes using PyPDFLoader"""
        docs = []
        
        for i, pdf_bytes in enumerate(pdf_files):
            try:
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
                    temp.write(pdf_bytes)
                    temp.flush()
                    
                    # Load using PyPDFLoader
                    loader = PyPDFLoader(temp.name)
                    file_docs = loader.load()
                    
                    # Add metadata
                    for doc in file_docs:
                        doc.metadata.update({
                            "user_id": self.user_id,
                            "file_index": i,
                            "processed_at": datetime.utcnow().isoformat()
                        })
                    
                    docs.extend(file_docs)
                
                # Cleanup
                os.unlink(temp.name)
                
            except Exception as e:
                print(f"Error processing PDF {i}: {e}")
                continue
        
        return docs
    
    def get_relevant_chunks(self, question: str, max_chunks: int = 5) -> List[Dict]:
        """Get relevant document chunks using semantic similarity with Gemini"""
        try:
            # Get all user's document chunks
            user_chunks = list(documents.find(
                {"user_id": self.user_id},
                {"content": 1, "metadata": 1, "chunk_id": 1}
            ))
            
            if not user_chunks:
                return []
            
            # Use Gemini to score relevance of each chunk
            relevant_chunks = []
            
            for chunk in user_chunks:
                content = chunk["content"]
                
                # Skip very short chunks
                if len(content.strip()) < 50:
                    continue
                
                # Ask Gemini to score relevance (0-10)
                relevance_prompt = f"""
Rate how relevant this document chunk is to answering the question on a scale of 0-10.
Only respond with a single number (0-10).

Question: {question}

Document chunk: {content[:800]}...

Relevance score (0-10):"""
                
                try:
                    if model:
                        response = model.generate_content(relevance_prompt)  # type: ignore
                        score_text = response.text.strip() if hasattr(response, 'text') else "0"
                        
                        # Extract numeric score
                        score = 0
                        for char in score_text:
                            if char.isdigit():
                                score = int(char)
                                break
                        
                        if score >= 6:  # Only include reasonably relevant chunks
                            relevant_chunks.append({
                                "content": content,
                                "metadata": chunk["metadata"],
                                "chunk_id": chunk["chunk_id"],
                                "relevance_score": score
                            })
                
                except Exception as e:
                    print(f"Error scoring chunk relevance: {e}")
                    continue
            
            # Sort by relevance score and return top chunks
            relevant_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)
            return relevant_chunks[:max_chunks]
            
        except Exception as e:
            print(f"Error getting relevant chunks: {e}")
            return []
    
    def get_context_from_chunks(self, chunks: List[Dict], max_length: int = 4000) -> str:
        """Combine relevant chunks into context string"""
        if not chunks:
            return ""
        
        context_parts = []
        current_length = 0
        
        for chunk in chunks:
            content = chunk["content"].strip()
            chunk_info = f"[Document Section - Score: {chunk['relevance_score']}/10]\n{content}"
            
            if current_length + len(chunk_info) <= max_length:
                context_parts.append(chunk_info)
                current_length += len(chunk_info)
            else:
                # Add partial content if there's meaningful space left
                remaining_space = max_length - current_length
                if remaining_space > 200:
                    partial_content = chunk_info[:remaining_space] + "...\n[Content truncated]"
                    context_parts.append(partial_content)
                break
        
        return "\n\n" + "="*50 + "\n\n".join(context_parts) + "\n" + "="*50 + "\n"
    
    def has_documents(self) -> bool:
        """Check if user has uploaded documents"""
        return documents.count_documents({"user_id": self.user_id}) > 0
    
    def clear_documents(self) -> bool:
        """Clear user's documents"""
        try:
            result = documents.delete_many({"user_id": self.user_id})
            return result.deleted_count > 0 or True
        except Exception as e:
            print(f"Error clearing documents: {e}")
            return False
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about user's documents"""
        try:
            total_chunks = documents.count_documents({"user_id": self.user_id})
            
            if total_chunks == 0:
                return {"total_chunks": 0, "total_words": 0, "total_chars": 0}
            
            # Aggregate statistics
            pipeline = [
                {"$match": {"user_id": self.user_id}},
                {"$group": {
                    "_id": None,
                    "total_chunks": {"$sum": 1},
                    "total_words": {"$sum": "$metadata.word_count"},
                    "total_chars": {"$sum": "$metadata.char_count"}
                }}
            ]
            
            result = list(documents.aggregate(pipeline))
            return result[0] if result else {"total_chunks": 0, "total_words": 0, "total_chars": 0}
            
        except Exception as e:
            print(f"Error getting document stats: {e}")
            return {"total_chunks": 0, "total_words": 0, "total_chars": 0}

# Global storage for PDF processors (per user)
pdf_processors: Dict[str, PDFProcessor] = {}

class ChatMessage(BaseModel):
    user_id: str
    message: str

def get_or_create_processor(user_id: str) -> PDFProcessor:
    """Get existing processor or create new one for user"""
    if user_id not in pdf_processors:
        pdf_processors[user_id] = PDFProcessor(user_id)
    
    return pdf_processors[user_id]

def save_to_db(user_id: str, question: str, answer: str, source: str = "unknown") -> None:
    """Save chat to MongoDB"""
    try:
        messages.insert_one({
            "user_id": user_id,
            "question": question,
            "answer": answer,
            "source": source,
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
            history = []
            key = f"history:{user_id}"
            
            list_length = redis_client.llen(key)
            if list_length and isinstance(list_length, int) and list_length > 0:
                for i in range(min(list_length, 20)):
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
        "message": "Gemini-Powered RAG Chat API",
        "gemini_available": model is not None,
        "redis_available": redis_client is not None,
        "features": ["gemini_embeddings", "semantic_search", "document_chunking", "relevance_scoring"]
    }

@app.post("/upload")
async def upload_pdfs(user_id: str = Form(...), files: List[UploadFile] = File(...)):
    """Upload and process PDF files with Gemini-powered processing"""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    # Validate files
    pdf_bytes_list = []
    processed_files = 0
    
    for file in files:
        filename = getattr(file, 'filename', None)
        if not filename or not filename.endswith('.pdf'):
            raise HTTPException(
                status_code=400, 
                detail=f"File {filename or 'unknown'} is not a PDF"
            )
        
        try:
            pdf_bytes = await file.read()
            pdf_bytes_list.append(pdf_bytes)
            processed_files += 1
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Error reading {filename}: {str(e)}"
            )
    
    # Get or create processor for user
    processor = get_or_create_processor(user_id)
    
    # Process PDFs
    result = processor.process_pdfs(pdf_bytes_list)
    
    if not result["success"]:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing PDFs: {result.get('error', 'Unknown error')}"
        )
    
    # Cache upload info
    cache_message(user_id, f"SYSTEM: Uploaded and processed {processed_files} PDF(s) with Gemini")
    
    return {
        "message": f"Successfully processed {processed_files} PDF(s) with Gemini AI",
        "files_processed": processed_files,
        "documents_processed": result["documents_processed"],
        "chunks_created": result["chunks_created"],
        "total_text_length": result["total_text_length"]
    }

@app.post("/chat")
async def chat(input: ChatMessage):
    """Enhanced chat with Gemini-powered semantic search"""
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
        # Get processor for user
        processor = get_or_create_processor(user_id)
        
        # Check if user has uploaded documents
        if processor.has_documents():
            # RAG with Gemini-powered semantic search
            relevant_chunks = processor.get_relevant_chunks(question)
            
            if relevant_chunks:
                # Get context from relevant chunks
                context = processor.get_context_from_chunks(relevant_chunks)
                
                # Create enhanced prompt with context
                prompt = f"""
You are an AI assistant that answers questions based on uploaded document content. 

Here are the most relevant sections from the user's documents (ranked by relevance):

{context}

User Question: {question}

Instructions:
- Answer the question using the provided document content
- Be comprehensive and detailed in your response
- If the document content fully answers the question, provide a complete answer
- If some information is missing, clearly state what additional information would be needed
- Cite relevant sections when making specific claims
- If the question cannot be answered from the documents, clearly state this

Your response:"""
                
                response = model.generate_content(prompt)  # type: ignore
                final_answer = response.text if hasattr(response, 'text') else str(response)
                
                # Add relevance info
                chunk_count = len(relevant_chunks)
                max_score = max([chunk["relevance_score"] for chunk in relevant_chunks])
                final_answer += f"\n\n📄 *Based on {chunk_count} relevant document sections (max relevance: {max_score}/10)*"
                
                source = "gemini_rag"
            else:
                # No relevant context found
                final_answer = "❓ I couldn't find relevant information in your uploaded documents to answer this question. The content might not be related to your query, or you might want to try rephrasing your question."
                source = "no_context"
        else:
            # General chat with history
            history = get_cached_history(user_id)
            history_context = "\n".join(history[-5:]) if history else "No previous conversation"
            
            prompt = f"""
You are a helpful AI assistant. Here's the recent conversation context:

{history_context}

Current question: {question}

Please provide a helpful and informative response.
"""
            response = model.generate_content(prompt)  # type: ignore
            final_answer = response.text if hasattr(response, 'text') else str(response)
            source = "general_chat"
        
        # Save to database
        save_to_db(user_id, question, final_answer, source)
        
        # Cache the conversation
        cache_message(user_id, f"USER: {question}")
        cache_message(user_id, f"BOT: {final_answer}")
        
        return {
            "user_id": user_id,
            "question": question,
            "answer": final_answer,
            "source": source,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        error_msg = f"Sorry, I encountered an error processing your request: {str(e)}"
        save_to_db(user_id, question, error_msg, "error")
        return {
            "user_id": user_id,
            "question": question,
            "answer": error_msg,
            "source": "error",
            "timestamp": datetime.utcnow()
        }

@app.get("/history/{user_id}")
async def get_history(user_id: str):
    """Get chat history with document processing info"""
    try:
        # Get from MongoDB
        db_history = list(messages.find(
            {"user_id": user_id}, 
            {"_id": 0}
        ).sort("timestamp", -1).limit(10))
        
        # Get from Redis
        redis_history = get_cached_history(user_id)
        
        # Get document statistics
        processor = get_or_create_processor(user_id)
        doc_stats = processor.get_document_stats()
        
        return {
            "user_id": user_id,
            "database_history": db_history,
            "cached_history": redis_history,
            "document_stats": doc_stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/session/{user_id}")
async def clear_session(user_id: str):
    """Clear user session including documents"""
    try:
        # Clear documents
        processor = get_or_create_processor(user_id)
        processor.clear_documents()
        
        # Remove processor from memory
        if user_id in pdf_processors:
            del pdf_processors[user_id]
        
        # Clear Redis cache
        if redis_client:
            redis_client.delete(f"history:{user_id}")
        
        return {"message": f"Session and documents cleared for {user_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{user_id}")
async def get_document_info(user_id: str):
    """Get information about user's uploaded documents"""
    try:
        processor = get_or_create_processor(user_id)
        stats = processor.get_document_stats()
        
        # Get recent document chunks
        recent_chunks = list(documents.find(
            {"user_id": user_id},
            {"content": 1, "metadata": 1, "created_at": 1}
        ).sort("created_at", -1).limit(5))
        
        return {
            "user_id": user_id,
            "stats": stats,
            "recent_chunks": [
                {
                    "preview": chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"],
                    "word_count": chunk["metadata"].get("word_count", 0),
                    "created_at": chunk["created_at"]
                }
                for chunk in recent_chunks
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)