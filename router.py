# router.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Literal, List, Dict, Any
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv
import os

# === Load .env ===
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

# === MongoDB Setup ===
client = MongoClient(MONGO_URI)
db = client["RAG"]
messages = db["messages"]
documents = db["documents"]  # Store document chunks

# === Gemini Setup ===
try:
    import google.generativeai as genai  # type: ignore
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))  # type: ignore
    model = genai.GenerativeModel('gemini-2.0-flash')  # type: ignore
except ImportError:
    model = None
except Exception as e:
    print(f"Error configuring Gemini: {e}")
    model = None

# === Pydantic Models ===
class ChatRequest(BaseModel):
    user_id: str
    question: str

class ChatResponse(BaseModel):
    user_id: str
    question: str
    answer: str
    source: str
    timestamp: datetime

class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "memory"] = Field(
        description="Choose either memory or vectorstore based on the question context"
    )

# === MongoDB Functions ===
def save_message(user_id: str, question: str, answer: str, source: str = "unknown"):
    try:
        result = messages.insert_one({
            "user_id": user_id,
            "question": question,
            "answer": answer,
            "source": source,
            "timestamp": datetime.utcnow()
        })
        print(f"✅ Message inserted with ID: {result.inserted_id}")
    except Exception as e:
        print(f"❌ DB insert error: {e}")

def get_messages(user_id: str) -> List[dict]:
    try:
        return list(messages.find({"user_id": user_id}, {"_id": 0}).sort("timestamp", -1).limit(10))
    except Exception as e:
        print(f"❌ DB fetch error: {e}")
        return []

# === Document Functions ===
def has_user_documents(user_id: str) -> bool:
    """Check if user has uploaded documents"""
    try:
        return documents.count_documents({"user_id": user_id}) > 0
    except Exception as e:
        print(f"❌ Document check error: {e}")
        return False

def get_relevant_chunks(user_id: str, question: str, max_chunks: int = 4) -> List[Dict]:
    """Get relevant document chunks using Gemini for semantic similarity scoring"""
    if not model:
        return []
    
    try:
        # Get all user's document chunks
        user_chunks = list(documents.find(
            {"user_id": user_id},
            {"content": 1, "metadata": 1, "chunk_id": 1}
        ))
        
        if not user_chunks:
            return []
        
        relevant_chunks = []
        
        for chunk in user_chunks:
            content = chunk["content"]
            
            # Skip very short chunks
            if len(content.strip()) < 50:
                continue
            
            # Create relevance scoring prompt
            relevance_prompt = f"""
Rate how relevant this document chunk is for answering the given question.
Respond with only a single number from 0-10, where:
- 0-3: Not relevant
- 4-6: Somewhat relevant  
- 7-8: Very relevant
- 9-10: Extremely relevant

Question: {question}

Document chunk:
{content[:900]}

Relevance score (0-10):"""
            
            try:
                response = model.generate_content(relevance_prompt)  # type: ignore
                score_text = response.text.strip() if hasattr(response, 'text') else "0"
                
                # Extract numeric score (handle various response formats)
                score = 0
                for char in score_text:
                    if char.isdigit():
                        score = int(char)
                        break
                
                # Only include chunks with decent relevance
                if score >= 5:
                    relevant_chunks.append({
                        "content": content,
                        "metadata": chunk.get("metadata", {}),
                        "chunk_id": chunk.get("chunk_id", ""),
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

def build_context_from_chunks(chunks: List[Dict], max_context_length: int = 3500) -> str:
    """Build context string from relevant chunks"""
    if not chunks:
        return ""
    
    context_parts = []
    current_length = 0
    
    for i, chunk in enumerate(chunks, 1):
        content = chunk["content"].strip()
        score = chunk["relevance_score"]
        
        chunk_header = f"\n--- Document Section {i} (Relevance: {score}/10) ---\n"
        chunk_text = f"{chunk_header}{content}\n"
        
        if current_length + len(chunk_text) <= max_context_length:
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
        else:
            # Try to fit partial content if there's meaningful space
            remaining_space = max_context_length - current_length - len(chunk_header)
            if remaining_space > 150:
                partial_content = content[:remaining_space] + "...\n[Content truncated]"
                context_parts.append(f"{chunk_header}{partial_content}\n")
            break
    
    return "".join(context_parts)

# === Routing (Enhanced rule-based) ===
def route_query(question: str, user_id: str) -> RouteQuery:
    """Enhanced routing that checks both keywords and document availability"""
    # Check if user has documents
    if not has_user_documents(user_id):
        return RouteQuery(datasource="memory")
    
    # Keywords that suggest document-based queries
    document_keywords = [
        "document", "pdf", "file", "uploaded", "report", "paper",
        "this document", "the document", "in the text", "according to",
        "what does it say", "summarize", "explain from", "based on"
    ]
    
    question_lower = question.lower()
    if any(keyword in question_lower for keyword in document_keywords):
        return RouteQuery(datasource="vectorstore")
    
    # If user has documents but question doesn't explicitly reference them,
    # still try vectorstore first as it might be relevant
    return RouteQuery(datasource="vectorstore")

# === Memory Response (Enhanced with history) ===
def get_memory_response(question: str, history: List[str], user_id: str) -> str:
    """Enhanced memory response using Gemini with conversation history"""
    if not model:
        return "❌ Gemini AI service is not available for general chat"
    
    try:
        # Build conversation context
        history_context = ""
        if history:
            recent_history = history[:5]  # Last 5 messages
            history_context = "Recent conversation:\n" + "\n".join(recent_history)
        
        # Create prompt for general conversation
        prompt = f"""
You are a helpful AI assistant having a conversation with a user.

{history_context}

Current question: {question}

Please provide a helpful, informative, and engaging response. Be natural and conversational.
"""
        
        response = model.generate_content(prompt)  # type: ignore
        answer = response.text if hasattr(response, 'text') else str(response)
        return answer.strip()
        
    except Exception as e:
        print(f"❌ Memory response error: {e}")
        return f"I'm having trouble processing your question right now. Please try again."

# === Vectorstore Response (Gemini-powered RAG) ===
def get_vectorstore_response(user_id: str, question: str) -> str:
    """Get answer using Gemini-powered RAG with document chunks"""
    if not model:
        return "❌ Gemini AI service is not available"
    
    try:
        # Check if user has documents
        if not has_user_documents(user_id):
            return "❌ No uploaded documents found. Please upload a PDF first."
        
        # Get relevant chunks
        relevant_chunks = get_relevant_chunks(user_id, question)
        
        if not relevant_chunks:
            return "❓ I couldn't find relevant information in your uploaded documents to answer this question. The content might not be related to your query, or you might want to try rephrasing your question."
        
        # Build context from relevant chunks
        context = build_context_from_chunks(relevant_chunks)
        
        # Create comprehensive prompt
        prompt = f"""
You are an AI assistant that answers questions based on provided document content.

Here are the most relevant sections from the documents (ranked by relevance):

{context}

User Question: {question}

Instructions:
- Answer the question using ONLY the provided document content
- Be comprehensive and detailed in your response
- If the document content fully answers the question, provide a complete answer
- If some information is missing from the documents, clearly state what additional information would be needed
- Quote or reference specific sections when making claims
- If the question cannot be answered from the provided documents, clearly state this
- Maintain a helpful and informative tone

Your response:"""
        
        # Get response from Gemini
        response = model.generate_content(prompt)  # type: ignore
        answer = response.text if hasattr(response, 'text') else str(response)
        
        # Add metadata about the search
        chunk_count = len(relevant_chunks)
        max_score = max([chunk["relevance_score"] for chunk in relevant_chunks])
        metadata = f"\n\n📄 *Answer based on {chunk_count} relevant document sections (max relevance: {max_score}/10)*"
        
        return f"{answer.strip()}{metadata}"
        
    except Exception as e:
        print(f"❌ Vectorstore QA error: {e}")
        return f"❌ Error while processing your question: {str(e)}"

# === FastAPI Router ===
router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Enhanced chat endpoint with Gemini-powered routing and responses"""
    if not model:
        raise HTTPException(status_code=503, detail="Gemini AI service is not available")
    
    try:
        # Get user's message history
        msgs = get_messages(request.user_id)
        history = [f"Q: {m['question']} -> A: {m['answer']}" for m in msgs]

        # Route the query
        decision = route_query(request.question, request.user_id)
        
        # Get appropriate response
        if decision.datasource == "vectorstore":
            answer = get_vectorstore_response(request.user_id, request.question)
            source = "gemini_rag"
        else:
            answer = get_memory_response(request.question, history, request.user_id)
            source = "gemini_chat"

        # Save message to database
        save_message(request.user_id, request.question, answer, source)

        return ChatResponse(
            user_id=request.user_id,
            question=request.question,
            answer=answer,
            source=source,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        print(f"❌ Chat error: {e}")
        error_msg = f"Sorry, I encountered an error: {str(e)}"
        save_message(request.user_id, request.question, error_msg, "error")
        
        return ChatResponse(
            user_id=request.user_id,
            question=request.question,
            answer=error_msg,
            source="error",
            timestamp=datetime.utcnow()
        )

@router.get("/messages/{user_id}", response_model=List[ChatResponse])
async def fetch_messages(user_id: str):
    """Fetch user's message history"""
    msgs = get_messages(user_id)
    if not msgs:
        raise HTTPException(status_code=404, detail="No messages found for this user")
    
    return [
        ChatResponse(
            user_id=msg["user_id"],
            question=msg["question"],
            answer=msg["answer"],
            source=msg.get("source", "historical"),
            timestamp=msg["timestamp"]
        )
        for msg in msgs
    ]

@router.delete("/messages/{user_id}")
async def clear_messages(user_id: str):
    """Clear user's message history"""
    try:
        result = messages.delete_many({"user_id": user_id})
        return {"message": f"Deleted {result.deleted_count} messages for user {user_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete messages: {e}")

@router.get("/documents/{user_id}")
async def get_document_stats(user_id: str):
    """Get user's document statistics"""
    try:
        total_chunks = documents.count_documents({"user_id": user_id})
        
        if total_chunks == 0:
            return {
                "user_id": user_id,
                "has_documents": False,
                "total_chunks": 0,
                "total_words": 0,
                "total_chars": 0
            }
        
        # Aggregate statistics
        pipeline = [
            {"$match": {"user_id": user_id}},
            {"$group": {
                "_id": None,
                "total_chunks": {"$sum": 1},
                "total_words": {"$sum": "$metadata.word_count"},
                "total_chars": {"$sum": "$metadata.char_count"}
            }}
        ]
        
        result = list(documents.aggregate(pipeline))
        stats = result[0] if result else {"total_chunks": 0, "total_words": 0, "total_chars": 0}
        
        return {
            "user_id": user_id,
            "has_documents": True,
            **stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting document stats: {e}")

@router.delete("/documents/{user_id}")
async def clear_documents(user_id: str):
    """Clear user's uploaded documents"""
    try:
        result = documents.delete_many({"user_id": user_id})
        return {"message": f"Deleted {result.deleted_count} document chunks for user {user_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete documents: {e}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "gemini_available": model is not None,
        "mongodb_connected": client is not None,
        "features": ["gemini_chat", "gemini_rag", "document_chunking", "semantic_search"]
    }