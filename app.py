import streamlit as st
import requests
import json
from datetime import datetime
import time

# === Configuration ===
BACKEND_URL = "http://localhost:8000"
USER_ID = "demo_user"

# === Page Config ===
st.set_page_config(
    page_title="PDF Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Custom CSS for better styling ===
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        background-color: #f8f9fa;
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
    }
    
    .bot-message {
        background-color: #f3e5f5;
        border-left-color: #9c27b0;
    }
    
    .status-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9ff;
    }
</style>
""", unsafe_allow_html=True)

# === Initialize Session State ===
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False

# === Helper Functions ===
def check_backend():
    try:
        response = requests.get(f"{BACKEND_URL}/", timeout=5)
        return response.status_code == 200
    except:
        return False

def upload_pdfs(files):
    try:
        files_data = [("files", (f.name, f, "application/pdf")) for f in files]
        response = requests.post(
            f"{BACKEND_URL}/upload",
            data={"user_id": USER_ID},
            files=files_data,
            timeout=60
        )
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"Upload failed: {e}")
        return None

def send_message(message):
    try:
        response = requests.post(
            f"{BACKEND_URL}/chat",
            json={"user_id": USER_ID, "message": message},
            timeout=30
        )
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        return {"answer": f"Error: {e}", "source": "error"}

def get_history():
    try:
        response = requests.get(f"{BACKEND_URL}/history/{USER_ID}", timeout=10)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def clear_session():
    try:
        response = requests.delete(f"{BACKEND_URL}/session/{USER_ID}", timeout=10)
        return response.status_code == 200
    except:
        return False

# === Header ===
st.markdown("""
<div class="main-header">
    <h1>🤖 AI-Powered PDF Chat Assistant</h1>
    <p>Upload PDFs and chat with your documents using Google Gemini AI</p>
</div>
""", unsafe_allow_html=True)

# === Sidebar ===
with st.sidebar:
    st.markdown("### 📁 Document Upload")
    
    # Backend status
    if check_backend():
        st.success("🟢 Backend Connected")
    else:
        st.error("🔴 Backend Offline")
        st.stop()
    
    # File upload
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="Upload one or more PDF files to chat about"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Process", type="primary", use_container_width=True):
            if uploaded_files:
                with st.spinner("Processing PDFs..."):
                    result = upload_pdfs(uploaded_files)
                    if result:
                        st.session_state.pdf_uploaded = True
                        st.success(f"✅ Processed {result['files_processed']} files")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"📄 Successfully processed {result['files_processed']} PDF file(s). You can now ask questions about the content!"
                        })
                        st.rerun()
                    else:
                        st.error("❌ Processing failed")
            else:
                st.warning("Select PDF files first")
    
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            if clear_session():
                st.session_state.pdf_uploaded = False
                st.session_state.messages = []
                st.success("Session cleared!")
                st.rerun()
    
    # Status
    st.markdown("---")
    st.markdown("### 📊 Status")
    
    if st.session_state.pdf_uploaded:
        st.markdown("""
        <div class="status-card">
            <h4>📄 PDFs Loaded</h4>
            <p>Ready to answer questions!</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-card">
            <h4>💭 General Chat Mode</h4>
            <p>Upload PDFs for document-specific answers</p>
        </div>
        """, unsafe_allow_html=True)
    
    # History info
    history = get_history()
    if history:
        st.metric("💬 Total Messages", len(history.get('database_history', [])))
        st.metric("⚡ Cached Messages", len(history.get('cached_history', [])))

# === Main Chat Interface ===
st.markdown("### 💬 Chat Interface")

# Chat container
chat_container = st.container()

with chat_container:
    # Display messages
    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 You:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message bot-message">
                <strong>🤖 Assistant:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Type your message here...", key="chat_input"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get response
    with st.spinner("🤔 Thinking..."):
        response = send_message(prompt)
        
        if response:
            answer = response.get("answer", "No response received")
            source = response.get("source", "unknown")
            
            # Add source indicator
            source_emoji = {
                "pdf_rag": "📄",
                "general_chat": "💭",
                "error": "❌"
            }.get(source, "🤖")
            
            formatted_answer = f"{answer}\n\n<small>{source_emoji} *Source: {source}*</small>"
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": formatted_answer
            })
    
    st.rerun()

# === Footer ===
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🚀 <strong>Powered by Google Gemini AI</strong></p>
    <p>📄 Upload • 💬 Chat • 🔍 Discover</p>
</div>
""", unsafe_allow_html=True)

# === Quick Actions ===
if not st.session_state.messages:
    st.markdown("### 🚀 Quick Start")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Upload a PDF first", use_container_width=True):
            st.info("👆 Use the sidebar to upload PDF files")
    
    with col2:
        if st.button("💬 Start general chat", use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": "Hello! How can you help me today?"
            })
            st.rerun()
    
    with col3:
        if st.button("❓ What can you do?", use_container_width=True):
            st.session_state.messages.append({
                "role": "assistant",
                "content": """
                🤖 **I can help you with:**
                
                📄 **PDF Analysis**: Upload PDFs and ask questions about their content
                💬 **General Chat**: Have conversations on any topic
                🔍 **Smart Search**: Find specific information in your documents
                📊 **Data Extraction**: Pull key insights from uploaded files
                
                **Get started by uploading a PDF or just start chatting!**
                """
            })
            st.rerun()

# === Debug panel (collapsible) ===
with st.expander("🔧 Debug Information"):
    st.write("**Session State:**")
    st.json({
        "messages_count": len(st.session_state.messages),
        "pdf_uploaded": st.session_state.pdf_uploaded,
        "backend_url": BACKEND_URL,
        "user_id": USER_ID
    })
    
    if st.button("🔄 Refresh History"):
        history = get_history()
        if history:
            st.json(history)
        else:
            st.write("No history available")