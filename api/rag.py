import os
import json
import hashlib
import tempfile
from io import BytesIO
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv

# LangChain imports for document processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Import with type ignoring for Pylance issues
try:
    import google.generativeai as genai  # type: ignore
except ImportError:
    genai = None  # type: ignore

load_dotenv()

class PDFIngestor:
    """Gemini-powered PDF RAG system with semantic search and chunking"""
    
    def __init__(self, pdfs: List[Union[bytes, BytesIO, bytearray, memoryview]], user_id: str = "default"):
        self.pdfs = pdfs
        self.user_id = user_id
        
        # Configure Gemini
        self.model = None
        if genai:
            try:
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))  # type: ignore
                self.model = genai.GenerativeModel('gemini-2.0-flash')  # type: ignore
            except Exception as e:
                print(f"Error configuring Gemini: {e}")
                self.model = None
        
        if not self.model:
            raise ValueError("Gemini API is not available. Please check your API key.")
        
        # Load and process documents
        self.docs_list = self.load_documents()
        
        # Configure text splitter for optimal chunks
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Larger chunks for better context
            chunk_overlap=150,  # Good overlap for continuity
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        
        # Split documents into chunks
        self.doc_splits = self.text_splitter.split_documents(self.docs_list)
        
        # Process chunks with metadata
        self.processed_chunks = self._process_chunks()
        
        print(f"✅ Processed {len(self.docs_list)} documents into {len(self.processed_chunks)} chunks")
    
    def _extract_bytes_content(self, pdf_data: Union[bytes, BytesIO, bytearray, memoryview]) -> bytes:
        """Extract bytes content from various input types"""
        if isinstance(pdf_data, bytes):
            return pdf_data
        elif isinstance(pdf_data, bytearray):
            return bytes(pdf_data)
        elif isinstance(pdf_data, memoryview):
            return pdf_data.tobytes()
        elif isinstance(pdf_data, BytesIO):
            # Reset position to beginning and read all content
            pdf_data.seek(0)
            return pdf_data.read()
        elif hasattr(pdf_data, "getvalue"):
            # For StringIO/BytesIO objects
            return pdf_data.getvalue()
        elif hasattr(pdf_data, "read"):
            # For file-like objects
            return pdf_data.read()
        else:
            raise TypeError(f"Unsupported PDF data type: {type(pdf_data)}")
    
    def load_documents(self) -> List:
        """Load documents from PDF bytes using PyPDFLoader"""
        docs_list = []
        
        for i, pdf_data in enumerate(self.pdfs):
            try:
                # Extract bytes content using the helper method
                content = self._extract_bytes_content(pdf_data)
                
                # Validate that we have actual content
                if not content or len(content) == 0:
                    print(f"Warning: PDF {i} appears to be empty, skipping...")
                    continue
                
                # Create temporary file for PyPDFLoader
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                    temp_pdf.write(content)
                    temp_pdf.flush()
                    
                    # Load using PyPDFLoader
                    loader = PyPDFLoader(temp_pdf.name)
                    file_docs = loader.load()
                    
                    # Add metadata
                    for doc in file_docs:
                        doc.metadata.update({
                            "user_id": self.user_id,
                            "file_index": i,
                            "file_size": len(content),
                            "processed_at": datetime.utcnow().isoformat()
                        })
                    
                    docs_list.extend(file_docs)
                    print(f"✅ Successfully processed PDF {i} ({len(content)} bytes, {len(file_docs)} pages)")
                
                # Cleanup temporary file
                os.unlink(temp_pdf.name)
                
            except Exception as e:
                print(f"❌ Error processing PDF {i}: {e}")
                continue
        
        return docs_list
    
    def _process_chunks(self) -> List[Dict[str, Any]]:
        """Process document chunks with metadata and content hashes"""
        processed_chunks = []
        
        for i, chunk in enumerate(self.doc_splits):
            # Skip very short chunks
            if len(chunk.page_content.strip()) < 50:
                continue
            
            chunk_data = {
                "chunk_id": f"{self.user_id}_{i}",
                "content": chunk.page_content.strip(),
                "metadata": {
                    **chunk.metadata,
                    "chunk_index": i,
                    "word_count": len(chunk.page_content.split()),
                    "char_count": len(chunk.page_content),
                    "content_hash": hashlib.md5(chunk.page_content.encode()).hexdigest()
                },
                "created_at": datetime.utcnow()
            }
            
            processed_chunks.append(chunk_data)
        
        return processed_chunks
    
    def get_relevant_chunks(self, question: str, max_chunks: int = 4) -> List[Dict]:
        """Get relevant document chunks using Gemini for semantic similarity scoring"""
        if not self.processed_chunks:
            return []
        
        try:
            relevant_chunks = []
            
            for chunk in self.processed_chunks:
                content = chunk["content"]
                
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
                    response = self.model.generate_content(relevance_prompt)  # type: ignore
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
                            **chunk,
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
    
    def _build_context(self, chunks: List[Dict], max_context_length: int = 3500) -> str:
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
    
    def get_answer(self, question: str) -> str:
        """Get answer to question using Gemini with RAG context"""
        if not self.model:
            return "❌ Gemini AI service is not available"
        
        try:
            # Get relevant chunks
            relevant_chunks = self.get_relevant_chunks(question)
            
            if not relevant_chunks:
                return "❓ I couldn't find relevant information in the uploaded documents to answer your question. The content might not be related to your query, or you might want to try rephrasing your question."
            
            # Build context from relevant chunks
            context = self._build_context(relevant_chunks)
            
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
            response = self.model.generate_content(prompt)  # type: ignore
            answer = response.text if hasattr(response, 'text') else str(response)
            
            # Add metadata about the search
            chunk_count = len(relevant_chunks)
            max_score = max([chunk["relevance_score"] for chunk in relevant_chunks])
            metadata = f"\n\n📄 *Answer based on {chunk_count} relevant document sections (max relevance: {max_score}/10)*"
            
            return f"{answer.strip()}{metadata}"
            
        except Exception as e:
            return f"❌ Error processing your question: {str(e)}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the processed documents"""
        if not self.processed_chunks:
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "total_words": 0,
                "total_characters": 0,
                "average_chunk_size": 0
            }
        
        total_words = sum(chunk["metadata"]["word_count"] for chunk in self.processed_chunks)
        total_chars = sum(chunk["metadata"]["char_count"] for chunk in self.processed_chunks)
        
        return {
            "total_documents": len(self.docs_list),
            "total_chunks": len(self.processed_chunks),
            "total_words": total_words,
            "total_characters": total_chars,
            "average_chunk_size": total_words // len(self.processed_chunks) if self.processed_chunks else 0
        }
    
    def search_chunks(self, query: str, min_score: int = 6) -> List[Dict]:
        """Search for chunks matching a query with minimum relevance score"""
        relevant_chunks = self.get_relevant_chunks(query, max_chunks=10)
        return [chunk for chunk in relevant_chunks if chunk["relevance_score"] >= min_score]

# Example usage and testing
if __name__ == "__main__":
    # Example of how to use the PDFIngestor
    print("🤖 Gemini-powered PDF RAG System")
    print("=" * 50)
    
    # This would typically be called with actual PDF bytes
    # For testing, you'd pass actual PDF file bytes:
    # with open('sample.pdf', 'rb') as f:
    #     pdf_bytes = f.read()
    #     ingestor = PDFIngestor([pdf_bytes], user_id="test_user")
    #     answer = ingestor.get_answer("What is this document about?")
    #     print(f"Answer: {answer}")
    
    print("Ready to process PDF files with Gemini AI!")
    print("Usage:")
    print("  ingestor = PDFIngestor(pdf_bytes_list, user_id='your_user_id')")
    print("  answer = ingestor.get_answer('Your question here')")
    print("  stats = ingestor.get_stats()")
