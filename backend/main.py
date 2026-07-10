from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import sys
import os

# Import utils from same directory
from utils import load_vector_store, create_enhanced_rag_response
from admin_routes import router as admin_router
from auth_routes import router as auth_router

app = FastAPI(title="Indian Legal Assistant API")

# Include routes
app.include_router(auth_router)
app.include_router(admin_router)

# Enable CORS - Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models (JSON Structure)
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    language: str = "English"
    chat_history: List[ChatMessage] = []

class Reference(BaseModel):
    document: str
    content: str

class ChatResponse(BaseModel):
    answer: str
    references: List[Reference]

# Global variable retriever
retriever = None   # Stores the RAG system (vector database searcher). None initially, loaded on startup

@app.on_event("startup")
async def startup_event():
    """Load vector store on startup"""
    global retriever
    try:
        # Import supabase_storage from backend directory
        import sys
        backend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        if backend_path not in sys.path:
            sys.path.insert(0, backend_path)
        
        from supabase_storage import restore_chroma_from_supabase
        
        # Change to backend directory
        os.chdir(backend_path)
        
        # Restore ChromaDB from Supabase if not exists
        restore_chroma_from_supabase()
        
        print("✅ API started. Vector store will load on first request to save memory.")
    except Exception as e:
        print(f"❌ Error during startup: {e}")
        import traceback
        traceback.print_exc()
        print(f"Current directory: {os.getcwd()}")

@app.get("/")
async def root():
    return {"message": "Indian Legal Assistant API is running", "embeddings": "OpenAI"}

@app.get("/reset-db")
async def reset_database():
    """Reset ChromaDB - use this after switching embedding models"""
    import shutil
    try:
        chroma_path = "chroma_db"
        if os.path.exists(chroma_path):
            shutil.rmtree(chroma_path)
            os.makedirs(chroma_path, exist_ok=True)
            
            # Also delete from Supabase
            try:
                import sys
                backend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
                if backend_path not in sys.path:
                    sys.path.insert(0, backend_path)
                
                from supabase_storage import supabase, BUCKET_NAME, CHROMA_ZIP
                supabase.storage.from_(BUCKET_NAME).remove([CHROMA_ZIP])
            except:
                pass
            
            return {"message": "ChromaDB reset successfully. You can now upload documents with new embeddings."}
        else:
            os.makedirs(chroma_path, exist_ok=True)
            return {"message": "ChromaDB directory created."}
    except Exception as e:
        return {"error": str(e)}

@app.get("/sync-db")
async def sync_database():
    """Manually sync ChromaDB to Supabase"""
    try:
        import sys
        backend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        if backend_path not in sys.path:
            sys.path.insert(0, backend_path)
        
        from supabase_storage import sync_chroma_to_supabase
        success = sync_chroma_to_supabase()
        if success:
            return {"message": "ChromaDB synced to Supabase successfully"}
        else:
            return {"error": "Failed to sync ChromaDB to Supabase"}
    except Exception as e:
        return {"error": str(e)}

@app.options("/chat")  # Purpose: Handle browser's CORS preflight check. Browser asks "Can I POST to /chat?" → Server says "OK"
async def chat_options():
    return {"message": "OK"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint - This is where frontend calls the API"""
    global retriever
    
    # Lazy load retriever on first request
    if retriever is None:
        try:
            vector_store = load_vector_store()
            retriever = vector_store.as_retriever(search_kwargs={"k": 5})
            print("✅ Vector store loaded on first request")
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Vector store not available: {str(e)}")
    
    try:
        # Format chat history
        chat_history = ""
        for msg in request.chat_history:
            role = "Human" if msg.role == "user" else "Assistant"
            chat_history += f"{role}: {msg.content}\n\n"
        
        # Get RAG response
        response = create_enhanced_rag_response(
            retriever=retriever,
            question=request.message,
            chat_history=chat_history,
            language=request.language
        )
        
        # Format references
        references = [
            Reference(document=ref["document"], content=ref["content"])
            for ref in response["references"]
        ]
        
        return ChatResponse(  # Return: Send AI answer + legal references back to frontend
            answer=response["answer"],
            references=references
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)