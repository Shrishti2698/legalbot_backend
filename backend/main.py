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

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
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
        # Change to parent directory to access chroma_db
        os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        vector_store = load_vector_store()
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        print("✅ Vector store loaded successfully")
    except Exception as e:
        print(f"❌ Error loading vector store: {e}")
        print(f"Current directory: {os.getcwd()}")

@app.get("/")
async def root():
    return {"message": "Indian Legal Assistant API is running"}

@app.options("/chat")  # Purpose: Handle browser's CORS preflight check. Browser asks "Can I POST to /chat?" → Server says "OK"
async def chat_options():
    return {"message": "OK"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint - This is where frontend calls the API"""
    if retriever is None:
        raise HTTPException(status_code=503, detail="Vector store not available")
    
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