# Backend Setup Guide

## How the API Works

### 1. Backend Structure
```
backend/
├── main.py          # FastAPI app with routes
├── start.bat        # Startup script
└── requirements.txt # Dependencies
```

### 2. API Route Flow

**Frontend → Backend Communication:**

1. **Frontend sends request** (main.js line ~80):
   ```javascript
   fetch('http://localhost:8000/chat', {
     method: 'POST',
     body: JSON.stringify({
       message: "What is Article 21?",
       language: "English", 
       chat_history: []
     })
   })
   ```

2. **Backend receives at `/chat` endpoint** (main.py line ~40):
   ```python
   @app.post("/chat")
   async def chat_endpoint(request: ChatRequest):
   ```

3. **Backend processes with RAG** (main.py line ~50):
   ```python
   response = create_enhanced_rag_response(
       retriever=retriever,
       question=request.message,
       chat_history=chat_history,
       language=request.language
   )
   ```

4. **Backend returns JSON response**:
   ```json
   {
     "answer": "Article 21 of the Indian Constitution...",
     "references": [
       {
         "document": "Constitution of India",
         "content": "No person shall be deprived..."
       }
     ]
   }
   ```

## Quick Start

### Terminal 1 - Start Backend:
```bash
cd backend
conda activate legal-chatbot
python main.py
```
Backend runs on: http://localhost:8000

### Terminal 2 - Start Frontend:
```bash
cd legal-advisor-frontend  
npm run dev
```
Frontend runs on: http://localhost:5173

## API Endpoints

- `GET /` - Health check
- `POST /chat` - Main chat endpoint (used by frontend)

## Troubleshooting

1. **"Vector store not available"** → Run `python ingest.py` first
2. **CORS errors** → Backend allows localhost:5173
3. **Connection refused** → Ensure backend is running on port 8000