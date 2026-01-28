# Legal Advisor Backend API

FastAPI-based backend for the Legal Advisor system. Provides RAG-based legal question answering using Chroma vector database.

## Setup

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. **Clone the repository**
   ```bash
   git clone <backend-repo-url>
   cd backend
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env and fill in your API keys and configurations
   ```

5. **Run the server**
   ```bash
   python main.py
   # or
   uvicorn main:app --reload
   ```

The API will be available at `http://localhost:8000`

## API Endpoints

- `POST /chat` - Send a legal question and get RAG-based response
- `GET /` - Health check

## Environment Variables

See `.env.example` for all available configuration options.

## Deployment

For production deployment:
1. Set `DEBUG=false` in `.env`
2. Use a production ASGI server (Gunicorn + Uvicorn recommended)
3. Configure proper CORS origins
4. Use strong JWT secrets
