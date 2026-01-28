# Legal Assistant API Documentation

Base URL: `http://localhost:8000`

---

## 1. PDF Upload & Processing APIs

### 1.1 Upload and Process PDF

**Endpoint:** `POST /upload`

**What it does:**
This API handles the complete document ingestion pipeline:
1. Receives PDF file upload
2. Saves PDF to appropriate folder in `data/` directory based on document type
3. Extracts text from PDF using PyPDF library
4. Splits text into chunks using RecursiveCharacterTextSplitter with configurable size/overlap
5. Generates embeddings for each chunk using HuggingFace sentence-transformers model
6. Stores embeddings in ChromaDB vector store with metadata (source path, page number)
7. Returns processing statistics

**Use Case:** When a new law (e.g., law2.pdf) is enacted and needs to be added to the legal knowledge base

**Request:**
```
POST http://localhost:8000/upload
Content-Type: multipart/form-data

Form Data:
- file: law2.pdf (required) - The PDF file to upload
- document_type: bns (required) - Category for organizing in data folder
- chunk_size: 1000 (optional) - Number of characters per chunk
- chunk_overlap: 200 (optional) - Overlapping characters between chunks
```

**document_type Options:**
- `constitution` → saves to `data/constitutionOfIndia.pdf`
- `ipc` → saves to `data/penal_code_India.pdf`
- `crpc` → saves to `data/` (criminal procedure)
- `bns` → saves to `data/bns_data/`
- `bnss` → saves to `data/bns_data/`
- `bsa` → saves to `data/bns_data/`
- `supreme_court` → saves to `data/`
- `high_court` → saves to `data/`
- `other` → saves to `data/`

**Response:**
```json
{
  "status": "success",
  "message": "Document processed and added to vector store",
  "filename": "law2.pdf",
  "saved_path": "data/bns_data/law2.pdf",
  "processing_stats": {
    "pages_extracted": 250,
    "total_characters": 500000,
    "chunks_created": 150,
    "embeddings_generated": 150,
    "vectors_added_to_chromadb": 150,
    "embedding_dimension": 384
  },
  "chunk_config_used": {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "separator": "\n\n"
  },
  "processing_time_seconds": 45.2
}
```

**Error Response:**
```json
{
  "status": "error",
  "error_code": "INVALID_FILE_TYPE",
  "message": "Only PDF files are allowed",
  "details": "File extension must be .pdf"
}
```

---

### 1.2 List All Documents

**Endpoint:** `GET /documents`

**What it does:**
- Scans the `data/` folder and all subfolders for PDF files
- For each PDF, checks if it exists in ChromaDB by querying metadata
- Counts how many chunks/embeddings exist for each document
- Returns file information including size, modification date, and vector store status

**Use Case:** View all legal documents in the system and verify which ones are indexed

**Request:**
```
GET http://localhost:8000/documents?folder=bns_data
```

**Query Parameters:**
- `folder` (optional) - Filter by subfolder name (e.g., "bns_data", "language_const")

**Response:**
```json
{
  "status": "success",
  "total_documents": 35,
  "documents": [
    {
      "filename": "bns_2024.pdf",
      "full_path": "data/bns_data/bns_2024.pdf",
      "size_mb": 2.5,
      "size_bytes": 2621440,
      "modified_date": "2025-01-15T10:30:00Z",
      "document_type": "bns",
      "in_vectorstore": true,
      "chunk_count": 150,
      "first_chunk_preview": "Section 1: General Provisions..."
    },
    {
      "filename": "constitutionOfIndia.pdf",
      "full_path": "data/constitutionOfIndia.pdf",
      "size_mb": 5.2,
      "size_bytes": 5452595,
      "modified_date": "2024-12-01T08:15:00Z",
      "document_type": "constitution",
      "in_vectorstore": true,
      "chunk_count": 320
    }
  ],
  "summary": {
    "total_pdfs": 35,
    "total_chunks_in_chromadb": 5000,
    "documents_not_indexed": 0
  }
}
```

---

### 1.3 Delete Document

**Endpoint:** `DELETE /documents`

**What it does:**
1. Locates the PDF file in the `data/` folder
2. Queries ChromaDB to find all chunks with matching source path in metadata
3. Deletes all embeddings/chunks from ChromaDB collection
4. Deletes the physical PDF file from disk
5. Returns count of deleted items

**Use Case:** Remove outdated law or incorrect document from the system

**Request:**
```
DELETE http://localhost:8000/documents
Content-Type: application/json

{
  "filename": "law2.pdf",
  "folder": "bns_data"
}
```

**Body Parameters:**
- `filename` (required) - Name of PDF file to delete
- `folder` (optional) - Subfolder path (if not provided, searches all folders)

**Response:**
```json
{
  "status": "success",
  "message": "Document and embeddings deleted successfully",
  "file_deleted": "data/bns_data/law2.pdf",
  "chunks_removed_from_chromadb": 120,
  "disk_space_freed_mb": 1.8
}
```

**Error Response:**
```json
{
  "status": "error",
  "error_code": "FILE_NOT_FOUND",
  "message": "PDF file not found",
  "searched_paths": [
    "data/law2.pdf",
    "data/bns_data/law2.pdf"
  ]
}
```

---

## 2. Chunking Configuration APIs

### 2.1 Reprocess Document with New Chunk Settings

**Endpoint:** `POST /reprocess`

**What it does:**
1. Locates existing PDF in `data/` folder
2. Queries ChromaDB to find and delete all old chunks for this document
3. Re-extracts text from PDF
4. Applies new chunking parameters (size, overlap, separator)
5. Generates new embeddings with updated chunks
6. Stores new embeddings in ChromaDB
7. Compares old vs new chunk counts

**Use Case:** When you realize chunks are too large/small and want to re-chunk an existing document without re-uploading

**Request:**
```
POST http://localhost:8000/reprocess
Content-Type: application/json

{
  "filename": "law2.pdf",
  "folder": "bns_data",
  "chunk_config": {
    "chunk_size": 1500,
    "chunk_overlap": 300,
    "separator": "\n\n"
  }
}
```

**Body Parameters:**
- `filename` (required) - PDF to reprocess
- `folder` (optional) - Subfolder location
- `chunk_config` (required):
  - `chunk_size` (int) - Characters per chunk (recommended: 500-2000)
  - `chunk_overlap` (int) - Overlap to maintain context (recommended: 10-20% of chunk_size)
  - `separator` (string) - Text separator for splitting (default: "\n\n")

**Response:**
```json
{
  "status": "success",
  "message": "Document reprocessed with new chunking settings",
  "filename": "law2.pdf",
  "old_chunks_removed": 120,
  "new_chunks_added": 95,
  "chunk_comparison": {
    "old_config": {
      "chunk_size": 1000,
      "chunk_overlap": 200
    },
    "new_config": {
      "chunk_size": 1500,
      "chunk_overlap": 300
    },
    "chunks_reduced_by": 25,
    "percentage_change": -20.8
  },
  "processing_time_seconds": 38.5
}
```

---

### 2.2 Get Current Default Chunk Settings

**Endpoint:** `GET /config/chunking`

**What it does:**
- Returns the current default chunking configuration stored in system config
- These defaults are used when uploading new documents without specifying custom settings

**Request:**
```
GET http://localhost:8000/config/chunking
```

**Response:**
```json
{
  "status": "success",
  "chunk_config": {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "separator": "\n\n",
    "length_function": "character",
    "description": "Default settings for RecursiveCharacterTextSplitter"
  },
  "recommendations": {
    "small_documents": {
      "chunk_size": 500,
      "chunk_overlap": 100
    },
    "large_documents": {
      "chunk_size": 1500,
      "chunk_overlap": 300
    }
  }
}
```

---

### 2.3 Update Default Chunk Settings

**Endpoint:** `PUT /config/chunking`

**What it does:**
- Updates the system-wide default chunking configuration
- New uploads will use these settings unless overridden
- Does NOT affect existing documents (use /reprocess for that)

**Request:**
```
PUT http://localhost:8000/config/chunking
Content-Type: application/json

{
  "chunk_size": 1500,
  "chunk_overlap": 300,
  "separator": "\n\n"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Default chunking configuration updated",
  "previous_config": {
    "chunk_size": 1000,
    "chunk_overlap": 200
  },
  "new_config": {
    "chunk_size": 1500,
    "chunk_overlap": 300,
    "separator": "\n\n"
  },
  "note": "Existing documents not affected. Use /reprocess to update them."
}
```

---

## 3. Embedding Configuration APIs

### 3.1 Get Current Embedding Model

**Endpoint:** `GET /config/embedding`

**What it does:**
- Returns information about the currently loaded embedding model
- Shows model name, device (CPU/GPU), dimension, and normalization settings
- Displays model performance metrics

**Request:**
```
GET http://localhost:8000/config/embedding
```

**Response:**
```json
{
  "status": "success",
  "embedding_config": {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "model_type": "HuggingFace",
    "device": "cpu",
    "normalize_embeddings": true,
    "embedding_dimension": 384,
    "max_sequence_length": 256
  },
  "model_info": {
    "model_size_mb": 90,
    "avg_embedding_time_ms": 50,
    "loaded_at": "2025-01-16T08:00:00Z"
  },
  "alternative_models": [
    "sentence-transformers/all-mpnet-base-v2 (768 dim, better quality)",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (384 dim, multilingual)"
  ]
}
```

---

### 3.2 Change Embedding Model

**Endpoint:** `PUT /config/embedding`

**What it does:**
1. Validates the new model name
2. Downloads model from HuggingFace if not cached
3. Loads new model into memory
4. Updates system configuration
5. **Important:** Existing embeddings become incompatible - requires vector store rebuild

**Use Case:** Switch to a better embedding model or multilingual model

**Request:**
```
PUT http://localhost:8000/config/embedding
Content-Type: application/json

{
  "model_name": "sentence-transformers/all-mpnet-base-v2",
  "device": "cpu",
  "normalize_embeddings": true
}
```

**Response:**
```json
{
  "status": "warning",
  "message": "Embedding model changed successfully",
  "previous_model": {
    "name": "sentence-transformers/all-MiniLM-L6-v2",
    "dimension": 384
  },
  "new_model": {
    "name": "sentence-transformers/all-mpnet-base-v2",
    "dimension": 768,
    "download_size_mb": 420,
    "loaded": true
  },
  "action_required": {
    "warning": "Existing embeddings (384-dim) incompatible with new model (768-dim)",
    "required_action": "Call POST /vectorstore/rebuild to regenerate all embeddings",
    "estimated_rebuild_time_minutes": 30
  }
}
```

---

## 4. ChromaDB Vector Store APIs

### 4.1 Get Vector Store Statistics

**Endpoint:** `GET /vectorstore/stats`

**What it does:**
- Queries ChromaDB collection for total count of stored vectors
- Groups chunks by document source to count documents
- Calculates storage size and embedding dimensions
- Provides breakdown by document type

**Request:**
```
GET http://localhost:8000/vectorstore/stats
```

**Response:**
```json
{
  "status": "success",
  "vectorstore_stats": {
    "collection_name": "default",
    "total_chunks": 5000,
    "total_documents": 35,
    "embedding_dimension": 384,
    "chroma_db_size_mb": 125.5,
    "chroma_db_path": "chroma_db/",
    "last_updated": "2025-01-16T10:30:00Z"
  },
  "documents_by_type": {
    "constitution": {
      "count": 1,
      "chunks": 800
    },
    "bns": {
      "count": 4,
      "chunks": 750
    },
    "ipc": {
      "count": 1,
      "chunks": 650
    },
    "supreme_court": {
      "count": 15,
      "chunks": 1200
    },
    "high_court": {
      "count": 10,
      "chunks": 900
    },
    "other": {
      "count": 4,
      "chunks": 700
    }
  },
  "health": {
    "status": "healthy",
    "index_built": true,
    "queryable": true
  }
}
```

---

### 4.2 Search Vector Store (Test Query)

**Endpoint:** `POST /vectorstore/search`

**What it does:**
1. Converts query text to embedding using current embedding model
2. Performs similarity search in ChromaDB
3. Returns top-k most similar chunks with metadata and scores
4. Useful for testing retrieval quality before using in chat

**Use Case:** Test if a specific legal topic is properly indexed and retrievable

**Request:**
```
POST http://localhost:8000/vectorstore/search
Content-Type: application/json

{
  "query": "What is Article 21 of Indian Constitution?",
  "k": 5,
  "score_threshold": 0.7
}
```

**Body Parameters:**
- `query` (required) - Search query text
- `k` (optional, default: 5) - Number of results to return
- `score_threshold` (optional) - Minimum similarity score (0-1)

**Response:**
```json
{
  "status": "success",
  "query": "What is Article 21 of Indian Constitution?",
  "results_count": 5,
  "results": [
    {
      "rank": 1,
      "content": "Article 21: Protection of life and personal liberty - No person shall be deprived of his life or personal liberty except according to procedure established by law.",
      "metadata": {
        "source": "data/constitutionOfIndia.pdf",
        "page": 15,
        "document_type": "constitution"
      },
      "similarity_score": 0.92,
      "distance": 0.08
    },
    {
      "rank": 2,
      "content": "The Supreme Court has interpreted Article 21 to include right to live with dignity, right to education, right to privacy...",
      "metadata": {
        "source": "data/Supreme-Court-Landmark-Cases-2024.pdf",
        "page": 45,
        "document_type": "supreme_court"
      },
      "similarity_score": 0.87,
      "distance": 0.13
    }
  ],
  "search_time_ms": 45
}
```

---

### 4.3 Rebuild Entire Vector Store

**Endpoint:** `POST /vectorstore/rebuild`

**What it does:**
1. Deletes entire ChromaDB collection (all embeddings)
2. Scans `data/` folder for all PDF files
3. Processes each PDF: extract → chunk → embed → store
4. Rebuilds vector store from scratch with current settings
5. Runs as background job with progress tracking

**Use Case:** 
- After changing embedding model
- After corrupted vector store
- To apply new chunking settings to all documents

**Request:**
```
POST http://localhost:8000/vectorstore/rebuild
Content-Type: application/json

{
  "confirm": true,
  "chunk_size": 1000,
  "chunk_overlap": 200,
  "process_folders": ["bns_data", "language_const"]
}
```

**Body Parameters:**
- `confirm` (required) - Must be `true` to proceed
- `chunk_size` (optional) - Chunk size for all documents
- `chunk_overlap` (optional) - Overlap for all documents
- `process_folders` (optional) - Specific folders to process (default: all)

**Response:**
```json
{
  "status": "processing",
  "message": "Vector store rebuild initiated",
  "job_id": "rebuild_abc123",
  "pdfs_to_process": 35,
  "estimated_time_minutes": 30,
  "status_check_url": "/vectorstore/rebuild/rebuild_abc123",
  "started_at": "2025-01-16T11:00:00Z"
}
```

---

### 4.4 Check Rebuild Job Status

**Endpoint:** `GET /vectorstore/rebuild/{job_id}`

**What it does:**
- Checks progress of background rebuild job
- Returns current file being processed and completion percentage
- Shows any errors encountered

**Request:**
```
GET http://localhost:8000/vectorstore/rebuild/rebuild_abc123
```

**Response (In Progress):**
```json
{
  "status": "processing",
  "job_id": "rebuild_abc123",
  "progress": {
    "processed": 20,
    "total": 35,
    "percentage": 57,
    "current_file": "data/supreme_court/case_123.pdf",
    "current_step": "generating_embeddings"
  },
  "statistics": {
    "total_chunks_created": 3200,
    "total_embeddings_generated": 3200,
    "failed_files": 0
  },
  "estimated_completion": "2025-01-16T11:25:00Z",
  "elapsed_time_minutes": 15
}
```

**Response (Completed):**
```json
{
  "status": "completed",
  "job_id": "rebuild_abc123",
  "message": "Vector store rebuild completed successfully",
  "final_statistics": {
    "pdfs_processed": 35,
    "total_chunks": 5000,
    "total_embeddings": 5000,
    "failed_files": 0,
    "total_time_minutes": 28
  },
  "completed_at": "2025-01-16T11:28:00Z"
}
```

---

### 4.5 Clear Vector Store

**Endpoint:** `DELETE /vectorstore/clear`

**What it does:**
- Deletes all embeddings from ChromaDB collection
- Keeps PDF files in `data/` folder intact
- Resets vector store to empty state

**Use Case:** Clean slate before rebuilding or testing

**Request:**
```
DELETE http://localhost:8000/vectorstore/clear
Content-Type: application/json

{
  "confirm": "DELETE_ALL_EMBEDDINGS"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Vector store cleared successfully",
  "chunks_deleted": 5000,
  "pdfs_preserved": 35,
  "chroma_db_size_before_mb": 125.5,
  "chroma_db_size_after_mb": 0.5
}
```

---

## 5. Retrieval Configuration APIs

### 5.1 Get Retrieval Settings

**Endpoint:** `GET /config/retrieval`

**What it does:**
- Returns current retrieval parameters used by the RAG system
- Shows how many chunks are retrieved per query
- Displays search algorithm and filtering settings

**Request:**
```
GET http://localhost:8000/config/retrieval
```

**Response:**
```json
{
  "status": "success",
  "retrieval_config": {
    "k": 5,
    "search_type": "similarity",
    "score_threshold": null,
    "fetch_k": 20,
    "lambda_mult": 0.5
  },
  "description": {
    "k": "Number of chunks returned to LLM",
    "search_type": "similarity (cosine) or mmr (maximal marginal relevance)",
    "score_threshold": "Minimum similarity score (null = no threshold)",
    "fetch_k": "Initial candidates for MMR (only used if search_type=mmr)",
    "lambda_mult": "Diversity vs relevance for MMR (0=diverse, 1=relevant)"
  }
}
```

---

### 5.2 Update Retrieval Settings

**Endpoint:** `PUT /config/retrieval`

**What it does:**
- Updates retrieval parameters for the RAG system
- Changes take effect immediately for new queries
- Affects quality and diversity of retrieved context

**Use Case:** Tune retrieval to get better context for LLM

**Request:**
```
PUT http://localhost:8000/config/retrieval
Content-Type: application/json

{
  "k": 10,
  "search_type": "mmr",
  "score_threshold": 0.7,
  "lambda_mult": 0.6
}
```

**Body Parameters:**
- `k` (int, 1-20) - More chunks = more context but slower
- `search_type` (string):
  - `similarity` - Pure cosine similarity (faster, may have redundancy)
  - `mmr` - Maximal Marginal Relevance (diverse results, slower)
- `score_threshold` (float, 0-1) - Filter low-quality matches
- `lambda_mult` (float, 0-1) - Only for MMR (0=max diversity, 1=max relevance)

**Response:**
```json
{
  "status": "success",
  "message": "Retrieval configuration updated",
  "previous_config": {
    "k": 5,
    "search_type": "similarity"
  },
  "new_config": {
    "k": 10,
    "search_type": "mmr",
    "score_threshold": 0.7,
    "lambda_mult": 0.6
  },
  "impact": {
    "context_size_change": "+100%",
    "expected_latency_change": "+50ms",
    "diversity_improvement": "high"
  }
}
```

---

## 6. System APIs

### 6.1 Health Check

**Endpoint:** `GET /health`

**What it does:**
- Checks if all system components are operational
- Tests ChromaDB connection
- Verifies embedding model is loaded
- Checks data folder accessibility

**Request:**
```
GET http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-16T12:00:00Z",
  "components": {
    "chromadb": {
      "status": "healthy",
      "queryable": true,
      "collection_exists": true
    },
    "embedding_model": {
      "status": "loaded",
      "model": "sentence-transformers/all-MiniLM-L6-v2"
    },
    "data_folder": {
      "status": "accessible",
      "path": "data/",
      "pdf_count": 35
    },
    "llm": {
      "status": "connected",
      "model": "gpt-4o-mini"
    }
  },
  "statistics": {
    "total_pdfs": 35,
    "total_chunks": 5000,
    "storage_used_mb": 125.5,
    "uptime_hours": 120
  }
}
```

---

### 6.2 Get All Configurations

**Endpoint:** `GET /config`

**What it does:**
- Returns all system configurations in one response
- Useful for backup or documentation

**Request:**
```
GET http://localhost:8000/config
```

**Response:**
```json
{
  "status": "success",
  "configurations": {
    "chunking": {
      "chunk_size": 1000,
      "chunk_overlap": 200,
      "separator": "\n\n"
    },
    "embedding": {
      "model_name": "sentence-transformers/all-MiniLM-L6-v2",
      "device": "cpu",
      "normalize_embeddings": true,
      "dimension": 384
    },
    "retrieval": {
      "k": 5,
      "search_type": "similarity",
      "score_threshold": null
    },
    "llm": {
      "model": "gpt-4o-mini",
      "temperature": 0.7
    }
  }
}
```

---

## Common Error Responses

### File Not Found
```json
{
  "status": "error",
  "error_code": "FILE_NOT_FOUND",
  "message": "PDF file not found in data folder",
  "details": {
    "filename": "law2.pdf",
    "searched_paths": ["data/law2.pdf", "data/bns_data/law2.pdf"]
  }
}
```

### ChromaDB Error
```json
{
  "status": "error",
  "error_code": "CHROMADB_ERROR",
  "message": "Failed to query vector store",
  "details": "Collection not found or corrupted"
}
```

### Processing Error
```json
{
  "status": "error",
  "error_code": "PROCESSING_ERROR",
  "message": "Failed to process PDF",
  "details": {
    "filename": "law2.pdf",
    "step": "text_extraction",
    "error": "PDF is encrypted or corrupted"
  }
}
```

---

## Postman Testing Workflow

### 1. Check System Health
```
GET http://localhost:8000/health
```

### 2. View Current Configurations
```
GET http://localhost:8000/config
```

### 3. Upload New Law Document
```
POST http://localhost:8000/upload
Form-data:
- file: law2.pdf
- document_type: bns
- chunk_size: 1000
- chunk_overlap: 200
```

### 4. Verify Upload
```
GET http://localhost:8000/documents?folder=bns_data
```

### 5. Test Search
```
POST http://localhost:8000/vectorstore/search
Body: {"query": "What is Section 1 of BNS?", "k": 5}
```

### 6. Reprocess if Needed
```
POST http://localhost:8000/reprocess
Body: {
  "filename": "law2.pdf",
  "folder": "bns_data",
  "chunk_config": {"chunk_size": 1500, "chunk_overlap": 300}
}
```

### 7. Delete if Wrong
```
DELETE http://localhost:8000/documents
Body: {"filename": "law2.pdf", "folder": "bns_data"}
```
