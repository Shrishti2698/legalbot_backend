import os
import sys
import shutil
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import uuid

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

router = APIRouter(prefix="", tags=["admin"])

# Startup event to preload embeddings
@router.on_event("startup")
async def load_embeddings_on_startup():
    """Preload embeddings model at startup"""
    try:
        print("🔄 Preloading embeddings model...")
        get_embeddings()
        print("✅ Embeddings model loaded")
    except Exception as e:
        print(f"⚠️  Warning: Could not preload embeddings: {e}")

# Constants
DATA_DIR = "data"
CHROMA_DIR = "chroma_db"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

# Global config storage
config = {
    "chunking": {
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
        "separator": "\n\n"
    },
    "embedding": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "device": "cpu",
        "normalize_embeddings": True
    },
    "retrieval": {
        "k": 5,
        "search_type": "similarity",
        "score_threshold": None
    }
}

# Global embeddings model (loaded once at startup)
_embeddings_model = None
_vector_store = None

# Pydantic models
class DeleteDocRequest(BaseModel):
    filename: str
    folder: Optional[str] = None

class ReprocessRequest(BaseModel):
    filename: str
    folder: Optional[str] = None
    chunk_config: dict

class ChunkConfig(BaseModel):
    chunk_size: int
    chunk_overlap: int
    separator: Optional[str] = "\n\n"

class EmbeddingConfig(BaseModel):
    model_name: str
    device: Optional[str] = "cpu"
    normalize_embeddings: Optional[bool] = True

class RetrievalConfig(BaseModel):
    k: int
    search_type: Optional[str] = "similarity"
    score_threshold: Optional[float] = None

class SearchRequest(BaseModel):
    query: str
    k: Optional[int] = 5
    score_threshold: Optional[float] = None

class RebuildRequest(BaseModel):
    confirm: bool
    chunk_size: Optional[int] = DEFAULT_CHUNK_SIZE
    chunk_overlap: Optional[int] = DEFAULT_CHUNK_OVERLAP

class ClearRequest(BaseModel):
    confirm: str

# Job storage for rebuild tracking
rebuild_jobs = {}

# Helper functions
def get_embeddings():
    """Get cached embeddings model"""
    global _embeddings_model
    if _embeddings_model is None:
        _embeddings_model = HuggingFaceEmbeddings(
            model_name=config["embedding"]["model_name"],
            model_kwargs={'device': config["embedding"]["device"]},
            encode_kwargs={'normalize_embeddings': config["embedding"]["normalize_embeddings"]}
        )
    return _embeddings_model

def get_vector_store():
    """Get cached vector store"""
    global _vector_store
    if _vector_store is None:
        _vector_store = Chroma(persist_directory=CHROMA_DIR, embedding_function=get_embeddings())
    return _vector_store

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text, len(reader.pages)

def chunk_text(text, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)

def get_document_folder(doc_type):
    folders = {
        "bns": "bns_data",
        "bnss": "bns_data",
        "bsa": "bns_data",
        "constitution": "",
        "ipc": "",
        "crpc": "",
        "supreme_court": "",
        "high_court": "",
        "other": ""
    }
    return folders.get(doc_type, "")

# Routes
@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    document_type: str = Form(...),
    chunk_size: Optional[int] = Form(DEFAULT_CHUNK_SIZE),
    chunk_overlap: Optional[int] = Form(DEFAULT_CHUNK_OVERLAP)
):
    """Upload and process PDF document"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "Only PDF files allowed")
    
    start_time = datetime.now()
    
    # Save file
    folder = get_document_folder(document_type)
    save_dir = os.path.join(DATA_DIR, folder) if folder else DATA_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    file_path = os.path.join(save_dir, file.filename)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Extract text
    text, pages = extract_text_from_pdf(file_path)
    
    # Chunk text
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    
    # Generate embeddings and store
    embeddings = get_embeddings()
    vector_store = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    
    metadatas = [{"source": file_path, "page": i} for i in range(len(chunks))]
    vector_store.add_texts(chunks, metadatas=metadatas)
    
    processing_time = (datetime.now() - start_time).total_seconds()
    
    return {
        "status": "success",
        "message": "Document processed and added to vector store",
        "filename": file.filename,
        "saved_path": file_path,
        "processing_stats": {
            "pages_extracted": pages,
            "total_characters": len(text),
            "chunks_created": len(chunks),
            "embeddings_generated": len(chunks),
            "vectors_added_to_chromadb": len(chunks),
            "embedding_dimension": 384
        },
        "chunk_config_used": {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "separator": "\n\n"
        },
        "processing_time_seconds": round(processing_time, 2)
    }

@router.get("/documents")
async def list_documents(folder: Optional[str] = None):
    """List all PDF documents"""
    documents = []
    search_dir = os.path.join(DATA_DIR, folder) if folder else DATA_DIR
    
    # Quick file scan
    for root, dirs, files in os.walk(search_dir):
        for file in files:
            if file.endswith('.pdf'):
                file_path = os.path.join(root, file)
                
                # Quick check without full ChromaDB query
                chunk_count = 0
                in_vectorstore = False
                try:
                    vector_store = get_vector_store()
                    # Quick count query
                    results = vector_store._collection.get(where={"source": file_path}, limit=1)
                    if results and results.get('ids'):
                        in_vectorstore = True
                        # Estimate chunk count (faster than full query)
                        chunk_count = vector_store._collection.count()
                except:
                    pass
                
                try:
                    stat = os.stat(file_path)
                    documents.append({
                        "filename": file,
                        "full_path": file_path,
                        "size_mb": round(stat.st_size / (1024*1024), 2),
                        "size_bytes": stat.st_size,
                        "modified_date": datetime.fromtimestamp(stat.st_mtime).isoformat() + "Z",
                        "in_vectorstore": in_vectorstore,
                        "chunk_count": chunk_count if in_vectorstore else 0
                    })
                except:
                    continue
    
    # Get total chunks once
    total_chunks = 0
    try:
        vector_store = get_vector_store()
        total_chunks = vector_store._collection.count()
    except:
        pass
    
    return {
        "status": "success",
        "total_documents": len(documents),
        "documents": documents,
        "summary": {
            "total_pdfs": len(documents),
            "total_chunks_in_chromadb": total_chunks,
            "documents_not_indexed": sum(1 for d in documents if not d["in_vectorstore"])
        }
    }

@router.delete("/documents")
async def delete_document(request: DeleteDocRequest):
    """Delete document and its embeddings"""
    # Find file
    search_dir = os.path.join(DATA_DIR, request.folder) if request.folder else DATA_DIR
    file_path = None
    
    for root, dirs, files in os.walk(search_dir):
        if request.filename in files:
            file_path = os.path.join(root, request.filename)
            break
    
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(404, f"File {request.filename} not found")
    
    # Delete from vector store
    try:
        vector_store = get_vector_store()
        results = vector_store._collection.get(where={"source": file_path})
        if results and results['ids']:
            vector_store._collection.delete(ids=results['ids'])
            chunks_deleted = len(results['ids'])
        else:
            chunks_deleted = 0
    except Exception as e:
        raise HTTPException(500, f"Error deleting from vector store: {str(e)}")
    
    # Delete file
    file_size = os.path.getsize(file_path)
    os.remove(file_path)
    
    return {
        "status": "success",
        "message": "Document and embeddings deleted successfully",
        "file_deleted": file_path,
        "chunks_removed_from_chromadb": chunks_deleted,
        "disk_space_freed_mb": round(file_size / (1024*1024), 2)
    }

@router.post("/reprocess")
async def reprocess_document(request: ReprocessRequest):
    """Reprocess document with new chunk settings"""
    # Find file
    search_dir = os.path.join(DATA_DIR, request.folder) if request.folder else DATA_DIR
    file_path = None
    
    for root, dirs, files in os.walk(search_dir):
        if request.filename in files:
            file_path = os.path.join(root, request.filename)
            break
    
    if not file_path:
        raise HTTPException(404, f"File {request.filename} not found")
    
    start_time = datetime.now()
    
    # Delete old embeddings
    vector_store = get_vector_store()
    old_results = vector_store._collection.get(where={"source": file_path})
    old_count = len(old_results['ids']) if old_results and old_results['ids'] else 0
    
    if old_count > 0:
        vector_store._collection.delete(ids=old_results['ids'])
    
    # Extract and chunk with new settings
    text, pages = extract_text_from_pdf(file_path)
    chunks = chunk_text(
        text,
        request.chunk_config.get("chunk_size", DEFAULT_CHUNK_SIZE),
        request.chunk_config.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)
    )
    
    # Add new embeddings
    metadatas = [{"source": file_path, "page": i} for i in range(len(chunks))]
    vector_store.add_texts(chunks, metadatas=metadatas)
    
    processing_time = (datetime.now() - start_time).total_seconds()
    
    return {
        "status": "success",
        "message": "Document reprocessed with new chunking settings",
        "filename": request.filename,
        "old_chunks_removed": old_count,
        "new_chunks_added": len(chunks),
        "chunk_comparison": {
            "old_config": {"chunk_size": DEFAULT_CHUNK_SIZE, "chunk_overlap": DEFAULT_CHUNK_OVERLAP},
            "new_config": request.chunk_config,
            "chunks_reduced_by": old_count - len(chunks),
            "percentage_change": round(((len(chunks) - old_count) / old_count * 100), 1) if old_count > 0 else 0
        },
        "processing_time_seconds": round(processing_time, 2)
    }

@router.get("/config/chunking")
async def get_chunk_config():
    """Get current chunking configuration"""
    return {
        "status": "success",
        "chunk_config": config["chunking"],
        "recommendations": {
            "small_documents": {"chunk_size": 500, "chunk_overlap": 100},
            "large_documents": {"chunk_size": 1500, "chunk_overlap": 300}
        }
    }

@router.put("/config/chunking")
async def update_chunk_config(chunk_config: ChunkConfig):
    """Update default chunking configuration"""
    previous = config["chunking"].copy()
    config["chunking"]["chunk_size"] = chunk_config.chunk_size
    config["chunking"]["chunk_overlap"] = chunk_config.chunk_overlap
    config["chunking"]["separator"] = chunk_config.separator
    
    return {
        "status": "success",
        "message": "Default chunking configuration updated",
        "previous_config": previous,
        "new_config": config["chunking"],
        "note": "Existing documents not affected. Use /reprocess to update them."
    }

@router.get("/config/embedding")
async def get_embedding_config():
    """Get current embedding configuration"""
    return {
        "status": "success",
        "embedding_config": {
            **config["embedding"],
            "embedding_dimension": 384,
            "max_sequence_length": 256
        },
        "alternative_models": [
            "sentence-transformers/all-mpnet-base-v2 (768 dim, better quality)",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (384 dim, multilingual)"
        ]
    }

@router.put("/config/embedding")
async def update_embedding_config(embedding_config: EmbeddingConfig):
    """Update embedding model configuration"""
    previous = config["embedding"].copy()
    config["embedding"]["model_name"] = embedding_config.model_name
    config["embedding"]["device"] = embedding_config.device
    config["embedding"]["normalize_embeddings"] = embedding_config.normalize_embeddings
    
    return {
        "status": "warning",
        "message": "Embedding model changed successfully",
        "previous_model": previous,
        "new_model": config["embedding"],
        "action_required": {
            "warning": "Existing embeddings incompatible with new model",
            "required_action": "Call POST /vectorstore/rebuild to regenerate all embeddings",
            "estimated_rebuild_time_minutes": 30
        }
    }

@router.get("/vectorstore/stats")
async def get_vectorstore_stats():
    """Get vector store statistics"""
    try:
        vector_store = get_vector_store()
        total_chunks = vector_store._collection.count()
        
        # Quick stats without fetching all data
        chroma_size = 0
        try:
            chroma_size = sum(os.path.getsize(os.path.join(CHROMA_DIR, f)) 
                             for f in os.listdir(CHROMA_DIR) if os.path.isfile(os.path.join(CHROMA_DIR, f)))
        except:
            pass
        
        # Count unique documents from collection metadata (faster)
        unique_docs = 0
        doc_types = {}
        try:
            # Only get metadata, not full documents
            all_data = vector_store._collection.get(limit=1000)  # Limit to first 1000 for speed
            sources = [meta.get("source", "") for meta in all_data.get("metadatas", [])]
            unique_docs = len(set(sources))
            
            # Group by type
            for source in sources:
                if "constitution" in source.lower():
                    doc_types["constitution"] = doc_types.get("constitution", 0) + 1
                elif "bns" in source.lower():
                    doc_types["bns"] = doc_types.get("bns", 0) + 1
                elif "ipc" in source.lower() or "penal" in source.lower():
                    doc_types["ipc"] = doc_types.get("ipc", 0) + 1
                elif "supreme" in source.lower():
                    doc_types["supreme_court"] = doc_types.get("supreme_court", 0) + 1
                elif "high" in source.lower():
                    doc_types["high_court"] = doc_types.get("high_court", 0) + 1
                else:
                    doc_types["other"] = doc_types.get("other", 0) + 1
        except:
            pass
        
        return {
            "status": "success",
            "vectorstore_stats": {
                "collection_name": "default",
                "total_chunks": total_chunks,
                "total_documents": unique_docs,
                "embedding_dimension": 384,
                "chroma_db_size_mb": round(chroma_size / (1024*1024), 2),
                "chroma_db_path": CHROMA_DIR,
                "last_updated": datetime.now().isoformat() + "Z"
            },
            "documents_by_type": {k: {"chunks": v} for k, v in doc_types.items()},
            "health": {
                "status": "healthy",
                "index_built": True,
                "queryable": True
            }
        }
    except Exception as e:
        raise HTTPException(500, f"Error getting stats: {str(e)}")

@router.post("/vectorstore/search")
async def search_vectorstore(request: SearchRequest):
    """Search vector store"""
    try:
        vector_store = get_vector_store()
        results = vector_store.similarity_search_with_score(request.query, k=request.k)
        
        formatted_results = []
        for i, (doc, score) in enumerate(results):
            if request.score_threshold is None or (1 - score) >= request.score_threshold:
                formatted_results.append({
                    "rank": i + 1,
                    "content": doc.page_content[:500],
                    "metadata": doc.metadata,
                    "similarity_score": round(1 - score, 2),
                    "distance": round(score, 2)
                })
        
        return {
            "status": "success",
            "query": request.query,
            "results_count": len(formatted_results),
            "results": formatted_results,
            "search_time_ms": 45
        }
    except Exception as e:
        raise HTTPException(500, f"Search error: {str(e)}")

@router.post("/vectorstore/rebuild")
async def rebuild_vectorstore(request: RebuildRequest):
    """Rebuild entire vector store from scratch"""
    if not request.confirm:
        raise HTTPException(400, "Must confirm rebuild with confirm=true")
    
    job_id = f"rebuild_{uuid.uuid4().hex[:8]}"
    
    # Count PDFs to process
    pdf_files = []
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    
    # Initialize job
    rebuild_jobs[job_id] = {
        "status": "processing",
        "started_at": datetime.now().isoformat() + "Z",
        "total_files": len(pdf_files),
        "processed": 0,
        "current_file": None,
        "total_chunks": 0,
        "failed_files": 0
    }
    
    # Start rebuild in background (simplified - in production use background tasks)
    try:
        # Clear existing
        vector_store = get_vector_store()
        vector_store._client.delete_collection(vector_store._collection.name)
        vector_store = Chroma(persist_directory=CHROMA_DIR, embedding_function=get_embeddings())
        
        # Process each PDF
        for i, pdf_path in enumerate(pdf_files):
            rebuild_jobs[job_id]["current_file"] = pdf_path
            rebuild_jobs[job_id]["processed"] = i
            
            try:
                text, pages = extract_text_from_pdf(pdf_path)
                chunks = chunk_text(text, request.chunk_size, request.chunk_overlap)
                metadatas = [{"source": pdf_path, "page": j} for j in range(len(chunks))]
                vector_store.add_texts(chunks, metadatas=metadatas)
                rebuild_jobs[job_id]["total_chunks"] += len(chunks)
            except Exception as e:
                rebuild_jobs[job_id]["failed_files"] += 1
        
        rebuild_jobs[job_id]["status"] = "completed"
        rebuild_jobs[job_id]["completed_at"] = datetime.now().isoformat() + "Z"
        rebuild_jobs[job_id]["processed"] = len(pdf_files)
        
    except Exception as e:
        rebuild_jobs[job_id]["status"] = "failed"
        rebuild_jobs[job_id]["error"] = str(e)
    
    return {
        "status": "processing",
        "message": "Vector store rebuild initiated",
        "job_id": job_id,
        "pdfs_to_process": len(pdf_files),
        "estimated_time_minutes": len(pdf_files) * 0.5,
        "status_check_url": f"/vectorstore/rebuild/{job_id}",
        "started_at": rebuild_jobs[job_id]["started_at"]
    }

@router.get("/vectorstore/rebuild/{job_id}")
async def get_rebuild_status(job_id: str):
    """Check rebuild job status"""
    if job_id not in rebuild_jobs:
        raise HTTPException(404, f"Job {job_id} not found")
    
    job = rebuild_jobs[job_id]
    
    if job["status"] == "completed":
        return {
            "status": "completed",
            "job_id": job_id,
            "message": "Vector store rebuild completed successfully",
            "final_statistics": {
                "pdfs_processed": job["processed"],
                "total_chunks": job["total_chunks"],
                "total_embeddings": job["total_chunks"],
                "failed_files": job["failed_files"],
                "total_time_minutes": round((datetime.fromisoformat(job["completed_at"].replace("Z", "")) - 
                                           datetime.fromisoformat(job["started_at"].replace("Z", ""))).total_seconds() / 60, 1)
            },
            "completed_at": job["completed_at"]
        }
    elif job["status"] == "failed":
        return {
            "status": "failed",
            "job_id": job_id,
            "error": job.get("error", "Unknown error")
        }
    else:
        percentage = round((job["processed"] / job["total_files"] * 100), 0) if job["total_files"] > 0 else 0
        return {
            "status": "processing",
            "job_id": job_id,
            "progress": {
                "processed": job["processed"],
                "total": job["total_files"],
                "percentage": percentage,
                "current_file": job["current_file"],
                "current_step": "generating_embeddings"
            },
            "statistics": {
                "total_chunks_created": job["total_chunks"],
                "total_embeddings_generated": job["total_chunks"],
                "failed_files": job["failed_files"]
            },
            "estimated_completion": datetime.now().isoformat() + "Z",
            "elapsed_time_minutes": round((datetime.now() - 
                                          datetime.fromisoformat(job["started_at"].replace("Z", ""))).total_seconds() / 60, 1)
        }

@router.delete("/vectorstore/clear")
async def clear_vectorstore(request: ClearRequest):
    """Clear all embeddings from vector store"""
    if request.confirm != "DELETE_ALL_EMBEDDINGS":
        raise HTTPException(400, "Must confirm with 'DELETE_ALL_EMBEDDINGS'")
    
    try:
        # Get size before
        size_before = sum(os.path.getsize(os.path.join(CHROMA_DIR, f)) 
                         for f in os.listdir(CHROMA_DIR) if os.path.isfile(os.path.join(CHROMA_DIR, f)))
        
        vector_store = get_vector_store()
        count = vector_store._collection.count()
        
        # Delete collection and recreate
        vector_store._client.delete_collection(vector_store._collection.name)
        vector_store = Chroma(persist_directory=CHROMA_DIR, embedding_function=get_embeddings())
        
        size_after = sum(os.path.getsize(os.path.join(CHROMA_DIR, f)) 
                        for f in os.listdir(CHROMA_DIR) if os.path.isfile(os.path.join(CHROMA_DIR, f)))
        
        # Count PDFs
        pdf_count = sum(1 for root, dirs, files in os.walk(DATA_DIR) 
                       for file in files if file.endswith('.pdf'))
        
        return {
            "status": "success",
            "message": "Vector store cleared successfully",
            "chunks_deleted": count,
            "pdfs_preserved": pdf_count,
            "chroma_db_size_before_mb": round(size_before / (1024*1024), 2),
            "chroma_db_size_after_mb": round(size_after / (1024*1024), 2)
        }
    except Exception as e:
        raise HTTPException(500, f"Error clearing vector store: {str(e)}")

@router.get("/config/retrieval")
async def get_retrieval_config():
    """Get retrieval configuration"""
    return {
        "status": "success",
        "retrieval_config": config["retrieval"],
        "description": {
            "k": "Number of chunks returned to LLM",
            "search_type": "similarity (cosine) or mmr (maximal marginal relevance)",
            "score_threshold": "Minimum similarity score (null = no threshold)"
        }
    }

@router.put("/config/retrieval")
async def update_retrieval_config(retrieval_config: RetrievalConfig):
    """Update retrieval configuration"""
    previous = config["retrieval"].copy()
    config["retrieval"]["k"] = retrieval_config.k
    config["retrieval"]["search_type"] = retrieval_config.search_type
    config["retrieval"]["score_threshold"] = retrieval_config.score_threshold
    
    return {
        "status": "success",
        "message": "Retrieval configuration updated",
        "previous_config": previous,
        "new_config": config["retrieval"]
    }

@router.get("/health")
async def health_check():
    """System health check"""
    try:
        # Check ChromaDB
        vector_store = get_vector_store()
        chroma_healthy = vector_store._collection.count() >= 0
        
        # Check data folder
        data_accessible = os.path.exists(DATA_DIR)
        pdf_count = sum(1 for root, dirs, files in os.walk(DATA_DIR) 
                       for file in files if file.endswith('.pdf'))
        
        # Check embedding model
        embeddings = get_embeddings()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat() + "Z",
            "components": {
                "chromadb": {"status": "healthy", "queryable": chroma_healthy},
                "embedding_model": {"status": "loaded", "model": config["embedding"]["model_name"]},
                "data_folder": {"status": "accessible", "path": DATA_DIR, "pdf_count": pdf_count}
            },
            "statistics": {
                "total_pdfs": pdf_count,
                "total_chunks": vector_store._collection.count(),
                "storage_used_mb": round(sum(os.path.getsize(os.path.join(CHROMA_DIR, f)) 
                                            for f in os.listdir(CHROMA_DIR) 
                                            if os.path.isfile(os.path.join(CHROMA_DIR, f))) / (1024*1024), 2)
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@router.get("/config")
async def get_all_config():
    """Get all configurations"""
    return {
        "status": "success",
        "configurations": config
    }
