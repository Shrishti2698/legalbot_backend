# Admin API Testing Guide

## Setup

1. **Install dependencies:**
```bash
cd backend
pip install -r requirements_admin.txt
```

2. **Start the backend:**
```bash
python main.py
```

Server runs on: `http://localhost:8000`

---

## Postman Testing Sequence

### 1. Health Check
```
GET http://localhost:8000/health
```
Expected: Status 200, all components healthy

---

### 2. View Current Configurations
```
GET http://localhost:8000/config
```
Expected: All default configs (chunking, embedding, retrieval)

---

### 3. List Existing Documents
```
GET http://localhost:8000/documents
```
Expected: List of all PDFs in data folder with vector store status

---

### 4. Upload New Document

**Request:**
```
POST http://localhost:8000/upload
Content-Type: multipart/form-data

Form Data:
- file: [Select your law2.pdf file]
- document_type: bns
- chunk_size: 1000
- chunk_overlap: 200
```

**Expected Response:**
```json
{
  "status": "success",
  "message": "Document processed and added to vector store",
  "filename": "law2.pdf",
  "saved_path": "data/bns_data/law2.pdf",
  "processing_stats": {
    "pages_extracted": 250,
    "chunks_created": 150,
    "embeddings_generated": 150
  },
  "processing_time_seconds": 45.2
}
```

---

### 5. Verify Upload
```
GET http://localhost:8000/documents?folder=bns_data
```
Expected: law2.pdf should appear with in_vectorstore: true

---

### 6. Test Search
```
POST http://localhost:8000/vectorstore/search
Content-Type: application/json

{
  "query": "What is Section 1 of BNS?",
  "k": 5
}
```
Expected: Relevant chunks from law2.pdf

---

### 7. Get Vector Store Stats
```
GET http://localhost:8000/vectorstore/stats
```
Expected: Updated chunk count including new document

---

### 8. Reprocess with Different Settings

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

**Expected Response:**
```json
{
  "status": "success",
  "old_chunks_removed": 150,
  "new_chunks_added": 95,
  "chunk_comparison": {
    "chunks_reduced_by": 55,
    "percentage_change": -36.7
  }
}
```

---

### 9. Update Default Chunk Settings
```
PUT http://localhost:8000/config/chunking
Content-Type: application/json

{
  "chunk_size": 1500,
  "chunk_overlap": 300,
  "separator": "\n\n"
}
```
Expected: Config updated, note about existing documents

---

### 10. Update Retrieval Settings
```
PUT http://localhost:8000/config/retrieval
Content-Type: application/json

{
  "k": 10,
  "search_type": "similarity",
  "score_threshold": 0.7
}
```
Expected: Retrieval config updated

---

### 11. Delete Document

**Request:**
```
DELETE http://localhost:8000/documents
Content-Type: application/json

{
  "filename": "law2.pdf",
  "folder": "bns_data"
}
```

**Expected Response:**
```json
{
  "status": "success",
  "file_deleted": "data/bns_data/law2.pdf",
  "chunks_removed_from_chromadb": 95,
  "disk_space_freed_mb": 1.8
}
```

---

### 12. Clear Vector Store (Optional - Dangerous!)
```
DELETE http://localhost:8000/vectorstore/clear
Content-Type: application/json

{
  "confirm": "DELETE_ALL_EMBEDDINGS"
}
```
Expected: All embeddings deleted, PDFs preserved

---

## Common Errors

### 1. File Not Found
```json
{
  "detail": "File law2.pdf not found"
}
```
**Solution:** Check filename and folder parameter

### 2. Invalid File Type
```json
{
  "detail": "Only PDF files allowed"
}
```
**Solution:** Upload only .pdf files

### 3. ChromaDB Error
```json
{
  "detail": "Error getting stats: ..."
}
```
**Solution:** Ensure chroma_db folder exists and is accessible

---

## Testing Tips

1. **Start with health check** to ensure all components are working
2. **List documents first** to see what's already indexed
3. **Test with small PDF** (few pages) for faster testing
4. **Use search endpoint** to verify document is properly indexed
5. **Check stats** after each operation to verify changes

---

## Postman Collection Variables

Set these in Postman environment:

```
base_url: http://localhost:8000
```

---

## Expected File Structure After Upload

```
legal-advisor/
├── backend/
│   ├── main.py
│   └── admin_routes.py
├── data/
│   ├── bns_data/
│   │   └── law2.pdf  ← Your uploaded file
│   ├── constitutionOfIndia.pdf
│   └── ...
└── chroma_db/  ← Vector embeddings stored here
```

---

## Performance Notes

- **Upload**: ~30-60 seconds for 100-page PDF
- **Reprocess**: ~20-40 seconds (depends on chunk count)
- **Search**: <100ms
- **Delete**: <1 second
- **Stats**: <500ms

---

## Next Steps

After testing admin APIs:
1. Integrate with frontend admin panel
2. Add authentication/authorization
3. Add batch upload functionality
4. Add progress tracking for long operations
5. Add document preview functionality
