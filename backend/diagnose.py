import os
import sys
import time
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 60)
print("ADMIN PANEL PERFORMANCE DIAGNOSTIC")
print("=" * 60)
print()

# Test 1: Check data folder
print("1. Checking data folder...")
start = time.time()
data_dir = "../data"
pdf_count = 0
total_size = 0

try:
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.pdf'):
                pdf_count += 1
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)
    
    elapsed = time.time() - start
    print(f"   ✅ Found {pdf_count} PDFs")
    print(f"   ✅ Total size: {total_size / (1024*1024):.2f} MB")
    print(f"   ⏱️  Time: {elapsed:.2f}s")
    
    if pdf_count > 50:
        print(f"   ⚠️  WARNING: {pdf_count} PDFs is a lot! This may slow down queries.")
except Exception as e:
    print(f"   ❌ Error: {e}")

print()

# Test 2: Check ChromaDB
print("2. Checking ChromaDB...")
start = time.time()
chroma_dir = "../chroma_db"

try:
    if not os.path.exists(chroma_dir):
        print(f"   ❌ ChromaDB directory not found!")
        print(f"   💡 Solution: Upload a document first to create ChromaDB")
    else:
        chroma_size = sum(os.path.getsize(os.path.join(chroma_dir, f)) 
                         for f in os.listdir(chroma_dir) 
                         if os.path.isfile(os.path.join(chroma_dir, f)))
        elapsed = time.time() - start
        print(f"   ✅ ChromaDB exists")
        print(f"   ✅ Size: {chroma_size / (1024*1024):.2f} MB")
        print(f"   ⏱️  Time: {elapsed:.2f}s")
        
        if chroma_size > 500 * 1024 * 1024:  # 500MB
            print(f"   ⚠️  WARNING: ChromaDB is large! This will slow down queries.")
except Exception as e:
    print(f"   ❌ Error: {e}")

print()

# Test 3: Load ChromaDB and test queries
print("3. Testing ChromaDB queries...")

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    
    # Test loading embeddings
    print("   3a. Loading embedding model...")
    start = time.time()
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    elapsed = time.time() - start
    print(f"      ✅ Loaded in {elapsed:.2f}s")
    
    if elapsed > 5:
        print(f"      ⚠️  WARNING: Embedding model loading is slow!")
    
    # Test loading vector store
    print("   3b. Loading vector store...")
    start = time.time()
    vector_store = Chroma(persist_directory=chroma_dir, embedding_function=embeddings)
    elapsed = time.time() - start
    print(f"      ✅ Loaded in {elapsed:.2f}s")
    
    if elapsed > 3:
        print(f"      ⚠️  WARNING: Vector store loading is slow!")
    
    # Test count query
    print("   3c. Testing count query...")
    start = time.time()
    count = vector_store._collection.count()
    elapsed = time.time() - start
    print(f"      ✅ Total chunks: {count}")
    print(f"      ⏱️  Time: {elapsed:.2f}s")
    
    if elapsed > 2:
        print(f"      ⚠️  WARNING: Count query is slow!")
        print(f"      💡 This is the bottleneck!")
    
    # Test get query with limit
    print("   3d. Testing get query (limit 10)...")
    start = time.time()
    results = vector_store._collection.get(limit=10)
    elapsed = time.time() - start
    print(f"      ✅ Retrieved {len(results['ids'])} items")
    print(f"      ⏱️  Time: {elapsed:.2f}s")
    
    if elapsed > 1:
        print(f"      ⚠️  WARNING: Get query is slow!")
    
    # Test get query with limit 1000
    print("   3e. Testing get query (limit 1000)...")
    start = time.time()
    results = vector_store._collection.get(limit=1000)
    elapsed = time.time() - start
    print(f"      ✅ Retrieved {len(results['ids'])} items")
    print(f"      ⏱️  Time: {elapsed:.2f}s")
    
    if elapsed > 5:
        print(f"      ⚠️  WARNING: Large get query is VERY slow!")
        print(f"      💡 This is likely the main bottleneck!")
    
    # Test metadata query
    print("   3f. Testing metadata query...")
    start = time.time()
    results = vector_store._collection.get(limit=100)
    sources = [meta.get("source", "") for meta in results.get("metadatas", [])]
    unique_sources = len(set(sources))
    elapsed = time.time() - start
    print(f"      ✅ Found {unique_sources} unique documents")
    print(f"      ⏱️  Time: {elapsed:.2f}s")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 4: Test actual endpoints
print("4. Testing API endpoints...")

try:
    import requests
    
    # Test health endpoint
    print("   4a. Testing /health...")
    start = time.time()
    response = requests.get("http://localhost:8000/health", timeout=30)
    elapsed = time.time() - start
    
    if response.status_code == 200:
        print(f"      ✅ Status: {response.status_code}")
        print(f"      ⏱️  Time: {elapsed:.2f}s")
        if elapsed > 5:
            print(f"      ⚠️  WARNING: Health endpoint is slow!")
    else:
        print(f"      ❌ Status: {response.status_code}")
    
    # Test vectorstore stats endpoint
    print("   4b. Testing /vectorstore/stats...")
    start = time.time()
    try:
        response = requests.get("http://localhost:8000/vectorstore/stats", timeout=30)
        elapsed = time.time() - start
        
        if response.status_code == 200:
            print(f"      ✅ Status: {response.status_code}")
            print(f"      ⏱️  Time: {elapsed:.2f}s")
            if elapsed > 10:
                print(f"      ⚠️  WARNING: Stats endpoint is VERY slow!")
                print(f"      💡 This is the bottleneck for Dashboard!")
        else:
            print(f"      ❌ Status: {response.status_code}")
    except requests.Timeout:
        print(f"      ❌ TIMEOUT after 30s")
        print(f"      💡 This is definitely the bottleneck!")
    
    # Test documents endpoint
    print("   4c. Testing /documents...")
    start = time.time()
    try:
        response = requests.get("http://localhost:8000/documents", timeout=30)
        elapsed = time.time() - start
        
        if response.status_code == 200:
            print(f"      ✅ Status: {response.status_code}")
            print(f"      ⏱️  Time: {elapsed:.2f}s")
            if elapsed > 10:
                print(f"      ⚠️  WARNING: Documents endpoint is VERY slow!")
                print(f"      💡 This is the bottleneck for Documents page!")
        else:
            print(f"      ❌ Status: {response.status_code}")
    except requests.Timeout:
        print(f"      ❌ TIMEOUT after 30s")
        print(f"      💡 This is definitely the bottleneck!")
        
except Exception as e:
    print(f"   ❌ Error: {e}")
    print(f"   💡 Make sure backend is running: python backend/main.py")

print()
print("=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60)
print()
print("📊 SUMMARY:")
print()
print("Check the warnings (⚠️) above to identify bottlenecks.")
print()
print("Common issues:")
print("1. Too many PDFs (>50) → Slow file scanning")
print("2. Large ChromaDB (>500MB) → Slow queries")
print("3. Slow embedding model loading → First-time delay")
print("4. Slow ChromaDB queries → Main bottleneck")
print()
print("💡 RECOMMENDATIONS:")
print()
print("If ChromaDB queries are slow (>5s):")
print("   → Option 1: Skip ChromaDB queries (instant loading)")
print("   → Option 2: Use lazy loading (load page first, stats later)")
print()
print("If file scanning is slow:")
print("   → Reduce number of PDFs in data folder")
print("   → Add pagination")
print()
print("If embedding model is slow:")
print("   → This is normal on first load")
print("   → Subsequent loads should be faster")
print()
