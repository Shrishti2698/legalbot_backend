#!/usr/bin/env python3
"""
Quick script to check what's actually in ChromaDB
"""
import os
import sys
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

CHROMA_DIR = "chroma_db"

def check_chromadb():
    try:
        print("🔍 Checking ChromaDB...")
        
        # Initialize embeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Load vector store
        vector_store = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        
        # Get basic stats
        total_chunks = vector_store._collection.count()
        print(f"📊 Total chunks in ChromaDB: {total_chunks}")
        
        if total_chunks > 0:
            # Get sample data
            sample_data = vector_store._collection.get(limit=5)
            print(f"📄 Sample documents:")
            
            if sample_data and sample_data.get('metadatas'):
                sources = set()
                for meta in sample_data['metadatas']:
                    source = meta.get('source', 'Unknown')
                    sources.add(source)
                
                print(f"📚 Unique sources found: {len(sources)}")
                for source in list(sources)[:5]:
                    print(f"  - {source}")
            
            # Test a simple search
            print("\n🔍 Testing search...")
            results = vector_store.similarity_search("constitution", k=2)
            print(f"Search results: {len(results)} found")
            for i, doc in enumerate(results):
                print(f"  {i+1}. {doc.page_content[:100]}...")
        else:
            print("❌ ChromaDB is empty!")
            
    except Exception as e:
        print(f"❌ Error checking ChromaDB: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_chromadb()