import os
import shutil

def reset_chromadb():
    chroma_dir = "chroma_db"
    
    print("Resetting ChromaDB...")
    
    # Remove entire chroma_db directory
    if os.path.exists(chroma_dir):
        shutil.rmtree(chroma_dir)
        print(f"Deleted {chroma_dir}")
    
    # Recreate empty directory
    os.makedirs(chroma_dir, exist_ok=True)
    print(f"Created fresh {chroma_dir}")
    
    print("ChromaDB reset complete!")
    print("Now you can:")
    print("   1. Start your backend: python main.py")
    print("   2. Upload documents via admin panel")

if __name__ == "__main__":
    reset_chromadb()