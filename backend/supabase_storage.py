import os
import zipfile
import tempfile
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# Supabase credentials from environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BUCKET_NAME = os.getenv("SUPABASE_BUCKET", "files")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

CHROMA_ZIP = "chroma_db.zip"
CHROMA_DIR = "chroma_db"

def upload_chroma_to_supabase():
    """Zip and upload chroma_db folder to Supabase"""
    try:
        if not os.path.exists(CHROMA_DIR):
            print("⚠️ ChromaDB directory doesn't exist, skipping upload")
            return False
        
        print("📦 Zipping ChromaDB...")
        # Create zip file
        with zipfile.ZipFile(CHROMA_ZIP, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(CHROMA_DIR):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, os.path.dirname(CHROMA_DIR))
                    zipf.write(file_path, arcname)
        
        print("☁️ Uploading to Supabase...")
        # Upload to Supabase
        with open(CHROMA_ZIP, 'rb') as f:
            supabase.storage.from_(BUCKET_NAME).upload(
                file=f,
                path=CHROMA_ZIP,
                file_options={"content-type": "application/zip", "upsert": "true"}
            )
        
        # Clean up zip file
        os.remove(CHROMA_ZIP)
        
        print("✅ ChromaDB uploaded to Supabase successfully")
        return True
    except Exception as e:
        print(f"❌ Error uploading to Supabase: {e}")
        if os.path.exists(CHROMA_ZIP):
            os.remove(CHROMA_ZIP)
        return False

def download_chroma_from_supabase():
    """Download and extract chroma_db from Supabase"""
    try:
        print("☁️ Downloading ChromaDB from Supabase...")
        
        # Download from Supabase
        response = supabase.storage.from_(BUCKET_NAME).download(CHROMA_ZIP)
        
        if not response:
            print("⚠️ No ChromaDB backup found in Supabase")
            return False
        
        # Save zip file
        with open(CHROMA_ZIP, 'wb') as f:
            f.write(response)
        
        print("📦 Extracting ChromaDB...")
        # Extract zip file
        with zipfile.ZipFile(CHROMA_ZIP, 'r') as zipf:
            zipf.extractall()
        
        # Clean up zip file
        os.remove(CHROMA_ZIP)
        
        print("✅ ChromaDB downloaded from Supabase successfully")
        return True
    except Exception as e:
        print(f"⚠️ Error downloading from Supabase: {e}")
        if os.path.exists(CHROMA_ZIP):
            os.remove(CHROMA_ZIP)
        return False

def sync_chroma_to_supabase():
    """Sync local ChromaDB to Supabase (called after uploads)"""
    return upload_chroma_to_supabase()

def restore_chroma_from_supabase():
    """Restore ChromaDB from Supabase (called on startup)"""
    if not os.path.exists(CHROMA_DIR) or len(os.listdir(CHROMA_DIR)) == 0:
        print("🔄 Restoring ChromaDB from Supabase...")
        return download_chroma_from_supabase()
    else:
        print("✅ ChromaDB already exists locally")
        return True
