import os
import json
from dotenv import load_dotenv

def check_setup():
    """Verify all requirements for RAGAS evaluation"""
    print("=" * 60)
    print("RAGAS Evaluation Setup Checker")
    print("=" * 60)
    
    all_good = True
    
    # 1. Check .env file
    print("\n[1/6] Checking environment variables...")
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print("  ✓ OPENAI_API_KEY found")
    else:
        print("  ✗ OPENAI_API_KEY not found in .env")
        all_good = False
    
    # 2. Check ground truth file
    print("\n[2/6] Checking ground truth data...")
    gt_path = "data/RAGAS_groundTruth.json"
    if os.path.exists(gt_path):
        with open(gt_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"  ✓ Ground truth file found ({len(data)} QnA pairs)")
    else:
        print(f"  ✗ Ground truth file not found: {gt_path}")
        all_good = False
    
    # 3. Check vector store
    print("\n[3/6] Checking vector store...")
    if os.path.exists("chroma_db"):
        print("  ✓ ChromaDB vector store found")
    else:
        print("  ✗ Vector store not found (run ingest.py first)")
        all_good = False
    
    # 4. Check required modules
    print("\n[4/6] Checking required packages...")
    required_packages = [
        'ragas',
        'datasets',
        'langchain',
        'langchain_openai',
        'chromadb',
        'pandas',
        'matplotlib',
        'seaborn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} not installed")
            missing_packages.append(package)
            all_good = False
    
    # 5. Check utils.py
    print("\n[5/6] Checking utils module...")
    if os.path.exists("utils.py"):
        try:
            from utils import load_vector_store, create_enhanced_rag_response
            print("  ✓ utils.py found and importable")
        except Exception as e:
            print(f"  ✗ Error importing utils: {e}")
            all_good = False
    else:
        print("  ✗ utils.py not found")
        all_good = False
    
    # 6. Test vector store loading
    print("\n[6/6] Testing vector store loading...")
    try:
        from utils import load_vector_store
        vector_store = load_vector_store()
        retriever = vector_store.as_retriever(search_kwargs={"k": 1})
        test_docs = retriever.invoke("What is Article 21?")
        print(f"  ✓ Vector store loaded successfully ({len(test_docs)} docs retrieved)")
    except Exception as e:
        print(f"  ✗ Error loading vector store: {e}")
        all_good = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_good:
        print("✓ ALL CHECKS PASSED - Ready to run evaluation!")
        print("\nRun evaluation with:")
        print("  python evaluate_ragas.py")
        print("  or")
        print("  run_ragas_eval.bat")
    else:
        print("✗ SOME CHECKS FAILED - Please fix issues above")
        if missing_packages:
            print("\nInstall missing packages:")
            print(f"  pip install {' '.join(missing_packages)}")
    print("=" * 60)
    
    return all_good

if __name__ == "__main__":
    check_setup()
