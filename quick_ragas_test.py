import json
import os
from utils import load_vector_store, create_enhanced_rag_response
from datetime import datetime

def quick_ragas_test():
    """Quick test of RAGAS evaluation with minimal samples"""
    print("🚀 Quick RAGAS Test - Legal Bot Evaluation")
    print("=" * 50)
    
    # Load test data
    test_data_path = "data/Test_data/IndicLegalQA Dataset_10K_Revised.json"
    
    try:
        with open(test_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Loaded {len(data)} test cases")
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return
    
    # Take first 5 samples for quick test
    sample_data = data[:5]
    print(f"Testing with {len(sample_data)} samples")
    
    # Setup RAG system
    try:
        vector_store = load_vector_store()
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        print("✅ RAG system loaded")
    except Exception as e:
        print(f"❌ Error loading RAG system: {e}")
        return
    
    # Test each sample
    results = []
    for i, item in enumerate(sample_data):
        print(f"\n--- Test {i+1}/5 ---")
        question = item['question']
        ground_truth = item['answer']
        case_name = item.get('case_name', 'Unknown')
        
        print(f"Question: {question[:100]}...")
        print(f"Case: {case_name}")
        
        try:
            # Generate RAG response
            rag_response = create_enhanced_rag_response(
                retriever=retriever,
                question=question,
                chat_history="",
                language="English"
            )
            
            # Get retrieved contexts
            retrieved_docs = retriever.invoke(question)
            contexts = [doc.page_content[:200] + "..." for doc in retrieved_docs]
            
            print(f"Generated Answer: {rag_response['answer'][:150]}...")
            print(f"Retrieved {len(contexts)} context chunks")
            print(f"References: {len(rag_response.get('references', []))}")
            
            results.append({
                'question': question,
                'ground_truth': ground_truth,
                'generated_answer': rag_response['answer'],
                'contexts': contexts,
                'references': rag_response.get('references', []),
                'case_name': case_name
            })
            
        except Exception as e:
            print(f"❌ Error processing question: {e}")
            continue
    
    # Simple analysis
    print(f"\n{'='*50}")
    print("QUICK ANALYSIS RESULTS")
    print(f"{'='*50}")
    print(f"Total Questions Processed: {len(results)}")
    print(f"Average Answer Length: {sum(len(r['generated_answer']) for r in results) / len(results):.0f} characters")
    print(f"Average Contexts Retrieved: {sum(len(r['contexts']) for r in results) / len(results):.1f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"quick_ragas_test_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to {filename}")
    
    # Show sample comparison
    if results:
        print(f"\n{'='*50}")
        print("SAMPLE COMPARISON")
        print(f"{'='*50}")
        sample = results[0]
        print(f"Question: {sample['question']}")
        print(f"\nGround Truth: {sample['ground_truth'][:200]}...")
        print(f"\nGenerated Answer: {sample['generated_answer'][:200]}...")
        print(f"\nRetrieved Context: {sample['contexts'][0] if sample['contexts'] else 'None'}")
    
    print(f"\n🎉 Quick test completed! Check {filename} for detailed results.")

if __name__ == "__main__":
    quick_ragas_test()