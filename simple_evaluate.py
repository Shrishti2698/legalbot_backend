import json
from utils import load_vector_store, create_enhanced_rag_response
from datetime import datetime

def load_ground_truth(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    print("=" * 60)
    print("Simple RAG Evaluation for Indian Legal Assistant")
    print("=" * 60)
    
    # Load vector store
    print("\n[1/3] Loading vector store...")
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    print("✓ Vector store loaded")
    
    # Load ground truth
    print("\n[2/3] Loading ground truth data...")
    ground_truth_data = load_ground_truth("data/RAGAS_groundTruth.json")
    print(f"✓ Loaded {len(ground_truth_data)} QnA pairs")
    
    # Generate responses
    print("\n[3/3] Generating RAG responses...")
    results = []
    
    for i, item in enumerate(ground_truth_data, 1):
        question = item['question']
        ground_truth = item['answer']
        
        print(f"  [{i}/{len(ground_truth_data)}] {question[:50]}...")
        
        response = create_enhanced_rag_response(retriever, question, "", "English")
        answer = response['answer']
        retrieved_docs = retriever.invoke(question)
        contexts = [doc.page_content for doc in retrieved_docs]
        
        results.append({
            'question': question,
            'answer': answer,
            'ground_truth': ground_truth,
            'contexts': contexts,
            'num_contexts': len(contexts)
        })
    
    print("✓ All responses generated")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"rag_evaluation_{timestamp}.json"
    
    output_data = {
        "timestamp": timestamp,
        "total_questions": len(results),
        "results": results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to: {output_file}")
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total Questions: {len(results)}")
    print(f"Average Contexts Retrieved: {sum(r['num_contexts'] for r in results) / len(results):.1f}")
    print("\nSample Results (First 3):")
    for i, r in enumerate(results[:3], 1):
        print(f"\n[{i}] Q: {r['question']}")
        print(f"    Ground Truth: {r['ground_truth']}")
        print(f"    RAG Answer: {r['answer'][:100]}...")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
