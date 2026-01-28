import json
import os
from dotenv import load_dotenv
from datasets import Dataset
from utils import load_vector_store, create_enhanced_rag_response
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Import RAGAS with proper error handling
try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    RAGAS_VERSION = "new"
except ImportError:
    from ragas import evaluate
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
    RAGAS_VERSION = "old"

load_dotenv()

def load_ground_truth(file_path):
    """Load ground truth QnA from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_rag_responses(ground_truth_data, retriever):
    """Generate RAG responses for all questions"""
    results = []
    
    for i, item in enumerate(ground_truth_data, 1):
        question = item['question']
        ground_truth = item['answer']
        
        print(f"Processing {i}/{len(ground_truth_data)}: {question[:50]}...")
        
        # Get RAG response
        response = create_enhanced_rag_response(retriever, question, "", "English")
        answer = response['answer']
        
        # Get contexts from retriever
        retrieved_docs = retriever.invoke(question)
        contexts = [doc.page_content for doc in retrieved_docs]
        
        results.append({
            'question': question,
            'answer': answer,
            'contexts': contexts,
            'ground_truth': ground_truth
        })
    
    return results

def main():
    print("=" * 60)
    print("RAGAS Evaluation for Indian Legal Assistant")
    print("=" * 60)
    
    # Load vector store and create retriever
    print("\n[1/4] Loading vector store...")
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    print("✓ Vector store loaded")
    
    # Load ground truth
    print("\n[2/4] Loading ground truth data...")
    ground_truth_path = "data/RAGAS_groundTruth.json"
    ground_truth_data = load_ground_truth(ground_truth_path)
    print(f"✓ Loaded {len(ground_truth_data)} QnA pairs")
    
    # Generate RAG responses
    print("\n[3/4] Generating RAG responses...")
    results = generate_rag_responses(ground_truth_data, retriever)
    print("✓ All responses generated")
    
    # Create dataset for RAGAS
    print("\n[4/4] Running RAGAS evaluation...")
    dataset = Dataset.from_list(results)
    
    # Evaluate with RAGAS metrics - use the correct format
    if RAGAS_VERSION == "new":
        evaluation_result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
        )
    else:
        evaluation_result = evaluate(
            dataset,
            metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()],
            llm=ChatOpenAI(model="gpt-4o-mini"),
            embeddings=OpenAIEmbeddings()
        )
    
    # Display results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    # Handle both old and new RAGAS result formats
    if hasattr(evaluation_result, 'to_pandas'):
        # New format - convert to dict
        results_df = evaluation_result.to_pandas()
        metrics_dict = results_df.mean().to_dict()
    else:
        # Old format - already a dict
        metrics_dict = evaluation_result
    
    print(f"\nFaithfulness:       {metrics_dict.get('faithfulness', 0):.4f}")
    print(f"Answer Relevancy:   {metrics_dict.get('answer_relevancy', 0):.4f}")
    print(f"Context Precision:  {metrics_dict.get('context_precision', 0):.4f}")
    print(f"Context Recall:     {metrics_dict.get('context_recall', 0):.4f}")
    
    # Save detailed results
    output_file = "ragas_evaluation_results.json"
    output_data = {
        "metrics": {
            "faithfulness": float(metrics_dict.get('faithfulness', 0)),
            "answer_relevancy": float(metrics_dict.get('answer_relevancy', 0)),
            "context_precision": float(metrics_dict.get('context_precision', 0)),
            "context_recall": float(metrics_dict.get('context_recall', 0))
        },
        "results": results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Detailed results saved to: {output_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()
