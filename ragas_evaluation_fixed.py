import json
import pandas as pd
import os
from typing import List, Dict
from datasets import Dataset
from ragas import evaluate
from ragas.metrics.collections import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from utils import load_vector_store, create_enhanced_rag_response
import time
from datetime import datetime

def run_ragas_evaluation_fixed():
    """Fixed RAGAS evaluation that handles the new API properly"""
    print("🚀 Starting Legal RAG System RAGAS Evaluation (Fixed Version)")
    print("=" * 60)
    
    # Load test data
    test_data_path = "data/Test_data/IndicLegalQA Dataset_10K_Revised.json"
    sample_size = 10  # Smaller sample for reliability
    
    try:
        with open(test_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        test_data = data[:sample_size]
        print(f"✅ Loaded {len(test_data)} test cases from {len(data)} total cases")
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return
    
    # Setup RAG system
    try:
        vector_store = load_vector_store()
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        print("✅ RAG system setup complete")
    except Exception as e:
        print(f"❌ Error setting up RAG system: {e}")
        return
    
    # Generate RAG responses
    print(f"Generating RAG responses for {len(test_data)} questions...")
    rag_results = []
    
    for i, item in enumerate(test_data):
        try:
            question = item['question']
            ground_truth = item['answer']
            case_name = item.get('case_name', 'Unknown Case')
            
            print(f"Processing {i+1}/{len(test_data)}: {question[:50]}...")
            
            # Generate RAG response
            rag_response = create_enhanced_rag_response(
                retriever=retriever,
                question=question,
                chat_history="",
                language="English"
            )
            
            # Extract contexts from retrieved documents
            retrieved_docs = retriever.invoke(question)
            contexts = [doc.page_content for doc in retrieved_docs]
            
            result = {
                'question': question,
                'answer': rag_response['answer'],
                'contexts': contexts,
                'ground_truth': ground_truth,
                'case_name': case_name
            }
            
            rag_results.append(result)
            time.sleep(0.1)  # Small delay
            
        except Exception as e:
            print(f"Error processing question {i+1}: {e}")
            continue
    
    print(f"✅ Generated {len(rag_results)} RAG responses")
    
    if not rag_results:
        print("❌ No RAG results generated. Exiting.")
        return
    
    # Prepare dataset for RAGAS
    dataset_dict = {
        'question': [item['question'] for item in rag_results],
        'answer': [item['answer'] for item in rag_results],
        'contexts': [item['contexts'] for item in rag_results],
        'ground_truth': [item['ground_truth'] for item in rag_results]
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    
    # Setup LLM and embeddings
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings()
    
    # Define metrics (initialize as objects)
    metrics = [
        faithfulness(),
        answer_relevancy(),
        context_recall(),
        context_precision()
    ]
    
    print("Running RAGAS evaluation...")
    try:
        # Run evaluation
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings
        )
        
        print("✅ RAGAS evaluation completed")
        
        # Extract results properly
        if hasattr(result, 'to_pandas'):
            df = result.to_pandas()
            print("\n" + "=" * 60)
            print("RAGAS EVALUATION RESULTS")
            print("=" * 60)
            
            # Calculate and display metrics
            metrics_summary = {}
            for column in df.columns:
                if column not in ['question', 'answer', 'contexts', 'ground_truth']:
                    score = df[column].mean()
                    if not pd.isna(score):
                        metrics_summary[column] = float(score)
                        print(f"{column.upper()}: {score:.4f}")
            
            # Save detailed results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save DataFrame as CSV
            df.to_csv(f"ragas_results_{timestamp}.csv", index=False)
            
            # Save summary
            summary = {
                'evaluation_date': datetime.now().isoformat(),
                'total_questions': len(rag_results),
                'metrics': metrics_summary,
                'sample_results': rag_results[:3]  # First 3 for inspection
            }
            
            with open(f"ragas_summary_{timestamp}.json", 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            print(f"\n✅ Results saved:")
            print(f"  - Detailed: ragas_results_{timestamp}.csv")
            print(f"  - Summary: ragas_summary_{timestamp}.json")
            
            # Generate interpretation
            print(f"\n" + "=" * 60)
            print("INTERPRETATION")
            print("=" * 60)
            
            for metric, score in metrics_summary.items():
                interpretation = interpret_score(metric, score)
                print(f"{metric}: {interpretation}")
            
            print(f"\nTotal Questions Evaluated: {len(rag_results)}")
            print("Evaluation completed successfully! 🎉")
            
        else:
            print("❌ Could not extract results from evaluation")
            
    except Exception as e:
        print(f"❌ Error during RAGAS evaluation: {e}")
        print("This might be due to API rate limits or network issues.")

def interpret_score(metric_name: str, score: float) -> str:
    """Interpret RAGAS metric scores"""
    if score >= 0.8:
        return f"{score:.4f} - Excellent ✅"
    elif score >= 0.6:
        return f"{score:.4f} - Good 👍"
    elif score >= 0.4:
        return f"{score:.4f} - Fair ⚠️"
    else:
        return f"{score:.4f} - Poor ❌"

if __name__ == "__main__":
    run_ragas_evaluation_fixed()