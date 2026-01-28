import json
import pandas as pd
import os
from typing import List, Dict
from datasets import Dataset
from ragas import evaluate
from ragas.metrics.collections import faithfulness, answer_relevancy
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from utils import load_vector_store, create_enhanced_rag_response
import time
from datetime import datetime

def ragas_without_ground_truth():
    """RAGAS evaluation focusing on faithfulness and answer relevancy without ground truth"""
    print("🚀 RAGAS Evaluation - No Ground Truth Required")
    print("=" * 60)
    
    # Load test questions only (ignore ground truth answers)
    test_data_path = "data/Test_data/IndicLegalQA Dataset_10K_Revised.json"
    sample_size = 8  # Small sample for reliability
    
    try:
        with open(test_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        test_data = data[:sample_size]
        print(f"✅ Loaded {len(test_data)} test questions (ignoring ground truth)")
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return
    
    # Setup RAG system
    try:
        vector_store = load_vector_store()
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
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
                'case_name': case_name,
                'references_count': len(rag_response.get('references', []))
            }
            
            rag_results.append(result)
            time.sleep(0.2)  # Rate limiting
            
        except Exception as e:
            print(f"Error processing question {i+1}: {e}")
            continue
    
    print(f"✅ Generated {len(rag_results)} RAG responses")
    
    if not rag_results:
        print("❌ No RAG results generated. Exiting.")
        return
    
    # Prepare dataset for RAGAS (without ground truth)
    dataset_dict = {
        'question': [item['question'] for item in rag_results],
        'answer': [item['answer'] for item in rag_results],
        'contexts': [item['contexts'] for item in rag_results]
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    
    # Setup LLM and embeddings
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings()
    
    print("Running RAGAS evaluation (faithfulness & answer relevancy only)...")
    
    try:
        # Define metrics that don't require ground truth
        metrics = [
            faithfulness,      # Does the answer stick to the retrieved context?
            answer_relevancy   # Is the answer relevant to the question?
        ]
        
        # Run evaluation
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings
        )
        
        print("✅ RAGAS evaluation completed")
        
        # Process results
        try:
            if hasattr(result, 'to_pandas'):
                df = result.to_pandas()
            else:
                # Handle different result formats
                df = pd.DataFrame(result)
            
            print(f"\n{'='*60}")
            print("RAGAS EVALUATION RESULTS")
            print(f"{'='*60}")
            
            # Calculate metrics
            metrics_summary = {}
            for column in df.columns:
                if column not in ['question', 'answer', 'contexts']:
                    try:
                        score = pd.to_numeric(df[column], errors='coerce').mean()
                        if not pd.isna(score):
                            metrics_summary[column] = float(score)
                            interpretation = interpret_score(column, float(score))
                            print(f"{column.upper()}: {score:.4f} - {interpretation}")
                    except Exception as e:
                        print(f"Could not process metric {column}: {e}")
            
            # Additional analysis
            print(f"\n{'='*60}")
            print("SYSTEM ANALYSIS")
            print(f"{'='*60}")
            print(f"Questions Evaluated: {len(rag_results)}")
            print(f"Average Answer Length: {sum(len(r['answer']) for r in rag_results) / len(rag_results):.0f} characters")
            print(f"Average Contexts Retrieved: {sum(len(r['contexts']) for r in rag_results) / len(rag_results):.1f}")
            print(f"Average References Found: {sum(r['references_count'] for r in rag_results) / len(rag_results):.1f}")
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save detailed CSV
            df.to_csv(f"ragas_no_ground_truth_{timestamp}.csv", index=False)
            
            # Save summary JSON
            summary = {
                'evaluation_date': datetime.now().isoformat(),
                'evaluation_type': 'RAGAS without ground truth',
                'total_questions': len(rag_results),
                'metrics': metrics_summary,
                'sample_results': rag_results[:2],  # First 2 for inspection
                'analysis': {
                    'avg_answer_length': sum(len(r['answer']) for r in rag_results) / len(rag_results),
                    'avg_contexts_retrieved': sum(len(r['contexts']) for r in rag_results) / len(rag_results),
                    'avg_references_found': sum(r['references_count'] for r in rag_results) / len(rag_results)
                }
            }
            
            with open(f"ragas_summary_no_gt_{timestamp}.json", 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            print(f"\n✅ Results saved:")
            print(f"  - Detailed: ragas_no_ground_truth_{timestamp}.csv")
            print(f"  - Summary: ragas_summary_no_gt_{timestamp}.json")
            
            # Generate recommendations
            print(f"\n{'='*60}")
            print("RECOMMENDATIONS")
            print(f"{'='*60}")
            
            recommendations = []
            
            if 'faithfulness' in metrics_summary:
                faithfulness_score = metrics_summary['faithfulness']
                if faithfulness_score < 0.7:
                    recommendations.append("🔧 Improve faithfulness: Refine prompts to stick closer to retrieved context")
                elif faithfulness_score >= 0.8:
                    recommendations.append("✅ Excellent faithfulness: Answers stick well to retrieved context")
            
            if 'answer_relevancy' in metrics_summary:
                relevancy_score = metrics_summary['answer_relevancy']
                if relevancy_score < 0.7:
                    recommendations.append("🔧 Improve answer relevancy: Better question understanding needed")
                elif relevancy_score >= 0.8:
                    recommendations.append("✅ Excellent answer relevancy: Answers are highly relevant")
            
            if not recommendations:
                recommendations.append("✅ System performance is good across evaluated metrics")
            
            for rec in recommendations:
                print(f"  {rec}")
            
            # Show sample result
            print(f"\n{'='*60}")
            print("SAMPLE RESULT")
            print(f"{'='*60}")
            if rag_results:
                sample = rag_results[0]
                print(f"Question: {sample['question']}")
                print(f"Generated Answer: {sample['answer'][:200]}...")
                print(f"Contexts Retrieved: {len(sample['contexts'])}")
                print(f"References Found: {sample['references_count']}")
                if metrics_summary:
                    scores_text = ", ".join([f"{k}={v:.3f}" for k, v in metrics_summary.items()])
                    print(f"Quality Scores: {scores_text}")
            
            print(f"\n🎉 RAGAS evaluation completed successfully!")
            
        except Exception as e:
            print(f"Error processing results: {e}")
            print("Raw result type:", type(result))
            if hasattr(result, '__dict__'):
                print("Result attributes:", list(result.__dict__.keys()))
            
    except Exception as e:
        print(f"❌ Error during RAGAS evaluation: {e}")
        print("This might be due to API rate limits or version compatibility issues.")

def interpret_score(metric_name: str, score: float) -> str:
    """Interpret RAGAS metric scores"""
    if score >= 0.8:
        return "Excellent ✅"
    elif score >= 0.6:
        return "Good 👍"
    elif score >= 0.4:
        return "Fair ⚠️"
    else:
        return "Poor ❌"

if __name__ == "__main__":
    ragas_without_ground_truth()