import json
import os
from utils import load_vector_store, create_enhanced_rag_response
from datetime import datetime
from langchain_openai import ChatOpenAI
import time

def simple_rag_evaluation():
    """Simple RAG evaluation without RAGAS dependencies"""
    print("🚀 Simple RAG Evaluation for Legal Bot")
    print("=" * 50)
    
    # Load test data
    test_data_path = "data/Test_data/IndicLegalQA Dataset_10K_Revised.json"
    sample_size = 5  # Small sample for quick test
    
    try:
        with open(test_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        test_data = data[:sample_size]
        print(f"✅ Loaded {len(test_data)} test cases")
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
    
    # Setup LLM for evaluation
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Generate and evaluate responses
    results = []
    total_scores = {'relevancy': 0, 'accuracy': 0, 'completeness': 0}
    
    print(f"\nEvaluating {len(test_data)} questions...")
    
    for i, item in enumerate(test_data):
        try:
            question = item['question']
            ground_truth = item['answer']
            case_name = item.get('case_name', 'Unknown Case')
            
            print(f"\n--- Question {i+1}/{len(test_data)} ---")
            print(f"Case: {case_name}")
            print(f"Q: {question}")
            
            # Generate RAG response
            rag_response = create_enhanced_rag_response(
                retriever=retriever,
                question=question,
                chat_history="",
                language="English"
            )
            
            generated_answer = rag_response['answer']
            references = rag_response.get('references', [])
            
            print(f"Generated Answer: {generated_answer[:150]}...")
            print(f"Ground Truth: {ground_truth[:150]}...")
            print(f"References Found: {len(references)}")
            
            # Simple evaluation using LLM
            evaluation_prompt = f"""
            Evaluate the generated answer against the ground truth for this legal question.
            Rate each aspect from 1-10:

            Question: {question}
            Ground Truth: {ground_truth}
            Generated Answer: {generated_answer}

            Please rate:
            1. Relevancy (how relevant is the answer to the question): X/10
            2. Accuracy (how factually correct compared to ground truth): X/10  
            3. Completeness (how complete is the answer): X/10

            Respond in format:
            Relevancy: X/10
            Accuracy: X/10
            Completeness: X/10
            Brief explanation: [your explanation]
            """
            
            try:
                eval_response = llm.invoke(evaluation_prompt)
                eval_text = eval_response.content
                
                # Extract scores
                relevancy = extract_score(eval_text, "Relevancy")
                accuracy = extract_score(eval_text, "Accuracy") 
                completeness = extract_score(eval_text, "Completeness")
                
                print(f"Scores - Relevancy: {relevancy}/10, Accuracy: {accuracy}/10, Completeness: {completeness}/10")
                
                total_scores['relevancy'] += relevancy
                total_scores['accuracy'] += accuracy
                total_scores['completeness'] += completeness
                
                results.append({
                    'question': question,
                    'ground_truth': ground_truth,
                    'generated_answer': generated_answer,
                    'case_name': case_name,
                    'scores': {
                        'relevancy': relevancy,
                        'accuracy': accuracy,
                        'completeness': completeness
                    },
                    'references_count': len(references),
                    'evaluation': eval_text
                })
                
            except Exception as e:
                print(f"Error in evaluation: {e}")
                continue
                
            time.sleep(1)  # Rate limiting
            
        except Exception as e:
            print(f"Error processing question {i+1}: {e}")
            continue
    
    # Calculate averages
    num_results = len(results)
    if num_results > 0:
        avg_scores = {
            'relevancy': total_scores['relevancy'] / num_results,
            'accuracy': total_scores['accuracy'] / num_results,
            'completeness': total_scores['completeness'] / num_results
        }
        
        overall_score = sum(avg_scores.values()) / 3
        
        print(f"\n{'='*50}")
        print("EVALUATION RESULTS")
        print(f"{'='*50}")
        print(f"Questions Evaluated: {num_results}")
        print(f"Average Relevancy: {avg_scores['relevancy']:.2f}/10")
        print(f"Average Accuracy: {avg_scores['accuracy']:.2f}/10") 
        print(f"Average Completeness: {avg_scores['completeness']:.2f}/10")
        print(f"Overall Score: {overall_score:.2f}/10")
        
        # Interpretation
        if overall_score >= 8:
            print("🎉 Excellent Performance!")
        elif overall_score >= 6:
            print("👍 Good Performance")
        elif overall_score >= 4:
            print("⚠️ Fair Performance - Needs Improvement")
        else:
            print("❌ Poor Performance - Major Issues")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simple_rag_evaluation_{timestamp}.json"
        
        summary = {
            'evaluation_date': datetime.now().isoformat(),
            'total_questions': num_results,
            'average_scores': avg_scores,
            'overall_score': overall_score,
            'detailed_results': results
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Results saved to {filename}")
        
        # Show sample result
        if results:
            print(f"\n{'='*50}")
            print("SAMPLE RESULT")
            print(f"{'='*50}")
            sample = results[0]
            print(f"Question: {sample['question']}")
            print(f"Generated: {sample['generated_answer'][:200]}...")
            print(f"Ground Truth: {sample['ground_truth'][:200]}...")
            print(f"Scores: {sample['scores']}")
    
    else:
        print("❌ No results to analyze")

def extract_score(text, metric):
    """Extract score from evaluation text"""
    try:
        lines = text.split('\n')
        for line in lines:
            if metric.lower() in line.lower() and '/10' in line:
                # Extract number before /10
                parts = line.split('/10')[0]
                score_part = parts.split(':')[-1].strip()
                return int(score_part)
        return 5  # Default score if not found
    except:
        return 5  # Default score on error

if __name__ == "__main__":
    simple_rag_evaluation()