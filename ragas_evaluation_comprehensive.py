"""
Comprehensive RAGAS Evaluation Script for Legal Chatbot
Evaluates RAG system performance using IndicLegalQA dataset with multiple metrics
"""

import json
import os
from datetime import datetime
from typing import List, Dict
import pandas as pd
from tqdm import tqdm

# RAGAS imports
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness
)
from datasets import Dataset

# Local imports
from utils import load_vector_store, create_enhanced_rag_response
from langchain_openai import ChatOpenAI

class RAGASEvaluator:
    """Comprehensive RAGAS evaluation for legal chatbot"""
    
    def __init__(self, dataset_path: str, sample_size: int = 100):
        """
        Initialize RAGAS evaluator
        
        Args:
            dataset_path: Path to IndicLegalQA JSON dataset
            sample_size: Number of questions to evaluate (default 100)
        """
        self.dataset_path = dataset_path
        self.sample_size = sample_size
        self.results_dir = "evaluation_results"
        self.viz_dir = os.path.join(self.results_dir, "visualizations")
        
        # Create directories
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # Load vector store
        print("[*] Loading vector store...")
        vector_store = load_vector_store()
        self.retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        
        # Initialize LLM for evaluation
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        
        print(f"[OK] Initialized evaluator with sample size: {sample_size}")
    
    def load_dataset(self) -> List[Dict]:
        """Load and sample the IndicLegalQA dataset"""
        print(f"[*] Loading dataset from {self.dataset_path}...")
        
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"[OK] Loaded {len(data)} total records")
        
        # Sample the dataset
        import random
        random.seed(42)  # For reproducibility
        sampled_data = random.sample(data, min(self.sample_size, len(data)))
        
        print(f"[*] Sampled {len(sampled_data)} questions for evaluation")
        return sampled_data
    
    def generate_rag_responses(self, questions_data: List[Dict]) -> List[Dict]:
        """
        Generate RAG responses for all questions
        
        Returns:
            List of dicts with question, answer, contexts, ground_truth
        """
        print("\n[*] Generating RAG responses...")
        
        evaluation_data = []
        
        for item in tqdm(questions_data, desc="Processing questions", ascii=True):
            question = item['question']
            ground_truth = item['answer']
            
            try:
                # Get RAG response with context
                response = create_enhanced_rag_response(
                    retriever=self.retriever,
                    question=question,
                    chat_history="",
                    language="English"
                )
                
                # Get retrieved documents for context
                retrieved_docs = self.retriever.invoke(question)
                contexts = [doc.page_content for doc in retrieved_docs]
                
                evaluation_data.append({
                    "question": question,
                    "answer": response["answer"],
                    "contexts": contexts,
                    "ground_truth": ground_truth,
                    "case_name": item.get('case_name', 'Unknown'),
                    "judgement_date": item.get('judgement_date', 'Unknown')
                })
                
            except Exception as e:
                print(f"\n[WARNING] Error processing question: {question[:50]}... Error: {e}")
                continue
        
        print(f"\n[OK] Generated {len(evaluation_data)} responses successfully")
        return evaluation_data
    
    def run_ragas_evaluation(self, evaluation_data: List[Dict]) -> Dict:
        """
        Run RAGAS evaluation with all metrics
        
        Args:
            evaluation_data: List of dicts with question, answer, contexts, ground_truth
            
        Returns:
            Dictionary with evaluation results
        """
        print("\n[*] Running RAGAS evaluation...")
        
        # Convert to RAGAS dataset format
        dataset_dict = {
            "question": [item["question"] for item in evaluation_data],
            "answer": [item["answer"] for item in evaluation_data],
            "contexts": [item["contexts"] for item in evaluation_data],
            "ground_truth": [item["ground_truth"] for item in evaluation_data]
        }
        
        dataset = Dataset.from_dict(dataset_dict)
        
        # Run evaluation with all metrics
        print("[*] Calculating metrics (this may take several minutes)...")
        
        try:
            result = evaluate(
                dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                    answer_correctness
                ],
                llm=self.llm
            )
            
            print("\n[OK] RAGAS evaluation completed!")
            return result
            
        except Exception as e:
            print(f"\n[ERROR] Error during RAGAS evaluation: {e}")
            raise
    
    def save_results(self, evaluation_data: List[Dict], ragas_results: Dict) -> str:
        """
        Save detailed results to JSON file
        
        Returns:
            Path to saved results file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.results_dir, f"ragas_evaluation_{timestamp}.json")
        
        # Combine evaluation data with RAGAS scores
        detailed_results = []
        
        # Convert ragas_results to DataFrame for easier access
        results_df = ragas_results.to_pandas()
        
        for i, item in enumerate(evaluation_data):
            detailed_results.append({
                "question": item["question"],
                "answer": item["answer"],
                "ground_truth": item["ground_truth"],
                "case_name": item["case_name"],
                "judgement_date": item["judgement_date"],
                "contexts": item["contexts"],
                "metrics": {
                    "faithfulness": float(results_df.iloc[i]["faithfulness"]) if "faithfulness" in results_df.columns else None,
                    "answer_relevancy": float(results_df.iloc[i]["answer_relevancy"]) if "answer_relevancy" in results_df.columns else None,
                    "context_precision": float(results_df.iloc[i]["context_precision"]) if "context_precision" in results_df.columns else None,
                    "context_recall": float(results_df.iloc[i]["context_recall"]) if "context_recall" in results_df.columns else None,
                    "answer_correctness": float(results_df.iloc[i]["answer_correctness"]) if "answer_correctness" in results_df.columns else None
                }
            })
        
        # Save to JSON
        output = {
            "timestamp": timestamp,
            "sample_size": len(evaluation_data),
            "overall_metrics": {
                metric: float(ragas_results[metric]) for metric in ragas_results.keys()
            },
            "detailed_results": detailed_results
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\n[OK] Detailed results saved to: {results_file}")
        
        # Also save summary
        summary_file = os.path.join(self.results_dir, f"metrics_summary_{timestamp}.json")
        summary = {
            "timestamp": timestamp,
            "sample_size": len(evaluation_data),
            "metrics": output["overall_metrics"]
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"[OK] Summary saved to: {summary_file}")
        
        return results_file
    
    def print_summary(self, ragas_results):
        """Print evaluation summary to console"""
        print("\n" + "="*60)
        print("RAGAS EVALUATION SUMMARY")
        print("="*60)
        
        # Convert to dict if it's an EvaluationResult object
        if hasattr(ragas_results, 'to_pandas'):
            results_dict = ragas_results.to_pandas().mean().to_dict()
        else:
            results_dict = ragas_results
        
        for metric, score in results_dict.items():
            print(f"{metric:.<40} {score:.4f}")
        
        print("="*60 + "\n")


def main(sample_size: int = 100, test_mode: bool = False):
    """
    Main execution function
    
    Args:
        sample_size: Number of questions to evaluate
        test_mode: If True, use very small sample for testing
    """
    if test_mode:
        sample_size = 5
        print("[TEST MODE] Running with 5 samples")
    
    # Dataset path
    dataset_path = r"c:\Users\USER\Documents\legal_bot_project\legal-advisor\data\Test_data\IndicLegalQA Dataset_10K_Revised.json"
    
    # Initialize evaluator
    evaluator = RAGASEvaluator(dataset_path, sample_size)
    
    # Load dataset
    questions_data = evaluator.load_dataset()
    
    # Generate RAG responses
    evaluation_data = evaluator.generate_rag_responses(questions_data)
    
    if len(evaluation_data) == 0:
        print("[ERROR] No evaluation data generated. Exiting.")
        return
    
    # Run RAGAS evaluation
    ragas_results = evaluator.run_ragas_evaluation(evaluation_data)
    
    # CRITICAL: Save results FIRST before any display operations
    results_file = None
    try:
        results_file = evaluator.save_results(evaluation_data, ragas_results)
        print(f"\n[OK] Results saved successfully to: {results_file}")
    except Exception as e:
        print(f"\n[ERROR] Failed to save results: {e}")
        import traceback
        traceback.print_exc()
    
    # Try to print summary (non-critical if it fails)
    try:
        evaluator.print_summary(ragas_results)
    except Exception as e:
        print(f"\n[WARNING] Could not print summary: {e}")
        print("Results are saved in JSON files.")
    
    if results_file:
        print(f"\n[OK] Evaluation complete! Results saved to: {results_file}")
        print(f"[*] Next step: Run visualize_ragas_results.py to generate graphs")
    else:
        print("\n[ERROR] Evaluation completed but results were not saved properly.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on legal chatbot")
    parser.add_argument("--sample-size", type=int, default=100, help="Number of questions to evaluate")
    parser.add_argument("--test-mode", action="store_true", help="Run in test mode with 5 samples")
    
    args = parser.parse_args()
    
    main(sample_size=args.sample_size, test_mode=args.test_mode)
