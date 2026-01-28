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
    context_precision,
    answer_correctness,
    answer_similarity
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from utils import load_vector_store, create_enhanced_rag_response
import time
from datetime import datetime

class LegalRAGASEvaluator:
    def __init__(self, test_data_path: str, sample_size: int = 50):
        """
        Initialize RAGAS evaluator for legal bot
        
        Args:
            test_data_path: Path to IndicLegalQA dataset
            sample_size: Number of samples to evaluate (default 50 for quick testing)
        """
        self.test_data_path = test_data_path
        self.sample_size = sample_size
        self.vector_store = None
        self.retriever = None
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.embeddings = OpenAIEmbeddings()
        
        # Load test data
        self.test_data = self._load_test_data()
        
    def _load_test_data(self) -> List[Dict]:
        """Load and sample test data from IndicLegalQA dataset"""
        try:
            with open(self.test_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Sample data for evaluation
            sampled_data = data[:self.sample_size] if len(data) > self.sample_size else data
            print(f"Loaded {len(sampled_data)} test cases from {len(data)} total cases")
            return sampled_data
            
        except Exception as e:
            print(f"Error loading test data: {e}")
            return []
    
    def setup_rag_system(self):
        """Setup RAG system with vector store"""
        try:
            self.vector_store = load_vector_store()
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
            print("✅ RAG system setup complete")
        except Exception as e:
            print(f"❌ Error setting up RAG system: {e}")
            raise
    
    def generate_rag_responses(self) -> List[Dict]:
        """Generate RAG responses for test questions"""
        results = []
        
        print(f"Generating RAG responses for {len(self.test_data)} questions...")
        
        for i, item in enumerate(self.test_data):
            try:
                question = item['question']
                ground_truth = item['answer']
                case_name = item.get('case_name', 'Unknown Case')
                
                print(f"Processing {i+1}/{len(self.test_data)}: {question[:50]}...")
                
                # Generate RAG response
                rag_response = create_enhanced_rag_response(
                    retriever=self.retriever,
                    question=question,
                    chat_history="",
                    language="English"
                )
                
                # Extract contexts from retrieved documents
                retrieved_docs = self.retriever.invoke(question)
                contexts = [doc.page_content for doc in retrieved_docs]
                
                result = {
                    'question': question,
                    'answer': rag_response['answer'],
                    'contexts': contexts,
                    'ground_truth': ground_truth,
                    'case_name': case_name,
                    'references': rag_response.get('references', [])
                }
                
                results.append(result)
                
                # Small delay to avoid rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error processing question {i+1}: {e}")
                continue
        
        print(f"✅ Generated {len(results)} RAG responses")
        return results
    
    def run_ragas_evaluation(self, rag_results: List[Dict]):
        """Run RAGAS evaluation on generated responses"""
        print("Running RAGAS evaluation...")
        
        # Convert to RAGAS dataset format
        dataset_dict = {
            'question': [item['question'] for item in rag_results],
            'answer': [item['answer'] for item in rag_results],
            'contexts': [item['contexts'] for item in rag_results],
            'ground_truth': [item['ground_truth'] for item in rag_results]
        }
        
        dataset = Dataset.from_dict(dataset_dict)
        
        # Define metrics to evaluate
        metrics = [
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
            answer_correctness,
            answer_similarity
        ]
        
        try:
            # Run evaluation
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=self.llm,
                embeddings=self.embeddings
            )
            
            print("✅ RAGAS evaluation completed")
            return result
            
        except Exception as e:
            print(f"❌ Error during RAGAS evaluation: {e}")
            return None
    
    def analyze_results(self, evaluation_result, rag_results: List[Dict]) -> Dict:
        """Analyze and summarize evaluation results"""
        print("Analyzing evaluation results...")
        
        # Extract metrics - handle new RAGAS EvaluationResult format
        metrics_summary = {}
        if evaluation_result:
            # Convert EvaluationResult to dict if needed
            if hasattr(evaluation_result, 'to_pandas'):
                df = evaluation_result.to_pandas()
                for column in df.columns:
                    if column not in ['question', 'answer', 'contexts', 'ground_truth']:
                        score = df[column].mean()
                        if not pd.isna(score):
                            metrics_summary[column] = {
                                'score': round(float(score), 4),
                                'interpretation': self._interpret_score(column, float(score))
                            }
            else:
                # Fallback for older format
                for metric_name, score in evaluation_result.items():
                    if isinstance(score, (int, float)):
                        metrics_summary[metric_name] = {
                            'score': round(score, 4),
                            'interpretation': self._interpret_score(metric_name, score)
                        }
        
        # Analyze by case types
        case_analysis = self._analyze_by_case_type(rag_results, evaluation_result)
        
        # Find best and worst performing questions
        performance_analysis = self._analyze_performance(rag_results, evaluation_result)
        
        analysis = {
            'evaluation_summary': {
                'total_questions': len(rag_results),
                'evaluation_date': datetime.now().isoformat(),
                'metrics': metrics_summary
            },
            'case_type_analysis': case_analysis,
            'performance_analysis': performance_analysis,
            'recommendations': self._generate_recommendations(metrics_summary)
        }
        
        return analysis
    
    def _interpret_score(self, metric_name: str, score: float) -> str:
        """Interpret RAGAS metric scores"""
        interpretations = {
            'faithfulness': {
                0.8: "Excellent - Answers are highly faithful to retrieved context",
                0.6: "Good - Most answers are faithful with minor hallucinations",
                0.4: "Fair - Some hallucinations present, needs improvement",
                0.0: "Poor - Significant hallucinations, major concern"
            },
            'answer_relevancy': {
                0.8: "Excellent - Answers are highly relevant to questions",
                0.6: "Good - Most answers are relevant",
                0.4: "Fair - Some answers lack relevance",
                0.0: "Poor - Many irrelevant answers"
            },
            'context_recall': {
                0.8: "Excellent - Retrieved context covers most ground truth",
                0.6: "Good - Context covers important information",
                0.4: "Fair - Some important information missing",
                0.0: "Poor - Context misses key information"
            },
            'context_precision': {
                0.8: "Excellent - Retrieved context is highly precise",
                0.6: "Good - Most retrieved context is relevant",
                0.4: "Fair - Some irrelevant context retrieved",
                0.0: "Poor - Much irrelevant context retrieved"
            },
            'answer_correctness': {
                0.8: "Excellent - Answers are highly accurate",
                0.6: "Good - Most answers are correct",
                0.4: "Fair - Some inaccuracies present",
                0.0: "Poor - Many incorrect answers"
            },
            'answer_similarity': {
                0.8: "Excellent - Answers very similar to ground truth",
                0.6: "Good - Answers reasonably similar",
                0.4: "Fair - Some similarity to ground truth",
                0.0: "Poor - Answers differ significantly from ground truth"
            }
        }
        
        if metric_name in interpretations:
            thresholds = sorted(interpretations[metric_name].keys(), reverse=True)
            for threshold in thresholds:
                if score >= threshold:
                    return interpretations[metric_name][threshold]
        
        return "Score interpretation not available"
    
    def _analyze_by_case_type(self, rag_results: List[Dict], evaluation_result: Dict) -> Dict:
        """Analyze performance by case type"""
        case_types = {}
        
        for item in rag_results:
            case_name = item.get('case_name', 'Unknown')
            # Extract case type from case name (simplified)
            if 'vs.' in case_name:
                case_type = 'Civil Case'
            elif 'Union of India' in case_name:
                case_type = 'Government Case'
            elif 'Commissioner' in case_name or 'Income Tax' in case_name:
                case_type = 'Tax Case'
            else:
                case_type = 'Other'
            
            if case_type not in case_types:
                case_types[case_type] = []
            case_types[case_type].append(item)
        
        return {case_type: len(cases) for case_type, cases in case_types.items()}
    
    def _analyze_performance(self, rag_results: List[Dict], evaluation_result: Dict) -> Dict:
        """Analyze best and worst performing questions"""
        # This is a simplified analysis - in practice, you'd need individual scores
        return {
            'total_evaluated': len(rag_results),
            'avg_context_length': sum(len(' '.join(item['contexts'])) for item in rag_results) / len(rag_results),
            'avg_answer_length': sum(len(item['answer']) for item in rag_results) / len(rag_results)
        }
    
    def _generate_recommendations(self, metrics_summary: Dict) -> List[str]:
        """Generate recommendations based on evaluation results"""
        recommendations = []
        
        for metric, data in metrics_summary.items():
            score = data['score']
            
            if metric == 'faithfulness' and score < 0.7:
                recommendations.append("Improve faithfulness by better prompt engineering and context filtering")
            
            if metric == 'context_recall' and score < 0.7:
                recommendations.append("Enhance retrieval system - consider increasing chunk overlap or improving embeddings")
            
            if metric == 'context_precision' and score < 0.7:
                recommendations.append("Improve context precision by fine-tuning retrieval parameters")
            
            if metric == 'answer_relevancy' and score < 0.7:
                recommendations.append("Improve answer relevancy through better prompt templates")
        
        if not recommendations:
            recommendations.append("System performance is good across all metrics")
        
        return recommendations
    
    def save_results(self, analysis: Dict, rag_results: List[Dict], filename: str = None):
        """Save evaluation results to files"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"legal_ragas_evaluation_{timestamp}"
        
        # Save detailed results
        detailed_results = {
            'analysis': analysis,
            'detailed_results': rag_results[:10]  # Save first 10 for inspection
        }
        
        with open(f"{filename}_detailed.json", 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        # Save summary report
        summary_report = self._generate_summary_report(analysis)
        with open(f"{filename}_summary.txt", 'w', encoding='utf-8') as f:
            f.write(summary_report)
        
        print(f"✅ Results saved to {filename}_detailed.json and {filename}_summary.txt")
    
    def _generate_summary_report(self, analysis: Dict) -> str:
        """Generate human-readable summary report"""
        report = []
        report.append("=" * 60)
        report.append("LEGAL RAG SYSTEM - RAGAS EVALUATION REPORT")
        report.append("=" * 60)
        report.append(f"Evaluation Date: {analysis['evaluation_summary']['evaluation_date']}")
        report.append(f"Total Questions Evaluated: {analysis['evaluation_summary']['total_questions']}")
        report.append("")
        
        report.append("METRICS SUMMARY:")
        report.append("-" * 30)
        for metric, data in analysis['evaluation_summary']['metrics'].items():
            report.append(f"{metric.upper()}: {data['score']:.4f}")
            report.append(f"  → {data['interpretation']}")
            report.append("")
        
        report.append("CASE TYPE DISTRIBUTION:")
        report.append("-" * 30)
        for case_type, count in analysis['case_type_analysis'].items():
            report.append(f"{case_type}: {count} cases")
        report.append("")
        
        report.append("RECOMMENDATIONS:")
        report.append("-" * 30)
        for i, rec in enumerate(analysis['recommendations'], 1):
            report.append(f"{i}. {rec}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)

def main():
    """Main function to run RAGAS evaluation"""
    print("🚀 Starting Legal RAG System RAGAS Evaluation")
    print("=" * 50)
    
    # Initialize evaluator
    test_data_path = "data/Test_data/IndicLegalQA Dataset_10K_Revised.json"
    evaluator = LegalRAGASEvaluator(test_data_path, sample_size=20)  # Small sample for testing
    
    if not evaluator.test_data:
        print("❌ No test data loaded. Exiting.")
        return
    
    # Setup RAG system
    evaluator.setup_rag_system()
    
    # Generate RAG responses
    rag_results = evaluator.generate_rag_responses()
    
    if not rag_results:
        print("❌ No RAG results generated. Exiting.")
        return
    
    # Run RAGAS evaluation
    evaluation_result = evaluator.run_ragas_evaluation(rag_results)
    
    # Analyze results
    analysis = evaluator.analyze_results(evaluation_result, rag_results)
    
    # Save results
    evaluator.save_results(analysis, rag_results)
    
    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION COMPLETED!")
    print("=" * 50)
    
    if analysis['evaluation_summary']['metrics']:
        print("\nKey Metrics:")
        for metric, data in analysis['evaluation_summary']['metrics'].items():
            print(f"  {metric}: {data['score']:.4f}")
    
    print(f"\nTotal Questions Evaluated: {len(rag_results)}")
    print("\nCheck the generated files for detailed results!")

if __name__ == "__main__":
    main()