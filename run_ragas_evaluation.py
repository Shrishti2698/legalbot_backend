"""
Quick Runner Script for RAGAS Evaluation
Simplified interface to run evaluation and generate visualizations
"""

import sys
import os

def print_banner():
    """Print welcome banner"""
    print("\n" + "="*70)
    print("🏛️  LEGAL CHATBOT - RAGAS EVALUATION SYSTEM")
    print("="*70 + "\n")

def main():
    """Main execution"""
    print_banner()
    
    print("This script will:")
    print("  1. Load the IndicLegalQA dataset (10,000 Q&A pairs)")
    print("  2. Sample questions for evaluation")
    print("  3. Generate RAG responses")
    print("  4. Calculate RAGAS metrics")
    print("  5. Generate visualizations and reports\n")
    
    # Get sample size from user
    print("How many questions would you like to evaluate?")
    print("  - Small test: 5-10 questions (~2 minutes)")
    print("  - Quick eval: 50 questions (~10 minutes)")
    print("  - Standard: 100 questions (~20 minutes)")
    print("  - Comprehensive: 500+ questions (1+ hours)\n")
    
    while True:
        try:
            sample_size = input("Enter sample size (default 100): ").strip()
            if sample_size == "":
                sample_size = 100
            else:
                sample_size = int(sample_size)
            
            if sample_size < 1:
                print("❌ Sample size must be at least 1")
                continue
            
            break
        except ValueError:
            print("❌ Please enter a valid number")
    
    print(f"\n✅ Will evaluate {sample_size} questions\n")
    
    # Run evaluation
    print("="*70)
    print("STEP 1: Running RAGAS Evaluation")
    print("="*70 + "\n")
    
    import ragas_evaluation_comprehensive
    ragas_evaluation_comprehensive.main(sample_size=sample_size, test_mode=False)
    
    # Generate visualizations
    print("\n" + "="*70)
    print("STEP 2: Generating Visualizations")
    print("="*70 + "\n")
    
    import visualize_ragas_results
    visualize_ragas_results.main()
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE!")
    print("="*70)
    print("\n📂 Check the 'evaluation_results' folder for:")
    print("   - Detailed JSON results")
    print("   - Visualization graphs (PNG)")
    print("   - HTML report\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
