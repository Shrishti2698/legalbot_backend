# RAGAS Evaluation for Legal Bot

This directory contains RAGAS (Retrieval-Augmented Generation Assessment) evaluation tools for the Indian Legal Assistant chatbot using the IndicLegalQA dataset.

## 📋 Overview

RAGAS evaluates your RAG system across multiple dimensions:
- **Faithfulness**: How well answers stick to retrieved context
- **Answer Relevancy**: How relevant answers are to questions  
- **Context Recall**: How well retrieval covers ground truth
- **Context Precision**: How precise retrieved context is
- **Answer Correctness**: How factually correct answers are
- **Answer Similarity**: How similar answers are to ground truth

## 🗂️ Files

- `ragas_evaluation.py` - Main comprehensive evaluation script
- `quick_ragas_test.py` - Quick test with 5 samples
- `run_ragas_evaluation.bat` - Windows batch script to run evaluation
- `data/Test_data/IndicLegalQA Dataset_10K_Revised.json` - Test dataset

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Activate your conda environment
conda activate legal-chatbot

# Install RAGAS dependencies
pip install ragas datasets pandas numpy scikit-learn

# Ensure you have vector store ready
python ingest.py  # If not already done
```

### 2. Quick Test (5 samples)

```bash
python quick_ragas_test.py
```

### 3. Full Evaluation (customizable sample size)

```bash
python ragas_evaluation.py
```

Or use the batch script:
```bash
run_ragas_evaluation.bat
```

## 📊 Understanding Results

### Metric Scores (0.0 - 1.0)

| Score Range | Interpretation |
|-------------|----------------|
| 0.8 - 1.0   | Excellent ✅   |
| 0.6 - 0.8   | Good 👍        |
| 0.4 - 0.6   | Fair ⚠️        |
| 0.0 - 0.4   | Poor ❌        |

### Key Metrics Explained

**Faithfulness (Most Important)**
- Measures hallucinations
- High score = answers stick to retrieved context
- Low score = model making things up

**Answer Relevancy** 
- How well answers address the question
- High score = answers are on-topic
- Low score = answers are off-topic

**Context Recall**
- How much ground truth info is in retrieved context
- High score = retrieval finds relevant info
- Low score = important info missing from retrieval

**Context Precision**
- How much retrieved context is actually relevant
- High score = clean, relevant context
- Low score = noisy, irrelevant context

## 📁 Output Files

After evaluation, you'll get:

1. **`legal_ragas_evaluation_YYYYMMDD_HHMMSS_detailed.json`**
   - Complete results with all data
   - Individual question responses
   - Retrieved contexts and references

2. **`legal_ragas_evaluation_YYYYMMDD_HHMMSS_summary.txt`**
   - Human-readable summary report
   - Metric scores and interpretations
   - Recommendations for improvement

## ⚙️ Configuration

### Sample Size
Edit `ragas_evaluation.py`:
```python
evaluator = LegalRAGASEvaluator(test_data_path, sample_size=50)  # Change this number
```

### Retrieval Parameters
Edit the retrieval setup:
```python
self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})  # Change k value
```

### Metrics Selection
Choose which metrics to evaluate:
```python
metrics = [
    faithfulness,        # Keep this - most important
    answer_relevancy,    # Keep this - very important  
    context_recall,      # Optional - slower
    context_precision,   # Optional - slower
    answer_correctness,  # Optional - requires ground truth
    answer_similarity    # Optional - requires ground truth
]
```

## 🔧 Troubleshooting

### Common Issues

**"Vector store not found"**
```bash
python ingest.py  # Create vector store first
```

**"Test data not found"**
- Ensure `IndicLegalQA Dataset_10K_Revised.json` is in `data/Test_data/`

**"OpenAI API Error"**
- Check your `.env` file has `OPENAI_API_KEY`
- Ensure you have API credits

**"Memory Error"**
- Reduce sample size in evaluation script
- Use `quick_ragas_test.py` instead

**"Slow Evaluation"**
- Reduce sample size
- Remove slower metrics (context_recall, context_precision)
- Use smaller k value for retrieval

### Performance Tips

1. **Start Small**: Use `quick_ragas_test.py` first
2. **Gradual Scale**: 5 → 20 → 50 → 100 samples
3. **Focus on Key Metrics**: Faithfulness + Answer Relevancy
4. **Monitor Costs**: RAGAS uses OpenAI API calls

## 📈 Interpreting Results

### Good Performance Indicators
- Faithfulness > 0.7 (minimal hallucinations)
- Answer Relevancy > 0.7 (answers are on-topic)
- Context Recall > 0.6 (retrieval finds key info)

### Red Flags
- Faithfulness < 0.5 (too many hallucinations)
- Answer Relevancy < 0.5 (answers off-topic)
- Context Precision < 0.3 (too much noise in retrieval)

### Improvement Strategies

**Low Faithfulness**
- Improve prompts to stick to context
- Add "only use provided context" instructions
- Filter out low-quality retrieved chunks

**Low Answer Relevancy**
- Improve question understanding in prompts
- Better retrieval query processing
- Add question-answer alignment checks

**Low Context Recall**
- Increase chunk overlap in document processing
- Improve embedding model
- Increase retrieval k value

**Low Context Precision**
- Better chunk splitting strategy
- Improve retrieval scoring
- Add context filtering

## 🎯 Example Usage

```python
from ragas_evaluation import LegalRAGASEvaluator

# Initialize with custom settings
evaluator = LegalRAGASEvaluator(
    test_data_path="data/Test_data/IndicLegalQA Dataset_10K_Revised.json",
    sample_size=20
)

# Setup and run
evaluator.setup_rag_system()
rag_results = evaluator.generate_rag_responses()
evaluation_result = evaluator.run_ragas_evaluation(rag_results)
analysis = evaluator.analyze_results(evaluation_result, rag_results)
evaluator.save_results(analysis, rag_results)
```

## 📚 Dataset Information

**IndicLegalQA Dataset**
- 10,000+ legal Q&A pairs
- Indian Supreme Court cases
- Various legal domains (civil, criminal, tax, etc.)
- High-quality ground truth answers

## 🤝 Contributing

To improve the evaluation:
1. Add more diverse test cases
2. Implement domain-specific metrics
3. Add multilingual evaluation support
4. Create automated benchmarking

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section
2. Verify all prerequisites are installed
3. Start with `quick_ragas_test.py`
4. Check OpenAI API key and credits