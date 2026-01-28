# RAGAS Evaluation Setup

This guide explains how to evaluate your Indian Legal Assistant chatbot using RAGAS (Retrieval Augmented Generation Assessment).

## What is RAGAS?

RAGAS is a framework for evaluating RAG (Retrieval-Augmented Generation) systems. It measures:

1. **Faithfulness** - Whether the answer is factually consistent with the retrieved context
2. **Answer Relevancy** - How relevant the answer is to the question
3. **Context Precision** - How precise the retrieved context is (less noise)
4. **Context Recall** - How much of the ground truth is covered by retrieved context

## Files Overview

- `data/RAGAS_groundTruth.json` - 31 ground truth QnA pairs covering Indian law
- `evaluate_ragas.py` - Main evaluation script
- `visualize_ragas.py` - Visualization and reporting script
- `run_ragas_eval.bat` - Quick run batch file

## Setup

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Set OpenAI API Key**
Make sure your `.env` file contains:
```
OPENAI_API_KEY=your_api_key_here
```

3. **Ensure Vector Store Exists**
The evaluation requires your ChromaDB vector store. If not created:
```bash
python ingest.py
```

## Running Evaluation

### Method 1: Using Batch File (Easiest)
```bash
run_ragas_eval.bat
```

### Method 2: Using Python
```bash
python evaluate_ragas.py
```

This will:
- Load your vector store
- Process all 31 ground truth questions
- Generate RAG responses
- Calculate RAGAS metrics
- Save results to `ragas_evaluation_results.json`

**Expected Runtime:** 5-10 minutes (depends on API speed)

## Viewing Results

After evaluation completes, visualize the results:

```bash
python visualize_ragas.py
```

This generates:
- **Bar charts** showing metric scores
- **Radar chart** for metric distribution
- **Performance summary table**
- **Key insights and recommendations**
- Saved as `ragas_visualization_YYYYMMDD_HHMMSS.png`

## Understanding Metrics

### Score Interpretation

| Score Range | Status | Meaning |
|-------------|--------|---------|
| 0.7 - 1.0 | ✓ Good | Excellent performance |
| 0.5 - 0.7 | ⚠ Fair | Acceptable, room for improvement |
| 0.0 - 0.5 | ✗ Poor | Needs significant improvement |

### Metric Details

**Faithfulness (0-1)**
- Measures factual consistency
- Higher = answers are more grounded in retrieved context
- Low score → Model hallucinating or adding unsupported info

**Answer Relevancy (0-1)**
- Measures how well answer addresses the question
- Higher = more relevant and focused answers
- Low score → Answers are off-topic or verbose

**Context Precision (0-1)**
- Measures quality of retrieved documents
- Higher = less irrelevant documents retrieved
- Low score → Retrieval bringing too much noise

**Context Recall (0-1)**
- Measures coverage of ground truth in retrieved context
- Higher = retrieved docs contain necessary information
- Low score → Important information not being retrieved

## Ground Truth Coverage

The evaluation dataset covers:

| Category | Questions | Topics |
|----------|-----------|--------|
| Constitutional Law | 7 | Articles 13, 14, 21, 21A, 22, 39A, 124 |
| Indian Penal Code | 9 | Murder, theft, suicide, intimidation |
| Bharatiya Nyaya Sanhita | 9 | New criminal code, mob lynching, terrorism |
| Criminal Procedure | 6 | FIR, arrest, bail, custody |

## Improving Scores

### Low Faithfulness
- Improve prompt engineering to stick to context
- Adjust retrieval parameters (k value)
- Add more relevant documents to vector store

### Low Answer Relevancy
- Refine system prompts
- Improve question understanding
- Filter out verbose responses

### Low Context Precision
- Improve embedding model
- Adjust chunk size and overlap
- Add metadata filtering

### Low Context Recall
- Increase k (number of retrieved docs)
- Improve document chunking strategy
- Add more comprehensive documents

## Customization

### Add More Questions

Edit `data/RAGAS_groundTruth.json`:
```json
[
  {
    "question": "Your question here?",
    "answer": "Expected answer here."
  }
]
```

### Change Retrieval Parameters

In `evaluate_ragas.py`, modify:
```python
retriever = vector_store.as_retriever(search_kwargs={"k": 5})  # Change k value
```

### Use Different LLM

In `evaluate_ragas.py`, change:
```python
llm=ChatOpenAI(model="gpt-4o-mini")  # Try gpt-4, gpt-3.5-turbo, etc.
```

## Output Files

- `ragas_evaluation_results.json` - Detailed results with all QnA pairs
- `ragas_visualization_*.png` - Visual charts and insights

## Troubleshooting

**Error: Vector store not found**
```bash
python ingest.py  # Create vector store first
```

**Error: OpenAI API key not found**
- Check `.env` file exists
- Verify `OPENAI_API_KEY` is set correctly

**Error: Module not found**
```bash
pip install -r requirements.txt  # Reinstall dependencies
```

**Slow evaluation**
- Normal for 31 questions (5-10 min)
- Each question requires LLM calls for evaluation
- Consider reducing questions for quick tests

## Best Practices

1. **Run evaluation regularly** - After major changes to:
   - RAG pipeline
   - Prompt templates
   - Vector store updates
   - Retrieval parameters

2. **Track metrics over time** - Save results with timestamps

3. **Focus on weakest metric** - Prioritize improvements

4. **Test with real queries** - Supplement with actual user questions

5. **Balance metrics** - Don't optimize one at expense of others

## Example Results

Good performance typically shows:
```
Faithfulness:       0.8500
Answer Relevancy:   0.7800
Context Precision:  0.7200
Context Recall:     0.8100
```

## Additional Resources

- [RAGAS Documentation](https://docs.ragas.io/)
- [LangChain RAG Guide](https://python.langchain.com/docs/use_cases/question_answering/)
- [ChromaDB Documentation](https://docs.trychroma.com/)

## Support

For issues or questions:
1. Check this README
2. Review error messages carefully
3. Verify all dependencies installed
4. Ensure vector store exists and is populated
