# RAGAS Evaluation System for Legal Chatbot

## Overview

This evaluation system uses **RAGAS** (Retrieval Augmented Generation Assessment) to comprehensively evaluate the legal chatbot's RAG pipeline using the **IndicLegalQA dataset** containing 10,000 legal Q&A pairs from Indian court cases.

## 📊 Evaluation Metrics

The system calculates 5 key RAGAS metrics:

1. **Faithfulness** - Measures if the answer is grounded in the retrieved context
2. **Answer Relevancy** - Measures how relevant the answer is to the question
3. **Context Precision** - Measures if relevant chunks are ranked higher
4. **Context Recall** - Measures if all relevant information is retrieved
5. **Answer Correctness** - Compares generated answer with ground truth

## 🚀 Quick Start

### Option 1: Interactive Runner (Recommended)

```bash
cd c:\Users\USER\Documents\legal_bot_project\legal-advisor
python run_ragas_evaluation.py
```

This will:
- Prompt you for sample size
- Run the evaluation
- Generate all visualizations automatically
- Create an HTML report

### Option 2: Command Line

**Small test (5 questions, ~2 minutes):**
```bash
python ragas_evaluation_comprehensive.py --test-mode
```

**Standard evaluation (100 questions, ~20 minutes):**
```bash
python ragas_evaluation_comprehensive.py --sample-size 100
```

**Then generate visualizations:**
```bash
python visualize_ragas_results.py
```

## 📁 Output Structure

After running evaluation, you'll find:

```
evaluation_results/
├── ragas_evaluation_YYYYMMDD_HHMMSS.json      # Detailed results
├── metrics_summary_YYYYMMDD_HHMMSS.json       # Summary statistics
└── visualizations/
    ├── overall_metrics_YYYYMMDD_HHMMSS.png
    ├── metrics_distribution_YYYYMMDD_HHMMSS.png
    ├── correlation_heatmap_YYYYMMDD_HHMMSS.png
    ├── metrics_boxplot_YYYYMMDD_HHMMSS.png
    ├── statistics_table_YYYYMMDD_HHMMSS.png
    ├── performance_radar_YYYYMMDD_HHMMSS.png
    └── evaluation_report_YYYYMMDD_HHMMSS.html  # 📄 Open this!
```

## 📈 Generated Visualizations

1. **Overall Metrics Bar Chart** - Shows all 5 metrics at a glance
2. **Distribution Histograms** - Shows score distribution for each metric
3. **Correlation Heatmap** - Shows relationships between metrics
4. **Box Plots** - Compares metric ranges and outliers
5. **Statistics Table** - Detailed mean, median, std dev, min, max
6. **Performance Radar** - 360° view of all metrics
7. **HTML Report** - Interactive report with all visualizations

## 🔧 Configuration

### Sample Size Recommendations

- **Test**: 5-10 questions (~2 minutes) - Quick sanity check
- **Quick**: 50 questions (~10 minutes) - Fast iteration
- **Standard**: 100 questions (~20 minutes) - Good balance
- **Comprehensive**: 500+ questions (1+ hours) - Full evaluation

### Custom Sample Size

```bash
python ragas_evaluation_comprehensive.py --sample-size 250
```

## 📝 Understanding the Results

### Metric Interpretation

- **0.8 - 1.0**: Excellent performance
- **0.6 - 0.8**: Good performance
- **0.4 - 0.6**: Moderate performance (needs improvement)
- **0.0 - 0.4**: Poor performance (requires attention)

### What to Look For

1. **High Faithfulness** - Answers are grounded in retrieved documents
2. **High Answer Relevancy** - Answers directly address the questions
3. **High Context Precision** - Most relevant chunks ranked at top
4. **High Context Recall** - All relevant information is retrieved
5. **High Answer Correctness** - Generated answers match ground truth

## 🔍 Troubleshooting

### Common Issues

**Issue**: `FileNotFoundError: No RAGAS evaluation results found!`
- **Solution**: Run `ragas_evaluation_comprehensive.py` first before visualizations

**Issue**: Evaluation is very slow
- **Solution**: Reduce sample size or check OpenAI API rate limits

**Issue**: `ModuleNotFoundError: No module named 'ragas'`
- **Solution**: Install dependencies: `pip install ragas datasets matplotlib seaborn`

## 📚 Dataset Information

**IndicLegalQA Dataset**
- **Location**: `data/Test_data/IndicLegalQA Dataset_10K_Revised.json`
- **Total Records**: 10,000 Q&A pairs
- **Source**: Indian Supreme Court and High Court judgements
- **Fields**: case_name, judgement_date, question, answer

## 🎯 Next Steps

After reviewing results:

1. **Identify Weak Metrics** - Focus on metrics with scores < 0.6
2. **Improve RAG Pipeline**:
   - Adjust chunk size/overlap for better context
   - Tune retrieval parameters (k, search type)
   - Improve prompt engineering
3. **Re-evaluate** - Run evaluation again to measure improvements
4. **Track Progress** - Compare results over time using timestamps

## 📞 Support

For issues or questions about the evaluation system, check:
- Implementation plan: `implementation_plan.md`
- Task checklist: `task.md`
