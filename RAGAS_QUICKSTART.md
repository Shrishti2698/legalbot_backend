# RAGAS Evaluation - Quick Start Guide

## 🚀 Quick Start (3 Steps)

### Step 1: Check Setup
```bash
python check_ragas_setup.py
```
This verifies:
- ✓ OpenAI API key configured
- ✓ Ground truth data exists (31 QnA pairs)
- ✓ Vector store is ready
- ✓ All packages installed

### Step 2: Run Evaluation
```bash
python evaluate_ragas.py
```
or simply double-click:
```
run_ragas_eval.bat
```

**Time:** ~5-10 minutes for 31 questions

### Step 3: View Results
```bash
python visualize_ragas.py
```

This shows:
- 📊 Metric scores (bar chart)
- 🎯 Performance radar
- 📋 Summary table
- 💡 Insights & recommendations

---

## 📊 What Gets Evaluated?

### 4 Key Metrics

| Metric | What it Measures | Good Score |
|--------|------------------|------------|
| **Faithfulness** | Answer accuracy vs context | > 0.7 |
| **Answer Relevancy** | How well answer fits question | > 0.7 |
| **Context Precision** | Quality of retrieved docs | > 0.7 |
| **Context Recall** | Coverage of ground truth | > 0.7 |

### 31 Test Questions Cover:

- ⚖️ **Constitutional Law** (7 questions)
  - Fundamental rights, Articles 14, 21, 21A, 22, 39A, 124
  
- 📜 **Indian Penal Code** (9 questions)
  - Murder, theft, suicide, criminal intimidation
  
- 🆕 **Bharatiya Nyaya Sanhita** (9 questions)
  - New criminal code, mob lynching, terrorism, organized crime
  
- 🏛️ **Criminal Procedure** (6 questions)
  - FIR, arrest, bail, police custody

---

## 📁 Files Created

```
legal-advisor/
├── data/
│   └── RAGAS_groundTruth.json          # 31 QnA pairs
├── evaluate_ragas.py                    # Main evaluation script
├── visualize_ragas.py                   # Visualization script
├── check_ragas_setup.py                 # Setup checker
├── run_ragas_eval.bat                   # Quick run batch file
├── RAGAS_EVALUATION_GUIDE.md            # Detailed guide
└── RAGAS_QUICKSTART.md                  # This file
```

---

## 🎯 Expected Output

### Console Output
```
====================================
RAGAS Evaluation Results
====================================

Faithfulness:       0.8234
Answer Relevancy:   0.7891
Context Precision:  0.7456
Context Recall:     0.8012

✓ Detailed results saved to: ragas_evaluation_results.json
```

### Generated Files
- `ragas_evaluation_results.json` - Full results with all QnA
- `ragas_visualization_YYYYMMDD_HHMMSS.png` - Charts and insights

---

## 🔧 Troubleshooting

### "Vector store not found"
```bash
python ingest.py  # Create vector store first
```

### "OPENAI_API_KEY not found"
Create/edit `.env` file:
```
OPENAI_API_KEY=sk-your-key-here
```

### "Module not found"
```bash
pip install -r requirements.txt
```

### Evaluation too slow?
- Normal: 5-10 minutes for 31 questions
- Each question needs multiple LLM calls
- For quick test, reduce questions in ground truth file

---

## 💡 Tips

1. **First time?** Run `check_ragas_setup.py` first
2. **Track progress** - Save results with different timestamps
3. **Compare versions** - Run after each major change
4. **Focus improvements** - Target lowest scoring metric
5. **Real-world test** - Add your own questions to ground truth

---

## 📈 Interpreting Results

### Excellent (0.8+)
```
✓ System performing very well
✓ Ready for production use
✓ Minor optimizations only
```

### Good (0.7-0.8)
```
✓ Solid performance
⚠ Some room for improvement
→ Focus on lowest metric
```

### Fair (0.5-0.7)
```
⚠ Acceptable but needs work
→ Review retrieval strategy
→ Improve prompts
→ Add more documents
```

### Poor (<0.5)
```
✗ Significant issues
→ Check vector store quality
→ Review chunking strategy
→ Verify ground truth accuracy
```

---

## 🎓 Next Steps

After evaluation:

1. **Review detailed results** in JSON file
2. **Identify weakest metric**
3. **Apply targeted improvements**:
   - Low Faithfulness → Better prompts
   - Low Relevancy → Refine responses
   - Low Precision → Better retrieval
   - Low Recall → More/better docs
4. **Re-evaluate** to measure improvement
5. **Iterate** until satisfied

---

## 📚 Resources

- Full Guide: `RAGAS_EVALUATION_GUIDE.md`
- RAGAS Docs: https://docs.ragas.io/
- Ground Truth: `data/RAGAS_groundTruth.json`

---

## ✅ Checklist

Before running evaluation:
- [ ] OpenAI API key configured
- [ ] Vector store exists (chroma_db/)
- [ ] Ground truth file present
- [ ] All packages installed
- [ ] Run check_ragas_setup.py

Ready to evaluate:
- [ ] Run evaluate_ragas.py
- [ ] Wait 5-10 minutes
- [ ] Check results JSON
- [ ] Run visualize_ragas.py
- [ ] Review charts and insights

---

**Need help?** Check `RAGAS_EVALUATION_GUIDE.md` for detailed documentation.
