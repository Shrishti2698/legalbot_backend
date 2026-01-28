@echo off
echo Starting RAGAS Evaluation for Indian Legal Assistant...
echo.

call conda activate legal-chatbot

echo Installing required packages...
pip install ragas datasets pandas matplotlib seaborn

echo.
echo Running evaluation...
python evaluate.py

echo.
echo Evaluation completed! Check the generated files:
echo - rag_evaluation_detailed_results.csv
echo - rag_evaluation_summary.csv  
echo - rag_evaluation_latex_table.tex
echo - rag_evaluation_results.png

pause