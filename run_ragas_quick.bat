@echo off
REM Quick RAGAS evaluation with 20 samples (~5 minutes)

echo Activating conda environment: legal-chatbot
call conda activate legal-chatbot

echo.
echo Running RAGAS evaluation with 20 samples...
echo Estimated time: 5-7 minutes
echo.

python ragas_evaluation_comprehensive.py --sample-size 20

echo.
echo Evaluation complete! Now generating visualizations...
echo.

python visualize_ragas_results.py

echo.
echo All done! Check evaluation_results folder.
pause
