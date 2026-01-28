@echo off
REM Batch script to run RAGAS evaluation with 100 samples

echo Activating conda environment: legal-chatbot
call conda activate legal-chatbot

echo.
echo Running RAGAS evaluation with 100 samples...
echo This will take approximately 20-30 minutes.
echo.

python ragas_evaluation_comprehensive.py --sample-size 100

echo.
echo Evaluation complete! Now generating visualizations...
echo.

python visualize_ragas_results.py

echo.
echo All done! Check evaluation_results folder for outputs.
pause
