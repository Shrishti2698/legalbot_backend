@echo off
REM Batch script to run RAGAS evaluation in conda environment

echo Activating conda environment: legal-chatbot
call conda activate legal-chatbot

echo.
echo Running RAGAS evaluation in TEST MODE (5 samples)...
echo.

python ragas_evaluation_comprehensive.py --test-mode

echo.
echo Evaluation complete!
pause
