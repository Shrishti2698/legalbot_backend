@echo off
echo ========================================
echo Legal RAG System - RAGAS Evaluation
echo ========================================
echo.

echo Checking if virtual environment is activated...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please activate your conda environment first.
    echo Run: conda activate legal-chatbot
    pause
    exit /b 1
)

echo.
echo Checking required dependencies...
python -c "import ragas" >nul 2>&1
if errorlevel 1 (
    echo Installing RAGAS dependencies...
    pip install ragas datasets pandas numpy scikit-learn
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
)

echo.
echo Checking if vector store exists...
if not exist "chroma_db" (
    echo ERROR: Vector store not found!
    echo Please run 'python ingest.py' first to create the vector store.
    pause
    exit /b 1
)

echo.
echo Checking if test data exists...
if not exist "data\Test_data\IndicLegalQA Dataset_10K_Revised.json" (
    echo ERROR: Test data not found!
    echo Please ensure the IndicLegalQA dataset is in data/Test_data/ folder.
    pause
    exit /b 1
)

echo.
echo Starting RAGAS evaluation...
echo This may take several minutes depending on the sample size...
echo.

python ragas_evaluation.py

if errorlevel 1 (
    echo.
    echo ERROR: Evaluation failed!
    pause
    exit /b 1
) else (
    echo.
    echo ========================================
    echo Evaluation completed successfully!
    echo Check the generated files for results.
    echo ========================================
)

pause