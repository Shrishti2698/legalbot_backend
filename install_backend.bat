@echo off
echo Installing backend dependencies...
echo Make sure you have activated your conda environment first:
echo conda activate legal-chatbot
echo.
pip install fastapi uvicorn
echo.
echo Installation complete! Now run:
echo cd backend
echo python main.py
pause