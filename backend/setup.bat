@echo off
REM Backend Setup Script

echo Setting up Legal Advisor Backend...

REM Create virtual environment
python -m venv venv
call venv\Scripts\activate

REM Install dependencies
pip install -r requirements.txt

REM Create .env file if it doesn't exist
if not exist .env (
    copy .env.example .env
    echo.
    echo Created .env file. Please edit it with your configuration before running the server.
    echo.
) else (
    echo .env file already exists.
)

echo.
echo Setup complete! To start the server, run:
echo   python main.py
echo or
echo   uvicorn main:app --reload
