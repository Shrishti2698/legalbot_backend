@echo off
echo Activating legal-chatbot environment and starting Jupyter...

call conda activate legal-chatbot
pip install jupyter ipykernel ragas datasets pandas matplotlib seaborn
jupyter notebook evaluate.ipynb

pause