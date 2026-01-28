@echo off
echo Setting up Jupyter kernel for legal-chatbot environment...

call conda activate legal-chatbot
pip install ipykernel
python -m ipykernel install --user --name=legal-chatbot --display-name="Python (legal-chatbot)"

echo Kernel setup completed!
echo You can now select "Python (legal-chatbot)" as your kernel in Jupyter.
pause