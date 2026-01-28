# Indian Legal Assistant Chatbot

A RAG-based chatbot that provides guidance on Indian legal matters using the Constitution of India, landmark Supreme Court cases, and other legal documents.

## Setup

1. Create and activate a conda environment:

```bash
conda create -n legal-chatbot python=3.10 -y
conda activate legal-chatbot
```

2. Install the requirements:

```bash
pip install -r requirements.txt
```

3. Make sure you have already run the ingestion process to create the vector store:

```bash
python ingest.py
```

4. Run the Streamlit app:

```bash
streamlit run app.py
```

## Features

- Chat interface with conversation history
- Retrieval-Augmented Generation (RAG) for accurate legal information
- Special weightage to the Constitution of India
- Citations to specific legal documents and cases

## Technologies Used

- ChromaDB for vector storage
- HuggingFace all-MiniLM-L6-v2 for embeddings
- GPT-4o-mini for response generation
- LangChain for the RAG pipeline
- Streamlit for the user interface

## Disclaimer

This chatbot provides general legal information and not legal advice. Always consult with a qualified legal professional for specific legal matters.