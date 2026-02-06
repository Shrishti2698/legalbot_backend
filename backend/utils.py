import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory

# Language configurations
LANGUAGES = {
    "English": "🇬🇧",
    "Hindi": "🇮🇳", 
    "Bengali": "🇧🇩"
}

# Legal code mappings for better understanding
LEGAL_CODES = {
    "IPC": "Indian Penal Code, 1860",
    "CrPC": "Code of Criminal Procedure, 1973", 
    "BNS": "Bharatiya Nyaya Sanhita, 2024",
    "BNSS": "Bharatiya Nagarik Suraksha Sanhita, 2024",
    "BSA": "Bharatiya Sakshya Adhiniyam, 2024"
}

# Load environment variables
load_dotenv()

# Constants
CHROMA_DIR = "chroma_db"

def get_embeddings_model():
    """Initialize and return the HuggingFace embeddings model"""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Initialize the embeddings model
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    return embeddings

def load_vector_store():
    """Load the existing vector store"""
    embeddings = get_embeddings_model()
    
    if not os.path.exists(CHROMA_DIR):
        raise ValueError(f"Vector store directory {CHROMA_DIR} does not exist. Please run ingest.py first.")
    
    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    
    return vector_store

def extract_document_name(source_path):
    """Extract document name from file path"""
    if not source_path:
        return "Unknown Document"
    
    filename = os.path.basename(source_path).lower()
    
    if "constitution" in filename:
        return "Constitution of India"
    elif "bns" in filename and "2024" in filename:
        return "Bharatiya Nyaya Sanhita (BNS) 2024"
    elif "bnss" in filename and "2024" in filename:
        return "Bharatiya Nagarik Suraksha Sanhita (BNSS) 2024"
    elif "bsa" in filename and "2024" in filename:
        return "Bharatiya Sakshya Adhiniyam (BSA) 2024"
    elif "penal" in filename or "ipc" in filename:
        return "Indian Penal Code (IPC) 1860"
    elif "crpc" in filename or "criminal" in filename:
        return "Code of Criminal Procedure (CrPC) 1973"
    elif "supreme" in filename or "sc" in filename:
        return "Supreme Court Judgments"
    elif "high" in filename or "hc" in filename:
        return "High Court Cases"
    else:
        return "Legal Document"

def generate_synthetic_reference(question, answer, language="English"):
    """Generate synthetic reference when no chunks are found"""
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    reference_prompt = f"""Based on the legal question and answer provided, generate a realistic legal reference citation.

Question: {question}
Answer: {answer}

Generate a reference in this format:
- Document: [Most likely Indian legal document name]
- Section/Article: [Relevant section or article number]
- Content: [Brief excerpt that would support the answer]

Use these document types when appropriate:
- Constitution of India (for fundamental rights, articles)
- Indian Penal Code (IPC) 1860 (for criminal offenses)
- Code of Criminal Procedure (CrPC) 1973 (for procedures)
- Bharatiya Nyaya Sanhita (BNS) 2024 (for new criminal laws)
- Supreme Court Judgments (for landmark cases)

Respond in {language}."""
    
    try:
        response = llm.invoke(reference_prompt)
        return response.content
    except:
        return "Reference: Based on general knowledge of Indian law"

def create_enhanced_rag_response(retriever, question, chat_history="", language="English"):
    """Create enhanced RAG response with references"""
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    # Language-specific instructions
    language_instructions = {
        "English": "Respond in English.",
        "Hindi": "Respond in Hindi (हिंदी में उत्तर दें).",
        "Bengali": "Respond in Bengali (বাংলায় উত্তর দিন)."
    }
    
    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(question)
    
    # Create context from retrieved documents
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # Create the main response prompt
    system_template = f"""You are an expert legal assistant specializing in Indian law. 
You MUST ONLY answer questions related to Indian law, legal matters, including but not limited to:
- The Indian Constitution and its provisions
- Indian Penal Code (IPC) sections and offenses
- Code of Criminal Procedure (CrPC) and procedural law
- Bharatiya Nyaya Sanhita (BNS) 2024 and new criminal laws
- Supreme Court and High Court judgments
- Legal rights, procedures, and remedies in India
- Civil and criminal law matters in Indian jurisdiction

For any questions not related to Indian legal matters, politely inform the user that you can only assist with Indian legal topics.

{language_instructions.get(language, "Respond in English.")}

Use the following pieces of context to answer the user's question about Indian legal matters.
Prioritize information from the provided context when available.
If the context doesn't contain the specific information needed, you can use your general knowledge about Indian law to provide a helpful response.
When referencing legal provisions, always specify:
- IPC Section numbers (e.g., "Section 302 IPC")
- CrPC Section numbers (e.g., "Section 154 CrPC")
- BNS Section numbers (e.g., "Section 103 BNS")
- Constitutional Articles (e.g., "Article 21")
- Specific case names and citations when available

When using your general knowledge, clearly indicate this in your response.

Previous conversation:
{chat_history}

Context:
{context}

Question: {question}"""
    
    # Generate main response
    response = llm.invoke(system_template)
    answer = response.content
    
    # Check if the response indicates an irrelevant query
    irrelevant_indicators = [
        "I can only assist you with questions related to Indian law",
        "I can only assist with Indian legal topics",
        "not related to Indian legal matters",
        "only assist with Indian legal",
        "Indian law and legal matters",
        "politely inform the user that you can only assist with Indian legal topics",
        "can only help with Indian law",
        "outside the scope of Indian law",
        "not within my expertise of Indian law",
        "I specialize in Indian legal matters",
        "my expertise is limited to Indian law"
    ]
    
    is_irrelevant = any(indicator.lower() in answer.lower() for indicator in irrelevant_indicators)
    
    # Process references only if query is relevant
    references = []
    if not is_irrelevant:
        seen_refs = set()
        
        if retrieved_docs:
            # Use actual retrieved documents as references
            for doc in retrieved_docs[:4]:  # Top 4 references
                doc_name = extract_document_name(doc.metadata.get('source', ''))
                content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                
                # Create unique identifier for deduplication
                ref_key = (doc_name, content_preview)
                
                if ref_key not in seen_refs:
                    seen_refs.add(ref_key)
                    references.append({
                        "document": doc_name,
                        "content": content_preview,
                        "type": "retrieved"
                    })
        else:
            # Generate synthetic reference if no chunks found but question is legal
            synthetic_ref = generate_synthetic_reference(question, answer, language)
            references.append({
                "document": "Generated Reference",
                "content": synthetic_ref,
                "type": "synthetic"
            })
    
    return {
        "answer": answer,
        "references": references
    }

def create_rag_chain(retriever, language="English"):
    """Create a RAG chain with the retriever and LLM (legacy function for compatibility)"""
    # This is kept for backward compatibility
    # The new enhanced function should be used instead
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    # Language-specific instructions
    language_instructions = {
        "English": "Respond in English.",
        "Hindi": "Respond in Hindi (हिंदी में उत्तर दें).",
        "Bengali": "Respond in Bengali (বাংলায় উত্তর দিন)."
    }
    
    # Create the prompt with conversation history context and language support
    system_template = f"""You are an expert legal assistant specializing in Indian law. 
You MUST ONLY answer questions related to Indian law, legal matters, including but not limited to:
- The Indian Constitution and its provisions
- Indian Penal Code (IPC) sections and offenses
- Code of Criminal Procedure (CrPC) and procedural law
- Bharatiya Nyaya Sanhita (BNS) 2024 and new criminal laws
- Supreme Court and High Court judgments
- Legal rights, procedures, and remedies in India
- Civil and criminal law matters in Indian jurisdiction

For any questions not related to Indian legal matters, politely inform the user that you can only assist with Indian legal topics.

{language_instructions.get(language, "Respond in English.")}

Use the following pieces of context to answer the user's question about Indian legal matters.
Prioritize information from the provided context when available.
If the context doesn't contain the specific information needed, you can use your general knowledge about Indian law to provide a helpful response.
When referencing legal provisions, always specify:
- IPC Section numbers (e.g., "Section 302 IPC")
- CrPC Section numbers (e.g., "Section 154 CrPC")
- BNS Section numbers (e.g., "Section 103 BNS")
- Constitutional Articles (e.g., "Article 21")
- Specific case names and citations when available

When using your general knowledge, clearly indicate this in your response.

Previous conversation:
{{chat_history}}

Context:
{{context}}

Question: {{input}}
"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
    ])
    
    # Create the document chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create the RAG chain
    rag_chain = create_retrieval_chain(retriever, document_chain)
    
    return rag_chain