import streamlit as st
from utils import load_vector_store, create_rag_chain, create_enhanced_rag_response, LANGUAGES
import time

# Set page configuration
st.set_page_config(
    page_title="Indian Legal Assistant",
    page_icon="⚖️",
    layout="wide"
)

# Force dark mode for better visibility
st.markdown("""
<script>
    var body = window.parent.document.querySelector('body');
    body.classList.add('dark');
</script>
""", unsafe_allow_html=True)

# Initialize session state for chat history and language
if "messages" not in st.session_state:
    st.session_state.messages = []
if "language" not in st.session_state:
    st.session_state.language = "English"

# Custom CSS for better styling with dark mode support
st.markdown("""
<style>
.main-header {
    font-family: 'Helvetica', sans-serif;
    text-align: center;
    color: #1E3A8A;
    padding: 1.5rem 0;
    border-bottom: 2px solid #E5E7EB;
    margin-bottom: 2rem;
    background-color: #F3F4F6;
    border-radius: 10px;
}
.description {
    text-align: center;
    margin-bottom: 2rem;
    font-size: 1.1rem;
    color: #4B5563;
}
.stChatMessage {
    padding: 1rem;
    border-radius: 15px;
    margin-bottom: 1rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.stChatMessageContent {
    border-radius: 12px;
}
.user-message {
    background-color: #E9F2FF;
    color: #1F2937 !important; /* Dark text for light background */
}
.assistant-message {
    background-color: #F0FFF4;
    color: #1F2937 !important; /* Dark text for light background */
}
/* Dark mode specific styles */
@media (prefers-color-scheme: dark) {
    .main-header {
        color: #93C5FD;
        background-color: #1F2937;
        border-bottom: 2px solid #374151;
    }
    .description {
        color: #D1D5DB;
    }
    .user-message {
        background-color: #1E40AF;
        color: #F3F4F6 !important;
    }
    .assistant-message {
        background-color: #065F46;
        color: #F3F4F6 !important;
    }
    .sidebar-header {
        color: #93C5FD;
        border-bottom: 1px solid #374151;
    }
    .sidebar-content {
        color: #D1D5DB;
    }
    .disclaimer {
        background-color: #78350F;
        color: #FBBF24;
    }
}
.sidebar-header {
    font-size: 1.5rem;
    font-weight: bold;
    color: #1E3A8A;
    margin-bottom: 1rem;
    border-bottom: 1px solid #E5E7EB;
    padding-bottom: 0.5rem;
}
.sidebar-content {
    color: #4B5563;
    font-size: 0.95rem;
}
.disclaimer {
    background-color: #FEF3C7;
    padding: 1rem;
    border-radius: 8px;
    margin-top: 2rem;
    font-size: 0.9rem;
    color: #92400E;
}
</style>
""", unsafe_allow_html=True)

# Multilingual titles and descriptions
titles = {
    "English": "🇮🇳 Indian Legal Assistant",
    "Hindi": "🇮🇳 भारतीय कानूनी सहायक",
    "Bengali": "🇮🇳 ভারতীয় আইনি সহায়ক"
}

descriptions = {
    "English": "This chatbot provides guidance on Indian legal matters based on the Constitution of India, landmark Supreme Court cases, and other legal documents.",
    "Hindi": "यह चैटबॉट भारतीय संविधान, सुप्रीम कोर्ट के ऐतिहासिक मामलों और अन्य कानूनी दस्तावेजों के आधार पर भारतीय कानूनी मामलों पर मार्गदर्शन प्रदान करता है।",
    "Bengali": "এই চ্যাটবট ভারতীয় সংবিধান, সুপ্রিম কোর্টের ঐতিহাসিক মামলা এবং অন্যান্য আইনি নথির ভিত্তিতে ভারতীয় আইনি বিষয়ে নির্দেশনা প্রদান করে।"
}

# App title and description
st.markdown(f'<div class="main-header"><h1>{titles.get(st.session_state.language, titles["English"])}</h1></div>', unsafe_allow_html=True)
st.markdown(f'<div class="description">{descriptions.get(st.session_state.language, descriptions["English"])}</div>', unsafe_allow_html=True)

# Language selector in sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-header">Language / भाषा / ভাষা</div>', unsafe_allow_html=True)
    selected_language = st.selectbox(
        "Choose your language:",
        options=list(LANGUAGES.keys()),
        format_func=lambda x: f"{LANGUAGES[x]} {x}",
        index=list(LANGUAGES.keys()).index(st.session_state.language)
    )
    
    # Update session state if language changed
    if selected_language != st.session_state.language:
        st.session_state.language = selected_language
        st.rerun()

# Load vector store and create retriever
@st.cache_resource
def load_retriever():
    try:
        vector_store = load_vector_store()
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        return retriever
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        return None

retriever = load_retriever()

# Function to format chat history for the RAG chain
def format_chat_history(messages):
    chat_history = ""
    for msg in messages:
        role = "Human" if msg["role"] == "user" else "Assistant"
        chat_history += f"{role}: {msg['content']}\n\n"
    return chat_history

# Display chat messages with enhanced styling
for message in st.session_state.messages:
    role_class = "user-message" if message["role"] == "user" else "assistant-message"
    with st.chat_message(message["role"]):
        st.markdown(f"<div class='{role_class}' style='color: inherit;'>{message['content']}</div>", unsafe_allow_html=True)
        
        # Display references for assistant messages
        if message["role"] == "assistant" and "references" in message:
            references = message["references"]
            if references:
                reference_labels = {
                    "English": "📚 References",
                    "Hindi": "📚 संदर्भ",
                    "Bengali": "📚 তথ্যসূত্র"
                }
                st.markdown(f"**{reference_labels.get(st.session_state.language, '📚 References')}:**")
                
                for ref in references:
                    st.markdown(f"""
                    <div style="
                        border: 1px solid #e0e0e0;
                        border-radius: 8px;
                        padding: 12px;
                        margin: 8px 0;
                        background-color: #f8f9fa;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    ">
                        <div style="font-weight: bold; color: #1f2937; margin-bottom: 8px;">
                            📖 {ref['document']}
                        </div>
                        <div style="color: #4b5563; font-size: 0.9em; line-height: 1.4;">
                            {ref['content']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# Multilingual chat input placeholders
chat_placeholders = {
    "English": "Ask about Indian legal matters...",
    "Hindi": "भारतीय कानूनी मामलों के बारे में पूछें...",
    "Bengali": "ভারতীয় আইনি বিষয়ে জিজ্ঞাসা করুন..."
}

# Chat input
if prompt := st.chat_input(chat_placeholders.get(st.session_state.language, "Ask about Indian legal matters...")):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        if retriever is None:
            error_messages = {
                "English": "Vector store not found. Please run the ingest.py script first.",
                "Hindi": "वेक्टर स्टोर नहीं मिला। कृपया पहले ingest.py स्क्रिप्ट चलाएं।",
                "Bengali": "ভেক্টর স্টোর পাওয়া যায়নি। দয়া করে প্রথমে ingest.py স্ক্রিপ্ট চালান।"
            }
            st.markdown(error_messages.get(st.session_state.language, "Vector store not found. Please run the ingest.py script first."))
        else:
            thinking_messages = {
        "English": "Thinking...",
        "Hindi": "सोच रहा हूँ...",
        "Bengali": "ভাবছি..."
    }
    with st.spinner(thinking_messages.get(st.session_state.language, "Thinking...")):
                # Format chat history for context
                chat_history = format_chat_history(st.session_state.messages[:-1])  # Exclude current message
                
                try:
                    # Use enhanced RAG response with references
                    response = create_enhanced_rag_response(
                        retriever, 
                        prompt, 
                        chat_history, 
                        st.session_state.language
                    )
                    answer = response["answer"]
                    references = response["references"]
                    
                    # Create a placeholder for the typing animation
                    message_placeholder = st.empty()
                    full_response = ""
                    
                    # Simulate typing with a simple animation
                    for chunk in answer.split():
                        full_response += chunk + " "
                        message_placeholder.markdown(f"<div class='assistant-message' style='color: inherit;'>{full_response}▌</div>", unsafe_allow_html=True)
                        time.sleep(0.05)  # Adjust speed as needed
                    
                    # Display the final answer
                    message_placeholder.markdown(f"<div class='assistant-message' style='color: inherit;'>{answer}</div>", unsafe_allow_html=True)
                    
                    # Display references in boxes
                    if references:
                        reference_labels = {
                            "English": "📚 References",
                            "Hindi": "📚 संदर्भ",
                            "Bengali": "📚 তথ্যসূত্র"
                        }
                        st.markdown(f"**{reference_labels.get(st.session_state.language, '📚 References')}:**")
                        
                        for i, ref in enumerate(references, 1):
                            with st.container():
                                st.markdown(f"""
                                <div style="
                                    border: 1px solid #e0e0e0;
                                    border-radius: 8px;
                                    padding: 12px;
                                    margin: 8px 0;
                                    background-color: #f8f9fa;
                                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                ">
                                    <div style="font-weight: bold; color: #1f2937; margin-bottom: 8px;">
                                        📖 {ref['document']}
                                    </div>
                                    <div style="color: #4b5563; font-size: 0.9em; line-height: 1.4;">
                                        {ref['content']}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": answer, "references": references})
                except Exception as e:
                    error_messages = {
                        "English": f"Error generating response: {e}",
                        "Hindi": f"उत्तर उत्पन्न करने में त्रुटि: {e}",
                        "Bengali": f"উত্তর তৈরিতে ত্রুটি: {e}"
                    }
                    st.error(error_messages.get(st.session_state.language, f"Error generating response: {e}"))

# Sidebar with information and enhanced styling
with st.sidebar:
    st.markdown('<div class="sidebar-header">About</div>', unsafe_allow_html=True)
    about_content = {
        "English": "This legal assistant uses Retrieval-Augmented Generation (RAG) to provide accurate information about Indian law, drawing from:<br><br>• <b>The Constitution of India</b><br>• <b>Supreme Court landmark judgments</b><br>• <b>High Court cases</b><br>• <b>Legal precedents and statutes</b>",
        "Hindi": "यह कानूनी सहायक भारतीय कानून के बारे में सटीक जानकारी प्रदान करने के लिए रिट्रीवल-ऑगमेंटेड जेनरेशन (RAG) का उपयोग करता है:<br><br>• <b>भारत का संविधान</b><br>• <b>सुप्रीम कोर्ट के ऐतिहासिक फैसले</b><br>• <b>हाई कोर्ट के मामले</b><br>• <b>कानूनी मिसालें और कानून</b>",
        "Bengali": "এই আইনি সহায়ক ভারতীয় আইন সম্পর্কে নির্ভুল তথ্য প্রদানের জন্য রিট্রিভাল-অগমেন্টেড জেনারেশন (RAG) ব্যবহার করে:<br><br>• <b>ভারতের সংবিধান</b><br>• <b>সুপ্রিম কোর্টের ঐতিহাসিক রায়</b><br>• <b>হাইকোর্টের মামলা</b><br>• <b>আইনি নজির ও আইন</b>"
    }
    st.markdown(f'<div class="sidebar-content">{about_content.get(st.session_state.language, about_content["English"])}</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-header">Features</div>', unsafe_allow_html=True)
    features_content = {
        "English": "• Answers questions about Indian law and legal rights<br>• Provides information on legal procedures<br>• Explains constitutional provisions<br>• References relevant court cases<br>• Maintains conversation context",
        "Hindi": "• भारतीय कानून और कानूनी अधिकारों के बारे में प्रश्नों के उत्तर देता है<br>• कानूनी प्रक्रियाओं की जानकारी प्रदान करता है<br>• संवैधानिक प्रावधानों की व्याख्या करता है<br>• संबंधित न्यायालयी मामलों का संदर्भ देता है<br>• बातचीत का संदर्भ बनाए रखता है",
        "Bengali": "• ভারতীয় আইন ও আইনি অধিকার সম্পর্কে প্রশ্নের উত্তর দেয়<br>• আইনি প্রক্রিয়া সম্পর্কে তথ্য প্রদান করে<br>• সাংবিধানিক বিধান ব্যাখ্যা করে<br>• প্রাসঙ্গিক আদালতের মামলার রেফারেন্স দেয়<br>• কথোপকথনের প্রসঙ্গ বজায় রাখে"
    }
    st.markdown(f'<div class="sidebar-content">{features_content.get(st.session_state.language, features_content["English"])}</div>', unsafe_allow_html=True)
    
    disclaimer_content = {
        "English": "⚠️ <b>Disclaimer:</b> This chatbot provides general legal information and not legal advice. Always consult with a qualified legal professional for specific legal matters.",
        "Hindi": "⚠️ <b>अस्वीकरण:</b> यह चैटबॉट सामान्य कानूनी जानकारी प्रदान करता है, कानूनी सलाह नहीं। विशिष्ट कानूनी मामलों के लिए हमेशा योग्य कानूनी पेशेवर से सलाह लें।",
        "Bengali": "⚠️ <b>দাবিত্যাग:</b> এই চ্যাটবট সাধারণ আইনি তথ্য প্রদান করে, আইনি পরামর্শ নয়। নির্দিষ্ট আইনি বিষয়ের জন্য সর্বদা যোগ্য আইনি পেশাদারের সাথে পরামর্শ করুন।"
    }
    st.markdown(f'<div class="disclaimer">{disclaimer_content.get(st.session_state.language, disclaimer_content["English"])}</div>', unsafe_allow_html=True)
    
    # Clear chat button with better styling
    col1, col2 = st.columns([1, 1])
    with col2:
        clear_button_text = {
            "English": "Clear Chat",
            "Hindi": "चैट साफ करें",
            "Bengali": "চ্যাট সাফ করুন"
        }
        if st.button(clear_button_text.get(st.session_state.language, "Clear Chat"), type="primary"):
            st.session_state.messages = []
            st.rerun()
    
    # Language info
    st.markdown('<div class="sidebar-header">Supported Languages</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-content">🇬🇧 English<br>🇮🇳 हिंदी (Hindi)<br>🇧🇩 বাংলা (Bengali)</div>', unsafe_allow_html=True)