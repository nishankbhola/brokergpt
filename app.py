import pysqlite3 as sqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import shutil
import json
import requests
import streamlit as st
import time
from PIL import Image
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Detect if running on Streamlit Cloud
def is_streamlit_cloud():
    return os.environ.get("HOME") == "/home/adminuser"

def create_chroma_vectorstore(vectorstore_path, max_retries=3):
    """Create Chroma vectorstore with retry logic"""
    for attempt in range(max_retries):
        try:
            if hasattr(st.session_state, 'chroma_client'):
                del st.session_state.chroma_client
            
            os.makedirs(vectorstore_path, exist_ok=True)
            
            vectorstore = Chroma(
                persist_directory=vectorstore_path,
                embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"),
                client_settings=None
            )
            
            vectorstore._client.heartbeat()
            return vectorstore
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                if os.path.exists(vectorstore_path):
                    for file in os.listdir(vectorstore_path):
                        if file.endswith('.sqlite3') or file.endswith('.db'):
                            try:
                                os.remove(os.path.join(vectorstore_path, file))
                            except:
                                pass
            else:
                raise e

def get_company_logo(company_name):
    """Get company logo if it exists"""
    logo_path = os.path.join("data/logos", f"{company_name}.png")
    if os.path.exists(logo_path):
        return Image.open(logo_path)
    return None

def get_available_companies():
    """Get list of available companies"""
    company_base_dir = "data/pdfs"
    if not os.path.exists(company_base_dir):
        return []
    
    company_folders = [f for f in os.listdir(company_base_dir) 
                      if os.path.isdir(os.path.join(company_base_dir, f))]
    return company_folders

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Broker-GPT",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ChatGPT-style CSS
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #f7f7f8;
    }
    
    /* Main container */
    .main-container {
        max-width: 900px;
        margin: 0 auto;
        padding: 0 20px;
    }
    
    /* Header */
    .chat-header {
        background: #ffffff;
        padding: 1rem 2rem;
        border-bottom: 1px solid #e5e5e5;
        display: flex;
        align-items: center;
        justify-content: space-between;
        position: sticky;
        top: 0;
        z-index: 1000;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .header-left {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .logo {
        font-size: 1.5rem;
        font-weight: 700;
        color: #10a37f;
    }
    
    .company-badge {
        background: #10a37f;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    /* Chat container */
    .chat-container {
        background: #ffffff;
        min-height: calc(100vh - 140px);
        display: flex;
        flex-direction: column;
    }
    
    /* Messages area */
    .messages-container {
        flex: 1;
        padding: 2rem;
        overflow-y: auto;
        max-height: calc(100vh - 200px);
    }
    
    /* Message styling */
    .message {
        margin-bottom: 2rem;
        display: flex;
        gap: 1rem;
    }
    
    .message-user {
        flex-direction: row-reverse;
    }
    
    .message-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 1.1rem;
        flex-shrink: 0;
    }
    
    .user-avatar {
        background: #10a37f;
        color: white;
    }
    
    .bot-avatar {
        background: #f7f7f8;
        color: #374151;
        border: 1px solid #e5e5e5;
    }
    
    .message-content {
        max-width: 70%;
        padding: 1rem 1.5rem;
        border-radius: 18px;
        word-wrap: break-word;
        line-height: 1.6;
    }
    
    .user-message {
        background: #10a37f;
        color: white;
        border-bottom-right-radius: 4px;
    }
    
    .bot-message {
        background: #f7f7f8;
        color: #374151;
        border-bottom-left-radius: 4px;
        border: 1px solid #e5e5e5;
    }
    
    /* Input area */
    .input-container {
        padding: 1rem 2rem 2rem;
        background: #ffffff;
        border-top: 1px solid #e5e5e5;
    }
    
    .input-wrapper {
        position: relative;
        max-width: 700px;
        margin: 0 auto;
    }
    
    .stTextArea textarea {
        border: 1px solid #e5e5e5 !important;
        border-radius: 25px !important;
        padding: 1rem 3rem 1rem 1.5rem !important;
        resize: none !important;
        background: #ffffff !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 1rem !important;
        line-height: 1.5 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
    }
    
    .stTextArea textarea:focus {
        outline: none !important;
        border-color: #10a37f !important;
        box-shadow: 0 0 0 2px rgba(16, 163, 127, 0.2) !important;
    }
    
    .send-button {
        position: absolute;
        right: 8px;
        top: 50%;
        transform: translateY(-50%);
        background: #10a37f;
        color: white;
        border: none;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: all 0.2s;
        font-size: 1.2rem;
    }
    
    .send-button:hover {
        background: #0d8a6b;
        transform: translateY(-50%) scale(1.05);
    }
    
    .send-button:disabled {
        background: #d1d5db;
        cursor: not-allowed;
        transform: translateY(-50%) scale(1);
    }
    
    /* Welcome message */
    .welcome-container {
        text-align: center;
        padding: 4rem 2rem;
        color: #6b7280;
    }
    
    .welcome-title {
        font-size: 2rem;
        font-weight: 600;
        color: #374151;
        margin-bottom: 1rem;
    }
    
    .welcome-subtitle {
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .suggestions {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        justify-content: center;
        margin-top: 2rem;
    }
    
    .suggestion {
        background: #f7f7f8;
        border: 1px solid #e5e5e5;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        cursor: pointer;
        transition: all 0.2s;
        font-size: 0.9rem;
    }
    
    .suggestion:hover {
        background: #f3f4f6;
        border-color: #10a37f;
    }
    
    /* Company selector */
    .company-selector {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 1001;
        background: white;
        border: 1px solid #e5e5e5;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        min-width: 200px;
    }
    
    .company-selector h4 {
        margin: 0 0 0.5rem 0;
        font-size: 0.9rem;
        color: #6b7280;
    }
    
    .stSelectbox > div > div {
        border-radius: 8px !important;
        border: 1px solid #e5e5e5 !important;
    }
    
    /* Loading animation */
    .loading-message {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #6b7280;
        font-style: italic;
    }
    
    .loading-dots {
        animation: blink 1.4s infinite;
    }
    
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0; }
    }
    
    /* Admin panel toggle */
    .admin-toggle {
        position: fixed;
        bottom: 20px;
        left: 20px;
        z-index: 1001;
        background: #374151;
        color: white;
        border: none;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        font-size: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        transition: all 0.2s;
    }
    
    .admin-toggle:hover {
        background: #4b5563;
        transform: scale(1.05);
    }
    
    /* Admin panel */
    .admin-panel {
        position: fixed;
        left: 0;
        top: 0;
        width: 350px;
        height: 100vh;
        background: #ffffff;
        border-right: 1px solid #e5e5e5;
        box-shadow: 4px 0 12px rgba(0,0,0,0.1);
        transform: translateX(-100%);
        transition: transform 0.3s ease;
        z-index: 1002;
        overflow-y: auto;
        padding: 2rem;
    }
    
    .admin-panel.open {
        transform: translateX(0);
    }
    
    .admin-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #e5e5e5;
    }
    
    .admin-close {
        background: none;
        border: none;
        font-size: 1.5rem;
        cursor: pointer;
        color: #6b7280;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-container {
            padding: 0 10px;
        }
        
        .chat-header {
            padding: 1rem;
        }
        
        .messages-container {
            padding: 1rem;
        }
        
        .input-container {
            padding: 1rem;
        }
        
        .message-content {
            max-width: 85%;
        }
        
        .company-selector {
            position: relative;
            top: auto;
            right: auto;
            margin-bottom: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'selected_company' not in st.session_state:
    st.session_state.selected_company = None
if 'admin_authenticated' not in st.session_state:
    st.session_state.admin_authenticated = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'show_admin_panel' not in st.session_state:
    st.session_state.show_admin_panel = False

# Create necessary directories
company_base_dir = "data/pdfs"
logos_dir = "data/logos"
os.makedirs(company_base_dir, exist_ok=True)
os.makedirs(logos_dir, exist_ok=True)

# Get available companies
available_companies = get_available_companies()

# Auto-select first company if none selected and companies exist
if not st.session_state.selected_company and available_companies:
    st.session_state.selected_company = available_companies[0]

# Header
st.markdown("""
<div class="chat-header">
    <div class="header-left">
        <div class="logo">🤖 Broker-GPT</div>
        <div class="company-badge">{}</div>
    </div>
</div>
""".format(st.session_state.selected_company or "No Company Selected"), unsafe_allow_html=True)

# Company selector (floating)
if available_companies:
    st.markdown('<div class="company-selector">', unsafe_allow_html=True)
    st.markdown('<h4>Select Company:</h4>', unsafe_allow_html=True)
    
    company_index = available_companies.index(st.session_state.selected_company) if st.session_state.selected_company in available_companies else 0
    selected_company_new = st.selectbox(
        "company",
        available_companies,
        index=company_index,
        key="company_selector_main",
        label_visibility="collapsed"
    )
    
    if selected_company_new != st.session_state.selected_company:
        st.session_state.selected_company = selected_company_new
        st.session_state.chat_history = []
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Admin toggle button
if st.button("⚙️", key="admin_toggle", help="Admin Panel"):
    st.session_state.show_admin_panel = not st.session_state.show_admin_panel
    st.rerun()

# Admin panel (if open)
if st.session_state.show_admin_panel:
    with st.container():
        st.markdown('<div class="admin-panel open">', unsafe_allow_html=True)
        
        # Admin header
        st.markdown("""
        <div class="admin-header">
            <h3>Admin Panel</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Admin authentication
        if not st.session_state.admin_authenticated:
            st.markdown("### 🔐 Admin Login")
            admin_password = st.text_input("Password:", type="password", key="admin_pass")
            if st.button("Login", key="admin_login"):
                if admin_password == "classmate":
                    st.session_state.admin_authenticated = True
                    st.success("✅ Admin access granted!")
                    st.rerun()
                else:
                    st.error("❌ Invalid password")
        else:
            # Admin controls
            st.success("🔓 Admin Mode Active")
            
            if st.button("🔒 Logout", key="admin_logout"):
                st.session_state.admin_authenticated = False
                st.rerun()
            
            st.markdown("### ➕ Add Company")
            new_company = st.text_input("Company Name:", key="new_company")
            logo_file = st.file_uploader("Logo (PNG):", type=['png', 'jpg', 'jpeg'], key="logo_upload")
            
            if st.button("Add Company", key="add_company_btn"):
                if new_company:
                    new_path = os.path.join(company_base_dir, new_company)
                    if not os.path.exists(new_path):
                        os.makedirs(new_path)
                        
                        if logo_file is not None:
                            logo_path = os.path.join(logos_dir, f"{new_company}.png")
                            with open(logo_path, "wb") as f:
                                f.write(logo_file.getbuffer())
                        
                        st.success(f"✅ Added: {new_company}")
                        st.rerun()
                    else:
                        st.warning("⚠️ Company already exists")
            
            # Upload PDFs
            if st.session_state.selected_company:
                st.markdown("### 📄 Upload PDFs")
                uploaded_pdf = st.file_uploader(
                    f"Upload to {st.session_state.selected_company}:",
                    type="pdf",
                    key="pdf_upload"
                )
                
                if uploaded_pdf:
                    save_path = os.path.join(company_base_dir, st.session_state.selected_company, uploaded_pdf.name)
                    with open(save_path, "wb") as f:
                        f.write(uploaded_pdf.getbuffer())
                    st.success(f"✅ Uploaded: {uploaded_pdf.name}")
                    st.rerun()
                
                # Relearn PDFs
                if st.button("🔄 Relearn PDFs", key="relearn_pdfs"):
                    try:
                        from ingest import ingest_company_pdfs
                        
                        VECTORSTORE_ROOT = "/mount/tmp/vectorstores" if is_streamlit_cloud() else "vectorstores"
                        vectorstore_path = os.path.join(VECTORSTORE_ROOT, st.session_state.selected_company)
                        
                        if os.path.exists(vectorstore_path):
                            shutil.rmtree(vectorstore_path, ignore_errors=True)
                        
                        time.sleep(1)
                        os.makedirs(vectorstore_path, exist_ok=True)
                        
                        ingest_company_pdfs(st.session_state.selected_company, persist_directory=vectorstore_path)
                        
                        if 'vectorstore' in st.session_state:
                            del st.session_state['vectorstore']
                        
                        st.success("✅ Knowledge base updated!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Main chat interface
st.markdown('<div class="main-container">', unsafe_allow_html=True)

if not available_companies:
    st.markdown("""
    <div class="welcome-container">
        <div class="welcome-title">Welcome to Broker-GPT</div>
        <div class="welcome-subtitle">No companies are currently configured.</div>
        <p>Please contact your administrator to set up companies and upload knowledge base documents.</p>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.selected_company:
    selected_company = st.session_state.selected_company
    
    # Check if knowledge base exists
    VECTORSTORE_ROOT = "/mount/tmp/vectorstores" if is_streamlit_cloud() else "vectorstores"
    vectorstore_path = os.path.join(VECTORSTORE_ROOT, selected_company)
    
    if not os.path.exists(vectorstore_path):
        st.markdown("""
        <div class="welcome-container">
            <div class="welcome-title">Knowledge Base Not Ready</div>
            <div class="welcome-subtitle">The knowledge base for this company needs to be set up.</div>
            <p>Please contact your administrator to upload PDFs and initialize the knowledge base.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Chat messages container
        st.markdown('<div class="messages-container">', unsafe_allow_html=True)
        
        # Display welcome message if no chat history
        if not st.session_state.chat_history:
            st.markdown("""
            <div class="welcome-container">
                <div class="welcome-title">How can I help you today?</div>
                <div class="welcome-subtitle">Ask me about insurance policies, underwriting, or company procedures</div>
                <div class="suggestions">
                    <div class="suggestion" onclick="document.querySelector('textarea').value='What are the underwriting guidelines for auto insurance?'; document.querySelector('textarea').focus();">
                        Underwriting Guidelines
                    </div>
                    <div class="suggestion" onclick="document.querySelector('textarea').value='How do I process a claim?'; document.querySelector('textarea').focus();">
                        Claims Processing
                    </div>
                    <div class="suggestion" onclick="document.querySelector('textarea').value='What coverage options are available?'; document.querySelector('textarea').focus();">
                        Coverage Options
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Display chat history
        for i, (query, response) in enumerate(st.session_state.chat_history):
            # User message
            st.markdown(f"""
            <div class="message message-user">
                <div class="message-content user-message">{query}</div>
                <div class="message-avatar user-avatar">U</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Bot response
            st.markdown(f"""
            <div class="message">
                <div class="message-avatar bot-avatar">🤖</div>
                <div class="message-content bot-message">{response}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Input area
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        st.markdown('<div class="input-wrapper">', unsafe_allow_html=True)
        
        # Chat input
        query = st.text_area(
            "Message",
            placeholder="Ask me anything about insurance...",
            height=60,
            key="chat_input_main",
            label_visibility="collapsed"
        )
        
        # Send button (styled with CSS)
        send_clicked = st.button("Send", key="send_btn", help="Send message")
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Process query
        if send_clicked and query.strip():
            # Add loading message
            st.markdown("""
            <div class="message">
                <div class="message-avatar bot-avatar">🤖</div>
                <div class="message-content bot-message loading-message">
                    <span>Thinking</span>
                    <span class="loading-dots">...</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            try:
                if 'vectorstore' not in st.session_state:
                    st.session_state['vectorstore'] = create_chroma_vectorstore(vectorstore_path)
                
                vectorstore = st.session_state['vectorstore']
                retriever = vectorstore.as_retriever()
                docs = retriever.get_relevant_documents(query)
                context = "\n\n".join([doc.page_content for doc in docs])

                GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

                payload = {
                    "contents": [{
                        "parts": [{
                            "text": f"""As a professional insurance broker assistant, answer the following question using the context provided.

Question: {query}

Context: {context}

Please provide a clear, professional response that would be helpful for insurance brokers and their clients. Format your response in a conversational, helpful manner.
"""
                        }]
                    }]
                }

                headers = {"Content-Type": "application/json"}
                response = requests.post(url, headers=headers, data=json.dumps(payload))

                if response.status_code == 200:
                    try:
                        answer = response.json()['candidates'][0]['content']['parts'][0]['text']
                        
                        # Add to chat history
                        st.session_state.chat_history.append((query, answer))
                        
                        # Clear input and rerun
                        st.rerun()
                        
                    except Exception as e:
                        st.error("❌ Error parsing response from Gemini")
                else:
                    st.error(f"❌ Gemini API Error: {response.status_code}")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                if 'vectorstore' in st.session_state:
                    del st.session_state['vectorstore']

st.markdown('</div>', unsafe_allow_html=True)
