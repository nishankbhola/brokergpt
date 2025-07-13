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
    page_title="🤖 Broker-GPT",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Claude-like CSS styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Company selector - Claude style */
    .company-selector {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Chat interface - Claude style */
    .chat-container {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 0;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        overflow: hidden;
    }
    
    .chat-input {
        border: none;
        border-top: 1px solid #e2e8f0;
        padding: 1rem;
        background: #f8fafc;
    }
    
    .chat-messages {
        max-height: 600px;
        overflow-y: auto;
        padding: 1rem;
    }
    
    /* Submit button - Claude style */
    .submit-btn {
        background: #2563eb;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s;
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2);
    }
    
    .submit-btn:hover {
        background: #1d4ed8;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(37, 99, 235, 0.3);
    }
    
    /* Sidebar styling */
    .sidebar-content {
        background: #f1f5f9;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Admin section styling */
    .admin-section {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .admin-success {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Response styling */
    .response-container {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* No company message */
    .no-company-message {
        background: linear-gradient(135deg, #fef3c7 0%, #f59e0b 100%);
        color: #92400e;
        border-radius: 12px;
        padding: 3rem;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        transition: all 0.2s;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
    
    /* Text area styling */
    .stTextArea > div > div > textarea {
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Form styling */
    .stForm {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Logo styling */
    .company-logo {
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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
<div class="main-header">
    <h2>🤖 Broker-GPT</h2>
    <p>Your AI-powered insurance broker assistant</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🏢 Company Management")
    
    # Company selection
    if available_companies:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.markdown("**Select Company:**")
        
        for company in available_companies:
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(f"📂 {company}", key=f"select_{company}", use_container_width=True):
                    st.session_state.selected_company = company
                    st.session_state.chat_history = []  # Clear chat history when switching
                    st.rerun()
            
            with col2:
                logo = get_company_logo(company)
                if logo:
                    st.image(logo, width=25)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Admin section
    if not st.session_state.admin_authenticated:
        st.markdown("### 🔐 Admin Access")
        with st.form("admin_login_form"):
            password = st.text_input("Password:", type="password", key="admin_password")
            login_btn = st.form_submit_button("Login", use_container_width=True)
            
            if login_btn:
                if password == "classmate":
                    st.session_state.admin_authenticated = True
                    st.success("✅ Admin access granted!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Invalid password")
    else:
        st.markdown('<div class="admin-success">', unsafe_allow_html=True)
        st.success("🔓 Admin Mode Active")
        
        if st.button("🔒 Logout", use_container_width=True):
            st.session_state.admin_authenticated = False
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Admin controls
        st.markdown("### ➕ Add Company")
        with st.form("add_company_form"):
            new_company = st.text_input("Company Name:")
            logo_file = st.file_uploader("Logo (PNG):", type=['png', 'jpg', 'jpeg'])
            add_btn = st.form_submit_button("Add Company", use_container_width=True)
            
            if add_btn and new_company:
                new_path = os.path.join(company_base_dir, new_company)
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                    
                    if logo_file is not None:
                        logo_path = os.path.join(logos_dir, f"{new_company}.png")
                        with open(logo_path, "wb") as f:
                            f.write(logo_file.getbuffer())
                    
                    st.success(f"✅ Added: {new_company}")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.warning("⚠️ Company exists")
        
        # Upload PDFs
        if st.session_state.selected_company:
            st.markdown("### 📄 Upload PDFs")
            uploaded_pdf = st.file_uploader(
                f"Upload to {st.session_state.selected_company}:", 
                type="pdf"
            )
            
            if uploaded_pdf:
                save_path = os.path.join(company_base_dir, st.session_state.selected_company, uploaded_pdf.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_pdf.getbuffer())
                st.success(f"✅ Uploaded: {uploaded_pdf.name}")
                time.sleep(1)
                st.rerun()
            
            # Admin actions
            st.markdown("### ⚙️ Admin Actions")
            
            if st.button("🔄 Relearn PDFs", use_container_width=True):
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
                    time.sleep(1)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
            
            # Delete company
            st.markdown('<div class="admin-section">', unsafe_allow_html=True)
            st.markdown("**⚠️ Danger Zone**")
            
            if st.button("🗑️ Delete Company", type="secondary", use_container_width=True):
                if st.button("⚠️ CONFIRM DELETE", key="confirm_delete", use_container_width=True):
                    try:
                        # Delete company data
                        company_path = os.path.join(company_base_dir, st.session_state.selected_company)
                        if os.path.exists(company_path):
                            shutil.rmtree(company_path)
                        
                        # Delete vectorstore
                        VECTORSTORE_ROOT = "/mount/tmp/vectorstores" if is_streamlit_cloud() else "vectorstores"
                        vectorstore_path = os.path.join(VECTORSTORE_ROOT, st.session_state.selected_company)
                        if os.path.exists(vectorstore_path):
                            shutil.rmtree(vectorstore_path)
                        
                        # Delete logo
                        logo_path = os.path.join(logos_dir, f"{st.session_state.selected_company}.png")
                        if os.path.exists(logo_path):
                            os.remove(logo_path)
                        
                        st.success(f"✅ Deleted: {st.session_state.selected_company}")
                        st.session_state.selected_company = None
                        st.session_state.chat_history = []
                        time.sleep(1)
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            
            st.markdown('</div>', unsafe_allow_html=True)

# Main content area
if not available_companies:
    st.markdown("""
    <div class="no-company-message">
        <h2>🏢 No Companies Available</h2>
        <p>There are currently no companies configured in the system.</p>
        <p><strong>Please contact the administrator</strong> to add companies and configure the knowledge base.</p>
        <hr>
        <p>📞 <strong>Need Help?</strong></p>
        <p>• Contact your system administrator<br>
        • Request company setup and PDF uploads<br>
        • Ensure proper knowledge base configuration</p>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.selected_company:
    selected_company = st.session_state.selected_company
    
    # Company selector with logo - Claude style
    st.markdown('<div class="company-selector">', unsafe_allow_html=True)
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.markdown("**Selected Company:**")
        
        # Company dropdown
        company_index = available_companies.index(selected_company) if selected_company in available_companies else 0
        selected_company_new = st.selectbox(
            "Choose company:",
            available_companies,
            index=company_index,
            key="company_selector",
            label_visibility="collapsed"
        )
        
        if selected_company_new != selected_company:
            st.session_state.selected_company = selected_company_new
            st.session_state.chat_history = []
            st.rerun()
    
    with col2:
        logo = get_company_logo(selected_company)
        if logo:
            st.image(logo, width=80, caption="", use_column_width=False)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Check if knowledge base exists
    VECTORSTORE_ROOT = "/mount/tmp/vectorstores" if is_streamlit_cloud() else "vectorstores"
    vectorstore_path = os.path.join(VECTORSTORE_ROOT, selected_company)
    
    if not os.path.exists(vectorstore_path):
        st.warning(f"📚 Knowledge base not ready for **{selected_company}**. Please contact admin to upload PDFs and click 'Relearn PDFs'.")
    else:
        # Chat interface - Claude style
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Chat messages area
        st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
        
        # Display chat history
        for i, (query, response) in enumerate(st.session_state.chat_history):
            st.markdown(f"**You:** {query}")
            st.markdown(f"**Broker-GPT:** {response}")
            if i < len(st.session_state.chat_history) - 1:
                st.markdown("---")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Chat input area
        st.markdown('<div class="chat-input">', unsafe_allow_html=True)
        
        with st.form("chat_form", clear_on_submit=True):
            col1, col2 = st.columns([5, 1])
            
            with col1:
                query = st.text_area(
                    "Message Broker-GPT...",
                    placeholder="Ask me anything about underwriting, policies, or company procedures...",
                    height=100,
                    key="chat_input",
                    label_visibility="collapsed"
                )
            
            with col2:
                st.write("")  # Spacing
                st.write("")  # Spacing
                submitted = st.form_submit_button("Send", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Process query
        if submitted and query:
            with st.spinner("🤖 Broker-GPT is thinking..."):
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

Please provide a clear, professional response that would be helpful for insurance brokers and their clients.
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
                            
                            # Rerun to show new message
                            st.rerun()
                            
                        except Exception as e:
                            st.error("❌ Error parsing response from Gemini")
                    else:
                        st.error(f"❌ Gemini API Error: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"❌ Error accessing knowledge base: {str(e)}")
                    if 'vectorstore' in st.session_state:
                        del st.session_state['vectorstore']

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #64748b; font-size: 14px; margin-top: 2rem;'>
        🤖 Broker-GPT | Powered by AI | Version 7.0.0 | 2025
    </div>
    """, 
    unsafe_allow_html=True
)
