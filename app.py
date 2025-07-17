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

# Load environment variables
load_dotenv()

# Configuration
COMPANY_BASE_DIR = "data/pdfs"
LOGOS_DIR = "data/logos"
VECTORSTORE_ROOT = "/mount/tmp/vectorstores" if os.environ.get("HOME") == "/home/adminuser" else "vectorstores"
ADMIN_PASSWORD = "classmate"

# Page configuration
st.set_page_config(
    page_title="🤖 Broker-GPT",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state with defaults
def init_session_state():
    defaults = {
        'selected_company': None,
        'admin_authenticated': False,
        'upload_success_message': None,
        'processed_files': set()
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# CSS styling
st.markdown("""
<style>
    .main-header { color: white; text-align: center; }
    .company-card { background: white; border: 2px solid #e0e0e0; border-radius: 10px; padding: 1rem; margin: 0.5rem 0; }
    .company-card:hover { border-color: #2a5298; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
    .metric-card { background: white; border: 1px solid #e0e0e0; border-radius: 8px; padding: 1rem; margin: 0.5rem; text-align: center; }
    .success-zone { background: #f0fff4; border: 2px solid #9ae6b4; border-radius: 10px; padding: 1rem; margin: 1rem 0; }
    .danger-zone { background: #fff5f5; border: 2px solid #feb2b2; border-radius: 10px; padding: 1rem; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# Utility functions
def ensure_directories():
    """Create necessary directories"""
    os.makedirs(COMPANY_BASE_DIR, exist_ok=True)
    os.makedirs(LOGOS_DIR, exist_ok=True)

def get_companies():
    """Get list of company folders"""
    return [f for f in os.listdir(COMPANY_BASE_DIR) if os.path.isdir(os.path.join(COMPANY_BASE_DIR, f))]

def get_company_logo(company_name):
    """Get company logo if it exists"""
    logo_path = os.path.join(LOGOS_DIR, f"{company_name}.png")
    return Image.open(logo_path) if os.path.exists(logo_path) else None

def get_uploaded_pdfs(company_name):
    """Get list of uploaded PDFs for a company"""
    company_pdf_dir = os.path.join(COMPANY_BASE_DIR, company_name)
    return [f for f in os.listdir(company_pdf_dir) if f.endswith(".pdf")] if os.path.exists(company_pdf_dir) else []

def create_vectorstore(company_name):
    """Create Chroma vectorstore with retry logic"""
    vectorstore_path = os.path.join(VECTORSTORE_ROOT, company_name)
    
    for attempt in range(3):
        try:
            os.makedirs(vectorstore_path, exist_ok=True)
            vectorstore = Chroma(
                persist_directory=vectorstore_path,
                embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
            )
            vectorstore._client.heartbeat()
            return vectorstore
        except Exception as e:
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
            else:
                raise e

def get_vectorstore(company_name):
    """Get or create company-specific vectorstore with caching"""
    cache_key = f'vectorstore_{company_name}'
    if cache_key not in st.session_state:
        st.session_state[cache_key] = create_vectorstore(company_name)
    return st.session_state[cache_key]

def clear_vectorstore_cache(company_name):
    """Clear vectorstore cache for a specific company"""
    cache_key = f'vectorstore_{company_name}'
    if cache_key in st.session_state:
        del st.session_state[cache_key]

def check_admin_auth():
    """Check admin authentication"""
    if not st.session_state.admin_authenticated:
        with st.form("admin_login"):
            st.subheader("🔐 Admin Access Required")
            password = st.text_input("Enter admin password:", type="password")
            if st.form_submit_button("Login"):
                if password == ADMIN_PASSWORD:
                    st.session_state.admin_authenticated = True
                    st.success("✅ Admin access granted!")
                    st.rerun()
                else:
                    st.error("❌ Invalid password")
        return False
    return True

def handle_company_selection(company):
    """Handle company selection logic"""
    if st.session_state.selected_company != company:
        if st.session_state.selected_company:
            clear_vectorstore_cache(st.session_state.selected_company)
        st.session_state.selected_company = company
        st.session_state.upload_success_message = None
        st.rerun()

def handle_file_upload(company, uploaded_file):
    """Handle PDF file upload"""
    if uploaded_file:
        file_id = f"{company}_{uploaded_file.name}_{uploaded_file.size}"
        if file_id not in st.session_state.processed_files:
            try:
                save_path = os.path.join(COMPANY_BASE_DIR, company, uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state.processed_files.add(file_id)
                st.session_state.upload_success_message = f"✅ Uploaded: {uploaded_file.name}"
                time.sleep(0.1)
            except Exception as e:
                st.error(f"❌ Error uploading file: {str(e)}")

def relearn_pdfs(company):
    """Relearn PDFs for a company"""
    try:
        from ingest import ingest_company_pdfs
        
        with st.spinner("🔄 Rebuilding knowledge base..."):
            vectorstore_path = os.path.join(VECTORSTORE_ROOT, company)
            clear_vectorstore_cache(company)
            
            if os.path.exists(vectorstore_path):
                shutil.rmtree(vectorstore_path, ignore_errors=True)
                time.sleep(2)
            
            os.makedirs(vectorstore_path, exist_ok=True)
            
            progress_bar = st.progress(0)
            progress_bar.progress(25)
            
            vectordb = ingest_company_pdfs(company, persist_directory=vectorstore_path)
            
            progress_bar.progress(100)
            st.success("✅ Knowledge base updated successfully!")
            time.sleep(1)
            st.rerun()
            
    except Exception as e:
        error_msg = str(e)
        if "no such table: tenants" in error_msg:
            st.error("❌ Database corruption detected. Please try again.")
        else:
            st.error(f"❌ Error: {error_msg}")
        clear_vectorstore_cache(company)

def delete_company_data(company):
    """Delete all company data"""
    try:
        clear_vectorstore_cache(company)
        
        # Delete PDFs
        company_path = os.path.join(COMPANY_BASE_DIR, company)
        if os.path.exists(company_path):
            shutil.rmtree(company_path)
        
        # Delete vectorstore
        vectorstore_path = os.path.join(VECTORSTORE_ROOT, company)
        if os.path.exists(vectorstore_path):
            shutil.rmtree(vectorstore_path)
        
        # Delete logo
        logo_path = os.path.join(LOGOS_DIR, f"{company}.png")
        if os.path.exists(logo_path):
            os.remove(logo_path)
        
        # Clear processed files
        st.session_state.processed_files = {
            f for f in st.session_state.processed_files 
            if not f.startswith(f"{company}_")
        }
        
        st.success(f"✅ Deleted all data for {company}")
        st.session_state.selected_company = None
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error deleting: {str(e)}")

def render_sidebar():
    """Render sidebar with company management"""
    with st.sidebar:
        st.header("🏢 Company Management")
        
        # Admin section
        st.markdown("### 🔧 Admin Controls")
        if st.button("🔐 Admin Access"):
            st.session_state.admin_authenticated = False
        
        if check_admin_auth():
            st.markdown('<div class="success-zone">✅ Admin Mode Active</div>', unsafe_allow_html=True)
            
            # Add new company
            with st.form("add_company_form"):
                st.markdown("#### ➕ Add New Company")
                new_company = st.text_input("Company Name:")
                logo_file = st.file_uploader("Company Logo (PNG):", type=['png', 'jpg', 'jpeg'])
                
                if st.form_submit_button("Add Company") and new_company:
                    new_path = os.path.join(COMPANY_BASE_DIR, new_company)
                    if not os.path.exists(new_path):
                        os.makedirs(new_path)
                        if logo_file:
                            logo_path = os.path.join(LOGOS_DIR, f"{new_company}.png")
                            with open(logo_path, "wb") as f:
                                f.write(logo_file.getbuffer())
                        st.success(f"✅ Added company: {new_company}")
                        st.rerun()
                    else:
                        st.warning("⚠️ Company already exists")
        
        # Company selection
        st.markdown("---")
        st.markdown("### 📁 Select Company")
        
        companies = get_companies()
        if not companies:
            st.warning("⚠️ No companies found. Add one to begin.")
            return
        
        for company in companies:
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button(f"📂 {company}", key=f"select_{company}"):
                    handle_company_selection(company)
            with col2:
                logo = get_company_logo(company)
                if logo:
                    st.image(logo, width=30)
        
        # Company-specific admin controls
        if st.session_state.selected_company and st.session_state.admin_authenticated:
            company = st.session_state.selected_company
            
            st.markdown("---")
            st.markdown("### 📄 Upload PDFs")
            
            # Display current PDFs
            current_pdfs = get_uploaded_pdfs(company)
            if current_pdfs:
                st.markdown("**Current PDFs:**")
                for pdf in current_pdfs:
                    st.markdown(f"• {pdf}")
            
            # File uploader
            uploaded_pdf = st.file_uploader(
                f"Upload PDF to {company}:", 
                type="pdf", 
                key=f"pdf_uploader_{company}"
            )
            handle_file_upload(company, uploaded_pdf)
            
            # Show upload success message
            if st.session_state.upload_success_message:
                st.success(st.session_state.upload_success_message)
                if st.button("✅ Continue", key="clear_upload_msg"):
                    st.session_state.upload_success_message = None
                    st.rerun()
            
            # Admin actions
            st.markdown("---")
            st.markdown("### ⚙️ Admin Actions")
            
            if st.button("🔄 Relearn PDFs"):
                relearn_pdfs(company)
            
            # Delete company data
            st.markdown('<div class="danger-zone">', unsafe_allow_html=True)
            st.markdown("#### 🗑️ Danger Zone")
            if st.button("🗑️ Delete All Company Data"):
                if st.button("⚠️ CONFIRM DELETE", key="confirm_delete"):
                    delete_company_data(company)
            st.markdown('</div>', unsafe_allow_html=True)

def render_main_content():
    """Render main content area"""
    if not st.session_state.selected_company:
        st.info("👆 Please select a company from the sidebar to continue.")
        return
    
    company = st.session_state.selected_company
    vectorstore_path = os.path.join(VECTORSTORE_ROOT, company)
    
    if not os.path.exists(vectorstore_path):
        st.info(f"📚 Upload PDFs for **{company}** and use admin access to click 'Relearn PDFs' to start.")
        return
    
    # Display company info
    st.markdown("---")
    logo = get_company_logo(company)
    if logo:
        col1, col2 = st.columns([1, 4])
        with col1:
            st.image(logo, width=150)
    
    st.subheader(f"💬 Ask {company} Questions")
    
    # Query interface
    query = st.text_input("🔍 Enter your question:", placeholder="Ask me anything about underwriting...")
    
    if query:
        with st.spinner("🤖 Broker-GPT is analyzing your question..."):
            try:
                vectorstore = get_vectorstore(company)
                retriever = vectorstore.as_retriever()
                docs = retriever.get_relevant_documents(query)
                context = "\n\n".join([doc.page_content for doc in docs])

                # Gemini API call
                GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
                
                payload = {
                    "contents": [{
                        "parts": [{
                            "text": f"""As a professional insurance broker assistant, answer the following question using ONLY the context provided for {company}.

Question: {query}

Context from {company}: {context}

Please provide a clear, professional response that would be helpful for insurance brokers and their clients. Base your answer ONLY on the provided context from {company}.
"""
                        }]
                    }]
                }

                response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(payload))

                st.markdown("---")
                if response.status_code == 200:
                    try:
                        answer = response.json()['candidates'][0]['content']['parts'][0]['text']
                        st.markdown("### 🤖 Broker-GPT Response")
                        st.markdown(f"**Company:** {company}")
                        st.markdown(f"**Question:** {query}")
                        st.markdown("**Answer:**")
                        st.success(answer)
                        
                        # Show source documents
                        if docs:
                            with st.expander("📚 Source Documents"):
                                for i, doc in enumerate(docs[:3]):
                                    st.markdown(f"**Source {i+1}:**")
                                    st.text(doc.page_content[:500] + "...")
                                    st.markdown("---")
                                    
                    except Exception as e:
                        st.error("❌ Error parsing response from Gemini")
                else:
                    st.error(f"❌ Gemini API Error: {response.status_code}")
                    
            except Exception as e:
                error_msg = str(e)
                if "no such table: tenants" in error_msg:
                    st.error("❌ Database error detected. Please use admin access to click 'Relearn PDFs' to rebuild the knowledge base.")
                else:
                    st.error(f"❌ Error accessing knowledge base: {error_msg}")
                clear_vectorstore_cache(company)

def main():
    """Main application function"""
    # Header
    st.markdown('<div class="main-header"><h4>🤖 Broker-GPT</h4></div>', unsafe_allow_html=True)
    
    # Initialize
    init_session_state()
    ensure_directories()
    
    # Render components
    render_sidebar()
    render_main_content()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "🤖 Broker-GPT | Powered by AI | Version 7.0.5 | 2025"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
