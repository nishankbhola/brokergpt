sdcsimport pysqlite3 as sqlite3
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

def create_chroma_vectorstore(vectorstore_path, company_name, max_retries=3):
    """Create Chroma vectorstore with retry logic and company-specific caching"""
    for attempt in range(max_retries):
        try:
            # Clear any existing chroma client for this company
            vectorstore_key = f'vectorstore_{company_name}'
            if vectorstore_key in st.session_state:
                del st.session_state[vectorstore_key]
            
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

def display_company_with_logo(company_name, size=50):
    """Display company name with logo if available"""
    logo = get_company_logo(company_name)
    if logo:
        col1, col2 = st.columns([1, 4])
        with col1:
            st.image(logo, width=size)
        #with col2:
            #st.markdown(f"**{company_name}**")
    else:
        st.markdown(f"🏢 **{company_name}**")

def check_admin_password():
    """Check if admin password is correct"""
    if 'admin_authenticated' not in st.session_state:
        st.session_state.admin_authenticated = False
    
    if not st.session_state.admin_authenticated:
        with st.form("admin_login"):
            st.subheader("🔐 Admin Access Required")
            password = st.text_input("Enter admin password:", type="password")
            submitted = st.form_submit_button("Login")
            
            if submitted:
                if password == "classmate":
                    st.session_state.admin_authenticated = True
                    st.success("✅ Admin access granted!")
                    st.rerun()
                else:
                    st.error("❌ Invalid password")
                    return False
        return False
    return True

def clear_company_vectorstore_cache(company_name):
    """Clear vectorstore cache for a specific company"""
    vectorstore_key = f'vectorstore_{company_name}'
    if vectorstore_key in st.session_state:
        del st.session_state[vectorstore_key]

def get_company_vectorstore(company_name, vectorstore_path):
    """Get or create company-specific vectorstore with proper caching"""
    vectorstore_key = f'vectorstore_{company_name}'
    
    # Check if we have a cached vectorstore for this specific company
    if vectorstore_key not in st.session_state:
        st.session_state[vectorstore_key] = create_chroma_vectorstore(vectorstore_path, company_name)
    
    return st.session_state[vectorstore_key]

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="🤖 Broker-GPT",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        /*background: linear-gradient(90deg, #1e3c72, #2a5298);*/
        color: white;
        padding: 0.1rem;
        border-radius: 1px;
        margin-bottom: 0.1rem;
        text-align: center;
    }
    .company-card {
        background: white;
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    .company-card:hover {
        border-color: #2a5298;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .selected-company {
        border-color: #2a5298;
        background: #f0f4ff;
    }
    .nav-tabs {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem;
        text-align: center;
    }
    .danger-zone {
        background: #fff5f5;
        border: 2px solid #feb2b2;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-zone {
        background: #f0fff4;
        border: 2px solid #9ae6b4;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h4>🤖 Broker-GPT </h4>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'selected_company' not in st.session_state:
    st.session_state.selected_company = None
if 'current_view' not in st.session_state:
    st.session_state.current_view = "Ask Questions"

# Create necessary directories
company_base_dir = "data/pdfs"
logos_dir = "data/logos"
os.makedirs(company_base_dir, exist_ok=True)
os.makedirs(logos_dir, exist_ok=True)

# Sidebar for company management
with st.sidebar:

    
    st.header("🏢 Company Management")
    
    # Admin section
    st.markdown("### 🔧 Admin Controls")
    if st.button("🔐 Admin Access"):
        st.session_state.admin_authenticated = False
    
    if check_admin_password():
        st.markdown('<div class="success-zone">', unsafe_allow_html=True)
        st.success("🔓 Admin Mode Active")
        
        # Add new company
        st.markdown("#### ➕ Add New Company")
        with st.form("add_company_form"):
            new_company = st.text_input("Company Name:")
            logo_file = st.file_uploader("Company Logo (PNG):", type=['png', 'jpg', 'jpeg'])
            add_submitted = st.form_submit_button("Add Company")
            
            if add_submitted and new_company:
                new_path = os.path.join(company_base_dir, new_company)
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                    
                    # Save logo if uploaded
                    if logo_file is not None:
                        logo_path = os.path.join(logos_dir, f"{new_company}.png")
                        with open(logo_path, "wb") as f:
                            f.write(logo_file.getbuffer())
                    
                    st.success(f"✅ Added company: {new_company}")
                    st.rerun()
                else:
                    st.warning("⚠️ Company already exists")
        
        st.markdown('</div>', unsafe_allow_html=True)

    
    
    # Company selection
    st.markdown("---")
    st.markdown("### 📁 Select Company")
    
    company_folders = [f for f in os.listdir(company_base_dir) 
                      if os.path.isdir(os.path.join(company_base_dir, f))]
    
    if not company_folders:
        st.warning("⚠️ No companies found. Add one to begin.")
        st.stop()
    
    # Display companies with logos
    for company in company_folders:
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button(f"📂 {company}", key=f"select_{company}"):
                # Clear vectorstore cache when switching companies
                if st.session_state.selected_company and st.session_state.selected_company != company:
                    clear_company_vectorstore_cache(st.session_state.selected_company)
                
                st.session_state.selected_company = company
                st.rerun()
        
        with col2:
            logo = get_company_logo(company)
            if logo:
                st.image(logo, width=30)
    
    if st.session_state.selected_company:
        st.markdown("---")
        st.markdown("### 📄 Upload PDFs")
        
        selected_company = st.session_state.selected_company
        uploaded_pdf = st.file_uploader(
            f"Upload PDF to {selected_company}:", 
            type="pdf", 
            key="pdf_uploader"
        )
        
        if uploaded_pdf:
            save_path = os.path.join(company_base_dir, selected_company, uploaded_pdf.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_pdf.getbuffer())
            st.success(f"✅ Uploaded: {uploaded_pdf.name}")
            st.rerun()
        
        # Admin controls for selected company
        if st.session_state.get('admin_authenticated', False):
            st.markdown("---")
            st.markdown("### ⚙️ Admin Actions")
            
            # Relearn PDFs
            if st.button("🔄 Relearn PDFs"):
                try:
                    from ingest import ingest_company_pdfs
                    
                    VECTORSTORE_ROOT = "/mount/tmp/vectorstores" if is_streamlit_cloud() else "vectorstores"
                    vectorstore_path = os.path.join(VECTORSTORE_ROOT, selected_company)
                    
                    if os.path.exists(vectorstore_path):
                        shutil.rmtree(vectorstore_path, ignore_errors=True)
                    
                    time.sleep(1)
                    os.makedirs(vectorstore_path, exist_ok=True)
                    
                    ingest_company_pdfs(selected_company, persist_directory=vectorstore_path)
                    
                    # Clear the cached vectorstore for this company
                    clear_company_vectorstore_cache(selected_company)
                    
                    st.success("✅ Knowledge base updated!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
            
            # Delete company data
            st.markdown('<div class="danger-zone">', unsafe_allow_html=True)
            st.markdown("#### 🗑️ Danger Zone")
            
            if st.button("🗑️ Delete All Company Data", type="secondary"):
                if st.button("⚠️ CONFIRM DELETE", key="confirm_delete"):
                    try:
                        # Clear vectorstore cache first
                        clear_company_vectorstore_cache(selected_company)
                        
                        # Delete PDFs
                        company_path = os.path.join(company_base_dir, selected_company)
                        if os.path.exists(company_path):
                            shutil.rmtree(company_path)
                        
                        # Delete vectorstore
                        VECTORSTORE_ROOT = "/mount/tmp/vectorstores" if is_streamlit_cloud() else "vectorstores"
                        vectorstore_path = os.path.join(VECTORSTORE_ROOT, selected_company)
                        if os.path.exists(vectorstore_path):
                            shutil.rmtree(vectorstore_path)
                        
                        # Delete logo
                        logo_path = os.path.join(logos_dir, f"{selected_company}.png")
                        if os.path.exists(logo_path):
                            os.remove(logo_path)
                        
                        st.success(f"✅ Deleted all data for {selected_company}")
                        st.session_state.selected_company = None
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error deleting: {str(e)}")
            
            st.markdown('</div>', unsafe_allow_html=True)

# Main content area
if st.session_state.selected_company:
    selected_company = st.session_state.selected_company

    # Path handling
    VECTORSTORE_ROOT = "/mount/tmp/vectorstores" if is_streamlit_cloud() else "vectorstores"
    vectorstore_path = os.path.join(VECTORSTORE_ROOT, selected_company)
    
    if not os.path.exists(vectorstore_path):
        st.info(f"📚 Upload PDFs for **{selected_company}** and use admin access to click 'Relearn PDFs' to start.") 
    else:
        if st.session_state.current_view == "Dashboard":
            st.markdown("---")
            st.subheader("📊 Company Dashboard")
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            
            company_pdf_dir = os.path.join(company_base_dir, selected_company)
            uploaded_pdfs = []
            if os.path.exists(company_pdf_dir):
                uploaded_pdfs = [f for f in os.listdir(company_pdf_dir) if f.endswith(".pdf")]
            
            pdf_count = len(uploaded_pdfs)
            db_exists = os.path.exists(vectorstore_path)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("📄 PDF Files", pdf_count)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("💾 Knowledge Base", "Ready" if db_exists else "Not Ready")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                if db_exists:
                    total_size = sum(
                        os.path.getsize(os.path.join(dp, f))
                        for dp, dn, filenames in os.walk(vectorstore_path)
                        for f in filenames
                    )
                    size_mb = round(total_size / 1024 / 1024, 2)
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("💽 DB Size", f"{size_mb} MB")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # PDF List
            if uploaded_pdfs:
                st.markdown("#### 📄 Uploaded Documents")
                for pdf in uploaded_pdfs:
                    st.markdown(f"• {pdf}")
        
        else:  # Ask Questions view
            st.markdown("---")
            display_company_with_logo(selected_company, size=150)
            st.subheader(f"💬 Ask {selected_company} Questions") 
            
            query = st.text_input("🔍 Enter your question:", placeholder="Ask me anything about underwriting...")
            
            if query:
                with st.spinner("🤖 Broker-GPT is analyzing your question..."):
                    try:
                        # Get company-specific vectorstore
                        vectorstore = get_company_vectorstore(selected_company, vectorstore_path)
                        
                        retriever = vectorstore.as_retriever()
                        docs = retriever.get_relevant_documents(query)
                        context = "\n\n".join([doc.page_content for doc in docs])

                        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
                        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

                        payload = {
                            "contents": [{
                                "parts": [{
                                    "text": f"""As a professional insurance broker assistant, answer the following question using ONLY the context provided for {selected_company}.

Question: {query}

Context from {selected_company}: {context}

Please provide a clear, professional response that would be helpful for insurance brokers and their clients. Base your answer ONLY on the provided context from {selected_company}.
"""
                                }]
                            }]
                        }

                        headers = {"Content-Type": "application/json"}
                        response = requests.post(url, headers=headers, data=json.dumps(payload))

                        st.markdown("---")
                        if response.status_code == 200:
                            try:
                                answer = response.json()['candidates'][0]['content']['parts'][0]['text']
                                st.markdown("### 🤖 Broker-GPT Response")
                                st.markdown(f"**Company:** {selected_company}")
                                st.markdown(f"**Question:** {query}")
                                st.markdown("**Answer:**")
                                st.success(answer)
                                
                                # Show source documents
                                if docs:
                                    with st.expander("📚 Source Documents"):
                                        for i, doc in enumerate(docs[:3]):  # Show top 3 sources
                                            st.markdown(f"**Source {i+1}:**")
                                            st.text(doc.page_content[:500] + "...")
                                            st.markdown("---")
                                            
                            except Exception as e:
                                st.error("❌ Error parsing response from Gemini")
                        else:
                            st.error(f"❌ Gemini API Error: {response.status_code}")
                            
                    except Exception as e:
                        st.error(f"❌ Error accessing knowledge base: {str(e)}")
                        st.info("💡 Try using admin access to click 'Relearn PDFs' to rebuild the knowledge base.")
                        # Clear the cached vectorstore for this company
                        clear_company_vectorstore_cache(selected_company)
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Ask Questions", key="nav_questions"):
            st.session_state.current_view = "Ask Questions"
    
    with col2:
        if st.button("📊 Dashboard", key="nav_dashboard"):
            st.session_state.current_view = "Dashboard"
    
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("👆 Please select a company from the sidebar to continue.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🤖 Broker-GPT | Powered by AI | Version 6.0.2 | 2025"
    "</div>", 
    unsafe_allow_html=True
)
