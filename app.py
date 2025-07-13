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

def display_company_with_logo(company_name, size=50):
    """Display company name with logo if available"""
    logo = get_company_logo(company_name)
    if logo:
        col1, col2 = st.columns([1, 4])
        with col1:
            st.image(logo, width=size)
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
    initial_sidebar_state="collapsed"  # Start with sidebar collapsed for regular users
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        color: white;
        padding: 0.1rem;
        border-radius: 1px;
        margin-bottom: 0.1rem;
        text-align: center;
    }
    .company-selector {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .query-container {
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .submit-button {
        background: #2a5298;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        cursor: pointer;
    }
    .company-dropdown {
        min-width: 200px;
    }
    .admin-section {
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
    .no-company-message {
        background: #fff8dc;
        border: 2px solid #ffd700;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
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
if 'admin_authenticated' not in st.session_state:
    st.session_state.admin_authenticated = False

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

# Admin sidebar (only visible to authenticated admins)
if st.session_state.admin_authenticated:
    with st.sidebar:
        st.header("🏢 Company Management")
        
        # Company selection for admin
        st.markdown("### 📁 Select Company")
        
        if available_companies:
            for company in available_companies:
                col1, col2 = st.columns([3, 1])
                with col1:
                    if st.button(f"📂 {company}", key=f"select_{company}"):
                        st.session_state.selected_company = company
                        st.rerun()
                
                with col2:
                    logo = get_company_logo(company)
                    if logo:
                        st.image(logo, width=30)
        
        # Add new company
        st.markdown("---")
        st.markdown("### ➕ Add New Company")
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
        
        # Upload PDFs (admin only)
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
                    
                    if 'vectorstore' in st.session_state:
                        del st.session_state['vectorstore']
                    
                    st.success("✅ Knowledge base updated!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
            
            # Delete company data
            st.markdown('<div class="admin-section">', unsafe_allow_html=True)
            st.markdown("#### 🗑️ Danger Zone")
            
            if st.button("🗑️ Delete All Company Data", type="secondary"):
                if st.button("⚠️ CONFIRM DELETE", key="confirm_delete"):
                    try:
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
if not available_companies:
    # No companies available
    st.markdown('<div class="no-company-message">', unsafe_allow_html=True)
    st.markdown("""
    ## 🏢 No Companies Available
    
    There are currently no companies configured in the system.
    
    **Please contact the administrator** to add companies and configure the knowledge base.
    
    ### 📞 Need Help?
    - Contact your system administrator
    - Request company setup and PDF uploads
    - Ensure proper knowledge base configuration
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Admin access button at bottom
    st.markdown("---")
    if st.button("🔐 Admin Access", key="admin_access_main"):
        if check_admin_password():
            st.rerun()

elif st.session_state.selected_company:
    selected_company = st.session_state.selected_company
    
    # Company selector dropdown (similar to Claude's model selector)
    st.markdown('<div class="company-selector">', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 🏢 Selected Company")
        
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
            st.rerun()
    
    with col2:
        logo = get_company_logo(selected_company)
        if logo:
            st.image(logo, width=60)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Path handling
    VECTORSTORE_ROOT = "/mount/tmp/vectorstores" if is_streamlit_cloud() else "vectorstores"
    vectorstore_path = os.path.join(VECTORSTORE_ROOT, selected_company)
    
    if not os.path.exists(vectorstore_path):
        st.warning(f"📚 Knowledge base not ready for **{selected_company}**. Please contact admin to set up PDFs and relearn the knowledge base.")
    else:
        # Query interface (similar to Claude's interface)
        st.markdown('<div class="query-container">', unsafe_allow_html=True)
        st.markdown("### 💬 Ask Questions")
        
        # Form for query submission
        with st.form("query_form"):
            query = st.text_area(
                "🔍 Enter your question:",
                placeholder="Ask me anything about underwriting, policies, or company procedures...",
                height=100,
                key="query_input"
            )
            
            # Submit button
            col1, col2, col3 = st.columns([1, 1, 4])
            with col2:
                submitted = st.form_submit_button("Submit", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Process query
        if submitted and query:
            with st.spinner("🤖 Broker-GPT is analyzing your question..."):
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

                    st.markdown("---")
                    if response.status_code == 200:
                        try:
                            answer = response.json()['candidates'][0]['content']['parts'][0]['text']
                            st.markdown("### 🤖 Broker-GPT Response")
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
                    st.info("💡 Please contact admin to rebuild the knowledge base.")
                    if 'vectorstore' in st.session_state:
                        del st.session_state['vectorstore']

# Admin access button at bottom (for non-admin users)
if not st.session_state.admin_authenticated:
    st.markdown("---")
    if st.button("🔐 Admin Access", key="admin_access_bottom"):
        if check_admin_password():
            st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🤖 Broker-GPT | Powered by AI | Version 6.1.0 | 2025"
    "</div>", 
    unsafe_allow_html=True
)
