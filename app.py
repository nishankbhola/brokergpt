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
import zipfile
import tempfile
from io import BytesIO

# --- NEW: Cached function to load the embedding model ---
@st.cache_resource
def load_embedding_model():
    """Loads the sentence transformer model only once."""
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Detect if running on Streamlit Cloud
def is_streamlit_cloud():
    return os.environ.get("HOME") == "/home/adminuser"

def create_chroma_vectorstore(vectorstore_path, company_name, max_retries=5):
    """Create Chroma vectorstore with enhanced retry logic and company-specific caching"""
    for attempt in range(max_retries):
        try:
            # Clear any existing chroma client for this company
            vectorstore_key = f'vectorstore_{company_name}'
            if vectorstore_key in st.session_state:
                del st.session_state[vectorstore_key]
            
            os.makedirs(vectorstore_path, exist_ok=True)
            
            # --- MODIFIED: Use the cached function to get the model ---
            embedding_function = load_embedding_model()
            
            vectorstore = Chroma(
                persist_directory=vectorstore_path,
                embedding_function=embedding_function,
                client_settings=None
            )
            
            vectorstore._client.heartbeat()
            return vectorstore
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 * (attempt + 1)
                time.sleep(wait_time)
                
                # More aggressive cleanup on retry
                if os.path.exists(vectorstore_path):
                    try:
                        for file in os.listdir(vectorstore_path):
                            if file.endswith('.sqlite3') or file.endswith('.db'):
                                file_path = os.path.join(vectorstore_path, file)
                                try:
                                    os.remove(file_path)
                                except:
                                    pass
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

def clear_company_vectorstore_cache(company_name):
    """Clear vectorstore cache for a specific company"""
    vectorstore_key = f'vectorstore_{company_name}'
    if vectorstore_key in st.session_state:
        del st.session_state[vectorstore_key]

def get_company_vectorstore(company_name, vectorstore_path):
    """Get or create company-specific vectorstore with proper caching"""
    vectorstore_key = f'vectorstore_{company_name}'
    
    if vectorstore_key not in st.session_state:
        st.session_state[vectorstore_key] = create_chroma_vectorstore(vectorstore_path, company_name)
    
    return st.session_state[vectorstore_key]

def get_uploaded_pdfs(company_name):
    """Get list of uploaded PDFs for a company"""
    company_pdf_dir = os.path.join("data/pdfs", company_name)
    if os.path.exists(company_pdf_dir):
        return [f for f in os.listdir(company_pdf_dir) if f.endswith(".pdf")]
    return []

def call_gemini_with_fallback(payload):
    """Call Gemini API with automatic model fallback on rate limit"""
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    headers = {"Content-Type": "application/json"}
    
    for attempt in range(len(GEMINI_MODELS)):
        current_model = GEMINI_MODELS[st.session_state.current_model_index]
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{current_model}:generateContent?key={GEMINI_API_KEY}"
        
        try:
            time.sleep(1)
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            
            if response.status_code == 200:
                return response, current_model
            elif response.status_code == 429:
                st.warning(f"⚠️ Rate limit reached for {current_model}, trying next model...")
                # Switch to next model
                st.session_state.current_model_index = (st.session_state.current_model_index + 1) % len(GEMINI_MODELS)
                time.sleep(2)  # Wait before trying next model
                continue
            else:
                return response, current_model
                
        except Exception as e:
            st.error(f"❌ Error with {current_model}: {str(e)}")
            st.session_state.current_model_index = (st.session_state.current_model_index + 1) % len(GEMINI_MODELS)
            continue
    
    # If all models failed, return the last response
    return response, current_model

# Add this function after the existing utility functions (around line 100)
def create_full_backup():
    """Create a complete backup of all companies' data and vectorstores"""
    try:
        # Create a temporary zip file in memory
        zip_buffer = BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Backup PDFs
            pdf_base_dir = "data/pdfs"
            if os.path.exists(pdf_base_dir):
                for root, dirs, files in os.walk(pdf_base_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        # Create archive path maintaining folder structure
                        arcname = os.path.relpath(file_path, os.path.dirname(pdf_base_dir))
                        zip_file.write(file_path, arcname)
            
            # Backup logos
            logos_dir = "data/logos"
            if os.path.exists(logos_dir):
                for root, dirs, files in os.walk(logos_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, os.path.dirname(logos_dir))
                        zip_file.write(file_path, arcname)
            
            # Backup vectorstores
            VECTORSTORE_ROOT = "/mount/tmp/vectorstores" if is_streamlit_cloud() else "vectorstores"
            if os.path.exists(VECTORSTORE_ROOT):
                for root, dirs, files in os.walk(VECTORSTORE_ROOT):
                    for file in files:
                        file_path = os.path.join(root, file)
                        # Create archive path maintaining folder structure
                        arcname = os.path.join("vectorstores", os.path.relpath(file_path, VECTORSTORE_ROOT))
                        zip_file.write(file_path, arcname)
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
        
    except Exception as e:
        st.error(f"❌ Error creating backup: {str(e)}")
        return None


def restore_from_backup(uploaded_file):
    """Restore all data from uploaded backup file"""
    try:
        # Create a temporary file to work with
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        extracted_companies = set()
        successful_extractions = 0
        failed_extractions = 0
        
        with zipfile.ZipFile(tmp_file_path, 'r') as zip_file:
            # Debug: Print all files in the zip
            st.info(f"📦 Zip file contains {len(zip_file.namelist())} files")
            
            # First pass: identify all companies in the backup
            for member in zip_file.namelist():
                # Handle both old format (data/pdfs/) and new format (pdfs/)
                if ((member.startswith('data/pdfs/') or member.startswith('pdfs/')) and 
                    not member.endswith('/')):
                    parts = member.split('/')
                    # For data/pdfs/COMPANY/... format
                    if member.startswith('data/pdfs/') and len(parts) >= 3:
                        company_name = parts[2]
                        extracted_companies.add(company_name)
                    # For pdfs/COMPANY/... format
                    elif member.startswith('pdfs/') and len(parts) >= 2:
                        company_name = parts[1]
                        extracted_companies.add(company_name)
            
            st.info(f"🏢 Found companies in backup: {', '.join(extracted_companies) if extracted_companies else 'None'}")
            
            # Extract everything
            for member in zip_file.namelist():
                # Skip directories
                if member.endswith('/'):
                    continue
                
                # Determine extraction path
                extract_path = None
                
                if member.startswith('data/'):
                    # For data files, use relative path from current directory
                    extract_path = member  # Keep data/ structure
                elif member.startswith('pdfs/'):
                    # Handle the pdfs/ format by converting to data/pdfs/
                    relative_path = member[5:]  # Remove 'pdfs/' prefix
                    extract_path = os.path.join("data", "pdfs", relative_path)
                elif member.startswith('logos/'):
                    # Handle logos/ format by converting to data/logos/
                    relative_path = member[6:]  # Remove 'logos/' prefix
                    extract_path = os.path.join("data", "logos", relative_path)
                elif member.startswith('vectorstores/'):
                    # For vectorstores, use the proper base directory
                    VECTORSTORE_ROOT = "/mount/tmp/vectorstores" if is_streamlit_cloud() else "vectorstores"
                    # Remove 'vectorstores/' prefix and join with base path
                    relative_path = member[12:]  # Remove 'vectorstores/' prefix
                    extract_path = os.path.join(VECTORSTORE_ROOT, relative_path)
                else:
                    st.warning(f"⚠️ Skipping unknown file: {member}")
                    continue
                
                if extract_path is None:
                    continue
                
                # Ensure we don't have problematic absolute paths
                if os.path.isabs(extract_path) and not extract_path.startswith(('/mount/tmp', '/tmp')):
                    # If it's an absolute path but not in allowed directories, make it relative
                    extract_path = extract_path.lstrip('/')
                    extract_path = os.path.join('.', extract_path)
                
                # Create directory if it doesn't exist
                extract_dir = os.path.dirname(extract_path)
                if extract_dir:  # Only create if there's actually a directory path
                    os.makedirs(extract_dir, exist_ok=True)
                
                # Extract file
                try:
                    with zip_file.open(member) as source, open(extract_path, 'wb') as target:
                        target.write(source.read())
                    successful_extractions += 1
                    
                    # Track companies for cache clearing (additional check for both formats)
                    if member.startswith('data/pdfs/'):
                        parts = member.split('/')
                        if len(parts) >= 3:  # data/pdfs/COMPANY/...
                            company_name = parts[2]
                            extracted_companies.add(company_name)
                    elif member.startswith('pdfs/'):
                        parts = member.split('/')
                        if len(parts) >= 2:  # pdfs/COMPANY/...
                            company_name = parts[1]
                            extracted_companies.add(company_name)
                            
                except Exception as file_error:
                    st.warning(f"⚠️ Could not extract {member}: {file_error}")
                    failed_extractions += 1
                    continue
        
        # Clear vectorstore cache for all restored companies
        for company in extracted_companies:
            clear_company_vectorstore_cache(company)
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        # Enhanced success reporting
        st.info(f"📊 Extraction Summary:")
        st.info(f"   • Successful extractions: {successful_extractions}")
        st.info(f"   • Failed extractions: {failed_extractions}")
        st.info(f"   • Companies found: {len(extracted_companies)}")
        
        return True, extracted_companies
        
    except Exception as e:
        # Clean up temp file in case of error
        try:
            if 'tmp_file_path' in locals():
                os.unlink(tmp_file_path)
        except:
            pass
        return False, str(e)

# Load environment variables
load_dotenv()

# Gemini model fallback configuration (ordered by preference)
GEMINI_MODELS = [
    "gemini-2.5-flash",           # 15 RPM, 1M TPM, 1000 RPD
    "gemini-2.5-flash-lite-preview-06-17",  # 15 RPM, 250K TPM, 1000 RPD  
    "gemini-2.0-flash",           # 10 RPM, 250K TPM, 250 RPD
    "gemini-2.0-flash-lite",      # 30 RPM, 1M TPM, 200 RPD
    "gemini-2.5-pro"              # 5 RPM, 250K TPM, 100 RPD
]

# Initialize session state for model tracking
if 'current_model_index' not in st.session_state:
    st.session_state.current_model_index = 0

# Initialize session state
if 'selected_company' not in st.session_state:
    st.session_state.selected_company = None
if 'current_view' not in st.session_state:
    st.session_state.current_view = "Ask Questions"
if 'upload_success_message' not in st.session_state:
    st.session_state.upload_success_message = None
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()

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
        color: white;
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
        
        # Backup/Restore Section
        st.markdown("#### 💾 Backup & Restore")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download Backup
            if st.button("📥 Create Full Backup", help="Download all PDFs and vectorstores"):
                with st.spinner("🔄 Creating backup..."):
                    backup_data = create_full_backup()
                    if backup_data:
                        from datetime import datetime
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"brokergpt_backup_{timestamp}.zip"
                        
                        st.download_button(
                            label="📥 Download Backup",
                            data=backup_data,
                            file_name=filename,
                            mime="application/zip",
                            help="Click to download the complete backup file"
                        )
                        st.success("✅ Backup created successfully!")
        
        with col2:
            # Upload Restore
            st.markdown("**Restore from Backup:**")
            backup_file = st.file_uploader(
                "Upload backup file:",
                type=['zip'],
                help="Upload a previously created backup file",
                key="backup_uploader"
            )
            
            if backup_file:
                if st.button("🔄 Restore Backup", type="secondary"):
                    with st.spinner("🔄 Restoring from backup..."):
                        success, result = restore_from_backup(backup_file)
                        
                        if success:
                            companies = result
                            st.success(f"✅ Successfully restored {len(companies)} companies!")
                            st.info(f"Restored companies: {', '.join(companies)}")
                            
                            # Clear all session state to force refresh
                            for key in list(st.session_state.keys()):
                                if key.startswith('vectorstore_'):
                                    del st.session_state[key]
                            
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error(f"❌ Restore failed: {result}")
        
        st.markdown("---")
        
        # Add new company (existing code)
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
                st.session_state.upload_success_message = None
                st.rerun()
        
        with col2:
            logo = get_company_logo(company)
            if logo:
                st.image(logo, width=30)
    
    if st.session_state.selected_company:
        
        # Admin controls for selected company
        if st.session_state.get('admin_authenticated', False):

            st.markdown("---")
            st.markdown("### 📄 Upload PDFs")
            
            selected_company = st.session_state.selected_company
            
            # Display current PDFs
            current_pdfs = get_uploaded_pdfs(selected_company)
            if current_pdfs:
                st.markdown("**Current PDFs:**")
                for pdf in current_pdfs:
                    st.markdown(f"• {pdf}")
            
            # File uploader
            uploaded_pdf = st.file_uploader(
                f"Upload PDF to {selected_company}:", 
                type="pdf", 
                key=f"pdf_uploader_{selected_company}"
            )
            
            # Handle file upload
            if uploaded_pdf:
                file_id = f"{selected_company}_{uploaded_pdf.name}_{uploaded_pdf.size}"
                
                # Only process if this file hasn't been processed yet
                if file_id not in st.session_state.processed_files:
                    try:
                        save_path = os.path.join(company_base_dir, selected_company, uploaded_pdf.name)
                        with open(save_path, "wb") as f:
                            f.write(uploaded_pdf.getbuffer())
                        
                        st.session_state.processed_files.add(file_id)
                        st.session_state.upload_success_message = f"✅ Uploaded: {uploaded_pdf.name}"
                        
                        time.sleep(0.1)
                        
                    except Exception as e:
                        st.error(f"❌ Error uploading file: {str(e)}")
            
            # Display upload success message
            if st.session_state.upload_success_message:
                st.success(st.session_state.upload_success_message)
                if st.button("✅ Continue", key="clear_upload_msg"):
                    st.session_state.upload_success_message = None
                    st.rerun()
            
            st.markdown("---")
            st.markdown("### ⚙️ Admin Actions")
            
            # Enhanced Relearn PDFs
            if st.button("🔄 Relearn PDFs"):
                try:
                    from ingest import ingest_company_pdfs
                    
                    with st.spinner("🔄 Rebuilding knowledge base..."):
                        VECTORSTORE_ROOT = "/mount/tmp/vectorstores" if is_streamlit_cloud() else "vectorstores"
                        vectorstore_path = os.path.join(VECTORSTORE_ROOT, selected_company)
                        
                        # Clear the cached vectorstore
                        clear_company_vectorstore_cache(selected_company)
                        
                        # Remove existing vectorstore
                        if os.path.exists(vectorstore_path):
                            try:
                                shutil.rmtree(vectorstore_path, ignore_errors=True)
                                time.sleep(2)
                            except Exception as cleanup_error:
                                st.warning(f"⚠️ Cleanup warning: {cleanup_error}")
                        
                        os.makedirs(vectorstore_path, exist_ok=True)
                        
                        # Progress indicator
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("📖 Processing PDFs...")
                        progress_bar.progress(25)
                        
                        # Run the ingestion
                        vectordb = ingest_company_pdfs(selected_company, persist_directory=vectorstore_path)
                        
                        progress_bar.progress(75)
                        status_text.text("✅ Finalizing...")
                        
                        time.sleep(1)
                        
                        progress_bar.progress(100)
                        status_text.text("🎉 Complete!")
                        
                        st.success("✅ Knowledge base updated successfully!")
                        
                        time.sleep(1)
                        st.rerun()
                        
                except Exception as e:
                    error_msg = str(e)
                    if "no such table: tenants" in error_msg:
                        st.error("❌ Database corruption detected. Please try again - this usually resolves the issue.")
                        st.info("💡 If the problem persists, try deleting and re-adding the company data.")
                    else:
                        st.error(f"❌ Error: {error_msg}")
                    
                    clear_company_vectorstore_cache(selected_company)
            
            # Delete company data
            st.markdown('<div class="danger-zone">', unsafe_allow_html=True)
            st.markdown("#### 🗑️ Danger Zone")
            
            if st.button("🗑️ Delete All Company Data", type="secondary"):
                if st.button("⚠️ CONFIRM DELETE", key="confirm_delete"):
                    try:
                        # Clear vectorstore cache
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
                        
                        # Clear processed files
                        st.session_state.processed_files = {
                            f for f in st.session_state.processed_files 
                            if not f.startswith(f"{selected_company}_")
                        }
                        
                        st.success(f"✅ Deleted all data for {selected_company}")
                        st.session_state.selected_company = None
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error deleting: {str(e)}")
            
            st.markdown('</div>', unsafe_allow_html=True)

# Main content area

# Add a radio button for view selection (Ask Questions or General Chat)
view_option = st.sidebar.radio("Select View", ("Ask Questions", "General Chat"))
st.session_state.current_view = view_option




if st.session_state.current_view == "General Chat":
    st.markdown("---")
    st.subheader("💬 General Chat (All Companies)")
    
    general_query = st.text_input("🔍 Enter your question for all companies:", placeholder="Ask a general question...")
    
    if general_query:
        st.info("Fetching responses from all companies...")
        
        company_base_dir = "data/pdfs"
        company_folders = [f for f in os.listdir(company_base_dir) 
                          if os.path.isdir(os.path.join(company_base_dir, f))]
        
        if not company_folders:
            st.warning("⚠️ No companies found to query.")
        else:
            VECTORSTORE_ROOT = "/mount/tmp/vectorstores" if is_streamlit_cloud() else "vectorstores"
            
            for company in company_folders:
                vectorstore_path = os.path.join(VECTORSTORE_ROOT, company)
                
                if os.path.exists(vectorstore_path):
                    st.markdown(f"### 🏢 Response from {company}")
                    try:
                        # Get company-specific vectorstore
                        vectorstore = get_company_vectorstore(company, vectorstore_path)
                        
                        retriever = vectorstore.as_retriever()
                        docs = retriever.get_relevant_documents(general_query)
                        context = """

""".join([doc.page_content for doc in docs])
                        
                        payload = {
                            "contents": [{
                                "parts": [{
                                    "text": f"""As a professional insurance broker assistant, answer the following question using ONLY the context provided for {company}.

Question: {general_query}

Context from {company}: {context}

Please provide a clear, professional response that would be helpful for insurance brokers and their clients. Base your answer ONLY on the provided context from {company}.
"""
                                }]
                            }]
                        }

                        response, used_model = call_gemini_with_fallback(payload)
                        st.info(f"🤖 Using model: {used_model}")

                        if response.status_code == 200:
                            try:
                                answer = response.json()['candidates'][0]['content']['parts'][0]['text']
                                st.success(answer)
                                
                                # Show source documents with download links
                                if docs:
                                    with st.expander("📚 Source Documents"):
                                        for i, doc in enumerate(docs[:3]):
                                            st.markdown(f"**Source {i+1}:**")
                                            st.text(doc.page_content[:500] + "...")
                                            
                                            # Add download link if source information is available
                                            if 'source' in doc.metadata:
                                                source_file = doc.metadata['source']
                                                file_path = os.path.join("data/pdfs", company, os.path.basename(source_file))
                                                if os.path.exists(file_path):
                                                    with open(file_path, "rb") as f:
                                                        st.download_button(
                                                            label=f"Download {os.path.basename(source_file)}",
                                                            data=f,
                                                            file_name=os.path.basename(source_file),
                                                            mime="application/pdf",
                                                            key=f"download_{company}_{i}"
                                                        )
                                            st.markdown("---")
                                            
                            except Exception as e:
                                st.error("❌ Error parsing response from Gemini")
                        else:
                            st.error(f"❌ Gemini API Error: {response.status_code}")
                            
                    except Exception as e:
                        error_msg = str(e)
                        if "no such table: tenants" in error_msg:
                            st.error("❌ Database error detected. Please use admin access to click 'Relearn PDFs' to rebuild the knowledge base.")
                            clear_company_vectorstore_cache(company)
                        else:
                            st.error(f"❌ Error accessing knowledge base: {error_msg}")
                            st.info("💡 Try using admin access to click 'Relearn PDFs' to rebuild the knowledge base.")
                            clear_company_vectorstore_cache(company)
                else:
                    st.warning(f"⚠️ Knowledge base not found for {company}.")
                st.markdown("---")

elif st.session_state.selected_company:
    selected_company = st.session_state.selected_company

    # Path handling
    VECTORSTORE_ROOT = "/mount/tmp/vectorstores" if is_streamlit_cloud() else "vectorstores"
    vectorstore_path = os.path.join(VECTORSTORE_ROOT, selected_company)
    
    # Check if vectorstore exists and has actual data
    vectorstore_exists = False
    if os.path.exists(vectorstore_path):
        # Check if the directory has any files (not just empty directory)
        try:
            files_in_vectorstore = []
            for root, dirs, files in os.walk(vectorstore_path):
                files_in_vectorstore.extend(files)
            
            # Consider vectorstore valid if it has files
            if files_in_vectorstore:
                vectorstore_exists = True
            else:
                st.warning(f"⚠️ Empty vectorstore found for {selected_company}. Please use 'Relearn PDFs'.")
        except Exception as e:
            st.warning(f"⚠️ Error checking vectorstore for {selected_company}: {str(e)}")
    
    if not vectorstore_exists:
        st.info(f"📚 Upload PDFs for **{selected_company}** and use admin access to click 'Relearn PDFs' to start.")
    else:
        if st.session_state.current_view == "Dashboard":
            st.markdown("---")
            st.subheader("📊 Company Dashboard")
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            
            uploaded_pdfs = get_uploaded_pdfs(selected_company)
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
                        context = """

""".join([doc.page_content for doc in docs])

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

                        response, used_model = call_gemini_with_fallback(payload)
                        st.info(f"🤖 Using model: {used_model}")

                        st.markdown("---")
                        if response.status_code == 429:
                            st.error("🚫 Rate limit reached. Please wait a moment and try again.")
                            st.info("💡 Try asking fewer questions or wait 1-2 minutes between requests.")
                        elif response.status_code == 200:
                            try:
                                answer = response.json()['candidates'][0]['content']['parts'][0]['text']
                                st.markdown("### 🤖 Broker-GPT Response")
                                st.markdown(f"**Company:** {selected_company}")
                                st.markdown(f"**Question:** {query}")
                                st.markdown("**Answer:**")
                                st.success(answer)
                                
                                # Show source documents with download links
                                if docs:
                                    with st.expander("📚 Source Documents"):
                                        for i, doc in enumerate(docs[:3]):
                                            st.markdown(f"**Source {i+1}:**")
                                            st.text(doc.page_content[:500] + "...")
                                            
                                            # Add download link if source information is available
                                            if 'source' in doc.metadata:
                                                source_file = doc.metadata['source']
                                                # Assuming the source metadata contains the full path within the data/pdfs structure
                                                # You might need to adjust this path based on how your source metadata is stored
                                                file_path = source_file
                                                if os.path.exists(file_path):
                                                     with open(file_path, "rb") as f:
                                                        st.download_button(
                                                            label=f"Download {os.path.basename(source_file)}",
                                                            data=f,
                                                            file_name=os.path.basename(source_file),
                                                            mime="application/pdf",
                                                            key=f"download_{selected_company}_{i}"
                                                        )
                                            st.markdown("---")
                                            
                            except Exception as e:
                                st.error("❌ Error parsing response from Gemini")
                        else:
                            st.error(f"❌ Gemini API Error: {response.status_code}")
                            
                    except Exception as e:
                        error_msg = str(e)
                        if "no such table: tenants" in error_msg:
                            st.error("❌ Database error detected. Please use admin access to click 'Relearn PDFs' to rebuild the knowledge base.")
                            clear_company_vectorstore_cache(selected_company)
                        else:
                            st.error(f"❌ Error accessing knowledge base: {error_msg}")
                            st.info("💡 Try using admin access to click 'Relearn PDFs' to rebuild the knowledge base.")
                            clear_company_vectorstore_cache(selected_company)
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Ask Questions", key="nav_questions"):
            st.session_state.current_view = "Ask Questions"
    
    with col2:
        if st.button("📊 Dashboard", key="nav_dashboard"):
            st.session_state.current_view = "Dashboard"

else:
    st.info("👆 Please select a company from the sidebar to continue.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🤖 Broker-GPT | Powered by AI | Version 16.0.5 | 2025"
    "</div>", 
    unsafe_allow_html=True
)
