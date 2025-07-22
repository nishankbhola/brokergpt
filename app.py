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
    st.session_state.current_view = "General Chat"
if 'upload_success_message' not in st.session_state:
    st.session_state.upload_success_message = None
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()

# Page configuration
st.set_page_config(
    page_title="🤖 BIBLIO",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        color: black;
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

    .tab-button {
    /* Base styling */
    padding: 12px 24px;
    border-radius: 16px;
    margin: 0 8px;
    border: none;
    cursor: pointer;
    font-weight: 600;
    font-size: 14px;
    position: relative;
    overflow: hidden;
    
    /* Modern gradient background */
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    
    /* Smooth transitions */
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    
    /* Subtle shadow */
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}

    .tab-button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s ease;
    }
    
    .tab-button:hover {
        /* Lift and glow effect */
        transform: translateY(-4px) scale(1.02);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        
        /* Brighter gradient on hover */
        background: linear-gradient(135deg, #7c8ef7 0%, #8a5cb8 100%);
    }
    
    .tab-button:hover::before {
        /* Shimmer effect */
        left: 100%;
    }
    
    .tab-button:active {
        transform: translateY(-2px) scale(1.01);
        transition: all 0.1s ease;
    }
    
    /* Active/selected state */
    .tab-button.active {
        background: linear-gradient(135deg, #ff6b6b 0%, #ff8e53 100%);
        box-shadow: 0 6px 20px rgba(255, 107, 107, 0.4);
    }
    
    .tab-button.active:hover {
        background: linear-gradient(135deg, #ff7979 0%, #ffa368 100%);
        box-shadow: 0 8px 25px rgba(255, 107, 107, 0.5);
    }
    
    /* Disabled state */
    .tab-button:disabled {
        background: linear-gradient(135deg, #e0e0e0 0%, #bdbdbd 100%);
        color: #757575;
        cursor: not-allowed;
        transform: none;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Alternative glass morphism style */
    .tab-button.glass {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: #333;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .tab-button.glass:hover {
        background: rgba(255, 255, 255, 0.2);
        border-color: rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h4>🤖BIBLIO </h4>
    <h6>Insurance broker assistant </h6>
</div>
""", unsafe_allow_html=True)

# Create necessary directories
company_base_dir = "data/pdfs"
logos_dir = "data/logos"
os.makedirs(company_base_dir, exist_ok=True)
os.makedirs(logos_dir, exist_ok=True)


# Sidebar for company management
with st.sidebar:
    st.header("BIBLIO")
    

    # Company selection
    st.markdown("---")
    st.markdown("### 📁 Select Company")
    
    company_folders = [f for f in os.listdir(company_base_dir) 
                      if os.path.isdir(os.path.join(company_base_dir, f))]
    
    if not company_folders:
        st.warning("⚠️ BIBLIO under maintenance")
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

# Tab navigation at the top

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("💬 General Chat", key="tab_general", use_container_width=True, 
                type="primary" if st.session_state.current_view == "General Chat" else "secondary"):
        st.session_state.current_view = "General Chat"
        st.rerun()

with col2:
    if st.button("🔍 Ask Questions", key="tab_questions", use_container_width=True,
                type="primary" if st.session_state.current_view == "Ask Questions" else "secondary"):
        st.session_state.current_view = "Ask Questions" 
        st.rerun()

with col3:
    if st.button("📚 Resources", key="tab_resources", use_container_width=True,
                type="primary" if st.session_state.current_view == "Resources" else "secondary"):
        st.session_state.current_view = "Resources"
        st.rerun()

st.markdown("---")






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
                                    "text": f""" In less than 20 words, As a professional insurance broker assistant, answer the following question using ONLY the context provided for {company}.

Question: {general_query}

Context from {company}: {context}

Please provide a clear, professional response in less than 20 words that would be helpful for insurance brokers and their clients. Base your answer ONLY on the provided context from {company}.
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
elif st.session_state.current_view == "Resources":
    st.markdown("---")
    st.subheader("📚 Resources - All Company PDFs")
    
    company_base_dir = "data/pdfs"
    company_folders = [f for f in os.listdir(company_base_dir) 
                      if os.path.isdir(os.path.join(company_base_dir, f))]
    
    if not company_folders:
        st.warning("⚠️ No companies found with uploaded PDFs.")
    else:
        # Create tabs or expandable sections for each company
        for company in company_folders:
            with st.expander(f"🏢 {company}", expanded=False):
                company_pdf_dir = os.path.join(company_base_dir, company)
                pdf_files = [f for f in os.listdir(company_pdf_dir) if f.endswith(".pdf")]
                
                if pdf_files:
                    st.info(f"Found {len(pdf_files)} PDF(s) for {company}")
                    
                    # Display company logo if available
                    logo = get_company_logo(company)
                    if logo:
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            st.image(logo, width=80)
                    
                    # Create columns for better layout
                    cols = st.columns(2)
                    
                    for i, pdf_file in enumerate(pdf_files):
                        with cols[i % 2]:  # Alternate between columns
                            pdf_path = os.path.join(company_pdf_dir, pdf_file)
                            
                            # Create a container for each PDF
                            with st.container():
                                st.markdown(f"**📄 {pdf_file}**")
                                
                                # Get file size
                                try:
                                    file_size = os.path.getsize(pdf_path)
                                    size_mb = round(file_size / (1024 * 1024), 2)
                                    st.caption(f"Size: {size_mb} MB")
                                except:
                                    st.caption("Size: Unknown")
                                
                                # Download button
                                try:
                                    with open(pdf_path, "rb") as f:
                                        pdf_data = f.read()
                                        st.download_button(
                                            label=f"⬇️ Download {pdf_file}",
                                            data=pdf_data,
                                            file_name=pdf_file,
                                            mime="application/pdf",
                                            key=f"download_resources_{company}_{pdf_file}",
                                            use_container_width=True
                                        )
                                except Exception as e:
                                    st.error(f"Error loading {pdf_file}: {str(e)}")
                                
                                st.markdown("---")
                else:
                    st.info(f"No PDFs found for {company}")
        
        # Summary statistics
        st.markdown("---")
        st.subheader("📊 Summary")
        
        total_pdfs = 0
        total_size = 0
        
        for company in company_folders:
            company_pdf_dir = os.path.join(company_base_dir, company)
            pdf_files = [f for f in os.listdir(company_pdf_dir) if f.endswith(".pdf")]
            total_pdfs += len(pdf_files)
            
            for pdf_file in pdf_files:
                try:
                    file_size = os.path.getsize(os.path.join(company_pdf_dir, pdf_file))
                    total_size += file_size
                except:
                    pass
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🏢 Total Companies", len(company_folders))
        
        with col2:
            st.metric("📄 Total PDFs", total_pdfs)
        
        with col3:
            total_size_mb = round(total_size / (1024 * 1024), 2)
            st.metric("💽 Total Size", f"{total_size_mb} MB")
            
elif st.session_state.selected_company:
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
                with st.spinner("🤖 BIBLIO is analyzing your question..."):
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
                                st.markdown("### 🤖 BIBLIO Response")
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
    
    


        
else:
    st.info("👆 Please select a company from the sidebar to continue.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🤖 BIBLIO | Powered by AI | Version 20.1.1 | 2025"
    "</div>", 
    unsafe_allow_html=True
)
