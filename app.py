import pysqlite3 as sqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import shutil
import json
import requests
import streamlit as st
import time
import gc
import psutil
from PIL import Image
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Detect if running on Streamlit Cloud
def is_streamlit_cloud():
    return os.environ.get("HOME") == "/home/adminuser"

@st.cache_resource
def get_embedding_model():
    """Cache the embedding model to avoid reloading"""
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def get_memory_usage():
    """Get current memory usage in MB"""
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    except:
        return 0

def force_garbage_collection():
    """Force garbage collection to free memory"""
    gc.collect()
    
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_company_folders():
    """Cache company folder list"""
    company_base_dir = "data/pdfs"
    if not os.path.exists(company_base_dir):
        return []
    return [f for f in os.listdir(company_base_dir) 
            if os.path.isdir(os.path.join(company_base_dir, f))]

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_uploaded_pdfs_cached(company_name):
    """Cached version of get_uploaded_pdfs"""
    company_pdf_dir = os.path.join("data/pdfs", company_name)
    if os.path.exists(company_pdf_dir):
        return [f for f in os.listdir(company_pdf_dir) if f.endswith(".pdf")]
    return []

@st.cache_resource
def get_cached_logo(company_name):
    """Cache company logos to avoid repeated file I/O"""
    logo_path = os.path.join("data/logos", f"{company_name}.png")
    if os.path.exists(logo_path):
        return Image.open(logo_path)
    return None

def create_chroma_vectorstore(vectorstore_path, company_name, max_retries=5):
    """Create Chroma vectorstore with enhanced retry logic and company-specific caching"""
    for attempt in range(max_retries):
        try:
            # Clear any existing chroma client for this company
            vectorstore_key = f'vectorstore_{company_name}'
            if vectorstore_key in st.session_state:
                # Properly close the old vectorstore
                try:
                    old_vectorstore = st.session_state[vectorstore_key]
                    if hasattr(old_vectorstore, '_client'):
                        old_vectorstore._client.reset()
                except:
                    pass
                del st.session_state[vectorstore_key]
                force_garbage_collection()
            
            os.makedirs(vectorstore_path, exist_ok=True)
            
            vectorstore = Chroma(
                persist_directory=vectorstore_path,
                embedding_function=get_embedding_model(),
                client_settings=None
            )
            
            vectorstore._client.heartbeat()
            return vectorstore
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 * (attempt + 1)  # Exponential backoff
                force_garbage_collection()  # Clean up memory before retry
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
    return get_cached_logo(company_name)

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
        # Properly close the vectorstore before deletion
        try:
            vectorstore = st.session_state[vectorstore_key]
            if hasattr(vectorstore, '_client'):
                vectorstore._client.reset()
        except:
            pass
        del st.session_state[vectorstore_key]
        force_garbage_collection()

@st.cache_resource
def get_company_vectorstore(_company_name, _vectorstore_path):
    """Get or create company-specific vectorstore with proper caching"""
    return create_chroma_vectorstore(_vectorstore_path, _company_name)

def get_uploaded_pdfs(company_name):
    """Get list of uploaded PDFs for a company"""
    return get_uploaded_pdfs_cached(company_name)

# Load environment variables
load_dotenv()

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
        /*background: linear-gradient(90deg, #1e3c72, #2a5298);
        
        padding: 0.1rem;
        border-radius: 1px;
        margin-bottom: 0.1rem;
        */
    }
    .memory-info {
        background: #f0f2f6;
        border-radius: 5px;
        padding: 0.5rem;
        margin: 0.5rem 0;
        font-size: 0.8rem;
        color: #666;
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

# Memory monitoring (only show in admin mode)
if st.session_state.get('admin_authenticated', False):
    memory_usage = get_memory_usage()
    if memory_usage > 0:
        memory_color = "🟢" if memory_usage < 500 else "🟡" if memory_usage < 800 else "🔴"
        st.markdown(f"""
        <div class="memory-info">
            {memory_color} Memory Usage: {memory_usage:.1f} MB / 1024 MB
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
    
    company_folders = get_company_folders()
    
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
                    force_garbage_collection()
                
                st.session_state.selected_company = company
                # Clear upload success message when switching companies
                st.session_state.upload_success_message = None
                # Clear cached data for new company
                get_uploaded_pdfs_cached.clear()
                get_company_folders.clear()
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
            
            # File uploader with unique key to prevent conflicts
            uploaded_pdf = st.file_uploader(
                f"Upload PDF to {selected_company}:", 
                type="pdf", 
                key=f"pdf_uploader_{selected_company}"
            )
            
            # Handle file upload without immediate rerun
            if uploaded_pdf:
                file_id = f"{selected_company}_{uploaded_pdf.name}_{uploaded_pdf.size}"
                
                # Only process if this file hasn't been processed yet
                if file_id not in st.session_state.processed_files:
                    try:
                        save_path = os.path.join(company_base_dir, selected_company, uploaded_pdf.name)
                        with open(save_path, "wb") as f:
                            f.write(uploaded_pdf.getbuffer())
                        
                        # Mark this file as processed
                        st.session_state.processed_files.add(file_id)
                        st.session_state.upload_success_message = f"✅ Uploaded: {uploaded_pdf.name}"
                        
                        # Small delay to ensure file is written
                        time.sleep(0.1)
                        
                    except Exception as e:
                        st.error(f"❌ Error uploading file: {str(e)}")
            
            # Display upload success message if exists
            if st.session_state.upload_success_message:
                st.success(st.session_state.upload_success_message)
                # Clear message after displaying
                if st.button("✅ Continue", key="clear_upload_msg"):
                    st.session_state.upload_success_message = None
                    st.rerun()
            
            st.markdown("---")
            st.markdown("### ⚙️ Admin Actions")
            
            # Enhanced Relearn PDFs with better error handling
            if st.button("🔄 Relearn PDFs"):
                try:
                    from ingest import ingest_company_pdfs
                    
                    with st.spinner("🔄 Rebuilding knowledge base..."):
                        VECTORSTORE_ROOT = "/mount/tmp/vectorstores" if is_streamlit_cloud() else "vectorstores"
                        vectorstore_path = os.path.join(VECTORSTORE_ROOT, selected_company)
                        
                        # Clear the cached vectorstore FIRST
                        clear_company_vectorstore_cache(selected_company)
                        
                        # Clear Streamlit cache for this company
                        get_company_vectorstore.clear()
                        get_uploaded_pdfs_cached.clear()
                        force_garbage_collection()
                        
                        # Remove existing vectorstore with better error handling
                        if os.path.exists(vectorstore_path):
                            try:
                                shutil.rmtree(vectorstore_path, ignore_errors=True)
                                time.sleep(2)  # Wait for cleanup
                                force_garbage_collection()
                            except Exception as cleanup_error:
                                st.warning(f"⚠️ Cleanup warning: {cleanup_error}")
                        
                        # Ensure directory exists
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
                        
                        # Small delay to ensure everything is written
                        time.sleep(1)
                        force_garbage_collection()
                        
                        progress_bar.progress(100)
                        status_text.text("🎉 Complete!")
                        
                        st.success("✅ Knowledge base updated successfully!")
                        
                        # Clear progress indicators
                        time.sleep(1)
                        st.rerun()
                        
                except Exception as e:
                    error_msg = str(e)
                    if "no such table: tenants" in error_msg:
                        st.error("❌ Database corruption detected. Please try again - this usually resolves the issue.")
                        st.info("💡 If the problem persists, try deleting and re-adding the company data.")
                    else:
                        st.error(f"❌ Error: {error_msg}")
                    
                    # Clear any cached data that might be causing issues
                    clear_company_vectorstore_cache(selected_company)
                    force_garbage_collection()
            
            # Delete company data
            st.markdown('<div class="danger-zone">', unsafe_allow_html=True)
            st.markdown("#### 🗑️ Danger Zone")
            
            if st.button("🗑️ Delete All Company Data", type="secondary"):
                if st.button("⚠️ CONFIRM DELETE", key="confirm_delete"):
                    try:
                        # Clear vectorstore cache first
                        clear_company_vectorstore_cache(selected_company)
                        
                        # Clear all related caches
                        get_company_vectorstore.clear()
                        get_uploaded_pdfs_cached.clear()
                        get_company_folders.clear()
                        get_cached_logo.clear()
                        force_garbage_collection()
                        
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
                        
                        # Clear processed files for this company
                        st.session_state.processed_files = {
                            f for f in st.session_state.processed_files 
                            if not f.startswith(f"{selected_company}_")
                        }
                        
                        force_garbage_collection()
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
                        
                        # Limit context size to prevent memory issues
                        max_context_length = 8000  # Reduced from unlimited
                        context_parts = []
                        current_length = 0
                        
                        for doc in docs:
                            if current_length + len(doc.page_content) > max_context_length:
                                break
                            context_parts.append(doc.page_content)
                            current_length += len(doc.page_content)
                        
                        context = "\n\n".join(context_parts)

                        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
                        
                        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

                        payload = {
                            "contents": [{
                                "parts": [{
                                    "text": f"""You are a professional insurance broker assistant. Answer the following question using the context provided for {selected_company}.

Question: {query}

Context from {selected_company}:
{context}

Instructions:
- Provide a clear, professional response
- Base your answer on the provided context
- If the context doesn't contain relevant information, say so
- Be helpful for insurance brokers and their clients"""
                                }]
                            }]
                        }

                        headers = {"Content-Type": "application/json"}
                        response = requests.post(url, headers=headers, data=json.dumps(payload))
                        
                        # Clean up variables to free memory
                        del context_parts, context, payload
                        force_garbage_collection()

                        st.markdown("---")
                        if response.status_code == 200:
                            response_data = response.json()
                            answer = response_data['candidates'][0]['content']['parts'][0]['text']
                            
                            st.markdown("### 🤖 Broker-GPT Response")
                            st.markdown(f"**Company:** {selected_company}")
                            st.markdown(f"**Question:** {query}")
                            st.markdown("**Answer:**")
                            st.success(answer)
                            
                            # Show source documents
                            if docs:
                                with st.expander("📚 Source Documents"):
                                    for i, doc in enumerate(docs[:2]):  # Reduced to top 2 sources
                                        st.markdown(f"**Source {i+1}:**")
                                        st.text(doc.page_content[:300] + "...")  # Reduced preview length
                                        st.markdown("---")
                            
                            # Clean up response data
                            del answer, response, response_data
                            force_garbage_collection()
                        else:
                            st.error(f"❌ Gemini API Error: {response.status_code}")
                            try:
                                error_data = response.json()
                                st.json(error_data)
                                
                                # Show specific error message if available
                                if 'error' in error_data:
                                    error_info = error_data['error']
                                    if 'message' in error_info:
                                        st.error(f"Error details: {error_info['message']}")
                            except:
                                st.text(f"Raw error response: {response.text}")
                            
                    except Exception as e:
                        error_msg = str(e)
                        if "no such table: tenants" in error_msg:
                            st.error("❌ Database error detected. Please use admin access to click 'Relearn PDFs' to rebuild the knowledge base.")
                            clear_company_vectorstore_cache(selected_company)
                            force_garbage_collection()
                        else:
                            st.error(f"❌ Error accessing knowledge base: {error_msg}")
                            st.info("💡 Try using admin access to click 'Relearn PDFs' to rebuild the knowledge base.")
                            # Clear the cached vectorstore for this company
                            clear_company_vectorstore_cache(selected_company)
                            force_garbage_collection()
                                st.markdown("### 🤖 Broker-GPT Response")
                                st.markdown(f"**Company:** {selected_company}")
                                st.markdown(f"**Question:** {query}")
                                st.markdown("**Answer:**")
                                st.success(answer)
                                
                                # Show source documents
                                if docs:
                                    with st.expander("📚 Source Documents"):
                                        for i, doc in enumerate(docs[:2]):  # Reduced to top 2 sources
                                            st.markdown(f"**Source {i+1}:**")
                                            st.text(doc.page_content[:300] + "...")  # Reduced preview length
                                            st.markdown("---")
                                
                                # Clean up response data
                                del answer, response, response_data
                                force_garbage_collection()
                                            
                            except Exception as e:
                                st.error(f"❌ Error parsing response from Gemini: {str(e)}")
                                try:
                                    st.json(response.json())  # Show raw response for debugging
                                except:
                                    st.text(f"Raw response: {response.text}")
                        else:
                            st.error(f"❌ Gemini API Error: {response.status_code}")
                            try:
                                error_data = response.json()
                                st.json(error_data)
                                
                                # Show specific error message if available
                                if 'error' in error_data:
                                    error_info = error_data['error']
                                    if 'message' in error_info:
                                        st.error(f"Error details: {error_info['message']}")
                            except:
                                st.text(f"Raw error response: {response.text}")
                            
                    except Exception as e:
                        error_msg = str(e)
                        if "no such table: tenants" in error_msg:
                            st.error("❌ Database error detected. Please use admin access to click 'Relearn PDFs' to rebuild the knowledge base.")
                            clear_company_vectorstore_cache(selected_company)
                            force_garbage_collection()
                        else:
                            st.error(f"❌ Error accessing knowledge base: {error_msg}")
                            st.info("💡 Try using admin access to click 'Relearn PDFs' to rebuild the knowledge base.")
                            # Clear the cached vectorstore for this company
                            clear_company_vectorstore_cache(selected_company)
                            force_garbage_collection()
    
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
    "🤖 Broker-GPT | Powered by AI | Version 7.0.5 | 2025"
    "</div>", 
    unsafe_allow_html=True
)
