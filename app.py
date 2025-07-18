import pysqlite3
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

# --- Cached function to load the embedding model ---
# This ensures the large model is loaded into memory only once and shared across all user sessions.
@st.cache_resource
def load_embedding_model():
    """Loads the sentence transformer model only once."""
    print("Loading embedding model...")
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# --- Function to initialize a Chroma client ---
# This function creates a lightweight client pointing to the on-disk vector store.
# It does NOT load the whole database into memory.
def get_chroma_client(vectorstore_path):
    """Initializes a Chroma client from a persisted directory."""
    if not os.path.exists(vectorstore_path):
        return None
    
    # --- Use the cached function to get the model ---
    embedding_function = load_embedding_model()
    
    vectorstore = Chroma(
        persist_directory=vectorstore_path,
        embedding_function=embedding_function
    )
    return vectorstore

# Detect if running on Streamlit Cloud
def is_streamlit_cloud():
    """Checks if the app is running on Streamlit Community Cloud."""
    return "STREAMLIT_SERVER_MODE" in os.environ or "HOME" in os.environ

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
        with col2:
            st.markdown(f"## {company_name}")
    else:
        st.markdown(f"## 🏢 {company_name}")

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
                # IMPORTANT: In a real application, use st.secrets for passwords.
                if password == os.getenv("ADMIN_PASSWORD", "classmate"):
                    st.session_state.admin_authenticated = True
                    st.success("✅ Admin access granted!")
                    st.rerun()
                else:
                    st.error("❌ Invalid password")
                    return False
        return False
    return True

def get_uploaded_pdfs(company_name):
    """Get list of uploaded PDFs for a company"""
    company_pdf_dir = os.path.join("data/pdfs", company_name)
    if os.path.exists(company_pdf_dir):
        return [f for f in os.listdir(company_pdf_dir) if f.endswith(".pdf")]
    return []

# Load environment variables from .env file
load_dotenv()

# Initialize session state variables
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

# Custom CSS for styling
st.markdown("""
<style>
    .main-header { text-align: center; }
    .stButton>button { width: 100%; }
    .danger-zone {
        background-color: #fff5f5;
        border: 1px solid #e53e3e;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<div class='main-header'><h1>🤖 Broker-GPT</h1></div>", unsafe_allow_html=True)

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
        # This button is used to bring up the password form
        st.session_state.admin_authenticated = False
        st.rerun()

    if check_admin_password():
        st.success("🔓 Admin Mode Active")
        
        st.markdown("#### ➕ Add New Company")
        with st.form("add_company_form"):
            new_company = st.text_input("Company Name:")
            logo_file = st.file_uploader("Company Logo (Optional PNG):", type=['png'])
            add_submitted = st.form_submit_button("Add Company")
            
            if add_submitted and new_company:
                new_path = os.path.join(company_base_dir, new_company)
                os.makedirs(new_path, exist_ok=True)
                
                if logo_file:
                    logo_path = os.path.join(logos_dir, f"{new_company}.png")
                    with open(logo_path, "wb") as f:
                        f.write(logo_file.getbuffer())
                
                st.success(f"✅ Added company: {new_company}")
                time.sleep(1)
                st.rerun()

    st.markdown("---")
    st.markdown("### 📁 Select Company")
    
    company_folders = sorted([f for f in os.listdir(company_base_dir) if os.path.isdir(os.path.join(company_base_dir, f))])
    
    if not company_folders:
        st.warning("⚠️ No companies found. Use Admin Access to add one.")
        st.stop()
    
    for company in company_folders:
        if st.button(f"📂 {company}", key=f"select_{company}"):
            st.session_state.selected_company = company
            st.session_state.upload_success_message = None # Clear message on switch
            st.rerun()
    
    if st.session_state.selected_company and st.session_state.get('admin_authenticated'):
        st.markdown("---")
        st.header(f"⚙️ Actions for {st.session_state.selected_company}")

        # PDF Upload Section
        st.markdown("### 📄 Upload PDFs")
        uploaded_pdfs = st.file_uploader(
            "Upload PDF documents", 
            type="pdf", 
            accept_multiple_files=True,
            key=f"pdf_uploader_{st.session_state.selected_company}"
        )
        
        if uploaded_pdfs:
            save_dir = os.path.join(company_base_dir, st.session_state.selected_company)
            for uploaded_pdf in uploaded_pdfs:
                save_path = os.path.join(save_dir, uploaded_pdf.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_pdf.getbuffer())
            st.success(f"✅ Uploaded {len(uploaded_pdfs)} file(s).")
            time.sleep(1)
            st.rerun()

        # Relearn PDFs Section
        st.markdown("### 🧠 Relearn PDFs")
        if st.button("🔄 Rebuild Knowledge Base"):
            try:
                from ingest import ingest_company_pdfs
                
                with st.spinner(f"🧠 Rebuilding knowledge base for {st.session_state.selected_company}... This may take a moment."):
                    VECTORSTORE_ROOT = "/mount/tmp/vectorstores" if is_streamlit_cloud() else "vectorstores"
                    vectorstore_path = os.path.join(VECTORSTORE_ROOT, st.session_state.selected_company)
                    
                    ingest_company_pdfs(st.session_state.selected_company, persist_directory=vectorstore_path)
                    
                st.success("✅ Knowledge base updated successfully!")
                time.sleep(2)
                st.rerun()
                    
            except Exception as e:
                st.error(f"❌ Ingestion Error: {e}")
        
        # Danger Zone for Deletion
        with st.container(border=True):
            st.markdown('<div class="danger-zone">', unsafe_allow_html=True)
            st.markdown("#### 🗑️ Danger Zone")
            
            if st.button("🗑️ Delete Company Data", type="primary"):
                try:
                    company = st.session_state.selected_company
                    # Delete PDFs
                    shutil.rmtree(os.path.join(company_base_dir, company), ignore_errors=True)
                    # Delete vectorstore
                    VECTORSTORE_ROOT = "/mount/tmp/vectorstores" if is_streamlit_cloud() else "vectorstores"
                    shutil.rmtree(os.path.join(VECTORSTORE_ROOT, company), ignore_errors=True)
                    # Delete logo
                    logo_path = os.path.join(logos_dir, f"{company}.png")
                    if os.path.exists(logo_path):
                        os.remove(logo_path)
                    
                    st.success(f"✅ Deleted all data for {company}")
                    st.session_state.selected_company = None
                    time.sleep(1)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error during deletion: {e}")
            st.markdown('</div>', unsafe_allow_html=True)


# Main content area
if st.session_state.selected_company:
    selected_company = st.session_state.selected_company
    display_company_with_logo(selected_company, size=100)

    # Define vector store path
    VECTORSTORE_ROOT = "/mount/tmp/vectorstores" if is_streamlit_cloud() else "vectorstores"
    vectorstore_path = os.path.join(VECTORSTORE_ROOT, selected_company)
    
    # Check if the knowledge base exists
    if not os.path.exists(vectorstore_path):
        st.warning(f"📚 The knowledge base for **{selected_company}** has not been created yet. Please upload PDFs and use the 'Rebuild Knowledge Base' button in the admin panel.")
    else:
        # Main question-answering interface
        st.subheader(f"💬 Ask a question about {selected_company}") 
        
        query = st.text_input("🔍 Enter your question:", placeholder="e.g., What is the policy on cyber insurance?", key=f"query_input_{selected_company}")
        
        if query:
            with st.spinner("🤖 Analyzing documents and formulating a response..."):
                try:
                    # --- MEMORY OPTIMIZATION ---
                    # Initialize the client on-demand. This is lightweight.
                    vectorstore = get_chroma_client(vectorstore_path)
                    
                    if vectorstore is None:
                        st.error("Vector store not found. Please rebuild the knowledge base.")
                    else:
                        # Retrieve relevant documents
                        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
                        docs = retriever.get_relevant_documents(query)
                        context = "\n\n".join([doc.page_content for doc in docs])

                        # Call Gemini API
                        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
                        if not GEMINI_API_KEY:
                            st.error("GEMINI_API_KEY not found. Please set it in your environment or a .env file.")
                        else:
                            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"
                            
                            prompt = f"""As a professional insurance broker assistant, answer the question based ONLY on the following context provided for {selected_company}. If the context does not contain the answer, state that you cannot answer based on the available documents.

Question: {query}

Context:
{context}
"""
                            payload = {"contents": [{"parts": [{"text": prompt}]}]}
                            headers = {"Content-Type": "application/json"}
                            
                            response = requests.post(api_url, headers=headers, data=json.dumps(payload))

                            st.markdown("---")
                            if response.status_code == 200:
                                result = response.json()
                                answer = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'No response found.')
                                
                                st.markdown("### 🤖 Response")
                                st.success(answer)
                                
                                with st.expander("📚 View Sources"):
                                    for i, doc in enumerate(docs):
                                        st.info(f"**Source {i+1}:** (From: {doc.metadata.get('source', 'N/A')})")
                                        st.text(doc.page_content)
                                        st.markdown("---")
                            else:
                                st.error(f"❌ Gemini API Error: {response.status_code} - {response.text}")
                        
                except Exception as e:
                    st.error(f"❌ An error occurred: {e}")
                    st.info("💡 If this persists, try rebuilding the knowledge base via the admin panel.")

else:
    st.info("👆 Please select a company from the sidebar to get started.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🤖 Broker-GPT | Memory Optimized Version | © 2025"
    "</div>", 
    unsafe_allow_html=True
)
