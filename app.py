import pysqlite3 as sqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import shutil
import json
import requests
import streamlit as st
import time
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
            # Clear any existing client settings
            if hasattr(st.session_state, 'chroma_client'):
                del st.session_state.chroma_client
            
            # Ensure directory exists
            os.makedirs(vectorstore_path, exist_ok=True)
            
            # Create vectorstore with explicit client settings
            vectorstore = Chroma(
                persist_directory=vectorstore_path,
                embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"),
                client_settings=None  # Use default client settings
            )
            
            # Test the connection
            vectorstore._client.heartbeat()
            return vectorstore
            
        except Exception as e:
            st.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retry
                # Clean up any partial database files
                if os.path.exists(vectorstore_path):
                    for file in os.listdir(vectorstore_path):
                        if file.endswith('.sqlite3') or file.endswith('.db'):
                            try:
                                os.remove(os.path.join(vectorstore_path, file))
                            except:
                                pass
            else:
                raise e

load_dotenv()

st.set_page_config(page_title="🤖 Broker-gpt", layout="wide")
st.title("🤝 Broker-gpt: Insurance Broker Assistant")

# === Sidebar UI ===
st.sidebar.header("🏢 Manage Companies")
company_base_dir = "data/pdfs"
os.makedirs(company_base_dir, exist_ok=True)

if "company_added" not in st.session_state:
    st.session_state["company_added"] = False

new_company = st.sidebar.text_input("➕ Create new company folder", key="new_company")
if st.sidebar.button("Add Company"):
    new_path = os.path.join(company_base_dir, new_company)
    if new_company and not os.path.exists(new_path):
        os.makedirs(new_path)
        st.sidebar.success(f"✅ Added company: {new_company}")
        st.session_state["company_added"] = True
    else:
        st.sidebar.warning("⚠️ Folder exists or name is empty")

company_folders = [f for f in os.listdir(company_base_dir) if os.path.isdir(os.path.join(company_base_dir, f))]
if not company_folders:
    st.warning("⚠️ No companies found. Add one to begin.")
    st.stop()

selected_company = st.sidebar.radio("📂 Select company", company_folders, key="selected_company")

uploaded_pdf = st.sidebar.file_uploader("📄 Upload PDF to selected company", type="pdf", key="uploader")
if uploaded_pdf:
    save_path = os.path.join(company_base_dir, selected_company, uploaded_pdf.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_pdf.getbuffer())
    st.sidebar.success(f"✅ Uploaded: {uploaded_pdf.name}")

# Path handling
VECTORSTORE_ROOT = "/mount/tmp/vectorstores" if is_streamlit_cloud() else "vectorstores"
vectorstore_path = os.path.join(VECTORSTORE_ROOT, selected_company)

# Relearn PDFs
if st.sidebar.button("🔄 Relearn PDFs"):
    try:
        from ingest import ingest_company_pdfs
        
        # Clean up old vectorstore completely
        if os.path.exists(vectorstore_path):
            shutil.rmtree(vectorstore_path, ignore_errors=True)
        
        # Small delay to ensure cleanup
        time.sleep(1)
        
        # Recreate directory
        os.makedirs(vectorstore_path, exist_ok=True)
        
        # Ingest PDFs
        ingest_company_pdfs(selected_company, persist_directory=vectorstore_path)
        
        # Clear any cached vectorstore
        if 'vectorstore' in st.session_state:
            del st.session_state['vectorstore']
            
        st.sidebar.success("✅ Re-ingested knowledge for " + selected_company)
        st.rerun()
        
    except Exception as e:
        st.sidebar.error(f"❌ Error during relearning: {str(e)}")

# View Mode
st.sidebar.markdown("---")
view_mode = st.sidebar.radio("📌 View Mode", ["🔍 Ask Questions", "📊 Dashboard"])

# === Main Area ===
if not os.path.exists(vectorstore_path):
    st.info(f"Upload PDFs for **{selected_company}** and click 'Relearn PDFs' to start.")
else:
    if view_mode == "📊 Dashboard":
        st.subheader("📊 Company Dashboard")

        company_pdf_dir = os.path.join(company_base_dir, selected_company)
        uploaded_pdfs = [f for f in os.listdir(company_pdf_dir) if f.endswith(".pdf")]
        pdf_count = len(uploaded_pdfs)
        db_exists = os.path.exists(vectorstore_path)

        st.markdown(f"- **Company**: `{selected_company}`")
        st.markdown(f"- **PDFs uploaded**: `{pdf_count}`")
        st.markdown(f"- **Vectorstore exists**: `{'Yes ✅' if db_exists else 'No ❌'}`")

        if db_exists:
            total_size = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, dn, filenames in os.walk(vectorstore_path)
                for f in filenames
            )
            size_mb = round(total_size / 1024 / 1024, 2)
            st.markdown(f"- **Vectorstore size**: `{size_mb} MB`")

    else:
        st.markdown("---")
        st.subheader(f"💬 Ask {selected_company} anything about their policies")
        query = st.text_input("Type your question:")

        if query:
            with st.spinner("🔍 Please wait while Broker-gpt searches and thinks..."):
                try:
                    # Use cached vectorstore if available
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
                                "text": f"""Answer the following question using the context below.

Question: {query}

Context: {context}
"""
                            }]
                        }]
                    }

                    headers = {"Content-Type": "application/json"}
                    response = requests.post(url, headers=headers, data=json.dumps(payload))

                    st.markdown("---")
                    st.markdown("### 🤖 Broker-gpt's Answer")
                    if response.status_code == 200:
                        try:
                            answer = response.json()['candidates'][0]['content']['parts'][0]['text']
                            st.success(answer)
                        except:
                            st.error("❌ Gemini replied but parsing failed.")
                    else:
                        st.error(f"❌ Gemini API Error: {response.status_code}")
                        st.json(response.json())
                        
                except Exception as e:
                    st.error(f"❌ Error accessing vectorstore: {str(e)}")
                    st.info("Try clicking 'Relearn PDFs' to rebuild the knowledge base.")
                    # Clear cached vectorstore on error
                    if 'vectorstore' in st.session_state:
                        del st.session_state['vectorstore']
