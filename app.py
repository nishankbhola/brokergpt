import pysqlite3 as sqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import shutil
import json
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

load_dotenv()

st.set_page_config(page_title="🤖 Broker-gpt", layout="wide")
st.title("🤝 Broker-gpt: Insurance Broker Assistant")

# === Sidebar UI ===
st.sidebar.header("🏢 Manage Companies")
company_base_dir = "data/pdfs"
os.makedirs(company_base_dir, exist_ok=True)

# Initialize session state for company refresh
if "company_added" not in st.session_state:
    st.session_state["company_added"] = False

# Add new company
new_company = st.sidebar.text_input("➕ Create new company folder", key="new_company")
if st.sidebar.button("Add Company"):
    new_path = os.path.join(company_base_dir, new_company)
    if new_company and not os.path.exists(new_path):
        os.makedirs(new_path)
        st.sidebar.success(f"✅ Added company: {new_company}")
        st.session_state["company_added"] = True
    else:
        st.sidebar.warning("⚠️ Folder exists or name is empty")

# Refresh company list
company_folders = [f for f in os.listdir(company_base_dir) if os.path.isdir(os.path.join(company_base_dir, f))]
if not company_folders:
    st.warning("⚠️ No companies found. Add one to begin.")
    st.stop()

# Select company
selected_company = st.sidebar.radio("📂 Select company", company_folders, key="selected_company")

# Upload PDF
uploaded_pdf = st.sidebar.file_uploader("📄 Upload PDF to selected company", type="pdf", key="uploader")
if uploaded_pdf:
    save_path = os.path.join(company_base_dir, selected_company, uploaded_pdf.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_pdf.getbuffer())
    st.sidebar.success(f"✅ Uploaded: {uploaded_pdf.name}")

# Relearn PDFs
if st.sidebar.button("🔄 Relearn PDFs"):
    from ingest import ingest_company_pdfs
    shutil.rmtree(os.path.join("vectorstores", selected_company), ignore_errors=True)
    ingest_company_pdfs(selected_company)
    st.sidebar.success("✅ Re-ingested knowledge for " + selected_company)

# === Main Area ===
vectorstore_path = os.path.join("vectorstores", selected_company)
if not os.path.exists(vectorstore_path):
    st.info(f"Upload PDFs for **{selected_company}** and click 'Relearn PDFs' to start.")
else:
    st.markdown("---")
    st.subheader(f"💬 Ask {selected_company} anything about their policies")
    query = st.text_input("Type your question:")

    if query:
        with st.spinner("🔍 Please wait while Broker-gpt searches and thinks..."):
            retriever = Chroma(
                persist_directory=vectorstore_path,
                embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
            ).as_retriever()

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

