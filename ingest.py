import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Helper to detect cloud
def is_streamlit_cloud():
    return os.environ.get("HOME") == "/home/adminuser"

def ingest_company_pdfs(company_name: str, persist_directory: str = None):
    pdf_folder = os.path.join("data/pdfs", company_name)

    # Always use a safe directory if none is passed
    if persist_directory is None:
        base_path = "/mount/tmp/vectorstores" if is_streamlit_cloud() else "vectorstores"
        persist_directory = os.path.join(base_path, company_name)

    print("🗂️ Using vectorstore path:", persist_directory)

    all_chunks = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_folder, filename)
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(pages)
            all_chunks.extend(chunks)

    # Ensure the target dir is writable
    os.makedirs(persist_directory, exist_ok=True)

    # Delete previous DB files if present
    for file in os.listdir(persist_directory):
        path = os.path.join(persist_directory, file)
        try:
            os.remove(path)
        except Exception as e:
            print("⚠️ Could not delete:", path, str(e))

    # Create new vectorstore
    vectordb = Chroma.from_documents(
        documents=all_chunks,
        embedding=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"),
        persist_directory=persist_directory
    )
    vectordb.persist()
    print(f"✅ Ingested {len(all_chunks)} chunks for company: {company_name}")
