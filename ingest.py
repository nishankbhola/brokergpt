import os
import sys
import pysqlite3 as sqlite3
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings


def ingest_company_pdfs(company_name: str):
    pdf_folder = os.path.join("data/pdfs", company_name) # Keep this as is
    
    # --- CHANGE START ---
    # Define the base directory for vectorstores within the /data persistent storage
    base_vectorstores_dir = "/data/vectorstores" 
    # Define the path for the company-specific vectorstore
    vectorstore_path = os.path.join(base_vectorstores_dir, company_name)
    # --- CHANGE END ---

    # Ensure the base /data/vectorstores directory exists and has correct permissions
    if not os.path.exists(base_vectorstores_dir):
        os.makedirs(base_vectorstores_dir, exist_ok=True)
    os.chmod(base_vectorstores_dir, 0o777) # Set permissions for full access

    # Ensure the company-specific vectorstore directory within /data/vectorstores exists
    os.makedirs(vectorstore_path, exist_ok=True)
    os.chmod(vectorstore_path, 0o777) # Set permissions for full access

    all_chunks = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_folder, filename)
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200
            )
            chunks = splitter.split_documents(pages)
            all_chunks.extend(chunks)

    vectordb = Chroma.from_documents(
        documents=all_chunks,
        embedding=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"),
        persist_directory=vectorstore_path 
    )
    
    vectordb.persist() 
    
    print(f"Ingested {len(all_chunks)} document chunks into vectorstore for {company_name}.")
