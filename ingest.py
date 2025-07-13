import pysqlite3 as sqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from langchain.embeddings import SentenceTransformerEmbeddings
 # --- NEW CODE START ---
    # Ensure the base 'vectorstores' directory exists and has correct permissions
    base_vectorstores_dir = "vectorstores"
    if not os.path.exists(base_vectorstores_dir):
        os.makedirs(base_vectorstores_dir, exist_ok=True)
        # Set permissions to rwx for owner, group, and others
        # This is 0o777 (octal), allowing full access
        os.chmod(base_vectorstores_dir, 0o777)
    else:
        # If it exists, ensure its permissions are correct
        os.chmod(base_vectorstores_dir, 0o777)

    # Ensure the specific company's vectorstore directory also exists and has correct permissions
    if not os.path.exists(vectorstore_path):
        os.makedirs(vectorstore_path, exist_ok=True)
        os.chmod(vectorstore_path, 0o777)
    else:
        # If it exists, ensure its permissions are correct
        os.chmod(vectorstore_path, 0o777)
    # --- NEW CODE END ---
def ingest_company_pdfs(company_name: str):
    pdf_folder = os.path.join("data/pdfs", company_name)
    vectorstore_path = os.path.join("vectorstores", company_name)

    all_chunks = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_folder, filename)
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            chunks = splitter.split_documents(pages)
            all_chunks.extend(chunks)

    os.makedirs(vectorstore_path, exist_ok=True)
    vectordb = Chroma.from_documents(
        documents=all_chunks,
        embedding=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"),
        persist_directory=vectorstore_path
        #persist_directory=None
    )
    vectordb.persist()
    print(f"Ingested {len(all_chunks)} document chunks into vectorstore for {company_name}.")
