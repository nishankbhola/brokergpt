import os
import sys
# Make sure these two lines are at the very top for pysqlite3
import pysqlite3 as sqlite3
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Updated LangChain imports to use langchain_community
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings


def ingest_company_pdfs(company_name: str):
    pdf_folder = os.path.join("data/pdfs", company_name) 

    # Define the full path to the company-specific vectorstore within the /data persistent storage
    # This will ensure the path is like /data/vectorstores/CompanyName
    vectorstore_path = os.path.join("/data", "vectorstores", company_name)

    print(f"Attempting to ensure vectorstore directory: {vectorstore_path}")
    try:
        # This will create all necessary parent directories (like /data/vectorstores)
        # if they don't already exist. exist_ok=True prevents an error if they do.
        os.makedirs(vectorstore_path, exist_ok=True)
        print(f"Successfully ensured directory exists: {vectorstore_path}")

        # Explicitly set permissions for the final directory to ensure full write access
        os.chmod(vectorstore_path, 0o777)
        print(f"Successfully set permissions for: {vectorstore_path} to 0o777")

    except Exception as e:
        print(f"CRITICAL ERROR: Could not create or set permissions for vectorstore path {vectorstore_path}.")
        print(f"Please check if the Hugging Face Space's /data directory is truly writable for your application.")
        print(f"Original error: {e}")
        raise # Re-raise the error as ingestion cannot proceed without this


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
