import os
import sys
# Make sure these two lines are at the very top for pysqlite3
import pysqlite3 as sqlite3
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Updated LangChain imports to use langchain_community
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter # This is still in 'langchain'
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings


def ingest_company_pdfs(company_name: str):
    pdf_folder = os.path.join("data/pdfs", company_name)
    vectorstore_path = os.path.join("vectorstores", company_name)

    # --- Permissions and directory creation for the base 'vectorstores' folder ---
    base_vectorstores_dir = "vectorstores"
    if not os.path.exists(base_vectorstores_dir):
        os.makedirs(base_vectorstores_dir, exist_ok=True)
    # Ensure the base 'vectorstores' directory is writable
    os.chmod(base_vectorstores_dir, 0o777) # Set permissions to rwx for owner, group, others

    # --- Permissions and directory creation for the company-specific vectorstore folder ---
    # This creates the specific company's directory if it doesn't exist
    os.makedirs(vectorstore_path, exist_ok=True)
    # Ensure the specific company's directory is writable
    os.chmod(vectorstore_path, 0o777) # Set permissions to rwx for owner, group, others
    # --- END Permissions and Directory Creation ---

    all_chunks = []
    # Loop through PDFs in the specific company's folder
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_folder, filename)
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            
            # Split documents into chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200
            )
            chunks = splitter.split_documents(pages)
            all_chunks.extend(chunks)

    # Create the Chroma vector store from documents
    vectordb = Chroma.from_documents(
        documents=all_chunks,
        embedding=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"),
        persist_directory=vectorstore_path # This tells Chroma where to save the data
    )
    
    # Explicitly persist the vector database.
    # While from_documents with persist_directory often auto-persists,
    # explicitly calling it ensures the save operation completes.
    vectordb.persist() 
    
    print(f"Ingested {len(all_chunks)} document chunks into vectorstore for {company_name}.")
