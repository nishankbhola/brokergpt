import os
import time
import shutil
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# This is a workaround for Streamlit Cloud's environment
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# --- Cached function to load the embedding model ---
# This ensures the large model is loaded into memory only once.
@st.cache_resource
def load_embedding_model():
    """Loads the sentence transformer model only once."""
    print("Loading embedding model for ingestion...")
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def clean_vectorstore_directory(persist_directory):
    """Robustly removes the existing vectorstore directory."""
    if os.path.exists(persist_directory):
        print(f"🧹 Cleaning up old vectorstore directory: {persist_directory}")
        try:
            shutil.rmtree(persist_directory)
            time.sleep(1) # Give a moment for the filesystem to catch up
        except OSError as e:
            print(f"⚠️ Error cleaning directory: {e}. Retrying...")
            # A simple retry can often solve filesystem lock issues
            time.sleep(2)
            shutil.rmtree(persist_directory, ignore_errors=True)
    
    # Ensure the directory exists for the new store
    os.makedirs(persist_directory, exist_ok=True)

def ingest_company_pdfs(company_name: str, persist_directory: str):
    """
    Ingests all PDF documents for a specific company into a persistent Chroma vector store.
    """
    pdf_folder = os.path.join("data/pdfs", company_name)
    print(f"🚀 Starting ingestion for company: {company_name}")
    print(f"📂 PDF source folder: {pdf_folder}")
    print(f"💾 Vectorstore destination: {persist_directory}")

    if not os.path.exists(pdf_folder):
        raise FileNotFoundError(f"PDF source folder not found: {pdf_folder}")
    
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]
    if not pdf_files:
        raise ValueError(f"No PDF files found in {pdf_folder}. Please upload documents first.")

    print(f"📄 Found {len(pdf_files)} PDF files to process.")

    # Load and process all PDFs into document chunks
    all_chunks = []
    for filename in pdf_files:
        file_path = os.path.join(pdf_folder, filename)
        print(f"  -> Processing: {filename}")
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200, 
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_documents(documents)
            all_chunks.extend(chunks)
            print(f"     ✅ Created {len(chunks)} chunks.")
        except Exception as e:
            print(f"     ❌ Error processing {filename}: {e}")
            continue

    if not all_chunks:
        raise ValueError("Failed to create any text chunks from the PDF files.")

    print(f"📊 Total chunks created: {len(all_chunks)}")

    # Get the cached embedding model
    print("🧠 Loading embedding model...")
    embeddings = load_embedding_model()
    print("✅ Embedding model loaded.")

    # Clean the target directory before creating the new vector store
    clean_vectorstore_directory(persist_directory)

    # Create and persist the Chroma vector store
    print("✨ Creating new vector store...")
    try:
        vectordb = Chroma.from_documents(
            documents=all_chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        print("💾 Persisting vector store to disk...")
        vectordb.persist()
        print(f"🎉 Ingestion complete for {company_name}!")
        return vectordb
    except Exception as e:
        print(f"❌❌ Critical error during vector store creation: {e}")
        raise

if __name__ == "__main__":
    # Example of how to run this script directly for testing
    print("Running ingestion script in standalone mode for testing.")
    
    # Create dummy data for testing
    TEST_COMPANY = "test_company"
    TEST_PDF_DIR = os.path.join("data/pdfs", TEST_COMPANY)
    TEST_VS_DIR = os.path.join("vectorstores", TEST_COMPANY)
    
    if not os.path.exists(TEST_PDF_DIR):
        os.makedirs(TEST_PDF_DIR)
        # You would need to place a test PDF file in this directory
        print(f"Created test directory: {TEST_PDF_DIR}")
        print("Please add a PDF to the test directory to run a full test.")
    
    try:
        ingest_company_pdfs(company_name=TEST_COMPANY, persist_directory=TEST_VS_DIR)
    except (ValueError, FileNotFoundError) as e:
        print(f"Could not run test: {e}")

