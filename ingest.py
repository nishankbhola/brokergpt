import os
import time
import shutil
import sqlite3
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

def force_cleanup_database_files(persist_directory):
    """Forcefully clean up all database files and locks"""
    print(f"🧹 Force cleaning directory: {persist_directory}")
    
    if os.path.exists(persist_directory):
        try:
            # First, try to close any open connections by finding and removing lock files
            for root, dirs, files in os.walk(persist_directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    if file.endswith(('.sqlite3', '.db', '.wal', '.shm', '.lock')):
                        try:
                            os.chmod(file_path, 0o777)  # Make writable
                            os.remove(file_path)
                            print(f"   Removed: {file}")
                        except Exception as e:
                            print(f"   Warning: Could not remove {file}: {e}")
            
            # Now remove the entire directory
            shutil.rmtree(persist_directory, ignore_errors=True)
            time.sleep(1)  # Give filesystem time to clean up
            
        except Exception as e:
            print(f"⚠️ Error during cleanup: {e}")
    
    # Ensure directory exists and is empty
    os.makedirs(persist_directory, exist_ok=True)
    print(f"✅ Directory cleaned and recreated: {persist_directory}")

def create_fresh_vectorstore(chunks, embeddings, persist_directory, max_retries=3):
    """Create a fresh vectorstore with proper error handling"""
    
    for attempt in range(max_retries):
        try:
            print(f"🔄 Creating vectorstore (attempt {attempt + 1}/{max_retries})")
            
            # Force cleanup before each attempt
            force_cleanup_database_files(persist_directory)
            
            # Create vectorstore with explicit settings
            vectordb = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=persist_directory,
                client_settings=None,
                collection_name="documents"  # Explicit collection name
            )
            
            # Test the connection immediately
            print("🔍 Testing vectorstore connection...")
            vectordb._client.heartbeat()
            
            # Try a simple operation
            test_results = vectordb.similarity_search("test", k=1)
            print(f"✅ Vectorstore test successful: {len(test_results)} results")
            
            # Persist the vectorstore
            vectordb.persist()
            print("💾 Vectorstore persisted successfully")
            
            return vectordb
            
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {str(e)}")
            
            if attempt < max_retries - 1:
                print("⏳ Waiting before retry...")
                time.sleep(3)  # Longer wait between retries
            else:
                print("❌ All attempts failed")
                raise Exception(f"Failed to create vectorstore after {max_retries} attempts: {str(e)}")

def ingest_company_pdfs(company_name: str, persist_directory: str = None):
    pdf_folder = os.path.join("data/pdfs", company_name)

    # Always use a safe directory if none is passed
    if persist_directory is None:
        base_path = "/mount/tmp/vectorstores" if is_streamlit_cloud() else "vectorstores"
        persist_directory = os.path.join(base_path, company_name)

    print("🗂️ Using vectorstore path:", persist_directory)

    # Check if PDF folder exists and has PDFs
    if not os.path.exists(pdf_folder):
        raise ValueError(f"PDF folder not found: {pdf_folder}")
    
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    if not pdf_files:
        raise ValueError(f"No PDF files found in: {pdf_folder}")

    print(f"📄 Found {len(pdf_files)} PDF files")

    # Load and process PDFs
    all_chunks = []
    for filename in pdf_files:
        print(f"📖 Processing: {filename}")
        file_path = os.path.join(pdf_folder, filename)
        
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            
            if not pages:
                print(f"⚠️ No pages found in {filename}")
                continue
                
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200,
                length_function=len
            )
            chunks = splitter.split_documents(pages)
            
            if chunks:
                all_chunks.extend(chunks)
                print(f"✅ Added {len(chunks)} chunks from {filename}")
            else:
                print(f"⚠️ No chunks created from {filename}")
                
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            continue

    if not all_chunks:
        raise ValueError("No chunks were created from any PDF files")

    print(f"📊 Total chunks to process: {len(all_chunks)}")

    # Create embeddings
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create vectorstore with improved error handling
    vectordb = create_fresh_vectorstore(all_chunks, embeddings, persist_directory)
    
    print(f"✅ Successfully created vectorstore for {company_name}")
    print(f"📈 Ingested {len(all_chunks)} chunks")
    
    return vectordb

if __name__ == "__main__":
    # Test function
    company_name = "test_company"
    ingest_company_pdfs(company_name)
