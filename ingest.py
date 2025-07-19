import os
import time
import shutil
import sqlite3
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from app import load_embedding_model

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# ADD THIS FUNCTION - This is the missing function causing NameError
def force_garbage_collection():
    """Force garbage collection to free up memory"""
    import gc
    gc.collect()
    print("🧹 Memory cleanup performed")

# Helper to detect cloud
def is_streamlit_cloud():
    return os.environ.get("HOME") == "/home/adminuser"

def clean_vectorstore_directory(persist_directory):
    """Clean up vectorstore directory completely with better error handling"""
    if os.path.exists(persist_directory):
        try:
            # Force close any open database connections
            for root, dirs, files in os.walk(persist_directory):
                for file in files:
                    if file.endswith('.sqlite3') or file.endswith('.db'):
                        db_path = os.path.join(root, file)
                        try:
                            # Try to close any open connections
                            conn = sqlite3.connect(db_path)
                            conn.close()
                        except:
                            pass
            
            # Wait a bit for connections to close
            time.sleep(1)
            
            # Remove the directory
            shutil.rmtree(persist_directory)
            print(f"🧹 Cleaned up directory: {persist_directory}")
        except Exception as e:
            print(f"⚠️ Error cleaning directory: {e}")
            # If we can't remove it, try to remove just the db files
            try:
                for root, dirs, files in os.walk(persist_directory):
                    for file in files:
                        if file.endswith('.sqlite3') or file.endswith('.db'):
                            os.remove(os.path.join(root, file))
                print("🧹 Cleaned up database files")
            except:
                pass
    
    # Ensure directory exists
    os.makedirs(persist_directory, exist_ok=True)

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

    # Clean up old vectorstore completely
    clean_vectorstore_directory(persist_directory)

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
            chunk_size=800, 
            chunk_overlap=100,
            length_function=len
            )
            chunks = splitter.split_documents(pages)
            
            if chunks:
                all_chunks.extend(chunks)
                print(f"✅ Added {len(chunks)} chunks from {filename}")
                
                # Clean up after processing each file
                del pages, chunks
                force_garbage_collection()  # This will now work
            else:
                print(f"⚠️ No chunks created from {filename}")
                
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            continue

    if not all_chunks:
        raise ValueError("No chunks were created from any PDF files")

    print(f"📊 Total chunks to process: {len(all_chunks)}")
    max_chunks = 5000  # Adjust based on your needs
    if len(all_chunks) > max_chunks:
        print(f"⚠️ Limiting chunks from {len(all_chunks)} to {max_chunks} to prevent memory issues")
        all_chunks = all_chunks[:max_chunks]

    # --- MODIFIED: Create embeddings using the cached function ---
    print("🧠 Loading embedding model...")
    embeddings = load_embedding_model()
    print("✅ Embedding model loaded.")

    # Create vectorstore with enhanced retry logic
    max_retries = 5
    for attempt in range(max_retries):
        try:
            print(f"🔄 Creating vectorstore (attempt {attempt + 1}/{max_retries})")
            
            # Extra cleanup on retry attempts
            if attempt > 0:
                clean_vectorstore_directory(persist_directory)
                time.sleep(2)  # Wait longer on retries
            
            # Create vectorstore with explicit settings
            vectordb = Chroma.from_documents(
                documents=all_chunks,
                embedding=embeddings,
                persist_directory=persist_directory,
                client_settings=None  # Use default settings
            )
            
            # Test the vectorstore immediately
            print("🔍 Testing vectorstore connection...")
            vectordb._client.heartbeat()
            
            # Do a quick search test
            test_results = vectordb.similarity_search("test", k=1)
            print(f"🔍 Vectorstore test: {len(test_results)} results found")
            
            # Persist the vectorstore
            print("💾 Persisting vectorstore...")
            vectordb.persist()
            
            # Final verification
            print("✅ Final verification...")
            vectordb._client.heartbeat()
            
            print(f"✅ Successfully created vectorstore for {company_name}")
            print(f"📈 Ingested {len(all_chunks)} chunks")
            
            # Clean up before returning
            del all_chunks
            force_garbage_collection()  # This will now work
            
            return vectordb
            
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {e}")
            
            # Check if it's the specific tenants table error
            if "no such table: tenants" in str(e):
                print("🔧 Detected tenants table error - doing deep cleanup...")
                # Force remove everything and wait longer
                clean_vectorstore_directory(persist_directory)
                time.sleep(3)
                force_garbage_collection()  # This will now work
            
            if attempt < max_retries - 1:
                print(f"⏳ Waiting {2 * (attempt + 1)} seconds before retry...")
                time.sleep(2 * (attempt + 1))  # Exponential backoff
            else:
                print("❌ All attempts failed")
                raise e

if __name__ == "__main__":
    # Test function
    company_name = "test_company"
    ingest_company_pdfs(company_name)
