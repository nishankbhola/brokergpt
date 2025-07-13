import os
import time
import shutil
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

def clean_vectorstore_directory(persist_directory):
    """Clean up vectorstore directory completely"""
    if os.path.exists(persist_directory):
        try:
            shutil.rmtree(persist_directory)
            print(f"🧹 Cleaned up directory: {persist_directory}")
        except Exception as e:
            print(f"⚠️ Error cleaning directory: {e}")
    
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

    # Create vectorstore with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"🔄 Creating vectorstore (attempt {attempt + 1}/{max_retries})")
            
            # Create vectorstore
            vectordb = Chroma.from_documents(
                documents=all_chunks,
                embedding=embeddings,
                persist_directory=persist_directory,
                client_settings=None  # Use default settings
            )
            
            # Test the vectorstore
            vectordb._client.heartbeat()
            
            # Persist the vectorstore
            vectordb.persist()
            
            print(f"✅ Successfully created vectorstore for {company_name}")
            print(f"📈 Ingested {len(all_chunks)} chunks")
            
            # Verify the vectorstore works
            test_results = vectordb.similarity_search("test", k=1)
            print(f"🔍 Vectorstore verification: {len(test_results)} results found")
            
            return vectordb
            
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {e}")
            
            if attempt < max_retries - 1:
                print("⏳ Waiting before retry...")
                time.sleep(2)
                # Clean up partial files
                clean_vectorstore_directory(persist_directory)
            else:
                print("❌ All attempts failed")
                raise e

if __name__ == "__main__":
    # Test function
    company_name = "test_company"
    ingest_company_pdfs(company_name)
