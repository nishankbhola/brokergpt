import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

def ingest_company_pdfs(company_name: str, persist_directory: str = None):
    pdf_folder = os.path.join("data/pdfs", company_name)

    # Use /mount/tmp as fallback if no path is provided
    if persist_directory is None:
        persist_directory = os.path.join("/mount/tmp/vectorstores", company_name)

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

    os.makedirs(persist_directory, exist_ok=True)

    vectordb = Chroma.from_documents(
        documents=all_chunks,
        embedding=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"),
        persist_directory=persist_directory
    )
    vectordb.persist()
    print(f"Ingested {len(all_chunks)} chunks into vectorstore for {company_name}.")
