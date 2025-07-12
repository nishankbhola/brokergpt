import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

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
    )
    vectordb.persist()
    print(f"Ingested {len(all_chunks)} document chunks into vectorstore for {company_name}.")
