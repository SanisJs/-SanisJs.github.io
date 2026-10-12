import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

os.makedirs("meus_pdfs", exist_ok=True)
loader = PyPDFDirectoryLoader("meus_pdfs")
documentos = loader.load()

if not documentos:
    print("Nenhum arquivo PDF encontrado na pasta 'meus_pdfs'.")
    exit()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250)
blocos = text_splitter.split_documents(documentos)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma.from_documents(blocos, embeddings, persist_directory="./banco_conhecimento")

print(f"Processo concluido. {len(blocos)} blocos de conhecimento salvos no banco vetorial.")