"""
alimentar_ia.py - Indexa os PDFs da pasta 'meus_pdfs' no banco vetorial (Chroma).

Rode de novo sempre que adicionar ou remover PDFs. O banco é recriado do zero (evita blocos duplicados).
Feche o server_agro.py antes de rodar (no Windows o arquivo do banco fica travado enquanto o servidor está aberto).
"""
import os
import shutil
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

BASE = Path(__file__).resolve().parent
PASTA_PDFS = BASE / "meus_pdfs"
PASTA_BANCO = BASE / "banco_conhecimento"
# Modelo multilíngue: o anterior (all-MiniLM-L6-v2) é treinado em inglês e recupera mal textos em português.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

PASTA_PDFS.mkdir(exist_ok=True)
pdfs = sorted(PASTA_PDFS.rglob("*.pdf"))
if not pdfs:
    print("Nenhum arquivo PDF encontrado na pasta 'meus_pdfs'.")
    raise SystemExit(1)

documentos = []
for pdf in pdfs:
    try:
        paginas = PyPDFLoader(str(pdf)).load()
    except Exception as e:
        print(f"[ERRO]  Não consegui ler {pdf.name}: {e}")
        continue
    paginas = [p for p in paginas if len(p.page_content.strip()) > 50]
    if not paginas:
        print(f"[AVISO] {pdf.name}: sem texto extraível (PDF escaneado? precisa de OCR). Ignorado.")
        continue
    for p in paginas:
        p.metadata["arquivo"] = pdf.name                       # o servidor cita estes campos nas respostas
        p.metadata["pagina"] = int(p.metadata.get("page", 0)) + 1
    documentos.extend(paginas)
    print(f"[OK]    {pdf.name}: {len(paginas)} página(s) com texto")

if not documentos:
    print("Nenhum PDF com texto utilizável.")
    raise SystemExit(1)

blocos = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250).split_documents(documentos)

if PASTA_BANCO.exists():
    try:
        shutil.rmtree(PASTA_BANCO)
    except PermissionError:
        print("Não consegui apagar o banco antigo. Feche o server_agro.py e rode de novo.")
        raise SystemExit(1)

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, encode_kwargs={"normalize_embeddings": True})
db = Chroma(persist_directory=str(PASTA_BANCO), embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"})
for i in range(0, len(blocos), 500):
    db.add_documents(blocos[i:i + 500])

print(f"Processo concluído. {len(blocos)} blocos de {len(pdfs)} PDF(s) salvos no banco vetorial.")