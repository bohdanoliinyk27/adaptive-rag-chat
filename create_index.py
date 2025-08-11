import os
import uuid
from typing import List
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone_client import index

load_dotenv()

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

def load_and_split(path: str) -> List[dict]:
    if path.lower().endswith(".pdf"):
        loader = PyPDFLoader(path)
    else:
        loader = TextLoader(path, encoding="utf-8")
    return loader.load_and_split(text_splitter=splitter)

def ingest_file(path: str, namespace: str) -> str:
    docs = load_and_split(path)
    records = [
        {"_id": str(uuid.uuid4()), "chunk_text": doc.page_content}
        for doc in docs
    ]
    index.upsert_records(namespace, records)
    stats = index.describe_index_stats(namespaces=[namespace])
    total = stats.get("namespaces", {}).get(namespace, {}).get("vectorCount", 0)
    return (
        f"Ingested {len(records)} chunks from {os.path.basename(path)} "
        f"in namespace '{namespace}'; total vectors now {total}"
    )
