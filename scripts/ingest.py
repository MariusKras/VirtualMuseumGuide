import os, json, glob
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX = os.environ["PINECONE_INDEX"]
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")

DATA_DIR = Path("../data")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=600, chunk_overlap=100
)

docs = []
for path in glob.glob(str(DATA_DIR / "*.json")):
    with open(path, "r", encoding="utf-8") as f:
        js = json.load(f)

    text = (js.get("aprašymas") or "").strip()
    if not text:
        continue

    meta = {k: v for k, v in js.items() if k != "aprašymas"}
    meta["filename"] = Path(path).name
    base_doc = Document(page_content=text, metadata=meta)

    for i, chunk in enumerate(splitter.split_documents([base_doc])):
        chunk.metadata = {
            **chunk.metadata,
            "chunk_text": chunk.page_content,
            "chunk_idx": i,
        }
        docs.append(chunk)

vectorstore = PineconeVectorStore(
    index_name=PINECONE_INDEX,
    embedding=embeddings,
    namespace=PINECONE_NAMESPACE,
)

ids = [f"{d.metadata['filename']}::chunk-{d.metadata['chunk_idx']}" for d in docs]
vectorstore.add_documents(docs, ids=ids)

print(
    f"Upserted {len(docs)} chunks into index '{PINECONE_INDEX}' (ns='{PINECONE_NAMESPACE}')."
)
