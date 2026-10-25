
import os
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
from pypdf import PdfReader

# Project configuration
BASE_DIR = Path(__file__).resolve().parent

load_dotenv(
    BASE_DIR / ".env",
    override=True
)

KNOWLEDGE_DIR = BASE_DIR / "knowledge_base"
DB_PATH = str(BASE_DIR / "chroma_db")
COLLECTION_NAME = "customer_support"


def load_document(path):
    """Read PDF, TXT, and Markdown files."""

    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))

        return "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

    if path.suffix.lower() in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")

    return ""


def chunk_text(text, chunk_size=1000, overlap=150):
    """Split documents into overlapping chunks."""

    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size].strip()

        if chunk:
            chunks.append(chunk)

    return chunks


def main():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing from .env"
        )

    # Initialize OpenAI embedding model
    embedding_fn = OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small"
    )

    # Connect to local ChromaDB
    client = chromadb.PersistentClient(
        path=DB_PATH
    )

    # Check for existing collection
    existing_collections = [
        collection.name
        for collection in client.list_collections()
    ]

    if COLLECTION_NAME in existing_collections:
        client.delete_collection(COLLECTION_NAME)
        print("Existing collection deleted.")

    # Create collection with cosine similarity
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        configuration={
            "hnsw": {
                "space": "cosine"
            }
        }
    )

    ids = []
    documents = []
    metadatas = []

    # Load documents
    for path in sorted(KNOWLEDGE_DIR.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in {
            ".txt", ".md", ".pdf"
        }:
            continue

        text = load_document(path)

        chunks = chunk_text(text)

        for index, chunk in enumerate(chunks):
            ids.append(f"chunk-{len(ids)}")
            documents.append(chunk)

            metadatas.append({
                "source": str(path.relative_to(BASE_DIR)),
                "chunk": index
            })

    if not documents:
        print("No readable document chunks found.")
        return

    # Store embeddings in batches
    batch_size = 100

    for start in range(0, len(documents), batch_size):
        end = start + batch_size

        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end]
        )

    print(
        f"Successfully indexed {len(documents)} chunks."
    )

    print(f"ChromaDB saved at: {DB_PATH}")


if __name__ == "__main__":
    main()
