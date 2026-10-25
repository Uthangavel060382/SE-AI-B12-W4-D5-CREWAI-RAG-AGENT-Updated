
import os
from pathlib import Path
from typing import Type

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from crewai.tools import BaseTool
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent

load_dotenv(
    BASE_DIR / ".env",
    override=True
)

# ChromaDB configuration
DB_PATH = str(BASE_DIR / "chroma_db")
COLLECTION_NAME = "customer_support"


# Define input schema for the RAG tool
class RAGToolInput(BaseModel):
    query: str = Field(
        ...,
        description="Customer question to search in the knowledge base"
    )


# Custom CrewAI RAG Tool
class CustomerSupportRAGTool(BaseTool):

    name: str = "customer_support_rag_tool"

    description: str = (
        "Search the official company knowledge base "
        "for customer support information about "
        "returns, refunds, shipping, warranties, "
        "and product policies."
    )

    args_schema: Type[BaseModel] = RAGToolInput

    def _run(self, query: str) -> str:

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            return "ERROR: OPENAI_API_KEY is missing."

        embedding_fn = OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name="text-embedding-3-small"
        )

        client = chromadb.PersistentClient(
            path=DB_PATH
        )

        try:
            collection = client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=embedding_fn
            )
        except Exception as e:
            return f"Error accessing ChromaDB: {e}"

        if collection.count() == 0:
            return "Knowledge base is empty."

        results = collection.query(
            query_texts=[query],
            n_results=min(3, collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        output = []

        for document, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            output.append(
                f"Source: {metadata.get('source', 'Unknown')}\n"
                f"Text: {document}\n"
                f"Distance: {distance:.4f}"
            )

        if not output:
            return "No relevant documents found."

        return "\n\n---\n\n".join(output)
