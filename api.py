import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from groq import Groq

import graph_rag

QDRANT_COLLECTION = graph_rag.QDRANT_COLLECTION

load_dotenv()


# ============================================================
# CONFIG
# ============================================================

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "nmamit_chunks")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ============================================================
# GLOBAL RAG OBJECTS
# ============================================================

driver = None
qdrant = None
embedding_model = None
groq_client = None
schema = None


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    question: str
    answer: str


# ============================================================
# STARTUP
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global driver
    global qdrant
    global embedding_model
    global groq_client
    global schema

    print("=" * 70)
    print("STARTING NMAMIT GRAPH RAG API")
    print("=" * 70)

    # ---------------- EMBEDDING MODEL ----------------

    print("\nLoading embedding model...")

    embedding_model = SentenceTransformer(
        graph_rag.EMBEDDING_MODEL
    )

    print("Embedding model loaded.")

    # ---------------- NEO4J ----------------

    print("Connecting to Neo4j...")

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )

    driver.verify_connectivity()

    print("Neo4j connected.")

    schema = graph_rag.load_schema(driver)

    print(
        f"Schema loaded: "
        f"{len(schema['rel_types'])} relationship types, "
        f"{len(schema['entity_types'])} entity types."
    )

    # ---------------- QDRANT ----------------

    print("Connecting to Qdrant...")

    qdrant = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )

    qdrant.get_collection(
        collection_name=QDRANT_COLLECTION
    )

    print(
        "Qdrant collection ready: "
        + QDRANT_COLLECTION
    )

    # ---------------- GROQ ----------------

    print("Connecting to Groq...")

    groq_client = Groq(
        api_key=GROQ_API_KEY
    )

    print("Groq client ready.")

    print("\n" + "=" * 70)
    print("NMAMIT GRAPH RAG API READY")
    print("=" * 70)

    yield

    # ---------------- SHUTDOWN ----------------

    print("\nShutting down...")

    if driver:
        driver.close()

    if qdrant:
        qdrant.close()

    print("Connections closed.")


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="NMAMIT Hybrid Graph RAG",
    description="NMAMIT information system using Neo4j, Qdrant and Groq.",
    version="1.0.0",
    lifespan=lifespan
)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():

    return {
        "status": "online",
        "service": "NMAMIT Hybrid Graph RAG",
        "message": "API is running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "neo4j": driver is not None,
        "qdrant": qdrant is not None,
        "embedding_model": embedding_model is not None,
        "groq": groq_client is not None
    }


# ============================================================
# ASK ENDPOINT
# ============================================================

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):

    if not request.question.strip():

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    if (
        driver is None
        or qdrant is None
        or embedding_model is None
        or groq_client is None
    ):

        raise HTTPException(
            status_code=503,
            detail="RAG system is still initializing."
        )

    try:

        # We want the existing RAG pipeline to return
        # the answer without changing its internal logic.

        answer = graph_rag.retrieve_and_answer(
            driver=driver,
            qdrant=qdrant,
            embedding_model=embedding_model,
            groq_client=groq_client,
            question=request.question,
            schema=schema
        )

        return AskResponse(
            question=request.question,
            answer=answer
        )

    except Exception as e:

        print("\nRAG ERROR:")
        print(e)

        raise HTTPException(
            status_code=500,
            detail="The RAG system could not process the question."
        )
