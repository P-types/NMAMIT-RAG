import os
from contextlib import asynccontextmanager

from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from groq import Groq

import graph_rag


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv(".env")


# ============================================================
# CONFIG
# ============================================================

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv(
    "QDRANT_COLLECTION",
    "nmamit_chunks"
)

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
# NEO4J DRIVER CREATION
# ============================================================

def create_neo4j_driver():
    """
    Create a Neo4j driver with connection settings suitable
    for a long-running FastAPI application.
    """

    if not NEO4J_URI:
        raise RuntimeError("NEO4J_URI is missing.")

    if not NEO4J_USERNAME:
        raise RuntimeError("NEO4J_USERNAME is missing.")

    if not NEO4J_PASSWORD:
        raise RuntimeError("NEO4J_PASSWORD is missing.")

    return GraphDatabase.driver(
        NEO4J_URI,
        auth=(
            NEO4J_USERNAME,
            NEO4J_PASSWORD
        ),
        connection_timeout=30,
        connection_acquisition_timeout=30,
        max_connection_lifetime=300,
        keep_alive=True,
    )


# ============================================================
# NEO4J CONNECTION CHECK
# ============================================================

def ensure_neo4j_connection():
    """
    Verify the current Neo4j connection.

    If the existing driver has become invalid, recreate it.
    """

    global driver

    if driver is None:
        print(
            "Neo4j driver does not exist. Creating one...",
            flush=True
        )

        driver = create_neo4j_driver()
        driver.verify_connectivity()

        print(
            "Neo4j connection created.",
            flush=True
        )

        return

    try:

        driver.verify_connectivity()

        print(
            "Neo4j connection verified.",
            flush=True
        )

    except Exception as e:

        print(
            "\nNeo4j connection check failed.",
            flush=True
        )

        print(
            f"Reason: {e}",
            flush=True
        )

        print(
            "Recreating Neo4j driver...",
            flush=True
        )

        try:
            driver.close()
        except Exception:
            pass

        driver = create_neo4j_driver()

        driver.verify_connectivity()

        print(
            "Neo4j connection restored.",
            flush=True
        )


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
    print(
        "STARTING NMAMIT GRAPH RAG API",
        flush=True
    )
    print("=" * 70)

    # ========================================================
    # EMBEDDING MODEL
    # ========================================================

    print(
        "\nLoading embedding model...",
        flush=True
    )

    embedding_model = SentenceTransformer(
        graph_rag.EMBEDDING_MODEL
    )

    print(
        "Embedding model loaded.",
        flush=True
    )

    # ========================================================
    # NEO4J
    # ========================================================

    print(
        "\nConnecting to Neo4j...",
        flush=True
    )

    driver = create_neo4j_driver()

    driver.verify_connectivity()

    print(
        "Neo4j connected.",
        flush=True
    )

    # ========================================================
    # LOAD NEO4J SCHEMA
    # ========================================================

    print(
        "Loading Neo4j schema...",
        flush=True
    )

    schema = graph_rag.load_schema(driver)

    print(
        f"Schema loaded: "
        f"{len(schema['rel_types'])} relationship types, "
        f"{len(schema['entity_types'])} entity types.",
        flush=True
    )

    # ========================================================
    # QDRANT
    # ========================================================

    print(
        "\nConnecting to Qdrant...",
        flush=True
    )

    qdrant = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )

    qdrant.get_collection(
        collection_name=QDRANT_COLLECTION
    )

    print(
        "Qdrant collection ready: "
        + QDRANT_COLLECTION,
        flush=True
    )

    # ========================================================
    # GROQ
    # ========================================================

    print(
        "\nConnecting to Groq...",
        flush=True
    )

    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is missing."
        )

    groq_client = Groq(
        api_key=GROQ_API_KEY
    )

    print(
        "Groq client ready.",
        flush=True
    )

    # ========================================================
    # READY
    # ========================================================

    print(
        "\n" + "=" * 70,
        flush=True
    )

    print(
        "NMAMIT GRAPH RAG API READY",
        flush=True
    )

    print(
        "=" * 70,
        flush=True
    )

    yield

    # ========================================================
    # SHUTDOWN
    # ========================================================

    print(
        "\nShutting down...",
        flush=True
    )

    if driver is not None:

        try:
            driver.close()
        except Exception:
            pass

    if qdrant is not None:

        try:
            qdrant.close()
        except Exception:
            pass

    print(
        "Connections closed.",
        flush=True
    )


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="NMAMIT Hybrid Graph RAG",
    description=(
        "NMAMIT information system using "
        "Neo4j, Qdrant and Groq."
    ),
    version="1.0.0",
    lifespan=lifespan
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "status": "online",
        "service": "NMAMIT Hybrid Graph RAG",
        "message": "API is running"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    neo4j_ok = False

    if driver is not None:

        try:
            driver.verify_connectivity()
            neo4j_ok = True
        except Exception:
            neo4j_ok = False

    return {
        "status": "healthy",
        "neo4j": neo4j_ok,
        "qdrant": qdrant is not None,
        "embedding_model": embedding_model is not None,
        "groq": groq_client is not None
    }


# ============================================================
# ASK ENDPOINT
# ============================================================

@app.post(
    "/ask",
    response_model=AskResponse
)
def ask(request: AskRequest):

    global driver

    # ========================================================
    # VALIDATE QUESTION
    # ========================================================

    if not request.question.strip():

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    # ========================================================
    # CHECK INITIALIZATION
    # ========================================================

    if (
        driver is None
        or qdrant is None
        or embedding_model is None
        or groq_client is None
    ):

        raise HTTPException(
            status_code=503,
            detail=(
                "RAG system is still initializing."
            )
        )

    # ========================================================
    # ENSURE NEO4J CONNECTION
    # ========================================================

    try:

        ensure_neo4j_connection()

    except Exception as e:

        print(
            "\nNEO4J CONNECTION ERROR:",
            flush=True
        )

        print(
            e,
            flush=True
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "Unable to connect to the Neo4j database."
            )
        )

    # ========================================================
    # RUN RAG
    # ========================================================

    try:

        print(
            "\n" + "=" * 70,
            flush=True
        )

        print(
            "PROCESSING QUESTION",
            flush=True
        )

        print(
            f"Question: {request.question}",
            flush=True
        )

        print(
            "=" * 70,
            flush=True
        )

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

    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        print(
            "\n" + "=" * 70,
            flush=True
        )

        print(
            "RAG ERROR",
            flush=True
        )

        print(
            "=" * 70,
            flush=True
        )

        print(
            repr(e),
            flush=True
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "The RAG system could not process "
                "the question."
            )
        )