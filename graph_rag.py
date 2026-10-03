"""
NMAMIT Hybrid Graph RAG  (Neo4j AuraDB + Qdrant Cloud + Groq)

Intent-aware retrieval version.

Pipeline:
  question
    -> query understanding (Groq: intent, entities, query_type, relation types)
    -> entity resolution (normalised scoring + controlled aliases, Neo4j)
    -> intent-aware graph retrieval (relation filtering, hop limits, scoring)
       or explicit path search for relationship questions
    -> Qdrant retrieval with the analysed query + reranking + de-duplication
    -> structured evidence context (strong vs supporting)
    -> grounded answer (Groq) + programmatic sources
"""

import os
import re
import json
import argparse
import difflib
from collections import defaultdict, OrderedDict

import numpy as np
from dotenv import load_dotenv
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
from groq import Groq

class FastEmbedAdapter:
    """
    Adapter that makes FastEmbed behave like the
    SentenceTransformer interface used by this RAG pipeline.
    """

    def __init__(self, model_name):
        self.model = TextEmbedding(model_name=model_name)

    def encode(self, texts, normalize_embeddings=True):
        vectors = np.asarray(
            list(self.model.embed(texts)),
            dtype=np.float32
        )

        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.clip(norms, 1e-12, None)

        return vectors


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "nmamit_chunks")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "openai/gpt-oss-20b"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# RAG_DEBUG=0 hides retrieval details, RAG_SHOW_CONTEXT=1 prints the exact
# context sent to Groq.
DEBUG = os.getenv("RAG_DEBUG", "1") != "0"
SHOW_CONTEXT = os.getenv("RAG_SHOW_CONTEXT", "0") == "1"


# ============================================================
# RETRIEVAL SETTINGS
# ============================================================

MODE_LIMITS = {
    "normal": {
        "seeds_per_entity": 2,
        "max_edges": 18,
        "per_type_cap": 6,
        "fetch_per_hop": 80,
        "vector_fetch": 24,
        "vector_keep": 5,
        "per_page": 2,
    },
    "exhaustive": {
        "seeds_per_entity": 2,
        "max_edges": 150,
        "per_type_cap": 150,
        "fetch_per_hop": 400,
        "vector_fetch": 40,
        "vector_keep": 10,
        "per_page": 3,
    },
}

MAX_CHUNK_CHARS = 3500          # stored/scored chunk size
MAX_CONTEXT_CHUNK_CHARS = 2200  # size sent to Groq per chunk
MIN_ENTITY_SCORE = 40
MIN_VECTOR_SCORE = 0.20         # below this a chunk needs strong other signals
MAX_PATH_HOPS = 3
DEFAULT_ENTITY_TERM = "NMAMIT"  # used when the question names no entity

INSUFFICIENT_MESSAGE = (
    "I couldn't find enough information in the NMAMIT knowledge base "
    "to answer that reliably."
)


# ============================================================
# CONTROLLED ALIASES
# ============================================================
# Names inside one group are treated as possible references to the same
# institution, but ONLY as search/scoring hints: an alias can lift an
# entity that actually exists in Neo4j, it never creates one, and other
# entities (NMAMIT DSC, NMAMIT Central Library, ...) are never merged.
# Edit this list only with names you have verified in your graph.

ALIAS_GROUPS = [
    [
        "NMAMIT",
        "NMAM Institute of Technology",
        "NMAM Institute",
        "NMAMIT Nitte",
    ],
]


# ============================================================
# TEXT HELPERS
# ============================================================

STOPWORDS = set("""
a an the is are was were be been of in on at to for from by with and or
what which who whom where when why how does do did has have had tell me
about give list all any there their it its this that these those can
could would should please are available
""".split())

GENERIC_TOKENS = {
    "university", "institute", "technology", "college", "department",
    "school", "centre", "center", "engineering", "program", "programme",
}


def normalize_text(text):
    """Lowercase, strip punctuation/hyphens, collapse whitespace."""
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def content_tokens(text):
    return [
        t for t in normalize_text(text).split()
        if len(t) >= 3 and t not in STOPWORDS
    ]


_ALIAS_GROUPS_NORM = [
    {normalize_text(name) for name in group}
    for group in ALIAS_GROUPS
]


def alias_names_for(term_norm):
    for group in _ALIAS_GROUPS_NORM:
        if term_norm in group:
            return group - {term_norm}
    return set()


_EMBED_CACHE = {}


def embed_texts(model, texts):
    """Batch-encode with a cache so the same text is never embedded twice."""
    missing = [t for t in dict.fromkeys(texts) if t not in _EMBED_CACHE]
    if missing:
        vectors = model.encode(missing, normalize_embeddings=True)
        for text, vec in zip(missing, vectors):
            _EMBED_CACHE[text] = np.asarray(vec, dtype=float)
    return [_EMBED_CACHE[t] for t in texts]


# ============================================================
# CONFIGURATION CHECK
# ============================================================

def check_config():

    required = {
        "NEO4J_URI": NEO4J_URI,
        "NEO4J_USERNAME": NEO4J_USERNAME,
        "NEO4J_PASSWORD": NEO4J_PASSWORD,
        "NEO4J_DATABASE": NEO4J_DATABASE,
        "QDRANT_URL": QDRANT_URL,
        "QDRANT_API_KEY": QDRANT_API_KEY,
        "GROQ_API_KEY": GROQ_API_KEY,
    }

    missing = [key for key, value in required.items() if not value]

    if missing:
        raise RuntimeError(
            "Missing environment variables: " + ", ".join(missing)
        )


# ============================================================
# SCHEMA DISCOVERY
# ============================================================
# Relationship types and entity types are READ from the live database so
# that nothing in retrieval assumes a type that does not exist.

def load_schema(driver):

    schema = {"rel_types": [], "entity_types": []}

    with driver.session(database=NEO4J_DATABASE) as session:

        try:
            schema["rel_types"] = sorted(
                r["relationshipType"]
                for r in session.run(
                    "CALL db.relationshipTypes() "
                    "YIELD relationshipType "
                    "RETURN relationshipType"
                )
            )
        except Exception as e:
            print(f"Schema warning (relationship types): {e}")

        try:
            schema["entity_types"] = sorted(
                r["t"]
                for r in session.run(
                    "MATCH (n:Entity) WHERE n.type IS NOT NULL "
                    "RETURN DISTINCT n.type AS t LIMIT 200"
                )
                if r["t"]
            )
        except Exception as e:
            print(f"Schema warning (entity types): {e}")

    return schema


# ============================================================
# QUERY TYPES AND PROFILES
# ============================================================

QUERY_TYPES = [
    "location", "relationship", "programs", "departments", "faculty",
    "facilities", "admissions", "fees", "placements", "research",
    "clubs", "events", "general_information", "comparison",
]

# Substring patterns used to map a query type onto relationship types that
# ACTUALLY exist in the database. Keep them tight: they only select among
# existing types, they never add new ones.
REL_KEYWORDS = {
    "location": ["LOCAT", "BASED", "SITUAT", "ADDRESS"],
    "relationship": [
        "AFFILIAT", "PART_OF", "BELONG", "ASSOCIAT", "CONNECT",
        "RELATED", "COLLABORAT", "UNDER", "MANAGED", "FOUNDED", "TRUST",
    ],
    "programs": ["PROGRAM", "OFFER", "COURSE", "DEGREE"],
    "departments": ["DEPARTMENT", "HAS_DEPT"],
    "faculty": ["FACULTY", "WORKS_AT", "TEACH", "HEAD_OF", "HOD", "PROFESSOR"],
    "facilities": ["FACILIT", "HAS_LAB", "LIBRARY", "HOSTEL"],
    "admissions": ["ADMISSION", "ELIGIB", "ENTRANCE"],
    "fees": ["FEE", "SCHOLARSHIP"],
    "placements": ["PLACE", "RECRUIT", "HIRED"],
    "research": ["RESEARCH", "PUBLICATION", "PATENT", "PROJECT", "CENTRE", "CENTER"],
    "clubs": ["CLUB", "CHAPTER", "SOCIETY"],
    "events": ["EVENT", "ORGANIZ", "ORGANIS", "HOST", "FEST"],
    "general_information": [],
    "comparison": [],
}

# direction: "out" follows edges leaving the seed, "both" also takes edges
# entering it. Deeper hops (hop > 1) always follow outgoing edges only, so
# a traversal can never wander into other institutions' neighbourhoods.
# neighbor_hints: soft entity-type hints (bonus only, never a hard filter).
QUERY_PROFILES = {
    "location": {"direction": "out", "hops": 3,
                 "neighbor_hints": ["place", "city", "location", "state", "country", "district", "region"]},
    "relationship": {"direction": "both", "hops": 1, "neighbor_hints": []},
    "programs": {"direction": "both", "hops": 1,
                 "neighbor_hints": ["program", "course", "degree"]},
    "departments": {"direction": "both", "hops": 1,
                    "neighbor_hints": ["department"]},
    "faculty": {"direction": "both", "hops": 1,
                "neighbor_hints": ["person", "faculty", "professor"]},
    "facilities": {"direction": "both", "hops": 1,
                   "neighbor_hints": ["facility", "lab", "library"]},
    "admissions": {"direction": "both", "hops": 1, "neighbor_hints": []},
    "fees": {"direction": "both", "hops": 1, "neighbor_hints": []},
    "placements": {"direction": "both", "hops": 1,
                   "neighbor_hints": ["company", "organization"]},
    "research": {"direction": "both", "hops": 1,
                 "neighbor_hints": ["research", "project", "centre", "center"]},
    "clubs": {"direction": "both", "hops": 1,
              "neighbor_hints": ["club", "organization", "society"]},
    "events": {"direction": "both", "hops": 1, "neighbor_hints": ["event"]},
    "general_information": {"direction": "both", "hops": 1, "neighbor_hints": []},
    "comparison": {"direction": "both", "hops": 1, "neighbor_hints": []},
}

# Generic topic vocabulary used ONLY to rerank document chunks.
INTENT_LEXICON = {
    "location": ["located", "location", "address", "situated", "campus", "km",
                 "distance", "airport", "railway", "station", "district",
                 "state", "country", "road", "reach"],
    "relationship": ["affiliated", "affiliation", "part", "constituent",
                     "trust", "managed", "under", "associated", "deemed"],
    "programs": ["program", "programme", "programs", "course", "courses",
                 "degree", "btech", "mtech", "mca", "mba", "phd", "bachelor",
                 "master", "intake"],
    "departments": ["department", "departments", "head", "hod", "branch"],
    "faculty": ["faculty", "professor", "associate", "assistant", "hod",
                "staff", "teaching", "qualification"],
    "facilities": ["facility", "facilities", "library", "hostel", "lab",
                   "laboratory", "sports", "canteen", "infrastructure",
                   "wifi", "auditorium"],
    "admissions": ["admission", "admissions", "eligibility", "apply",
                   "entrance", "cet", "comedk", "cutoff", "counselling"],
    "fees": ["fee", "fees", "tuition", "scholarship", "payment"],
    "placements": ["placement", "placements", "recruiters", "package",
                   "companies", "internship", "offers"],
    "research": ["research", "publication", "publications", "patent",
                 "project", "centre", "center", "funded", "lab"],
    "clubs": ["club", "clubs", "chapter", "society", "student", "activities"],
    "events": ["event", "events", "fest", "workshop", "seminar", "hackathon",
               "conference"],
    "general_information": ["established", "founded", "institute", "college",
                            "about", "autonomous", "accredited", "naac"],
    "comparison": [],
}


def classify_query_type(question):
    """Heuristic fallback used only when the Groq analyser fails."""
    q = question.lower()
    rules = [
        ("relationship", ["related to", "relationship", "relation between",
                          "connected to", "affiliat", "associated with",
                          "link between"]),
        ("comparison", [" vs ", "versus", "compare", "difference between"]),
        ("location", ["where is", "located", "location", "address", "nearest",
                      "how far", "how to reach", "distance"]),
        ("fees", ["fee", "tuition", "scholarship"]),
        ("placements", ["placement", "recruit", "package", "salary"]),
        ("admissions", ["admission", "eligib", "apply", "cutoff", "cet",
                        "comedk", "entrance"]),
        ("programs", ["program", "course", "branch", "degree", "btech",
                      "mtech", "mba", "mca"]),
        ("departments", ["department"]),
        ("faculty", ["faculty", "professor", "teacher", "hod", "staff"]),
        ("facilities", ["facilit", "hostel", "library", "sports",
                        "canteen", "infrastructure"]),
        ("research", ["research", "publication", "patent"]),
        ("clubs", ["club", "chapter", "society"]),
        ("events", ["event", "fest", "hackathon", "workshop", "seminar"]),
    ]
    for qtype, needles in rules:
        if any(n in q for n in needles):
            return qtype
    return "general_information"


def is_exhaustive_query(question):

    q = question.lower().strip()

    phrases = [
        "list all", "all the", "all programs", "all program",
        "all programme", "all programmes", "all courses", "all branches",
        "all departments", "all facilities", "all companies",
        "every program", "every programme", "every course",
        "every branch", "every department", "what programs",
        "what programme", "what programmes", "what courses",
        "what branches", "what departments", "which programs",
        "which programmes", "which branches", "which departments",
        "what facilities", "what clubs", "which clubs", "who are the",
        "what does nmamit offer", "what does nmamit have",
    ]

    if any(p in q for p in phrases):
        return True

    return bool(re.match(
        r"^(what|which)\s+(\w+\s+){0,3}(are|is)\s+(there|available)", q
    ))


# ============================================================
# STEP 1  QUERY UNDERSTANDING
# ============================================================

def _extract_json(text):
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object in model output")
    return json.loads(text[start:end + 1])


def resolve_relation_types(query_type, llm_types, actual_types):
    """
    Map the analyser's relation hints and the query type onto relationship
    types that really exist in Neo4j. Anything that does not exist is dropped.
    """
    actual = {t.upper(): t for t in actual_types}
    chosen = []

    for hinted in llm_types or []:
        h = str(hinted).strip().upper()
        if not h:
            continue
        if h in actual:
            chosen.append(actual[h])
            continue
        # tolerate near-misses such as HAS_PROGRAMS vs HAS_PROGRAM
        for up, real in actual.items():
            if len(h) >= 5 and (h in up or up in h):
                chosen.append(real)

    keywords = REL_KEYWORDS.get(query_type, [])
    for up, real in actual.items():
        if any(k in up for k in keywords):
            chosen.append(real)

    return list(dict.fromkeys(chosen))


def analyze_query(groq_client, question, schema):

    rel_types = schema.get("rel_types", [])
    shown_types = ", ".join(rel_types[:120]) if rel_types else "(unknown)"

    prompt = f"""
Analyze this user question for a Graph RAG system about NMAMIT
(NMAM Institute of Technology, Nitte).

USER QUESTION:
{question}

Relationship types that exist in the knowledge graph:
{shown_types}

Return ONLY valid JSON with exactly this structure:

{{
  "intent": "short description of what the user wants",
  "entities": ["important named entity 1", "important named entity 2"],
  "search_query": "concise expanded search query with the key concepts",
  "query_type": "one of: {', '.join(QUERY_TYPES)}",
  "is_list": true or false,
  "target_relation_types": ["relationship types from the list above that answer this question"]
}}

Rules:
1. Identify the actual intent and the kind of information requested.
2. entities: named things the user mentions, spelled as the user wrote them.
   Do not invent entities. Use [] if the question names none.
3. query_type "relationship" is for questions about how two things are
   connected; put BOTH entities in "entities", the first mentioned first.
4. is_list is true when the user wants a complete list of items.
5. target_relation_types must be copied from the list above, only those
   that would actually answer the question. Use [] if none fit. Never invent.
6. search_query must be useful for semantic search over documents.
7. Do not answer the question.
"""

    fallback_type = classify_query_type(question)
    fallback = {
        "intent": "general information",
        "entities": [],
        "search_query": question,
        "query_type": fallback_type,
        "is_list": False,
        "target_relation_types": [],
    }

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system",
                 "content": "You are a query analyzer for a Graph RAG system."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        result = _extract_json(response.choices[0].message.content or "")

    except Exception as e:
        print(f"\nQuery analysis warning: {e}")
        return fallback

    query_type = str(result.get("query_type", "")).strip().lower()
    if query_type in ("list", "exhaustive", "list/exhaustive"):
        query_type, result["is_list"] = fallback_type, True
    if query_type not in QUERY_TYPES:
        query_type = fallback_type

    entities = [
        str(e).strip() for e in (result.get("entities") or [])
        if isinstance(e, str) and e.strip()
    ]

    llm_types = result.get("target_relation_types") or []
    if not isinstance(llm_types, list):
        llm_types = []

    return {
        "intent": str(result.get("intent") or "general information"),
        "entities": entities,
        "search_query": str(result.get("search_query") or question),
        "query_type": query_type,
        "is_list": bool(result.get("is_list", False)),
        "target_relation_types": resolve_relation_types(
            query_type, llm_types, rel_types
        ),
    }


# ============================================================
# STEP 2  ENTITY RESOLUTION
# ============================================================

def score_entity_name(name, raw_term, term_norm, term_tokens, alias_norms):
    """
    exact > normalised exact > alias > all query tokens present (penalised
    for extra tokens) > entity tokens inside the query > fuzzy > substring.
    NMAMIT, NMAMIT DSC and NMAMIT Central Library therefore stay distinct
    and rank in that order for the term "NMAMIT".
    """
    n = normalize_text(name)
    if not n or not term_norm:
        return 0

    nt = n.split()

    if name.strip().lower() == raw_term.strip().lower():
        return 100
    if n == term_norm:
        return 95
    if n in alias_norms:
        return 88

    if term_tokens and set(term_tokens) <= set(nt):
        extra = len(nt) - len(term_tokens)
        return max(45, 75 - 6 * extra)

    if len(nt) >= 2 and set(nt) <= set(term_tokens):
        return 55

    if difflib.SequenceMatcher(None, n, term_norm).ratio() >= 0.88:
        return 50

    if term_norm in n or (len(n) >= 4 and n in term_norm):
        return 10

    return 0


def fetch_entity_candidates(session, contains_terms, exact_terms, limit=400):

    query = """
    MATCH (n:Entity)
    WHERE n.name IS NOT NULL
      AND any(t IN $contains WHERE toLower(n.name) CONTAINS t)
    RETURN
        n.id AS id,
        n.name AS name,
        n.type AS type,
        CASE WHEN toLower(n.name) IN $exact THEN 0 ELSE 1 END AS rank
    ORDER BY rank, size(n.name)
    LIMIT $limit
    """

    return [
        record.data()
        for record in session.run(
            query,
            contains=contains_terms,
            exact=exact_terms,
            limit=limit,
        )
    ]


def resolve_term(session, model, raw_term, seeds_per_entity):

    term_norm = normalize_text(raw_term)
    if not term_norm:
        return []

    term_tokens = term_norm.split()
    alias_norms = alias_names_for(term_norm)

    phrases = {raw_term.lower().strip(), term_norm} | alias_norms
    distinctive = {
        t for t in term_tokens
        if len(t) >= 4 and t not in STOPWORDS and t not in GENERIC_TOKENS
    }

    candidates = fetch_entity_candidates(
        session,
        contains_terms=sorted(phrases | distinctive),
        exact_terms=sorted(phrases),
    )

    scored = []
    for c in candidates:
        if c.get("id") is None:
            continue
        s = score_entity_name(
            c["name"], raw_term, term_norm, term_tokens, alias_norms
        )
        if s > 0:
            scored.append([float(s), c])

    scored.sort(key=lambda x: (-x[0], len(x[1]["name"])))
    top = scored[:25]

    # Light semantic tie-break, skipped when we already have an exact hit.
    if top and top[0][0] < 95 and model is not None:
        vecs = embed_texts(
            model,
            [term_norm] + [normalize_text(c["name"]) for _, c in top],
        )
        for i, item in enumerate(top):
            sim = float(np.dot(vecs[0], vecs[i + 1]))
            item[0] += max(0.0, sim - 0.5) * 20

        top.sort(key=lambda x: (-x[0], len(x[1]["name"])))

    if not top:
        return []

    cutoff = max(MIN_ENTITY_SCORE, top[0][0] - 12)

    return [
        {
            "id": c["id"],
            "name": c["name"],
            "type": c.get("type") or "Unknown",
            "score": round(s, 1),
            "matched_term": raw_term,
        }
        for s, c in top
        if s >= cutoff
    ][:seeds_per_entity]


def fallback_terms(question):
    return [t for t in content_tokens(question) if t not in GENERIC_TOKENS][:4]


def resolve_entities(session, model, analyzed_entities, question, cfg):
    """Returns OrderedDict: input term -> list of seed entities."""

    terms = list(dict.fromkeys(analyzed_entities)) or fallback_terms(question)

    by_term = OrderedDict()
    seen = set()

    for term in terms:
        seeds = [
            s for s in resolve_term(
                session, model, term, cfg["seeds_per_entity"]
            )
            if s["id"] not in seen
        ]
        if seeds:
            by_term[term] = seeds
            seen.update(s["id"] for s in seeds)

    if not by_term:
        seeds = resolve_term(
            session, model, DEFAULT_ENTITY_TERM, cfg["seeds_per_entity"]
        )
        if seeds:
            by_term[DEFAULT_ENTITY_TERM] = seeds

    return by_term


# ============================================================
# STEP 3  INTENT-AWARE GRAPH TRAVERSAL
# ============================================================

def fetch_edges(session, ids, types, direction, limit):

    if direction == "out":
        cond = "a.id IN $ids"
    elif direction == "in":
        cond = "b.id IN $ids"
    else:
        cond = "(a.id IN $ids OR b.id IN $ids)"

    if types:
        cond += " AND type(r) IN $types"

    query = f"""
    MATCH (a:Entity)-[r]->(b:Entity)
    WHERE {cond}
    RETURN
        a.id AS source_id,
        a.name AS source_name,
        a.type AS source_type,
        type(r) AS relation,
        b.id AS target_id,
        b.name AS target_name,
        b.type AS target_type
    LIMIT $limit
    """

    return [
        record.data()
        for record in session.run(
            query, ids=list(ids), types=list(types), limit=limit
        )
    ]


def expand_hops(session, seed_ids, types, hops, direction, fetch_limit):
    """
    Hop-by-hop bounded expansion (one small query per hop) instead of one
    unbounded variable-length Cypher pattern. Hop 1 uses `direction`; later
    hops only follow outgoing edges.
    """
    collected = []
    seen_edges = set()
    visited = set(seed_ids)
    frontier = list(seed_ids)

    for hop in range(1, hops + 1):

        if not frontier:
            break

        edges = fetch_edges(
            session,
            frontier,
            types,
            direction if hop == 1 else "out",
            fetch_limit,
        )

        next_frontier = []

        for e in edges:
            if e["source_id"] == e["target_id"]:
                continue
            if not e.get("source_name") or not e.get("target_name"):
                continue

            key = (e["source_id"], e["relation"], e["target_id"])
            if key in seen_edges:
                continue
            seen_edges.add(key)

            e["hop"] = hop
            collected.append(e)

            for endpoint in (e["source_id"], e["target_id"]):
                if endpoint not in visited:
                    visited.add(endpoint)
                    next_frontier.append(endpoint)

        frontier = next_frontier

    return collected


def score_edge(edge, seed_scores, target_types, question_tokens, hints):

    score = 0.0

    if target_types:
        score += 1.0 if edge["relation"] in target_types else 0.2
    else:
        score += 0.5

    anchor = max(
        seed_scores.get(edge["source_id"], 0),
        seed_scores.get(edge["target_id"], 0),
    )
    score += anchor / 200.0
    score -= 0.15 * (edge.get("hop", 1) - 1)

    qset = set(question_tokens)
    neighbors = (
        edge["target_name"]
        if edge["source_id"] in seed_scores
        else edge["source_name"]
    )
    overlap = len(qset & set(content_tokens(neighbors)))
    score += min(0.9, 0.3 * overlap)

    neighbor_type = normalize_text(
        edge["target_type"]
        if edge["source_id"] in seed_scores
        else edge["source_type"]
    )
    if hints and any(h in neighbor_type for h in hints):
        score += 0.3

    return round(score, 3)


def retrieve_graph_edges(
    session, seeds, query_type, target_types, cfg, question_tokens
):

    profile = QUERY_PROFILES.get(query_type, QUERY_PROFILES["general_information"])

    seed_scores = {s["id"]: s["score"] for s in seeds}
    seed_ids = list(seed_scores)

    raw = []
    fallback_used = False

    if seed_ids:
        raw = expand_hops(
            session, seed_ids, target_types,
            profile["hops"], profile["direction"], cfg["fetch_per_hop"],
        )

        if not raw and target_types:
            # No edge of the expected types exists: fall back to a small,
            # type-diverse neighbourhood and mark it as supporting only.
            fallback_used = True
            raw = expand_hops(
                session, seed_ids, [], 1, "both", cfg["fetch_per_hop"]
            )

    effective_types = [] if fallback_used else target_types

    for e in raw:
        e["score"] = score_edge(
            e, seed_scores, effective_types, question_tokens,
            profile["neighbor_hints"],
        )
        e["strong"] = bool(effective_types) and e["relation"] in effective_types

    raw.sort(key=lambda e: -e["score"])

    cap = cfg["per_type_cap"]
    if fallback_used or not target_types:
        cap = min(cap, 3)

    per_type = defaultdict(int)
    selected = []

    for e in raw:
        if per_type[e["relation"]] >= cap:
            continue
        per_type[e["relation"]] += 1
        selected.append(e)
        if len(selected) >= cfg["max_edges"]:
            break

    return selected, fallback_used


# ============================================================
# STEP 4  EXPLICIT PATH SEARCH (relationship questions)
# ============================================================

def find_entity_paths(session, seeds_by_term, max_paths=6):

    terms = list(seeds_by_term)
    if len(terms) < 2:
        return []

    query = f"""
    MATCH (a:Entity {{id: $a}}), (b:Entity {{id: $b}})
    MATCH p = shortestPath((a)-[*..{MAX_PATH_HOPS}]-(b))
    RETURN
        [n IN nodes(p) | {{id: n.id, name: n.name, type: n.type}}] AS nodes,
        [r IN relationships(p) |
            {{rel: type(r),
              source: startNode(r).name,
              target: endNode(r).name}}] AS rels,
        length(p) AS length
    """

    paths = []
    seen = set()

    for a in seeds_by_term[terms[0]][:2]:
        for b in seeds_by_term[terms[1]][:2]:
            if a["id"] == b["id"]:
                continue
            try:
                for record in session.run(query, a=a["id"], b=b["id"]):
                    data = record.data()
                    key = tuple(n["id"] for n in data["nodes"])
                    if key not in seen:
                        seen.add(key)
                        paths.append(data)
            except Exception as e:
                print(f"Path search warning: {e}")

    paths.sort(key=lambda p: p["length"])
    return paths[:max_paths]


def render_path(path):

    nodes, rels = path["nodes"], path["rels"]
    parts = [f"{nodes[0]['name']} [{nodes[0]['type']}]"]

    for i, rel in enumerate(rels):
        forward = rel["source"] == nodes[i]["name"]
        arrow = (
            f"--{rel['rel']}-->" if forward else f"<--{rel['rel']}--"
        )
        nxt = nodes[i + 1]
        parts.append(f"{arrow} {nxt['name']} [{nxt['type']}]")

    return " ".join(parts)


# ============================================================
# STEP 5  QDRANT RETRIEVAL + RERANKING
# ============================================================

def vector_search(qdrant, embedding_model, query, limit):

    query_vector = embed_texts(embedding_model, [query])[0].tolist()

    result = qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=limit,
        with_payload=True,
    )

    chunks = []

    for point in result.points:

        payload = point.payload or {}
        content = (payload.get("content") or "").strip()

        if not content:
            continue

        if len(content) > MAX_CHUNK_CHARS:
            content = content[:MAX_CHUNK_CHARS] + "..."

        chunks.append({
            "point_id": point.id,
            "score": float(point.score),
            "chunk_id": payload.get("chunk_id"),
            "content": content,
            "page_title": payload.get("page_title"),
            "section": payload.get("section"),
            "source_url": payload.get("source_url"),
            "department": payload.get("department"),
        })

    return chunks


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def rerank_chunks(
    chunks, query_type, entity_phrases, question_tokens, graph_terms, cfg
):
    """
    relevance = 0.50*vector + 0.15*entity overlap + 0.12*keyword overlap
              + 0.12*intent vocabulary + official-source/metadata bonus
              + graph agreement bonus - short-chunk penalty
    then drop near-duplicates and cap chunks per page.
    """
    lexicon = set(INTENT_LEXICON.get(query_type, []))
    qset = set(question_tokens)
    phrases = [normalize_text(p) for p in entity_phrases if normalize_text(p)]
    graph_phrases = [
        normalize_text(g) for g in graph_terms
        if len(normalize_text(g)) >= 4
    ]

    for c in chunks:

        text = normalize_text(
            f"{c['content']} {c.get('page_title') or ''} {c.get('section') or ''}"
        )
        tokens = set(text.split())
        c["_tokens"] = tokens

        ent = (
            sum(1 for p in phrases if p in text) / len(phrases)
            if phrases else 0.0
        )
        kw = len(qset & tokens) / max(1, len(qset))
        intent = (
            min(1.0, len(lexicon & tokens) / 4.0) if lexicon else 0.0
        )

        url = (c.get("source_url") or "").lower()
        meta = 0.0
        if "nmamit" in url or "nitte" in url:
            meta += 0.05
        if c.get("page_title") or c.get("section"):
            meta += 0.03

        agree = min(0.15, 0.05 * sum(1 for g in graph_phrases if g in text))
        quality = -0.1 if len(c["content"]) < 100 else 0.0

        c["relevance"] = round(
            0.50 * c["score"] + 0.15 * ent + 0.12 * kw + 0.12 * intent
            + meta + agree + quality,
            4,
        )

    ranked = sorted(chunks, key=lambda c: -c["relevance"])

    selected = []
    per_page = defaultdict(int)

    for c in ranked:

        if c["score"] < MIN_VECTOR_SCORE and c["relevance"] < 0.30:
            continue

        # drop chunks far weaker than the best one (off-topic tail)
        if c["relevance"] < 0.40 * ranked[0]["relevance"]:
            continue

        page = c.get("page_title") or c.get("source_url") or "?"
        if per_page[page] >= cfg["per_page"]:
            continue

        if any(jaccard(c["_tokens"], s["_tokens"]) >= 0.80 for s in selected):
            continue

        per_page[page] += 1
        c["strong"] = c["relevance"] >= 0.55
        selected.append(c)

        if len(selected) >= cfg["vector_keep"]:
            break

    return selected, len(chunks)


# ============================================================
# STEP 6  CONTEXT CONSTRUCTION
# ============================================================

def format_edge(e):
    return (
        f"{e['source_name']} [{e['source_type']}] "
        f"--{e['relation']}--> "
        f"{e['target_name']} [{e['target_type']}]"
    )


def render_edges(edges, seed_ids):
    """Group large fan-outs into compact lines; keep small ones explicit."""

    groups = OrderedDict()

    for e in edges:
        if e["target_id"] in seed_ids and e["source_id"] not in seed_ids:
            key = ("in", e["target_name"], e["target_type"], e["relation"])
        else:
            key = ("out", e["source_name"], e["source_type"], e["relation"])
        groups.setdefault(key, []).append(e)

    lines = []

    for (direction, name, ntype, relation), items in groups.items():

        marker = "[STRONG]" if items[0].get("strong") else "[SUPPORTING]"

        if len(items) < 4:
            lines += [f"- {marker} {format_edge(e)}" for e in items]
            continue

        if direction == "out":
            members = "; ".join(
                f"{e['target_name']} [{e['target_type']}]" for e in items
            )
            lines.append(
                f"- {marker} {name} [{ntype}] --{relation}--> "
                f"({len(items)} items): {members}"
            )
        else:
            members = "; ".join(
                f"{e['source_name']} [{e['source_type']}]" for e in items
            )
            lines.append(
                f"- {marker} ({len(items)} items): {members} "
                f"--{relation}--> {name} [{ntype}]"
            )

    return lines


def collect_sources(chunks, limit=4):

    sources = OrderedDict()

    for c in sorted(chunks, key=lambda c: -c["relevance"]):
        url = c.get("source_url")
        if url and url not in sources:
            sources[url] = c.get("page_title") or url
        if len(sources) >= limit:
            break

    return sources


def build_context(
    question, query_info, seeds_by_term, edges, paths, chunks,
    fallback_used, exhaustive
):

    seeds = [s for group in seeds_by_term.values() for s in group]
    seed_ids = {s["id"] for s in seeds}

    lines = [
        f"QUESTION: {question}",
        "",
        f"QUERY TYPE: {query_info['query_type']}"
        + (" (complete list requested)" if exhaustive else ""),
        f"INTENT: {query_info['intent']}",
        "",
        "ENTITIES MATCHED IN THE KNOWLEDGE GRAPH:",
    ]

    if seeds:
        lines += [f"- {s['name']} [{s['type']}]" for s in seeds]
    else:
        lines.append("- None matched.")

    lines += [
        "",
        "HOW TO READ THE EVIDENCE:",
        "- [STRONG] graph edges are relationship types that directly match "
        "the question; strong document chunks have high relevance.",
        "- [SUPPORTING] items are context only and must not be turned into "
        "claims the evidence does not establish.",
    ]

    lines += ["", "=" * 60, "GRAPH EVIDENCE", "=" * 60]

    if fallback_used:
        lines.append(
            "NOTE: no relationships of the expected type were found, so "
            "only a small general neighbourhood is shown (supporting only)."
        )

    direct = [e for e in edges if e.get("hop", 1) == 1]
    chained = [e for e in edges if e.get("hop", 1) > 1]

    lines += ["", "DIRECT RELATIONSHIPS:"]
    lines += render_edges(direct, seed_ids) or ["- None retrieved."]

    if chained:
        lines += [
            "",
            "CHAINED RELATIONSHIPS (continue from the direct ones above, "
            "e.g. place -> state -> country):",
        ]
        lines += render_edges(chained, seed_ids)

    if query_info["query_type"] == "relationship":
        lines += ["", "RELEVANT GRAPH PATHS BETWEEN THE ENTITIES:"]
        if paths:
            lines += [
                f"- ({p['length']} hop{'s' if p['length'] != 1 else ''}) "
                f"{render_path(p)}"
                for p in paths
            ]
        else:
            lines.append(
                f"- No graph path of up to {MAX_PATH_HOPS} hops was found "
                "between the two entities (or one of them was not matched)."
            )

    lines += ["", "=" * 60, "DOCUMENT EVIDENCE", "=" * 60]

    if not chunks:
        lines.append("No relevant document chunks were retrieved.")

    for i, c in enumerate(chunks, 1):
        label = "STRONG" if c.get("strong") else "SUPPORTING"
        content = c["content"]
        if len(content) > MAX_CONTEXT_CHUNK_CHARS:
            content = content[:MAX_CONTEXT_CHUNK_CHARS] + "..."
        lines += [
            f"[{label}] DOCUMENT CHUNK {i}",
            f"Page: {c.get('page_title')}",
            f"Section: {c.get('section')}",
            f"Source URL: {c.get('source_url')}",
            content,
            "",
        ]

    sources = collect_sources(chunks)
    lines += ["SOURCE URLS:"]
    lines += [f"- {t}: {u}" for u, t in sources.items()] or ["- None."]

    return "\n".join(lines)


# ============================================================
# STEP 7  ANSWER GENERATION
# ============================================================

def style_instructions(query_type, exhaustive):

    if exhaustive:
        return (
            "This asks for a complete list. Give every item supported by "
            "the evidence as a clean bullet list (group sensibly if there "
            "are many). Do not add items from outside the evidence. If the "
            "evidence cannot establish that the list is complete, say so "
            "in one sentence."
        )

    if query_type == "relationship":
        return (
            "Explain how the entities are connected. Say clearly which "
            "links are direct graph relationships and which are only "
            "implied by a multi-hop path (for example, a shared location "
            "is not the same as an affiliation). If no connection is "
            "evidenced, say so."
        )

    if query_type == "comparison":
        return (
            "Give a short structured comparison using only attributes "
            "present in the evidence for BOTH sides; say where one side "
            "has no evidence."
        )

    return (
        "Answer directly and concisely (about 2 to 5 sentences) unless the "
        "question genuinely needs more. Lead with the answer, then add the "
        "most useful supporting details."
    )


def ask_groq(client, question, query_info, context, exhaustive):

    system = f"""
You are the NMAMIT information and admissions assistant.

Answer using ONLY the evidence provided in the user message
(knowledge-graph relationships and document chunks).

GROUNDING RULES
1. Never use outside knowledge and never invent facts, numbers, names,
   URLs, rankings, facilities, programs, locations or affiliations.
2. Direct graph relationships and document text are evidence. A weak or
   unusual relationship must not be inflated into a stronger claim.
3. Prefer [STRONG] evidence. Use [SUPPORTING] evidence only to add
   context that it actually states.
4. Combine graph and document evidence into one natural answer. Do not
   dump raw relationships or repeat irrelevant graph data.
5. If the graph and the documents disagree, say that the retrieved
   sources differ and state both; never silently pick one.
6. Distinguish what the evidence states directly from what you are
   inferring, and flag inference briefly.
7. If the evidence is insufficient, reply that you couldn't find enough
   information in the NMAMIT knowledge base to answer reliably. If it
   covers only part of the question, answer that part and say what is
   missing.
8. Do not mention retrieval, Neo4j, Qdrant, chunks or "evidence" labels.
9. Do not write a Sources section; it is appended automatically.

STYLE
{style_instructions(query_info['query_type'], exhaustive)}
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": context},
        ],
        temperature=0.2,
    )

    return (response.choices[0].message.content or "").strip()


# ============================================================
# DEBUG OUTPUT
# ============================================================

def banner(title):
    if DEBUG:
        print("\n" + "=" * 70)
        print(title)
        print("=" * 70)


def dprint(*args):
    if DEBUG:
        print(*args)


# ============================================================
# STEP 8  COMPLETE PIPELINE
# ============================================================

def retrieve_and_answer(
    driver, qdrant, embedding_model, groq_client, question, schema=None
):

    schema = schema or {"rel_types": [], "entity_types": []}

    # ---------------- QUERY UNDERSTANDING ----------------

    query_info = analyze_query(groq_client, question, schema)
    query_type = query_info["query_type"]

    exhaustive = is_exhaustive_query(question) or query_info["is_list"]
    cfg = MODE_LIMITS["exhaustive" if exhaustive else "normal"]

    question_tokens = content_tokens(
        f"{question} {query_info['search_query']}"
    )

    banner("QUERY UNDERSTANDING")
    dprint(f"Original question : {question}")
    dprint(f"Intent            : {query_info['intent']}")
    dprint(f"Entities          : {query_info['entities']}")
    dprint(f"Search query      : {query_info['search_query']}")
    dprint(f"Query type        : {query_type}"
           f"{'  (exhaustive)' if exhaustive else ''}")
    dprint(f"Relation types    : {query_info['target_relation_types']}")

    # ---------------- GRAPH RETRIEVAL ----------------

    with driver.session(database=NEO4J_DATABASE) as session:

        seeds_by_term = resolve_entities(
            session, embedding_model, query_info["entities"], question, cfg
        )

        seeds = [s for group in seeds_by_term.values() for s in group]

        edges, fallback_used = retrieve_graph_edges(
            session, seeds, query_type,
            query_info["target_relation_types"], cfg, question_tokens,
        )

        paths = []
        if query_type == "relationship" and len(seeds_by_term) >= 2:
            paths = find_entity_paths(session, seeds_by_term)

    banner("GRAPH RETRIEVAL")
    dprint("Selected entities:")
    for term, group in seeds_by_term.items():
        for s in group:
            dprint(f"  - {s['name']} [{s['type']}]  "
                   f"score={s['score']}  (from '{term}')")
    if not seeds:
        dprint("  - none")

    dprint(f"Selected relationships ({len(edges)}):")
    for e in edges[:40]:
        dprint(f"  - {format_edge(e)}  "
               f"[score={e['score']}, hop={e.get('hop', 1)}"
               f"{', strong' if e.get('strong') else ''}]")
    if len(edges) > 40:
        dprint(f"  ... {len(edges) - 40} more")
    if fallback_used:
        dprint("  (no edges of the target types; general fallback used)")

    dprint(f"Selected paths ({len(paths)}):")
    for p in paths:
        dprint(f"  - ({p['length']} hops) {render_path(p)}")

    # ---------------- VECTOR RETRIEVAL ----------------

    vector_query = query_info["search_query"]
    if seeds:
        canonical = seeds[0]["name"]
        if (
            normalize_text(canonical) not in normalize_text(vector_query)
            and len(canonical) < 80
        ):
            vector_query = f"{vector_query} {canonical}"

    candidates = vector_search(
        qdrant, embedding_model, vector_query, cfg["vector_fetch"]
    )

    entity_phrases = list(query_info["entities"]) + [s["name"] for s in seeds]
    graph_terms = [
        e["target_name"] if e["source_id"] in {s["id"] for s in seeds}
        else e["source_name"]
        for e in edges
    ]

    chunks, considered = rerank_chunks(
        candidates, query_type, entity_phrases, question_tokens,
        graph_terms, cfg,
    )

    banner("VECTOR RETRIEVAL")
    dprint(f"Vector query: {vector_query}")
    dprint(f"Candidates: {considered}   kept after rerank/dedup: {len(chunks)}")
    for c in chunks:
        dprint(f"  - chunk {c['chunk_id']}  sim={c['score']:.4f}  "
               f"relevance={c['relevance']:.3f}"
               f"{'  strong' if c.get('strong') else ''}")
        dprint(f"      page   : {c.get('page_title')}")
        dprint(f"      section: {c.get('section')}")
        dprint(f"      source : {c.get('source_url')}")

    # ---------------- INSUFFICIENT EVIDENCE ----------------

    if not edges and not paths and not chunks:
        banner("FINAL ANSWER")
        print(INSUFFICIENT_MESSAGE)
        return INSUFFICIENT_MESSAGE

    # ---------------- CONTEXT + ANSWER ----------------

    context = build_context(
        question, query_info, seeds_by_term, edges, paths, chunks,
        fallback_used, exhaustive,
    )

    if SHOW_CONTEXT:
        banner("CONTEXT SENT TO GROQ")
        print(context)

    answer = ask_groq(groq_client, question, query_info, context, exhaustive)

    if not answer:
        answer = INSUFFICIENT_MESSAGE

    sources = collect_sources(chunks)
    declined = "couldn't find enough information" in answer.lower()

    if sources and not declined:
        answer += "\n\nSources:\n" + "\n".join(
            f"- {title}: {url}" for url, title in sources.items()
        )

    banner("FINAL ANSWER")
    print(answer)

    return answer


# ============================================================
# TEST QUESTIONS
# ============================================================

TEST_QUESTIONS = [
    "where is nmamit?",
    "how is nmamit related to nitte deemed university?",
    "what programs does nmamit offer?",
    "what departments are there in nmamit?",
    "what facilities does nmamit have?",
    "who are the faculty members?",
    "what research activities are happening at nmamit?",
    "what clubs are available?",
    "what is nmamit?",
    "what is the nearest airport to nmamit?",
]


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(description="NMAMIT Hybrid Graph RAG")
    parser.add_argument("--ask", help="answer one question and exit")
    parser.add_argument("--tests", action="store_true",
                        help="run the 10 built-in test questions")
    args = parser.parse_args()

    check_config()

    print("=" * 70)
    print("NMAMIT HYBRID GRAPH RAG")
    print("Neo4j AuraDB + Qdrant Cloud + Groq")
    print("=" * 70)

    print("\nLoading embedding model...")
    embedding_model = FastEmbedAdapter(EMBEDDING_MODEL)
    print("Embedding model loaded.")

    print("Connecting to Neo4j...")
    driver = GraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )
    driver.verify_connectivity()
    print("Neo4j connected.")

    schema = load_schema(driver)
    print(
        f"Schema: {len(schema['rel_types'])} relationship types, "
        f"{len(schema['entity_types'])} entity types."
    )
    if DEBUG:
        print("Relationship types:", ", ".join(schema["rel_types"]))

    print("Connecting to Qdrant...")
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    try:
        qdrant.get_collection(collection_name=QDRANT_COLLECTION)
        print("Qdrant collection ready: " + QDRANT_COLLECTION)
    except Exception as e:
        print("\nWARNING: Could not verify Qdrant collection.")
        print(e)

    print("Connecting to Groq...")
    groq_client = Groq(api_key=GROQ_API_KEY)
    print("Groq client ready.")

    def run(question):
        try:
            retrieve_and_answer(
                driver=driver,
                qdrant=qdrant,
                embedding_model=embedding_model,
                groq_client=groq_client,
                question=question,
                schema=schema,
            )
        except Exception as e:
            print("\n" + "=" * 70)
            print("ERROR")
            print("=" * 70)
            print(e)
            print("\nThe query could not be completed.")

    try:

        if args.ask:
            run(args.ask)
            return

        if args.tests:
            for q in TEST_QUESTIONS:
                print("\n" + "#" * 70)
                print(f"TEST: {q}")
                print("#" * 70)
                run(q)
            return

        print("\n" + "=" * 70)
        print("SYSTEM READY")
        print("=" * 70)
        print("Type: ask <your question>")
        print("Type: exit")
        print("=" * 70)

        while True:

            try:
                user_input = input("\ngraph-rag> ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nExiting...")
                break

            if user_input.lower().startswith("ask "):
                question = user_input[4:].strip()
                if not question:
                    print("Please enter a question.")
                    continue
                run(question)
            else:
                print("Use: ask <your question>")

    finally:
        driver.close()
        print("\nNeo4j connection closed.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()