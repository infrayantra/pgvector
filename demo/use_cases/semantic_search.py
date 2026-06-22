"""Semantic search over a document corpus."""

from db import EMBED_DIM, print_results, reset_schema
from embeddings import embed, embedding_backend


DOCUMENTS = [
    ("doc1", "How to install PostgreSQL on Ubuntu Linux"),
    ("doc2", "Machine learning fundamentals with Python"),
    ("doc3", "Setting up Docker containers for development"),
    ("doc4", "Neural networks and deep learning explained"),
    ("doc5", "Backup and restore strategies for databases"),
    ("doc6", "Introduction to natural language processing"),
    ("doc7", "Kubernetes deployment best practices"),
    ("doc8", "Vector embeddings for semantic search"),
]


def run(conn) -> None:
    print("\n--- Semantic Document Search ---")
    print(f"Embedding backend: {embedding_backend()}")
    print("Demonstrates: cosine similarity search over text corpus\n")

    reset_schema(conn)
    conn.execute(f"""
        CREATE TABLE documents (
            id text PRIMARY KEY,
            content text NOT NULL,
            embedding vector({EMBED_DIM})
        )
    """)
    conn.execute("""
        CREATE INDEX documents_hnsw ON documents
        USING hnsw (embedding vector_cosine_ops)
    """)

    texts = [d[1] for d in DOCUMENTS]
    vectors = embed(texts)
    for (doc_id, content), vec in zip(DOCUMENTS, vectors):
        conn.execute(
            "INSERT INTO documents (id, content, embedding) VALUES (%s, %s, %s)",
            (doc_id, content, vec),
        )

    queries = [
        "database administration tutorial",
        "artificial intelligence and transformers",
        "container orchestration",
    ]

    for query in queries:
        qvec = embed([query])[0]
        rows = conn.execute(
            """
            SELECT id, left(content, 45) AS content,
                   round((1 - (embedding <=> %s))::numeric, 4) AS similarity
            FROM documents
            ORDER BY embedding <=> %s
            LIMIT 3
            """,
            (qvec, qvec),
        ).fetchall()
        print(f'Query: "{query}"')
        print_results(rows, ["id", "content", "similarity"])
        print()
