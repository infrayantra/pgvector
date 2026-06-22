"""RAG retrieval simulation — chunk store + context assembly."""

from db import EMBED_DIM, print_results, reset_schema
from embeddings import embed


CHUNKS = [
    ("pgvector-intro", 0, "pgvector is an open-source extension for PostgreSQL that enables vector similarity search."),
    ("pgvector-intro", 1, "It supports exact and approximate nearest neighbor search using HNSW and IVFFlat indexes."),
    ("pgvector-intro", 2, "You can store embeddings alongside relational data and use standard SQL for queries."),
    ("rag-explained", 0, "Retrieval-Augmented Generation combines a retriever with a large language model."),
    ("rag-explained", 1, "The retriever fetches relevant documents which are passed as context to the LLM."),
    ("rag-explained", 2, "This reduces hallucinations by grounding answers in your own data."),
    ("indexing", 0, "HNSW indexes provide fast approximate search with tunable recall via ef_search."),
    ("indexing", 1, "IVFFlat divides vectors into lists using k-means and probes nearest lists at query time."),
    ("indexing", 2, "Create indexes after bulk loading data for best build performance."),
]


def run(conn) -> None:
    print("\n--- RAG Retrieval Simulation ---")
    print("Demonstrates: chunked knowledge base, top-k retrieval, context assembly\n")

    reset_schema(conn)
    conn.execute(f"""
        CREATE TABLE knowledge_chunks (
            id serial PRIMARY KEY,
            source text NOT NULL,
            chunk_index int NOT NULL,
            content text NOT NULL,
            embedding vector({EMBED_DIM}),
            UNIQUE (source, chunk_index)
        )
    """)
    conn.execute("""
        CREATE INDEX chunks_hnsw ON knowledge_chunks
        USING hnsw (embedding vector_cosine_ops)
    """)

    texts = [c[2] for c in CHUNKS]
    vectors = embed(texts)
    for (source, idx, content), vec in zip(CHUNKS, vectors):
        conn.execute(
            """
            INSERT INTO knowledge_chunks (source, chunk_index, content, embedding)
            VALUES (%s, %s, %s, %s)
            """,
            (source, idx, content, vec),
        )

    user_question = "How do I speed up vector search in PostgreSQL?"
    qvec = embed([user_question])[0]
    k = 4

    rows = conn.execute(
        """
        SELECT source, chunk_index, content,
               round((embedding <=> %s)::numeric, 4) AS distance
        FROM knowledge_chunks
        ORDER BY embedding <=> %s
        LIMIT %s
        """,
        (qvec, qvec, k),
    ).fetchall()

    print(f'User question: "{user_question}"')
    print(f"\nRetrieved top-{k} chunks:")
    print_results(rows, ["source", "chunk", "content", "distance"])

    context = "\n\n".join(f"[{r[0]}#{r[1]}] {r[2]}" for r in rows)
    print("\n--- Assembled LLM context ---")
    print(context)
    print("\n(In production, this context + question would be sent to an LLM API)")
