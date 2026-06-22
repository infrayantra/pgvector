"""Item-based recommendation via vector similarity."""

import numpy as np

from db import EMBED_DIM, print_results, reset_schema


ITEMS = [
    (1, "The Matrix", "sci-fi action"),
    (2, "Inception", "sci-fi thriller"),
    (3, "Interstellar", "sci-fi drama"),
    (4, "The Notebook", "romance drama"),
    (5, "Pride and Prejudice", "romance classic"),
    (6, "Blade Runner 2049", "sci-fi noir"),
    (7, "La La Land", "musical romance"),
    (8, "Dune", "sci-fi epic"),
]

# Hand-crafted genre vectors for reproducible demo (no ML model needed)
GENRE_DIMS = {"sci-fi": 0, "action": 1, "romance": 2, "drama": 3, "thriller": 4, "noir": 5, "musical": 6, "epic": 7}


def _genre_vector(tags: str) -> np.ndarray:
    vec = np.zeros(EMBED_DIM, dtype=np.float32)
    for tag in tags.split():
        if tag in GENRE_DIMS:
            vec[GENRE_DIMS[tag]] = 1.0
    # spread signal across dimensions for pgvector
    for i, tag in enumerate(tags.split()):
        vec[(i * 47) % EMBED_DIM] = 0.5
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def run(conn) -> None:
    print("\n--- Recommendation Engine ---")
    print("Demonstrates: item vectors, 'users who liked X' style recommendations\n")

    reset_schema(conn)
    conn.execute(f"""
        CREATE TABLE movies (
            id int PRIMARY KEY,
            title text NOT NULL,
            tags text,
            embedding vector({EMBED_DIM})
        )
    """)
    conn.execute("""
        CREATE INDEX movies_hnsw ON movies
        USING hnsw (embedding vector_cosine_ops)
    """)

    for mid, title, tags in ITEMS:
        conn.execute(
            "INSERT INTO movies (id, title, tags, embedding) VALUES (%s, %s, %s, %s)",
            (mid, title, tags, _genre_vector(tags)),
        )

    liked_movie_id = 1  # The Matrix
    liked = conn.execute("SELECT title, embedding FROM movies WHERE id = %s", (liked_movie_id,)).fetchone()
    print(f'User liked: "{liked[0]}"\n')

    rows = conn.execute(
        """
        SELECT m.title, m.tags,
               round((1 - (m.embedding <=> ref.embedding))::numeric, 4) AS similarity
        FROM movies m, (SELECT embedding FROM movies WHERE id = %s) ref
        WHERE m.id != %s
        ORDER BY m.embedding <=> ref.embedding
        LIMIT 5
        """,
        (liked_movie_id, liked_movie_id),
    ).fetchall()

    print("Recommended movies:")
    print_results(rows, ["title", "tags", "similarity"])

    # User preference vector = average of liked items
    print("\n--- User profile vector (avg of liked items) ---")
    user_liked_ids = [1, 2, 8]  # Matrix, Inception, Dune
    rows = conn.execute(
        """
        SELECT m.title,
               round((1 - (m.embedding <=> profile.avg_emb))::numeric, 4) AS similarity
        FROM movies m,
             (SELECT AVG(embedding) AS avg_emb FROM movies WHERE id = ANY(%s)) profile
        WHERE m.id != ALL(%s)
        ORDER BY m.embedding <=> profile.avg_emb
        LIMIT 5
        """,
        (user_liked_ids, user_liked_ids),
    ).fetchall()
    print(f"Based on liking: The Matrix, Inception, Dune")
    print_results(rows, ["title", "similarity"])
