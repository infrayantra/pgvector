"""Filtered vector search with metadata and iterative scans."""

from db import EMBED_DIM, print_results, reset_schema
from embeddings import embed


PRODUCTS = [
    ("Laptop Pro 15", "electronics", 1299.00, "High-performance laptop for developers"),
    ("Wireless Mouse", "electronics", 29.99, "Ergonomic wireless mouse"),
    ("Python Cookbook", "books", 45.00, "Advanced Python programming recipes"),
    ("Database Internals", "books", 55.00, "Deep dive into database storage engines"),
    ("USB-C Hub", "electronics", 49.99, "Multi-port hub for laptops"),
    ("ML Handbook", "books", 60.00, "Machine learning algorithms and practice"),
    ("Mechanical Keyboard", "electronics", 149.00, "RGB mechanical keyboard for coding"),
    ("SQL Antipatterns", "books", 40.00, "Common SQL mistakes and how to avoid them"),
]


def run(conn) -> None:
    print("\n--- Filtered Search (Metadata + Vector) ---")
    print("Demonstrates: B-tree filter + HNSW + iterative scan\n")

    reset_schema(conn)
    conn.execute(f"""
        CREATE TABLE products (
            id serial PRIMARY KEY,
            name text NOT NULL,
            category text NOT NULL,
            price numeric(10,2),
            description text,
            embedding vector({EMBED_DIM})
        )
    """)
    conn.execute("CREATE INDEX products_category ON products (category)")
    conn.execute("""
        CREATE INDEX products_hnsw ON products
        USING hnsw (embedding vector_cosine_ops)
    """)

    for name, cat, price, desc in PRODUCTS:
        vec = embed([f"{name}: {desc}"])[0]
        conn.execute(
            """
            INSERT INTO products (name, category, price, description, embedding)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (name, cat, price, desc, vec),
        )

    query = "programming and software development"
    qvec = embed([query])[0]

    print(f'Query: "{query}"\n')

    # Unfiltered
    rows = conn.execute(
        """
        SELECT name, category, round((embedding <=> %s)::numeric, 4) AS dist
        FROM products ORDER BY embedding <=> %s LIMIT 3
        """,
        (qvec, qvec),
    ).fetchall()
    print("Unfiltered top 3:")
    print_results(rows, ["name", "category", "distance"])

    # Category filter with iterative scan
    conn.execute("SET hnsw.iterative_scan = strict_order")
    rows = conn.execute(
        """
        SELECT name, category, round((embedding <=> %s)::numeric, 4) AS dist
        FROM products
        WHERE category = 'books'
        ORDER BY embedding <=> %s
        LIMIT 3
        """,
        (qvec, qvec),
    ).fetchall()
    print("\nFiltered (category = 'books') top 3:")
    print_results(rows, ["name", "category", "distance"])

    # Price + category
    rows = conn.execute(
        """
        SELECT name, price, round((embedding <=> %s)::numeric, 4) AS dist
        FROM products
        WHERE category = 'electronics' AND price < 100
        ORDER BY embedding <=> %s
        LIMIT 3
        """,
        (qvec, qvec),
    ).fetchall()
    print("\nFiltered (electronics, price < $100) top 3:")
    print_results(rows, ["name", "price", "distance"])

    conn.execute("RESET hnsw.iterative_scan")
