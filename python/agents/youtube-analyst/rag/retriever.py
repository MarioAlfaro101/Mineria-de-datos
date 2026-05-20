from .embeddings import create_embedding
from .vector_store import collection

def retrieve(query, k=5):
    embedding = create_embedding(query)

    results = collection.query(
        query_embeddings=[embedding],
        n_results=k
    )

    return results["documents"]
