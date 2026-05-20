from rag.retriever import retrieve
from rag.chunker import chunk_text
from rag.vector_store import add_chunks
from rag.embeddings import create_embedding

# Test 1: Crear embeddings
text = "Este es un transcript de YouTube sobre inteligencia artificial"
embedding = create_embedding(text)
print(f"✅ Embedding creado: {len(embedding)} dimensiones")

# Test 2: Chunear texto
chunks = chunk_text(text)
print(f"✅ Texto chuneado en {len(chunks)} partes")

# Test 3: Agregar a ChromaDB
embeddings = [create_embedding(chunk) for chunk in chunks]
add_chunks(chunks, embeddings)
print(f"✅ Chunks agregados a ChromaDB")

# Test 4: Recuperar con RAG
results = retrieve("inteligencia artificial", k=5)
print(f"✅ Resultados recuperados: {results}")
