1. Instala dependencias
   pip install -r requirements.txt

2. Prueba RAG aislado
   python test_rag.py

3. Prueba el agent con transcripts
   python -c "from agent import augment_with_rag; print(augment_with_rag('análisis de YouTube', 'base prompt'))"

4. Lanza el dashboard
   streamlit run dashboard/app.py
cd python/agents/youtube-analyst
python test_rag.py
