# ─────────────────────────────────────────
# Fermentation RAG Assistant API
# ─────────────────────────────────────────

# 1. Library imports
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from rag_pipeline import initialize_rag, get_answer

# 2. Create the app object
app = FastAPI()

# 3. Initialize RAG pipeline
retriever, llm = initialize_rag()

# 4. Pydantic models
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]

# 5. Index route — http://0.0.0.0:8000
@app.get('/')
def index():
    return {'message': 'Welcome to Fermentation Assistant!'}

# 6. Health check — http://0.0.0.0:8000/health
@app.get('/health')
def health():
    return {'status': 'running'}

# 7. Ask question — http://0.0.0.0:8000/ask
@app.post('/ask')
def ask_question(request: QueryRequest):
    result = get_answer(request.question, retriever, llm)
    return QueryResponse(
        question=request.question,
        answer=result['answer'],
        sources=result['sources']
    )

# 8. Run the API with uvicorn
#    Will run on http://0.0.0.0:8000
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
