import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from raganything import RAGAnything, QueryParam
from raganything.config import RAGAnythingConfig
from lightrag.utils import EmbeddingFunc

from adapters import OpenRouterLLMAdapter, OpenRouterEmbeddingAdapter
from config import EMBED_DIM

# Global state to hold the RAG instance
rag_instance = None

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager for the FastAPI application.
    Initializes the RAG-Anything system and its heavy dependencies once on startup.
    """
    global rag_instance
    print("Initializing RAG-Anything system...")
    
    try:
        llm = OpenRouterLLMAdapter()
        emb = OpenRouterEmbeddingAdapter()
        
        async def llm_complete(prompt: str, max_tokens: int = 2048) -> str:
            return await llm.chat(prompt)

        async def embed_func(texts: list) -> list:
            return [await emb.embed_text(t) for t in texts]
            
        config = RAGAnythingConfig(working_dir="rag_workdir")
        
        rag_instance = RAGAnything(
            config=config,
            llm_model_func=llm_complete,
            vision_model_func=llm.chat_vlm_with_image if hasattr(llm, "chat_vlm_with_image") else None,
            embedding_func=EmbeddingFunc(
                embedding_dim=EMBED_DIM,
                max_token_size=8192,
                func=embed_func
            ),
        )
        print("RAG-Anything initialized successfully.")
    except Exception as e:
        print(f"Error initializing RAG-Anything: {e}")
        raise e
        
    yield
    
    # Cleanup on shutdown (if needed)
    print("Shutting down RAG-Anything system...")

# Initialize FastAPI App
app = FastAPI(
    title="RAG-Anything API", 
    description="Multimodal RAG powered by Qwen and Milvus", 
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "rag_loaded": rag_instance is not None}

@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    """
    Main chat endpoint. Takes a query string and returns the RAG generated answer.
    """
    if rag_instance is None:
        raise HTTPException(status_code=503, detail="RAG system is not initialized yet.")
        
    try:
        # We use hybrid mode (graph + vector search)
        answer = await rag_instance.aquery(request.query, param=QueryParam(mode="hybrid"))
        return QueryResponse(answer=answer)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
