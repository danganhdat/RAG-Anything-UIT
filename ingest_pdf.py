import asyncio
import os
import shutil
from pathlib import Path
from rich import print

from raganything import RAGAnything
from raganything.config import RAGAnythingConfig

from lightrag.utils import EmbeddingFunc
from adapters import OpenRouterLLMAdapter, OpenRouterEmbeddingAdapter
from milvus_store import MilvusVectorStore
from config import EMBED_DIM

# Ensure we have a place to copy the file if needed for folder processing
WORKING_DIR = Path("rag_workdir")
PDF_PATH = Path("sample/docs/cam_nang_sau_dai_hoc_2025_0.pdf")

async def main():
    print(f"[bold cyan]Starting ingestion for {PDF_PATH}[/bold cyan]")
    
    if not PDF_PATH.exists():
        print(f"[red]Error: PDF not found at {PDF_PATH}[/red]")
        return

    # Initialize Adapters
    llm = OpenRouterLLMAdapter()
    emb = OpenRouterEmbeddingAdapter()
    store = MilvusVectorStore()  # Connects to standalone Milvus at tcp://localhost:19530
    
    async def llm_complete(prompt: str, max_tokens: int = 2048) -> str:
        # Utilizing DeepSeek-V4 via Aliyun DashScope (as configured in .env)
        return await llm.chat(prompt)

    async def embed_func(texts: list) -> list:
        # LightRAG/RAGAnything passes a list of texts; we need to return a list of embeddings
        # The AliyunEmbeddingAdapter embeds a single string currently, so we process each.
        return [await emb.embed_text(t) for t in texts]

    # Initialize RAG-Anything
    config = RAGAnythingConfig(working_dir=str(WORKING_DIR))
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_complete,
        vision_model_func=llm.chat_vlm_with_image if hasattr(llm, "chat_vlm_with_image") else None,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBED_DIM,
            max_token_size=8192,
            func=embed_func
        ),
    )

    # RAG-Anything ingestion
    print("[yellow]Ingesting document into the RAG-Anything Graph & Vector DB... This may take a while.[/yellow]")
    
    try:
        print("[yellow]Using RAGAnything's process_document_complete to parse and insert the PDF...[/yellow]")
        # This triggers the built-in parser (MinerU) and extracts multimodal content into the Vector Store
        await rag.process_document_complete(str(PDF_PATH), start_page=0, end_page=1)
            
    except Exception as e:
        print(f"[red]Error during ingestion: {e}[/red]")
        return
        
    print("[green]Ingestion completed successfully![/green]")

if __name__ == "__main__":
    asyncio.run(main())
