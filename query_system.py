import asyncio
import sys
from rich import print

from raganything import RAGAnything
from lightrag.base import QueryParam
from raganything.config import RAGAnythingConfig

from lightrag.utils import EmbeddingFunc
from adapters import OpenRouterLLMAdapter, OpenRouterEmbeddingAdapter
from config import EMBED_DIM

WORKING_DIR = "rag_workdir"

async def main(query_text: str):
    llm = OpenRouterLLMAdapter()
    emb = OpenRouterEmbeddingAdapter()
    
    async def llm_complete(prompt: str, max_tokens: int = 2048) -> str:
        return await llm.chat(prompt)

    async def embed_func(texts: list) -> list:
        return [await emb.embed_text(t) for t in texts]

    config = RAGAnythingConfig(working_dir=WORKING_DIR)
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

    print(f"\n[bold cyan]Querying Qwen (OpenRouter) for:[/bold cyan] {query_text}")
    print("[yellow]Retrieving context...[/yellow]")
    
    # Query using hybrid mode (vector + graph)
    answer = await rag.aquery(query_text, param=QueryParam(mode="hybrid"))
    
    print("\n[bold green]Answer:[/bold green]")
    print(answer)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        user_query = "Điều kiện tuyển sinh sau đại học năm 2025 là gì?"
    asyncio.run(main(user_query))
