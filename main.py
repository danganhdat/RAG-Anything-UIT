"""
Minimal Working Example:
- Insert a short text FAQ into LightRAG (text retrieval context)
- Insert an image description vector into Milvus (image retrieval context)
- Execute two example queries: one text QA and one image-based QA
"""
import asyncio
import uuid
from pathlib import Path
from rich import print
from raganything import RAGAnything, QueryParam
from raganything.config import RAGAnythingConfig
from lightrag.utils import EmbeddingFunc
from adapters import OpenRouterLLMAdapter, OpenRouterEmbeddingAdapter
from milvus_store import MilvusVectorStore
from config import EMBED_DIM

SAMPLE_DOC = Path("sample/docs/faq_milvus.txt")
SAMPLE_IMG = Path("sample/images/milvus_arch.png")

async def main():
    # 1) Initialize core components
    llm = OpenRouterLLMAdapter()
    emb = OpenRouterEmbeddingAdapter()
    store = MilvusVectorStore()

    # 2) Initialize LightRAG (for text-only retrieval)
    async def llm_complete(prompt: str, max_tokens: int = 1024) -> str:
        return await llm.chat(prompt)
    async def embed_func(text: str) -> list:
        return await emb.embed_text(text)
    config = RAGAnythingConfig(working_dir="rag_workdir_min")
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

    # 3) Insert text data
    if SAMPLE_DOC.exists():
        text = SAMPLE_DOC.read_text(encoding="utf-8")
        await rag.lightrag.ainsert(text)
        print("[green]Inserted FAQ text into RAG-Anything[/green]")
    else:
        print("[yellow] sample/docs/faq_milvus.txt not found[/yellow]")

    # 4) Insert image data (store description in Milvus)
    if SAMPLE_IMG.exists():
        # Use the VLM to generate a description as its semantic content
        desc = await llm.chat_vlm_with_image("Please briefly describe the key components of the Milvus architecture shown in the image.", str(SAMPLE_IMG))
        vec = await emb.embed_text(desc)  # Use text embeddings to maintain a consistent vector dimension, simplifying reuse
        store.upsert(
            ids=[str(uuid.uuid4())],
            vectors=[vec],
            contents=[desc],
            content_types=["image"],
            sources=[str(SAMPLE_IMG)]
        )
        print("[green]Inserted image description into Milvus（content_type=image）[/green]")
    else:
        print("[yellow] sample/images/milvus_arch.png not found[/yellow]")

    # 5) Query: Text-based QA (from RAG-Anything)
    q1 = "Does Milvus support simultaneous insertion and search? Give a short answer."
    ans1 = await rag.aquery(q1, param=QueryParam(mode="hybrid"))
    print("\\n[bold]Text QA[/bold]")
    print(ans1)
    
    # 6) Query: Image-related QA (from Milvus)
    q2 = "What are the key components of the Milvus architecture?"
    q2_vec = await emb.embed_text(q2)
    img_hits = store.search([q2_vec], top_k=3, content_type="image")
    print("\\n[bold]Image Retrieval (returns semantic image descriptions)[/bold]")
    print(img_hits[0] if img_hits else [])

if __name__ == "__main__":
    asyncio.run(main())
