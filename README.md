# RAG-Anything with Milvus Standalone & DeepSeek-V4

This repository contains a Multimodal RAG system built using **RAG-Anything**, **LightRAG**, **Milvus Standalone**, and **DeepSeek-V4** (via Aliyun DashScope). 

It is designed to ingest a massive PDF (`sample/docs/cam_nang_sau_dai_hoc_2025_0.pdf`) and provide intelligent text and multimodal querying.

## Setup Instructions

### 1. Host Milvus
You are using **Milvus Lite**. The vector database will automatically run locally within the application and store data in `milvus_lite.db` in your current folder. No external hosting or Docker setup is required!

### 2. Environment Variables
Your `.env` and `config.py` files have already been updated. Ensure you provide your Aliyun API Key in `.env`:
```env
DASHSCOPE_API_KEY=your_api_key_here
LLM_TEXT_MODEL=deepseek-v4
MILVUS_URI=milvus_lite.db
```
*Note: Make sure your API key has access to the `deepseek-v4` model on DashScope.*

### 3. Install Dependencies
Make sure you have all required dependencies installed, especially the RAG-Anything package.
```bash
pip install -r requirements.txt
pip install "raganything[all]" 
pip install pymupdf  # Used as a fallback parser if needed
```

## Running the Pipeline

### Step 1: Ingest the Massive PDF
To parse the PDF (`sample/docs/cam_nang_sau_dai_hoc_2025_0.pdf`) and build the vector database and knowledge graph:
```bash
python ingest_pdf.py
```
*Note: Depending on the PDF size, this may take several minutes to chunk, parse, embed, and insert into Milvus and RAG-Anything's local KV storage.*

### Step 2: Query the Knowledge Base
Once ingestion is complete, you can start querying your system:
```bash
# Default query (Vietnamese)
python query_system.py

# Custom query
python query_system.py "What are the English language requirements?"
```
The script will use `RAG-Anything` in hybrid mode (searching both vector embeddings and the knowledge graph) to retrieve the best context and generate a response using DeepSeek-V4.
