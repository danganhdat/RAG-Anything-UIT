from rag_app.core.logging import setup_logging

setup_logging()

from rag_app.api.app import create_app

app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("run_api:app", host="0.0.0.0", port=8000, reload=True)
