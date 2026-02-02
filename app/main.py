from contextlib import asynccontextmanager

import trafilatura
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from app.classifier import classify, get_classifier
from app.models import (
    ClassifyRequest,
    ClassifyResponse,
    ExtractRequest,
    ExtractResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_classifier()
    yield


app = FastAPI(title="Zero-Shot Classifier", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/classify", response_model=ClassifyResponse)
def classify_content(req: ClassifyRequest):
    results = classify(req.content, req.assertions)
    return ClassifyResponse(results=results)


@app.post("/extract", response_model=ExtractResponse)
def extract_content(req: ExtractRequest):
    downloaded = trafilatura.fetch_url(req.url)
    if not downloaded:
        raise HTTPException(status_code=422, detail="Could not fetch the URL")
    content = trafilatura.extract(downloaded)
    if not content:
        raise HTTPException(status_code=422, detail="Could not extract content")
    metadata = trafilatura.extract(downloaded, output_format="json", only_with_metadata=True)
    title = ""
    if metadata:
        import json
        meta = json.loads(metadata)
        title = meta.get("title", "")
    return ExtractResponse(content=content, title=title)


app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
