"""
Ingest endpoints — document upload and indexing.
POST /api/upload   →  upload PDF / TXT files to the data directory
POST /api/ingest   →  run the ingestion pipeline (load → chunk → embed → index)
"""

import os
from fastapi import APIRouter, HTTPException, UploadFile, File

from src.config import settings
from src.schemas import IngestResponse, UploadResponse
from src.ingest import ingest_pipeline, load_documents, chunk_documents

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_files(files: list[UploadFile] = File(...)):
    """Upload one or more PDF/TXT files to the data directory.

    Files are saved to disk but NOT ingested yet.
    Call POST /api/ingest afterwards to build the FAISS index.
    """
    allowed_extensions = (".pdf", ".txt")
    saved = []

    os.makedirs(settings.data_directory, exist_ok=True)

    for f in files:
        if not f.filename:
            continue
        if not f.filename.lower().endswith(allowed_extensions):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {f.filename}. Only PDF and TXT are allowed.",
            )

        filepath = os.path.join(settings.data_directory, f.filename)
        content = await f.read()
        with open(filepath, "wb") as out:
            out.write(content)
        saved.append(f.filename)

    return UploadResponse(
        message=f"Uploaded {len(saved)} file(s).",
        filenames=saved,
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest():
    """Run the full ingestion pipeline.

    Reads all documents from the data/ directory, chunks them,
    embeds with SentenceTransformer, and builds a FAISS index.
    """
    # Quick pre-check: any files in the data dir?
    data_dir = settings.data_directory
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        raise HTTPException(
            status_code=422,
            detail=f"No files found in '{data_dir}/'. Upload documents first via POST /api/upload.",
        )

    # Count documents and chunks for the response
    documents = load_documents(data_dir)
    if not documents:
        raise HTTPException(
            status_code=422,
            detail="No readable .txt or .pdf files found in the data directory.",
        )

    chunks = chunk_documents(documents)

    # Run full pipeline (this re-does loading + chunking, but also embeds + saves)
    index, _ = ingest_pipeline()

    if index is None:
        raise HTTPException(status_code=500, detail="Ingestion failed.")

    return IngestResponse(
        message="Ingestion complete.",
        num_documents=len(documents),
        num_chunks=len(chunks),
    )
