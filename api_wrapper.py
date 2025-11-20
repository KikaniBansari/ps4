# api_wrapper.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import uuid
import os
from typing import List, Optional
import pss  # your existing backend module (pss.py)

API_KEY_EXPECTED = "test123"  # keep same or load from env in production
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="PS S Wrapper")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock this to your pages domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

def check_api_key(key: Optional[str]):
    if key != API_KEY_EXPECTED:
        raise HTTPException(status_code=401, detail="Invalid API key")

@app.post("/api/upload")
async def upload(files: List[UploadFile] = File(...), x_api_key: Optional[str] = Header(None)):
    check_api_key(x_api_key)
    saved = []
    for up in files:
        file_id = str(uuid.uuid4())
        dest = os.path.join(UPLOAD_DIR, f"{file_id}__{up.filename}")
        with open(dest, "wb") as f:
            shutil.copyfileobj(up.file, f)
        saved.append({"id": file_id, "name": up.filename, "path": dest})
    # Trigger backend ingestion - this assumes you add pss.start_job(files) that returns job_id
    try:
        job_id = pss.start_job([s["path"] for s in saved])
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    return {"job_id": job_id, "files": [{"name": s["name"], "id": s["id"]} for s in saved], "status": "started"}

@app.get("/api/job_status")
def job_status(job_id: str, x_api_key: Optional[str] = Header(None)):
    check_api_key(x_api_key)
    try:
        status, files = pss.job_status(job_id)  # expect (status_str, [{name, status, summary}])
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    return {"status": status, "files": files}

@app.post("/api/scrape")
def scrape(payload: dict, x_api_key: Optional[str] = Header(None)):
    check_api_key(x_api_key)
    urls = payload.get("urls", [])
    if not isinstance(urls, list):
        raise HTTPException(status_code=400, detail="urls must be a list")
    try:
        job_id = pss.scrape_urls(urls)  # expect this function to return a job id
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    return {"job_id": job_id, "status": "started"}

@app.post("/api/ask")
def ask(payload: dict, x_api_key: Optional[str] = Header(None)):
    check_api_key(x_api_key)
    q = payload.get("question") or payload.get("query")
    if not q:
        raise HTTPException(status_code=400, detail="question required")
    try:
        answer = pss.answer_question(q)  # expect a single string answer
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    return {"answer": answer}

@app.get("/api/download")
def download(x_api_key: Optional[str] = Header(None)):
    check_api_key(x_api_key)
    # pss.get_submission_zip() should return path to zip file
    try:
        zip_path = pss.get_submission_zip()
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    if not os.path.exists(zip_path):
        raise HTTPException(status_code=404, detail="Zip not found")
    return FileResponse(zip_path, media_type="application/zip", filename=os.path.basename(zip_path))

@app.post("/api/push_memory")
def push_memory(payload: dict = {}, x_api_key: Optional[str] = Header(None)):
    check_api_key(x_api_key)
    try:
        res = pss.push_memory()
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    return {"status": "ok", "detail": res}
