"""FastAPI backend: serve processed clips, run the pipeline on uploads, and expose
the language half (events JSON -> commentary) as its own endpoint.

    uvicorn server.app:app --reload --port 8000
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import uuid

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from commentator.generation.generator import CommentaryGenerator, dedupe
from commentator.schema import CommentaryLine, Event, MatchInfo

logging.basicConfig(level=logging.INFO)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIPS = os.environ.get("CLIPS_DIR", os.path.join(ROOT, "outputs", "clips"))
DEMO = os.environ.get("DEMO_DIR", os.path.join(ROOT, "web", "public", "demo"))
DIST = os.path.join(ROOT, "web", "dist")
UPLOADS = os.path.join(ROOT, "outputs", "uploads")
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "200"))
ALLOW_UPLOADS = os.environ.get("ALLOW_UPLOADS", "1") == "1"

os.makedirs(CLIPS, exist_ok=True)
app = FastAPI(title="AI Football Commentary", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
JOBS: dict[str, dict] = {}
_lock = threading.Lock()


def _clip_dirs() -> dict[str, str]:
    out = {}
    for base in (DEMO, CLIPS):
        if os.path.isdir(base):
            for d in sorted(os.listdir(base)):
                if os.path.exists(os.path.join(base, d, "track.json")):
                    out[d] = os.path.join(base, d)
    return out


@app.get("/api/health")
def health():
    from commentator.generation import llm

    return {"ok": True, "llm": llm.available(), "uploads": ALLOW_UPLOADS,
            "spotter": os.path.exists(os.path.join(ROOT, "models", "spotter.pt"))}


@app.get("/api/clips")
def list_clips():
    items = []
    for cid, d in _clip_dirs().items():
        with open(os.path.join(d, "track.json")) as f:
            tr = json.load(f)
        items.append({"id": cid, "title": tr.get("stats", {}).get("title") or cid, "duration": tr["duration"],
                      "n_events": len(tr["events"]), "n_lines": len(tr["lines"]),
                      "has_video": bool(tr.get("video")), "has_audio": bool(tr.get("audio_track"))})
    return items


@app.get("/api/clips/{cid}")
def get_clip(cid: str):
    d = _clip_dirs().get(cid)
    if not d:
        raise HTTPException(404, "clip not found")
    return FileResponse(os.path.join(d, "track.json"), media_type="application/json")


@app.get("/media/{cid}/{path:path}")
def media(cid: str, path: str):
    d = _clip_dirs().get(cid)
    if not d:
        raise HTTPException(404)
    full = os.path.realpath(os.path.join(d, path))
    if not full.startswith(os.path.realpath(d) + os.sep) or not os.path.isfile(full):
        raise HTTPException(404)
    return FileResponse(full)


class CommentaryRequest(BaseModel):
    events: list[Event]
    match: MatchInfo = MatchInfo()
    language: str = "en"
    use_llm: bool = False


@app.post("/api/commentary", response_model=list[CommentaryLine])
def commentary(req: CommentaryRequest):
    """The language half on its own: structured events in, timed lines out."""
    gen = CommentaryGenerator(use_llm=req.use_llm, language=req.language)
    return gen.generate(dedupe(req.events), req.match)


def _run_job(job_id: str, video: str, events_path: str | None, match: MatchInfo, language: str):
    from commentator import pipeline

    try:
        JOBS[job_id]["status"] = "running"
        tr = pipeline.run_video(video, events_path, out_root=CLIPS, match=match, language=language)
        # keep the source video beside the track so the player can load it
        out = os.path.join(CLIPS, tr.clip_id)
        dst = os.path.join(out, "source" + os.path.splitext(video)[1])
        shutil.copy(video, dst)
        tr.video = os.path.basename(dst)  # the player overlays commentary.m4a itself
        if os.path.exists(os.path.join(out, "narrated.mp4")):
            tr.stats["narrated_video"] = "narrated.mp4"  # muxed file, for download
        with open(os.path.join(out, "track.json"), "w") as f:
            f.write(tr.model_dump_json(indent=1))
        JOBS[job_id].update(status="done", clip_id=tr.clip_id)
    except Exception as e:
        logging.exception("job failed")
        JOBS[job_id].update(status="error", error=str(e))


@app.post("/api/process")
async def process(background: BackgroundTasks, video: UploadFile = File(...),
                  events: UploadFile | None = File(None), home: str = Form("home"),
                  away: str = Form("away"), language: str = Form("en")):
    if not ALLOW_UPLOADS:
        raise HTTPException(403, "uploads disabled on this deployment")
    with _lock:
        if sum(j["status"] in ("queued", "running") for j in JOBS.values()) >= 2:
            raise HTTPException(429, "busy, try again shortly")
    job_id = uuid.uuid4().hex[:10]
    jd = os.path.join(UPLOADS, job_id)
    os.makedirs(jd, exist_ok=True)
    ext = os.path.splitext(video.filename or "clip.mp4")[1].lower() or ".mp4"
    if ext not in (".mp4", ".mov", ".mkv", ".webm", ".avi"):
        raise HTTPException(400, "unsupported video type")
    vpath = os.path.join(jd, f"clip-{job_id}{ext}")
    size = 0
    with open(vpath, "wb") as f:
        while chunk := await video.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_MB << 20:
                raise HTTPException(413, "video too large")
            f.write(chunk)
    epath = None
    if events is not None and events.filename:
        epath = os.path.join(jd, "events.json")
        raw = await events.read()
        try:
            [Event(**e) for e in json.loads(raw)]
        except Exception as e:
            raise HTTPException(400, f"invalid events JSON: {e}")
        with open(epath, "wb") as f:
            f.write(raw)
    JOBS[job_id] = {"status": "queued"}
    background.add_task(_run_job, job_id, vpath, epath, MatchInfo(home=home, away=away), language)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404)
    return JOBS[job_id]


if os.path.isdir(DIST):
    app.mount("/", StaticFiles(directory=DIST, html=True), name="web")
