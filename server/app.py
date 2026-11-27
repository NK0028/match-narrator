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
from fastapi.responses import FileResponse, StreamingResponse
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
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "4096"))  # full matches; lower it when hosting
VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi")
ALLOW_UPLOADS = os.environ.get("ALLOW_UPLOADS", "1") == "1"

os.makedirs(CLIPS, exist_ok=True)
app = FastAPI(title="AI Football Commentary", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
JOBS: dict[str, dict] = {}
LIVE: dict[str, "LiveSession"] = {}  # noqa: F821  (imported lazily: pulls in torch/ultralytics)
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

    return {"ok": True, "llm": llm.available(), "uploads": ALLOW_UPLOADS, "live": ALLOW_UPLOADS,
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


async def _save_upload(video: UploadFile, dst_no_ext: str) -> str:
    ext = os.path.splitext(video.filename or "clip.mp4")[1].lower() or ".mp4"
    if ext not in VIDEO_EXTS:
        raise HTTPException(400, "unsupported video type")
    os.makedirs(os.path.dirname(dst_no_ext), exist_ok=True)
    path, size = dst_no_ext + ext, 0
    with open(path, "wb") as f:
        while chunk := await video.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_MB << 20:
                f.close()
                os.remove(path)
                raise HTTPException(413, "video too large")
            f.write(chunk)
    return path


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
    vpath = await _save_upload(video, os.path.join(jd, f"clip-{job_id}"))
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


# ---- live mode ---------------------------------------------------------------------
def _new_live_session(sid, video, out_dir, match, language):  # patched in tests
    from commentator.live.session import LiveSession

    return LiveSession(sid, video, out_dir, match=match, language=language)


@app.post("/api/live")
async def live_start(video: UploadFile = File(...), home: str = Form("home"), away: str = Form("away"),
                     language: str = Form("en")):
    if not ALLOW_UPLOADS:
        raise HTTPException(403, "live mode disabled on this deployment")
    with _lock:
        if any(s.status in ("queued", "running") for s in LIVE.values()):
            raise HTTPException(429, "a live session is already running")
    sid = "live-" + uuid.uuid4().hex[:8]
    out = os.path.join(CLIPS, sid)
    vpath = await _save_upload(video, os.path.join(out, "source"))
    LIVE[sid] = _new_live_session(sid, vpath, out, MatchInfo(home=home, away=away, language=language),
                                  language).start()
    return {"session_id": sid}


def _live(sid: str):
    if sid not in LIVE:
        raise HTTPException(404, "no such live session")
    return LIVE[sid]


@app.get("/api/live/{sid}")
def live_snapshot(sid: str, since: int = 0):
    return _live(sid).snapshot(since)


@app.get("/api/live/{sid}/stream")
def live_stream(sid: str, since: int = 0):
    """Server-sent events. Replays lines from `since`, then streams new ones as they're final."""
    sess = _live(sid)

    def gen():
        sent, last_ready, last_stats = since, -1.0, None
        while True:
            with sess.cond:
                sess.cond.wait_for(lambda: len(sess.lines) > sent or sess.ready_t != last_ready
                                   or sess.status in ("done", "error"), timeout=15)
                lines = sess.lines[sent:]
                ready, status, stats = sess.ready_t, sess.status, sess.stats
                dur, kits, err, perf = sess.duration, sess.kits, sess.error, sess.perf
            for ln in lines:
                yield f"event: line\ndata: {ln.model_dump_json()}\n\n"
            sent += len(lines)
            if ready != last_ready:
                last_ready = ready
                yield f"event: progress\ndata: {json.dumps({'ready_t': ready, 'duration': dur, 'status': status, 'kits': kits})}\n\n"
            if stats != last_stats:
                last_stats = stats
                yield f"event: stats\ndata: {json.dumps(stats)}\n\n"
            if status == "error":
                yield f"event: error\ndata: {json.dumps({'error': err})}\n\n"
                return
            if status == "done":
                yield f"event: done\ndata: {json.dumps({'clip_id': sid, 'perf': perf})}\n\n"
                return
            if not lines and ready == last_ready:
                yield ": keep-alive\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if os.path.isdir(DIST):
    app.mount("/", StaticFiles(directory=DIST, html=True), name="web")
