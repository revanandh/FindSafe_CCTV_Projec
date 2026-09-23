from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import cv2
import numpy as np
import uuid
from datetime import datetime

from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
UPLOADS = BASE / "uploads"
RESULTS = BASE / "results"
UPLOADS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

app = FastAPI(title="FindSafe - Missing Person CCTV Prototype")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
app.mount("/results", StaticFiles(directory=RESULTS), name="results")
templates = Jinja2Templates(directory=BASE / "templates")

model = YOLO("yolo11n.pt")
jobs = {}

def color_histogram(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist

def compare_hist(a, b):
    return float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL))

def get_reference_hist(path):
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError("Invalid reference image")
    return color_histogram(img)

def process_video(video_path, reference_path, camera_name, location, threshold=0.55):
    job_id = str(uuid.uuid4())[:8]
    job_dir = RESULTS / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    ref_hist = get_reference_hist(reference_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Could not open CCTV video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_no = 0
    candidates = []
    sample_every = max(1, int(fps * 0.5))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        if frame_no % sample_every != 0:
            continue

        results = model.predict(frame, classes=[0], conf=0.35, verbose=False)
        result = results[0]
        if result.boxes is None:
            continue

        for box in result.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            score = compare_hist(ref_hist, color_histogram(crop))
            if score >= threshold:
                timestamp = frame_no / fps
                filename = f"candidate_{len(candidates)+1:03d}.jpg"
                out = frame.copy()
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(out, f"Possible match: {score:.2f}",
                            (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2)
                cv2.imwrite(str(job_dir / filename), out)
                candidates.append({
                    "camera": camera_name,
                    "location": location,
                    "time_seconds": round(timestamp, 2),
                    "score": round(score, 3),
                    "image": f"/results/{job_id}/{filename}",
                })

    cap.release()
    jobs[job_id] = {
        "job_id": job_id,
        "camera": camera_name,
        "location": location,
        "total_frames": total,
        "candidates": candidates,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "note": "Candidate scores are appearance-similarity signals, not identity verification."
    }
    return jobs[job_id]

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/search")
async def search(
    reference: UploadFile = File(...),
    cctv_video: UploadFile = File(...),
    camera_name: str = Form("Camera-01"),
    location: str = Form("Unknown"),
    threshold: float = Form(0.55),
):
    job_id = str(uuid.uuid4())[:8]
    ref_path = UPLOADS / f"{job_id}_reference_{reference.filename}"
    video_path = UPLOADS / f"{job_id}_video_{cctv_video.filename}"
    ref_path.write_bytes(await reference.read())
    video_path.write_bytes(await cctv_video.read())

    try:
        return JSONResponse(process_video(
            video_path, ref_path, camera_name, location,
            max(0.1, min(float(threshold), 0.95))
        ))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    return JSONResponse(jobs[job_id])
