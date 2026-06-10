from fastapi import FastAPI, UploadFile, File, HTTPException, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import os
from ocr_engine import EfficientTextReader

app = FastAPI()

API_KEY = os.environ.get("OCR_API_KEY", "dev-key-change-in-production")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    return key

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Loaded once when container starts — reused for every request
reader = EfficientTextReader(mobile_mode=True)

@app.post("/ocr")
async def run_ocr(
    file: UploadFile = File(...),
    _: str = Security(verify_key),
):
    # Reject files over 10MB
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large, max 10MB")

    # Decode image
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    # Run your existing OCR logic
    processed = reader._preprocess(frame)
    result = reader.ocr.ocr(processed, cls=False)
    items = result[0] if result and result[0] else []
    roi_offset = reader._get_roi_offset(frame)

    detections = []
    for item in items:
        if len(item) < 2:
            continue
        if not (isinstance(item[1], tuple) and len(item[1]) >= 2):
            continue
        text, confidence = item[1][0], item[1][1]
        if confidence < 0.5:
            continue
        adjusted = []
        for point in item[0]:
            if len(point) >= 2:
                adjusted.append([
                    int(point[0] / reader.roi_scale + roi_offset[0]),
                    int(point[1] / reader.roi_scale + roi_offset[1]),
                ])
        detections.append({
            "text": text,
            "confidence": round(float(confidence), 3),
            "bbox": adjusted,
        })

    return {"detections": detections, "count": len(detections)}

@app.get("/health")
def health():
    return {"status": "ok"}