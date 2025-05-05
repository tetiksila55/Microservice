from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO 
import numpy as np
import cv2 

app = FastAPI(
    title="YOLOv8 Object Detection API",
    version="1.0",
    description="Upload an image and receive detected objects."
)

model = YOLO('yolov8n.pt')

@app.get("/")
async def root():
    return {"message": "YOLOv8 Object Detection Service is up and running!"}

@app.post("/predict", summary="Detect object in an image", response_description="Detection results")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File is not an image.")
    
    contents = await file.read()
    nparr = np.frombuffer(contents, np.unit8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    
    results = model.predict(source=img, save=False, imgsz=640, conf=0.5)

    detections = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls)
            detections.append({
                "class_id": cls_id,
                "class_name": model.names[cls_id],
                "confidence": float(box.conf),
                "bbox": {
                    "xmin": float(box.xyxy[0][0]),
                    "ymin": float(box.xyxy[0][1]),
                    "xmax": float(box.xyxy[0][2]),
                    "ymax": float(box.xyxy[0][3]),
                }
            })

    return JSONResponse(content={"detections": detections})