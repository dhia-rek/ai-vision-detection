from fastapi import FastAPI, UploadFile, File
import shutil
from ultralytics import YOLO
import cv2

app = FastAPI()

model = YOLO("yolov8n.pt")

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model(file_path)

    img = results[0].plot()
    output_path = f"output_{file.filename}"
    cv2.imwrite(output_path, img)

    return {"result_image": output_path}