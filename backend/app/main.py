from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from pathlib import Path
from ultralytics import YOLO
import cv2

# Configuration paths
UPLOAD_DIR = "temp"
OUTPUT_DIR = "outputs"
STATIC_DIR = "static"

# Create directories if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(title="AI Vision Detection API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change to specific domains in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model
try:
    model = YOLO("app/yolov8n.pt")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the HTML interface"""
    try:
        with open(os.path.join(STATIC_DIR, "index.html"), "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return """
        <html>
            <body style="font-family: Arial; text-align: center; padding: 50px;">
                <h1>AI Vision Detection API</h1>
                <p>⚠️ Interface HTML not found.</p>
                <p>Make sure <code>static/index.html</code> exists.</p>
                <p><a href="/docs">API Documentation</a></p>
            </body>
        </html>
        """

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    """
    Upload an image and get YOLO detection results
    """
    if model is None:
        raise HTTPException(status_code=500, detail="YOLO model not loaded")
    
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Save uploaded file temporarily
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Run YOLO detection
        results = model(file_path)
        
        # Get the result
        result = results[0]
        
        # Plot the results (annotated image)
        annotated_frame = result.plot()
        
        # Save output image
        output_filename = f"result_{Path(file.filename).stem}.jpg"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        cv2.imwrite(output_path, annotated_frame)
        
        # Prepare response with statistics
        detections = []
        for box in result.boxes:
            detection = {
                "class": result.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": {
                    "x1": float(box.xyxy[0][0]),
                    "y1": float(box.xyxy[0][1]),
                    "x2": float(box.xyxy[0][2]),
                    "y2": float(box.xyxy[0][3])
                }
            }
            detections.append(detection)
        
        return {
            "result_image": f"outputs/{output_filename}",
            "detections": detections,
            "total_detections": len(detections),
            "model_used": "YOLOv8n"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up temporary file
        if os.path.exists(file_path):
            os.remove(file_path)

@app.get("/detect/history/")
async def get_history():
    """Get list of recent detections"""
    try:
        files = os.listdir(OUTPUT_DIR)
        files.sort(key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)), reverse=True)
        return {
            "recent_detections": files[:10]  # Return last 10 detections
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount outputs directory for serving images
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)