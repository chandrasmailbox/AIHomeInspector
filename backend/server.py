from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import uuid
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
import io
import base64
import torch
from transformers import CLIPProcessor, CLIPModel, pipeline
import tempfile
import asyncio
from concurrent.futures import ThreadPoolExecutor

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Global variables for models
clip_model = None
clip_processor = None
executor = ThreadPoolExecutor(max_workers=2)

# Initialize AI models
def load_models():
    global clip_model, clip_processor
    try:
        # Load CLIP model for defect classification
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        logging.info("AI models loaded successfully")
    except Exception as e:
        logging.error(f"Error loading models: {e}")

# Define Models
class DefectDetection(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    total_frames: int
    defects_found: List[Dict[str, Any]]
    summary: Dict[str, Any]

class DefectFrame(BaseModel):
    frame_number: int
    timestamp_in_video: float
    defects: List[Dict[str, Any]]
    confidence_score: float
    frame_image: str  # base64 encoded

# AI Detection Functions
def detect_cracks_opencv(image):
    """Detect cracks using OpenCV edge detection"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours that look like cracks (long, thin lines)
    crack_contours = []
    crack_boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Minimum area threshold
            # Check aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / min(w, h)
            if aspect_ratio > 3:  # Long, thin objects
                crack_contours.append(contour)
                crack_boxes.append((x, y, w, h))
    
    confidence = min(len(crack_contours) * 0.2, 1.0)
    return len(crack_contours) > 0, confidence, crack_contours, crack_boxes

def detect_water_damage(image):
    """Detect water damage using color analysis"""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Define ranges for water damage colors (browns, dark spots)
    lower_brown = np.array([10, 50, 20])
    upper_brown = np.array([20, 255, 200])
    
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 50])
    
    # Create masks
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)
    
    # Combine masks
    water_mask = cv2.bitwise_or(brown_mask, dark_mask)
    
    # Find contours for bounding boxes
    contours, _ = cv2.findContours(water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get bounding boxes for water damage areas
    damage_boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Minimum area for water damage
            x, y, w, h = cv2.boundingRect(contour)
            damage_boxes.append((x, y, w, h))
    
    # Calculate percentage of affected area
    total_pixels = image.shape[0] * image.shape[1]
    affected_pixels = cv2.countNonZero(water_mask)
    damage_percentage = affected_pixels / total_pixels
    
    has_damage = damage_percentage > 0.05  # 5% threshold
    confidence = min(damage_percentage * 5, 1.0)
    
    return has_damage, confidence, damage_boxes

def detect_defects_with_clip(image, clip_model, clip_processor):
    """Use CLIP model to classify various defects"""
    try:
        # Define defect categories to check
        defect_texts = [
            "mold on wall",
            "paint peeling off wall", 
            "water stains on ceiling",
            "damaged wood",
            "rust on metal",
            "broken tiles",
            "damaged flooring"
        ]
        
        # Prepare inputs
        inputs = clip_processor(
            text=defect_texts, 
            images=image, 
            return_tensors="pt", 
            padding=True
        )
        
        # Get predictions
        with torch.no_grad():
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        
        # Get results
        results = []
        for i, defect_type in enumerate(defect_texts):
            confidence = float(probs[0][i])
            if confidence > 0.15:  # Threshold for detection
                results.append({
                    "type": defect_type,
                    "confidence": confidence
                })
        
        return results
    except Exception as e:
        logging.error(f"CLIP detection error: {e}")
        return []

def analyze_frame(frame_data):
    """Analyze a single frame for defects"""
    frame_number, image_array = frame_data
    
    defects = []
    
    # 1. Crack detection with OpenCV
    has_cracks, crack_confidence, crack_contours = detect_cracks_opencv(image_array)
    if has_cracks:
        defects.append({
            "type": "cracks",
            "confidence": crack_confidence,
            "description": f"Potential cracks detected with {crack_confidence:.2f} confidence"
        })
    
    # 2. Water damage detection
    has_water_damage, water_confidence = detect_water_damage(image_array)
    if has_water_damage:
        defects.append({
            "type": "water_damage", 
            "confidence": water_confidence,
            "description": f"Water damage detected with {water_confidence:.2f} confidence"
        })
    
    # 3. CLIP-based defect detection
    if clip_model and clip_processor:
        clip_defects = detect_defects_with_clip(image_array, clip_model, clip_processor)
        defects.extend(clip_defects)
    
    # Calculate overall confidence
    overall_confidence = max([d["confidence"] for d in defects]) if defects else 0
    
    # Convert frame to base64 for frontend display
    pil_image = Image.fromarray(image_array)
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG')
    frame_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    return {
        "frame_number": frame_number,
        "defects": defects,
        "confidence_score": overall_confidence,
        "frame_image": frame_base64
    }

def extract_frames_from_video(video_bytes):
    """Extract frames from video for analysis"""
    try:
        # Save video to temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_file.write(video_bytes)
            temp_file_path = temp_file.name
        
        # Open video with OpenCV
        cap = cv2.VideoCapture(temp_file_path)
        
        frames = []
        frame_count = 0
        
        # Extract every 30th frame (for performance)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % 30 == 0:  # Sample every 30 frames
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append((frame_count, rgb_frame))
            
            frame_count += 1
        
        cap.release()
        os.unlink(temp_file_path)  # Clean up temp file
        
        return frames, frame_count
        
    except Exception as e:
        logging.error(f"Video processing error: {e}")
        raise HTTPException(status_code=400, detail=f"Error processing video: {e}")

async def process_video_async(video_bytes, filename):
    """Process video asynchronously"""
    # Extract frames
    frames, total_frames = extract_frames_from_video(video_bytes)
    
    if not frames:
        raise HTTPException(status_code=400, detail="No frames could be extracted from video")
    
    # Analyze frames in parallel
    loop = asyncio.get_event_loop()
    
    # Process frames in batches to avoid memory issues
    batch_size = 5
    all_results = []
    
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i+batch_size]
        tasks = [
            loop.run_in_executor(executor, analyze_frame, frame_data)
            for frame_data in batch
        ]
        batch_results = await asyncio.gather(*tasks)
        all_results.extend(batch_results)
    
    # Generate summary
    all_defects = []
    defect_types = set()
    high_confidence_frames = 0
    
    for result in all_results:
        all_defects.extend(result["defects"])
        for defect in result["defects"]:
            defect_types.add(defect["type"])
        if result["confidence_score"] > 0.7:
            high_confidence_frames += 1
    
    summary = {
        "total_defects_found": len(all_defects),
        "defect_types": list(defect_types),
        "frames_analyzed": len(frames),
        "high_confidence_detections": high_confidence_frames,
        "severity": "high" if high_confidence_frames > len(frames) * 0.3 else "medium" if high_confidence_frames > 0 else "low"
    }
    
    # Create detection record
    detection = DefectDetection(
        filename=filename,
        total_frames=total_frames,
        defects_found=all_results,
        summary=summary
    )
    
    # Save to database
    await db.inspections.insert_one(detection.dict())
    
    return detection

# API Routes
@api_router.get("/")
async def root():
    return {"message": "HomeInspector AI - Ready for video analysis"}

@api_router.post("/analyze-video", response_model=DefectDetection)
async def analyze_video(file: UploadFile = File(...)):
    """Analyze uploaded video for defects"""
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Only video files are supported")
    
    try:
        # Read video file
        video_bytes = await file.read()
        
        # Process video
        result = await process_video_async(video_bytes, file.filename)
        
        return result
        
    except Exception as e:
        logging.error(f"Video analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing video: {e}")

@api_router.get("/inspections", response_model=List[DefectDetection])
async def get_inspections():
    """Get all inspection results"""
    inspections = await db.inspections.find().to_list(100)
    return [DefectDetection(**inspection) for inspection in inspections]

@api_router.get("/inspection/{inspection_id}", response_model=DefectDetection)
async def get_inspection(inspection_id: str):
    """Get specific inspection result"""
    inspection = await db.inspections.find_one({"id": inspection_id})
    if not inspection:
        raise HTTPException(status_code=404, detail="Inspection not found")
    return DefectDetection(**inspection)

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_event():
    """Load AI models on startup"""
    load_models()

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
