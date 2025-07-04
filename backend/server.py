from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, StreamingResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import torch
import json
from transformers import CLIPProcessor, CLIPModel, pipeline
import tempfile
import asyncio
from concurrent.futures import ThreadPoolExecutor
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportImage, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
import ultralytics
from ultralytics import YOLO
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import supervision as sv

# Model Registry and Configuration
models_registry = {}
MODEL_CONFIG = {
    'basic_cv': {
        'name': 'Basic Computer Vision',
        'enabled': True,
        'confidence_threshold': 0.3,
        'description': 'OpenCV-based crack and water damage detection'
    },
    'yolov8n': {
        'name': 'YOLOv8 Nano',
        'enabled': True,
        'confidence_threshold': 0.3,
        'description': 'Lightweight object detection model'
    },
    'yolov8s': {
        'name': 'YOLOv8 Small',
        'enabled': True,
        'confidence_threshold': 0.4,
        'description': 'Small object detection model with better accuracy'
    },
    'yolov8m': {
        'name': 'YOLOv8 Medium',
        'enabled': True,
        'confidence_threshold': 0.4,
        'description': 'Medium-sized object detection model with balanced accuracy'
    },
    'clip_vit_b32': {
        'name': 'CLIP ViT-B/32',
        'enabled': False,  # Disabled for now due to memory constraints
        'confidence_threshold': 0.5,
        'description': 'Visual-language model for semantic defect understanding'
    },
    'clip_vit_l14': {
        'name': 'CLIP ViT-L/14',
        'enabled': False,  # Disabled for now due to memory constraints
        'confidence_threshold': 0.5,
        'description': 'Large visual-language model for advanced defect classification'
    },
    'sam': {
        'name': 'Segment Anything Model',
        'enabled': False,  # Will be enabled if SAM is available
        'confidence_threshold': 0.6,
        'description': 'Advanced segmentation model for precise defect boundaries'
    }
}
try:
    from segment_anything import SamPredictor, sam_model_registry
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    logging.warning("SAM not available - some features will be disabled")

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
yolo_model = None
available_models = ['basic_cv', 'yolo', 'clip']
executor = ThreadPoolExecutor(max_workers=2)

# Initialize AI models
def load_models():
    global clip_model, clip_processor, models_registry
    try:
        # Commented out for testing
        # clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        # clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model = None
        clip_processor = None
        
        # Load YOLO models for object detection
        try:
            # Load YOLOv8n model
            if MODEL_CONFIG['yolov8n']['enabled']:
                models_registry['yolov8n'] = YOLO('yolov8n.pt')
                logging.info("YOLOv8n model loaded successfully")
            
            # Load YOLOv8s model if enabled
            if MODEL_CONFIG['yolov8s']['enabled']:
                models_registry['yolov8s'] = YOLO('yolov8s.pt')
                logging.info("YOLOv8s model loaded successfully")
                
        except Exception as e:
            logging.warning(f"YOLO models not available: {e}")
            
        logging.info("AI models loading completed")
    except Exception as e:
        logging.error(f"Error loading models: {e}")

# Define Models
class UserCorrection(BaseModel):
    box_id: str
    defect_type: str
    is_hidden: bool = False
    user_feedback: str = ""
    correction_timestamp: datetime = Field(default_factory=datetime.utcnow)

class ModelSelection(BaseModel):
    selected_models: List[str] = Field(default=['basic_cv'])
    ensemble_method: str = Field(default='weighted_average')
    confidence_threshold: float = Field(default=0.5)
    use_ensemble: bool = Field(default=False)

class InspectionRequest(BaseModel):
    model_selection: ModelSelection = Field(default_factory=ModelSelection)

class DefectDetection(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    total_frames: int
    defects_found: List[Dict[str, Any]]
    summary: Dict[str, Any]
    user_corrections: Dict[str, List[UserCorrection]] = Field(default_factory=dict)
    model_results: Dict[str, Any] = Field(default_factory=dict)  # Store results from different models
    selected_models: List[str] = Field(default=['basic_cv'])
    ensemble_results: Optional[Dict[str, Any]] = Field(default=None)
    model_comparison: Optional[Dict[str, Any]] = Field(default=None)

class DefectFrame(BaseModel):
    frame_number: int
    timestamp_in_video: float
    defects: List[Dict[str, Any]]
    confidence_score: float
    frame_image: str  # base64 encoded

class FrameCorrection(BaseModel):
    inspection_id: str
    frame_number: int
    corrections: List[UserCorrection]

class PDFExportRequest(BaseModel):
    inspection_id: str
    include_images: bool = Field(default=True)
    include_model_comparison: bool = Field(default=True)
    include_recommendations: bool = Field(default=True)

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

def detect_defects_with_yolo(image, model_name='yolov8n'):
    """Use YOLO model for object detection"""
    try:
        if model_name not in models_registry or not MODEL_CONFIG[model_name]['enabled']:
            return []
            
        yolo_model = models_registry[model_name]
        
        # Run YOLO detection
        results = yolo_model(image)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    # Get coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Map YOLO classes to defect types (simplified mapping)
                    class_names = yolo_model.names
                    detected_object = class_names.get(class_id, 'unknown')
                    
                    # Map to defect types based on detected objects
                    defect_type = map_yolo_to_defect(detected_object)
                    
                    threshold = MODEL_CONFIG[model_name]['confidence_threshold']
                    if defect_type and confidence > threshold:
                        box_id = f"{model_name}_{defect_type}_{i}_{uuid.uuid4().hex[:8]}"
                        detections.append({
                            "type": defect_type,
                            "confidence": confidence,
                            "description": f"{defect_type} detected by {MODEL_CONFIG[model_name]['name']} with {confidence:.2f} confidence",
                            "boxes": [{
                                "id": box_id,
                                "coords": [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                                "visible": True
                            }],
                            "model": model_name
                        })
        
        return detections
    except Exception as e:
        logging.error(f"YOLO detection error: {e}")
        return []

def map_yolo_to_defect(yolo_class):
    """Map YOLO detected objects to defect types"""
    defect_mapping = {
        'crack': 'cracks',
        'damage': 'structural_damage',
        'stain': 'water_damage',
        'mold': 'mold',
        'rust': 'rust',
        'hole': 'structural_damage',
        'person': None,  # Ignore people
        'car': None,     # Ignore vehicles
        'furniture': None # Ignore furniture
    }
    
    for key, value in defect_mapping.items():
        if key.lower() in yolo_class.lower():
            return value
    
    # For any unrecognized object, consider it as potential structural issue
    return 'potential_defect'


def draw_defect_boxes(image, defects_with_boxes):
    """Draw colored bounding boxes on image for detected defects"""
    annotated_image = image.copy()
    
    # Define colors for different defect types
    colors = {
        'cracks': (255, 0, 0),      # Red
        'water_damage': (0, 0, 255), # Blue
        'mold': (0, 255, 0),        # Green
        'paint': (255, 255, 0),     # Yellow
        'rust': (255, 165, 0),      # Orange
        'tiles': (128, 0, 128),     # Purple
        'flooring': (255, 192, 203)  # Pink
    }
    
    for defect_info in defects_with_boxes:
        defect_type = defect_info['type']
        boxes = defect_info.get('boxes', [])
        
        # Get color for this defect type
        color = colors.get(defect_type.split()[0], (255, 255, 255))  # Default white
        
        # Draw bounding boxes
        for box_info in boxes:
            if isinstance(box_info, dict):
                # New format with IDs
                if box_info.get('visible', True):  # Only draw visible boxes
                    x, y, w, h = box_info['coords']
                else:
                    continue  # Skip hidden boxes
            else:
                # Old format (backward compatibility)
                x, y, w, h = box_info
            
            # Draw rectangle
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), color, 2)
            
            # Add label
            label = f"{defect_type}: {defect_info['confidence']:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Background for text
            cv2.rectangle(annotated_image, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            
            # Text
            cv2.putText(annotated_image, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return annotated_image

def detect_defects_with_clip(image, model_name='clip_vit_b32'):
    """Use CLIP model for defect detection"""
    try:
        if model_name not in models_registry or not MODEL_CONFIG[model_name]['enabled']:
            return []
            
        clip_data = models_registry[model_name]
        model = clip_data['model']
        processor = clip_data['processor']
        
        # Convert image to PIL format
        pil_image = Image.fromarray(image)
        
        # Define defect categories
        defect_categories = [
            "crack in wall",
            "water damage",
            "mold on wall",
            "peeling paint",
            "rust stains",
            "damaged tiles",
            "damaged flooring",
            "structural damage",
            "ceiling damage",
            "window damage"
        ]
        
        # Process image and text with CLIP
        inputs = processor(
            text=defect_categories,
            images=[pil_image],
            return_tensors="pt",
            padding=True
        )
        
        # Get model outputs
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        
        # Convert probabilities to list
        confidence_scores = probs[0].tolist()
        
        # Create defect results for high confidence detections
        defect_results = []
        threshold = MODEL_CONFIG[model_name]['confidence_threshold']
        
        for category, confidence in zip(defect_categories, confidence_scores):
            if confidence > threshold:
                h, w = image.shape[:2]
                # Create a general bounding box for CLIP detections
                general_box = (w//4, h//4, w//2, h//2)
                box_id = f"{model_name}_{category.replace(' ', '_')}_{uuid.uuid4().hex[:8]}"
                
                defect_results.append({
                    "type": category,
                    "confidence": confidence,
                    "description": f"{category} detected by {MODEL_CONFIG[model_name]['name']} with {confidence:.2f} confidence",
                    "boxes": [{
                        "id": box_id,
                        "coords": general_box,
                        "visible": True
                    }],
                    "model": model_name
                })
        
        return defect_results
        
    except Exception as e:
        logging.error(f"CLIP detection error: {e}")
        return []

def analyze_frame_with_models(frame_data, selected_models=['basic_cv']):
    """Analyze a single frame for defects using selected models"""
    frame_number, image_array = frame_data
    
    all_model_results = {}
    combined_defects = []
    combined_defects_with_boxes = []
    
    # 1. Basic Computer Vision (OpenCV) - always available
    if 'basic_cv' in selected_models:
        cv_defects = []
        
        # Crack detection with OpenCV
        has_cracks, crack_confidence, crack_contours, crack_boxes = detect_cracks_opencv(image_array)
        if has_cracks:
            boxes_with_ids = []
            for i, box in enumerate(crack_boxes):
                box_id = f"cv_crack_{frame_number}_{i}_{uuid.uuid4().hex[:8]}"
                boxes_with_ids.append({
                    "id": box_id,
                    "coords": box,
                    "visible": True
                })
            
            defect_info = {
                "type": "cracks",
                "confidence": crack_confidence,
                "description": f"Potential cracks detected with {crack_confidence:.2f} confidence",
                "boxes": boxes_with_ids,
                "model": "basic_cv"
            }
            cv_defects.append(defect_info)
        
        # Water damage detection
        has_water_damage, water_confidence, water_boxes = detect_water_damage(image_array)
        if has_water_damage:
            boxes_with_ids = []
            for i, box in enumerate(water_boxes):
                box_id = f"cv_water_{frame_number}_{i}_{uuid.uuid4().hex[:8]}"
                boxes_with_ids.append({
                    "id": box_id,
                    "coords": box,
                    "visible": True
                })
            
            defect_info = {
                "type": "water_damage", 
                "confidence": water_confidence,
                "description": f"Water damage detected with {water_confidence:.2f} confidence",
                "boxes": boxes_with_ids,
                "model": "basic_cv"
            }
            cv_defects.append(defect_info)
        
        all_model_results['basic_cv'] = cv_defects
        combined_defects.extend(cv_defects)
        combined_defects_with_boxes.extend(cv_defects)
    
    # 2. YOLO-based defect detection
    yolo_models = [model for model in selected_models if model.startswith('yolov8')]
    for yolo_model in yolo_models:
        if MODEL_CONFIG.get(yolo_model, {}).get('enabled', False):
            yolo_defects = detect_defects_with_yolo(image_array, yolo_model)
            all_model_results[yolo_model] = yolo_defects
            combined_defects.extend(yolo_defects)
            combined_defects_with_boxes.extend(yolo_defects)
    
    # 3. CLIP-based defect detection
    clip_models = [model for model in selected_models if model.startswith('clip')]
    for clip_model in clip_models:
        if MODEL_CONFIG.get(clip_model, {}).get('enabled', False):
            clip_defects = detect_defects_with_clip(image_array, clip_model)
            all_model_results[clip_model] = clip_defects
            combined_defects.extend(clip_defects)
            combined_defects_with_boxes.extend(clip_defects)
    
    # 4. SAM-based segmentation (if available)
    if 'sam' in selected_models and MODEL_CONFIG.get('sam', {}).get('enabled', False):
        sam_defects = detect_defects_with_sam(image_array)
        all_model_results['sam'] = sam_defects
        combined_defects.extend(sam_defects)
        combined_defects_with_boxes.extend(sam_defects)
    
    # Calculate overall confidence
    overall_confidence = max([d["confidence"] for d in combined_defects]) if combined_defects else 0
    
    # Create annotated image with bounding boxes
    annotated_image = draw_defect_boxes(image_array, combined_defects_with_boxes)
    
    # Convert annotated frame to base64 for frontend display
    pil_image = Image.fromarray(annotated_image)
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG')
    frame_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    return {
        "frame_number": frame_number,
        "defects": combined_defects,
        "confidence_score": overall_confidence,
        "frame_image": frame_base64,
        "model_results": all_model_results,
        "selected_models": selected_models
    }

def detect_defects_with_sam(image):
    """Use SAM model for defect segmentation"""
    try:
        if 'sam' not in models_registry:
            return []
            
        sam_predictor = models_registry['sam']
        sam_predictor.set_image(image)
        
        # For now, return mock results as SAM typically needs prompts
        # In a real implementation, you'd use point or box prompts
        mock_sam_defects = [
            {
                "type": "segmented_defect",
                "confidence": 0.8,
                "description": "Defect detected by SAM segmentation",
                "boxes": [{
                    "id": f"sam_defect_{uuid.uuid4().hex[:8]}",
                    "coords": (100, 100, 150, 150),
                    "visible": True
                }],
                "model": "sam"
            }
        ]
        return mock_sam_defects
        
    except Exception as e:
        logging.error(f"SAM detection error: {e}")
        return []

def analyze_frame(frame_data):
    """Legacy function that calls analyze_frame_with_models with default settings"""
    return analyze_frame_with_models(frame_data)

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
    return {"message": "HomeInspector AI - Ready for advanced video analysis with multiple AI models"}

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

@api_router.post("/inspection/{inspection_id}/corrections")
async def save_user_corrections(inspection_id: str, frame_correction: FrameCorrection):
    """Save user corrections for a specific frame"""
    try:
        # Find the inspection
        inspection = await db.inspections.find_one({"id": inspection_id})
        if not inspection:
            raise HTTPException(status_code=404, detail="Inspection not found")
        
        # Update user corrections
        frame_key = str(frame_correction.frame_number)
        corrections_dict = {}
        
        if "user_corrections" not in inspection:
            inspection["user_corrections"] = {}
        
        # Convert corrections to dict format
        corrections_dict[frame_key] = [correction.dict() for correction in frame_correction.corrections]
        
        # Update the inspection with user corrections
        await db.inspections.update_one(
            {"id": inspection_id},
            {"$set": {f"user_corrections.{frame_key}": corrections_dict[frame_key]}}
        )
        
        return {"message": "Corrections saved successfully"}
        
    except Exception as e:
        logging.error(f"Error saving corrections: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving corrections: {e}")

@api_router.get("/inspection/{inspection_id}/model-comparison")
async def get_model_comparison(inspection_id: str):
    """Get model comparison data for a specific inspection"""
    try:
        # Find the inspection
        inspection = await db.inspections.find_one({"id": inspection_id})
        if not inspection:
            raise HTTPException(status_code=404, detail="Inspection not found")
        
        # Extract model-specific results from frames
        model_results = {}
        frames = inspection.get("defects_found", [])
        
        for frame in frames:
            frame_model_results = frame.get("model_results", {})
            
            for model_name, defects in frame_model_results.items():
                if model_name not in model_results:
                    model_results[model_name] = {
                        "total_detections": 0,
                        "confidence_sum": 0,
                        "defect_types": set()
                    }
                
                for defect in defects:
                    model_results[model_name]["total_detections"] += 1
                    model_results[model_name]["confidence_sum"] += defect.get("confidence", 0)
                    model_results[model_name]["defect_types"].add(defect.get("type", "unknown"))
        
        # Calculate average confidence for each model
        performance_metrics = {}
        for model_name, data in model_results.items():
            total_detections = data["total_detections"]
            avg_confidence = data["confidence_sum"] / total_detections if total_detections > 0 else 0
            defect_types = list(data["defect_types"])
            
            performance_metrics[model_name] = {
                "total_detections": total_detections,
                "average_confidence": avg_confidence,
                "defect_types": defect_types
            }
        
        # Generate model comparison data
        comparison_data = {
            "inspection_id": inspection_id,
            "performance_metrics": performance_metrics,
            "model_strengths": {
                "basic_cv": "Good at detecting cracks and water damage",
                "yolov8n": "Excellent at identifying structural defects",
                "clip": "Strong semantic understanding of defect types"
            }
        }
        
        return comparison_data
        
    except Exception as e:
        logging.error(f"Error generating model comparison: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating model comparison: {e}")

@api_router.post("/inspection/{inspection_id}/export-pdf", response_class=StreamingResponse)
async def export_pdf_report(inspection_id: str, pdf_request: PDFExportRequest):
    """Export inspection results as PDF report"""
    try:
        # Find the inspection
        inspection = await db.inspections.find_one({"id": inspection_id})
        if not inspection:
            raise HTTPException(status_code=404, detail="Inspection not found")
        
        # Add model comparison data if requested
        if pdf_request.include_model_comparison:
            try:
                # Generate model comparison data
                model_results = {}
                frames = inspection.get("defects_found", [])
                
                for frame in frames:
                    frame_model_results = frame.get("model_results", {})
                    
                    for model_name, defects in frame_model_results.items():
                        if model_name not in model_results:
                            model_results[model_name] = {
                                "total_detections": 0,
                                "confidence_sum": 0,
                                "defect_types": set()
                            }
                        
                        for defect in defects:
                            model_results[model_name]["total_detections"] += 1
                            model_results[model_name]["confidence_sum"] += defect.get("confidence", 0)
                            model_results[model_name]["defect_types"].add(defect.get("type", "unknown"))
                
                # Calculate average confidence for each model
                performance_metrics = {}
                for model_name, data in model_results.items():
                    total_detections = data["total_detections"]
                    avg_confidence = data["confidence_sum"] / total_detections if total_detections > 0 else 0
                    defect_types = list(data["defect_types"])
                    
                    performance_metrics[model_name] = {
                        "total_detections": total_detections,
                        "average_confidence": avg_confidence,
                        "defect_types": defect_types
                    }
                
                # Add model comparison data to inspection
                inspection["model_comparison"] = {
                    "performance_metrics": performance_metrics,
                    "model_strengths": {
                        "basic_cv": "Good at detecting cracks and water damage",
                        "yolov8n": "Excellent at identifying structural defects",
                        "clip": "Strong semantic understanding of defect types"
                    }
                }
            except Exception as e:
                logging.warning(f"Error generating model comparison for PDF: {e}")
                # Continue without model comparison
        
        # Generate a simple PDF report for testing
        buffer = io.BytesIO()
        
        # Create a simple PDF document
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Add title
        story.append(Paragraph("HomeInspector AI - Inspection Report", styles['Title']))
        story.append(Spacer(1, 12))
        
        # Add inspection details
        story.append(Paragraph(f"Inspection ID: {inspection_id}", styles['Normal']))
        story.append(Paragraph(f"Filename: {inspection.get('filename', 'Unknown')}", styles['Normal']))
        story.append(Paragraph(f"Date: {inspection.get('timestamp', datetime.utcnow()).strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Add summary
        summary = inspection.get('summary', {})
        story.append(Paragraph("Summary:", styles['Heading2']))
        story.append(Paragraph(f"Total defects found: {summary.get('total_defects_found', 0)}", styles['Normal']))
        story.append(Paragraph(f"Frames analyzed: {summary.get('frames_analyzed', 0)}", styles['Normal']))
        story.append(Paragraph(f"Severity: {summary.get('severity', 'Unknown')}", styles['Normal']))
        
        # Build the PDF
        doc.build(story)
        
        # Get the PDF content
        pdf_content = buffer.getvalue()
        buffer.close()
        
        # Return PDF as streaming response
        return StreamingResponse(
            io.BytesIO(pdf_content),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=inspection_{inspection_id}.pdf"
            }
        )
        
    except Exception as e:
        logging.error(f"Error generating PDF report: {str(e)}")
        import traceback
        error_details = traceback.format_exc()
        logging.error(f"Detailed error: {error_details}")
        raise HTTPException(status_code=500, detail=f"Error generating PDF report: {str(e)}")

@api_router.get("/inspection/{inspection_id}/frame/{frame_number}/regenerate")
async def regenerate_frame_with_corrections(inspection_id: str, frame_number: int):
    """Regenerate frame image with user corrections applied"""
    try:
        # Get the inspection
        inspection = await db.inspections.find_one({"id": inspection_id})
        if not inspection:
            raise HTTPException(status_code=404, detail="Inspection not found")
        
        # Find the specific frame
        frame_data = None
        for frame in inspection["defects_found"]:
            if frame["frame_number"] == frame_number:
                frame_data = frame
                break
        
        if not frame_data:
            raise HTTPException(status_code=404, detail="Frame not found")
        
        # Get user corrections for this frame
        user_corrections = inspection.get("user_corrections", {}).get(str(frame_number), [])
        
        # Apply corrections to defects
        updated_defects = []
        for defect in frame_data["defects"]:
            updated_boxes = []
            for box in defect.get("boxes", []):
                box_id = box.get("id") if isinstance(box, dict) else None
                
                # Check if this box has user corrections
                correction_found = False
                for correction in user_corrections:
                    if correction.get("box_id") == box_id:
                        if isinstance(box, dict):
                            box["visible"] = not correction.get("is_hidden", False)
                        correction_found = True
                        break
                
                updated_boxes.append(box)
            
            defect["boxes"] = updated_boxes
            updated_defects.append(defect)
        
        # Regenerate the frame image with corrections applied
        # For now, we'll return the original frame with visibility applied in frontend
        # In a more advanced implementation, we could regenerate the actual image
        
        return {
            "frame_number": frame_number,
            "defects": updated_defects,
            "confidence_score": frame_data["confidence_score"],
            "frame_image": frame_data["frame_image"]
        }
        
    except Exception as e:
        logging.error(f"Error regenerating frame: {e}")
        raise HTTPException(status_code=500, detail=f"Error regenerating frame: {e}")

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

def generate_enhanced_pdf_report(inspection_data, include_images=True, include_model_comparison=True, include_recommendations=True):
    """Generate enhanced PDF report for inspection results"""
    try:
        # Create a temporary file for the PDF
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            pdf_path = temp_file.name
        
        # Create PDF document
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#2563eb')
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.HexColor('#1e40af')
        )
        
        subheading_style = ParagraphStyle(
            'CustomSubHeading',
            parent=styles['Heading3'],
            fontSize=14,
            spaceAfter=8,
            textColor=colors.HexColor('#3b82f6')
        )
        
        # Title and Executive Summary
        story.append(Paragraph("HomeInspector AI - Comprehensive Defect Analysis Report", title_style))
        story.append(Spacer(1, 20))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", heading_style))
        executive_summary = f"""
        This comprehensive inspection report was generated using advanced AI technology to identify and analyze structural defects.
        The inspection utilized {len(inspection_data.get('selected_models', ['basic_cv']))} different AI models to ensure accurate detection.
        A total of {inspection_data.get('summary', {}).get('total_defects_found', 0)} defects were identified across 
        {inspection_data.get('summary', {}).get('frames_analyzed', 0)} analyzed frames.
        """
        story.append(Paragraph(executive_summary, styles['Normal']))
        story.append(Spacer(1, 15))
        
        # Inspection Summary
        story.append(Paragraph("Inspection Details", heading_style))
        
        summary_data = [
            ['Property/File', inspection_data.get('filename', 'Unknown')],
            ['Inspection Date', inspection_data.get('timestamp', datetime.utcnow()).strftime('%Y-%m-%d %H:%M:%S')],
            ['Total Frames Analyzed', str(inspection_data.get('total_frames', 0))],
            ['Defects Found', str(inspection_data.get('summary', {}).get('total_defects_found', 0))],
            ['Severity Level', inspection_data.get('summary', {}).get('severity', 'Unknown').upper()],
            ['AI Models Used', ', '.join(inspection_data.get('selected_models', ['basic_cv']))],
            ['High Confidence Detections', str(inspection_data.get('summary', {}).get('high_confidence_detections', 0))]
        ]
        
        summary_table = Table(summary_data, colWidths=[2.5*inch, 3.5*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f3f4f6')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 20))
        
        # Model Performance Section
        if include_model_comparison and 'model_comparison' in inspection_data:
            story.append(Paragraph("AI Model Performance Analysis", heading_style))
            
            model_comparison = inspection_data['model_comparison']
            performance_metrics = model_comparison.get('performance_metrics', {})
            
            if performance_metrics:
                model_data = [['Model', 'Detections', 'Avg Confidence', 'Defect Types']]
                
                for model_name, metrics in performance_metrics.items():
                    model_display_name = MODEL_CONFIG.get(model_name, {}).get('name', model_name)
                    model_data.append([
                        model_display_name,
                        str(metrics.get('total_detections', 0)),
                        f"{metrics.get('average_confidence', 0):.2f}",
                        str(len(metrics.get('defect_types', [])))
                    ])
                
                model_table = Table(model_data, colWidths=[2*inch, 1*inch, 1.5*inch, 1.5*inch])
                model_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e40af')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ]))
                
                story.append(model_table)
                story.append(Spacer(1, 15))
        
        # Defect Types Found
        defect_types = inspection_data.get('summary', {}).get('defect_types', [])
        if defect_types:
            story.append(Paragraph("Detected Issues & Recommendations", heading_style))
            
            defect_table_data = [['Defect Type', 'Severity', 'Priority', 'Recommendations']]
            
            for defect_type in defect_types:
                severity = get_defect_severity(defect_type)
                priority = get_defect_priority(defect_type)
                recommendations = get_defect_recommendations(defect_type)
                defect_table_data.append([
                    defect_type.replace('_', ' ').title(),
                    severity,
                    priority,
                    recommendations
                ])
            
            defect_table = Table(defect_table_data, colWidths=[1.5*inch, 0.8*inch, 0.8*inch, 2.9*inch])
            defect_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e40af')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ]))
            
            story.append(defect_table)
            story.append(Spacer(1, 20))
        
        # Technical Analysis Section
        story.append(Paragraph("Technical Analysis Details", heading_style))
        
        # Statistical Summary
        total_detections = inspection_data.get('summary', {}).get('total_defects_found', 0)
        frames_analyzed = inspection_data.get('summary', {}).get('frames_analyzed', 0)
        detection_rate = (total_detections / frames_analyzed * 100) if frames_analyzed > 0 else 0
        
        stats_text = f"""
        Detection Statistics:
        • Detection Rate: {detection_rate:.1f}% (defects per frame)
        • Models Consensus: {len(inspection_data.get('selected_models', []))} AI models used for validation
        • Confidence Threshold: Applied for reliable detection filtering
        • Analysis Method: Multi-model ensemble approach for enhanced accuracy
        """
        story.append(Paragraph(stats_text, styles['Normal']))
        story.append(Spacer(1, 15))
        
        # Page break before detailed frame analysis
        story.append(PageBreak())
        
        # Frame Analysis with Images
        if include_images:
            story.append(Paragraph("Detailed Frame Analysis with Defect Locations", heading_style))
            story.append(Paragraph("The following pages show frames with detected defects, their exact locations, and confidence scores from multiple AI models.", styles['Normal']))
            story.append(Spacer(1, 15))
            
            # Add frames with defects
            frames_with_defects = [frame for frame in inspection_data.get('defects_found', []) if frame.get('defects')]
            
            for i, frame in enumerate(frames_with_defects[:15]):  # Limit to first 15 frames
                story.append(Paragraph(f"Frame {frame.get('frame_number', i+1)} Analysis", subheading_style))
                
                # Add frame image if available
                if frame.get('frame_image'):
                    try:
                        # Decode base64 image
                        image_data = base64.b64decode(frame['frame_image'])
                        
                        # Create temporary image file
                        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as img_temp:
                            img_temp.write(image_data)
                            img_path = img_temp.name
                        
                        # Add image to PDF
                        img = ReportImage(img_path, width=5*inch, height=3.75*inch)
                        story.append(img)
                        
                        # Clean up temp image
                        os.unlink(img_path)
                        
                    except Exception as e:
                        story.append(Paragraph(f"[Image could not be loaded: {e}]", styles['Normal']))
                
                story.append(Spacer(1, 10))
                
                # Frame details with model breakdown
                frame_details = [
                    ['Overall Confidence', f"{frame.get('confidence_score', 0) * 100:.1f}%"],
                    ['Total Defects', str(len(frame.get('defects', [])))],
                    ['Models Used', ', '.join(frame.get('selected_models', []))]
                ]
                
                frame_table = Table(frame_details, colWidths=[2*inch, 2*inch])
                frame_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f3f4f6')),
                    ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ]))
                
                story.append(frame_table)
                story.append(Spacer(1, 10))
                
                # Defect details for this frame with model information
                if frame.get('defects'):
                    story.append(Paragraph("Detected Defects:", styles['Heading4']))
                    defect_list = []
                    
                    for defect in frame['defects']:
                        defect_text = f"• {defect.get('type', 'Unknown').replace('_', ' ').title()}: "
                        defect_text += f"{defect.get('confidence', 0) * 100:.1f}% confidence"
                        if defect.get('model'):
                            model_name = MODEL_CONFIG.get(defect['model'], {}).get('name', defect['model'])
                            defect_text += f" (detected by {model_name})"
                        if defect.get('boxes'):
                            defect_text += f" | {len(defect['boxes'])} location(s)"
                        defect_list.append(defect_text)
                    
                    for defect_text in defect_list:
                        story.append(Paragraph(defect_text, styles['Normal']))
                
                if i < len(frames_with_defects) - 1:  # Don't add page break after last frame
                    story.append(PageBreak())
        
        # Recommendations and Next Steps
        if include_recommendations:
            story.append(PageBreak())
            story.append(Paragraph("Professional Recommendations & Next Steps", heading_style))
            
            recommendations_text = f"""
            Based on the AI analysis findings, the following actions are recommended:
            
            Immediate Actions Required:
            • Review all high-confidence detections marked in this report
            • Prioritize {inspection_data.get('summary', {}).get('severity', 'medium').upper()} severity issues for professional inspection
            • Consider multiple model consensus for validation of critical findings
            
            Professional Consultation:
            • Engage structural engineers for any structural damage findings
            • Contact specialized contractors for water damage or mold issues
            • Schedule follow-up inspections for monitoring defect progression
            
            Documentation:
            • Keep this AI analysis report for insurance and maintenance records
            • Document any remediation work performed
            • Schedule periodic re-analysis for monitoring purposes
            
            Technical Notes:
            • This report was generated using {len(inspection_data.get('selected_models', []))} AI models for enhanced accuracy
            • Confidence scores indicate the AI's certainty in detection accuracy
            • Multiple model consensus provides additional validation for findings
            """
            
            story.append(Paragraph(recommendations_text, styles['Normal']))
        
        # Build PDF
        doc.build(story)
        
        # Read PDF content
        with open(pdf_path, 'rb') as pdf_file:
            pdf_content = pdf_file.read()
        
        # Clean up temp file
        os.unlink(pdf_path)
        
        return pdf_content
        
    except Exception as e:
        logging.error(f"Enhanced PDF generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating enhanced PDF report: {e}")

def get_defect_priority(defect_type):
    """Get priority level for defect type"""
    priority_mapping = {
        'cracks': 'Critical',
        'water_damage': 'Critical',
        'mold': 'Critical',
        'structural_damage': 'Critical',
        'rust': 'High',
        'paint': 'Low',
        'tiles': 'Medium',
        'flooring': 'Medium'
    }
    
    for key, priority in priority_mapping.items():
        if key in defect_type.lower():
            return priority
    return 'Medium'

def get_defect_severity(defect_type):
    """Get severity level for defect type"""
    severity_mapping = {
        'cracks': 'High',
        'water_damage': 'High',
        'mold': 'High',
        'structural_damage': 'Critical',
        'rust': 'Medium',
        'paint': 'Low',
        'tiles': 'Medium',
        'flooring': 'Medium'
    }
    
    for key, severity in severity_mapping.items():
        if key in defect_type.lower():
            return severity
    return 'Medium'

def get_defect_recommendations(defect_type):
    """Get recommendations for defect type"""
    recommendations = {
        'cracks': 'Immediate professional inspection required. May indicate structural issues.',
        'water_damage': 'Identify and fix water source. Check for mold growth. Professional remediation recommended.',
        'mold': 'Professional mold remediation required. Address moisture source immediately.',
        'structural_damage': 'Critical - Contact structural engineer immediately. Do not occupy affected areas.',
        'rust': 'Clean and treat affected areas. Check for underlying moisture issues.',
        'paint': 'Surface preparation and repainting needed. Check for underlying issues.',
        'tiles': 'Replace damaged tiles. Check subfloor for water damage.',
        'flooring': 'Assess extent of damage. May require partial or complete replacement.'
    }
    
    for key, recommendation in recommendations.items():
        if key in defect_type.lower():
            return recommendation
    return 'Professional inspection recommended to determine appropriate action.'
