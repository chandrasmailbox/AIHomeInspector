
import requests
import sys
import os
import time
from datetime import datetime
import base64
import tempfile
import shutil
import json
import cv2
import numpy as np
from PIL import Image
import io
import PyPDF2

class HomeInspectorAPITester:
    def __init__(self, base_url="https://f9e22b5f-b6e7-48d4-b08f-ec20119bde78.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.test_video_path = None
        self.last_inspection_id = None

    def run_test(self, name, method, endpoint, expected_status, data=None, files=None):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}"
        headers = {}
        
        # Set content type for JSON data
        if data and isinstance(data, str):
            try:
                json.loads(data)  # Validate JSON
                headers['Content-Type'] = 'application/json'
            except:
                pass
        
        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers)
            elif method == 'POST':
                if headers.get('Content-Type') == 'application/json':
                    response = requests.post(url, data=data, headers=headers)
                else:
                    response = requests.post(url, data=data, files=files, headers=headers)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                return success, response.json() if response.content else {}
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                print(f"Response: {response.text}")
                return False, {}

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def create_test_video(self):
        """Create a small test video file for testing"""
        try:
            # Download a small sample video for testing
            print("Downloading sample video for testing...")
            sample_url = "https://filesamples.com/samples/video/mp4/sample_640x360.mp4"
            response = requests.get(sample_url, stream=True)
            
            if response.status_code == 200:
                # Create a temporary file
                temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
                self.test_video_path = temp_file.name
                
                # Write the content to the file
                with open(self.test_video_path, 'wb') as f:
                    response.raw.decode_content = True
                    shutil.copyfileobj(response.raw, f)
                
                print(f"Test video created at: {self.test_video_path}")
                return True
            else:
                print(f"Failed to download sample video: {response.status_code}")
                return False
        except Exception as e:
            print(f"Error creating test video: {str(e)}")
            return False

    def test_root_endpoint(self):
        """Test the root API endpoint"""
        success, response = self.run_test(
            "Root API Endpoint",
            "GET",
            "",
            200
        )
        if success:
            print(f"API Response: {response}")
        return success

    def test_analyze_video(self):
        """Test video analysis endpoint with focus on bounding box feature"""
        if not self.test_video_path or not os.path.exists(self.test_video_path):
            print("❌ No test video available")
            return False
            
        try:
            with open(self.test_video_path, 'rb') as video_file:
                files = {'file': ('test_video.mp4', video_file, 'video/mp4')}
                
                print("Uploading video for analysis (this may take some time)...")
                success, response = self.run_test(
                    "Video Analysis",
                    "POST",
                    "analyze-video",
                    200,
                    files=files
                )
                
                if success:
                    print("Video analysis completed successfully")
                    print(f"Defects found: {response.get('summary', {}).get('total_defects_found', 0)}")
                    print(f"Frames analyzed: {response.get('summary', {}).get('frames_analyzed', 0)}")
                    print(f"Severity: {response.get('summary', {}).get('severity', 'unknown')}")
                    
                    # Store the inspection ID for later use
                    self.last_inspection_id = response.get('id')
                    
                    # Check if we have defects_found in the response
                    if 'defects_found' in response and len(response['defects_found']) > 0:
                        # Test for bounding box data in the response
                        self.test_bounding_box_data(response['defects_found'])
                    
                return success
                
        except Exception as e:
            print(f"❌ Error testing video analysis: {str(e)}")
            return False

    def test_bounding_box_data(self, frames_data):
        """Test if frames contain bounding box data for defects"""
        print("\n🔍 Testing Bounding Box Data in Frames...")
        self.tests_run += 1
        
        try:
            has_boxes = False
            frames_with_boxes = 0
            total_boxes = 0
            defect_types_with_boxes = set()
            
            for frame in frames_data:
                if 'defects' in frame:
                    for defect in frame['defects']:
                        if 'boxes' in defect and defect['boxes']:
                            has_boxes = True
                            frames_with_boxes += 1
                            total_boxes += len(defect['boxes'])
                            defect_types_with_boxes.add(defect['type'])
                            break
            
            if has_boxes:
                self.tests_passed += 1
                print(f"✅ Passed - Found bounding box data in {frames_with_boxes} frames")
                print(f"Total boxes found: {total_boxes}")
                print(f"Defect types with boxes: {', '.join(defect_types_with_boxes)}")
                return True
            else:
                print("❌ Failed - No bounding box data found in any frames")
                return False
                
        except Exception as e:
            print(f"❌ Error testing bounding box data: {str(e)}")
            return False

    def test_frame_annotations(self, frames_data):
        """Test if frames have visual annotations (colored boxes)"""
        print("\n🔍 Testing Visual Annotations in Frames...")
        self.tests_run += 1
        
        try:
            frames_with_annotations = 0
            
            for frame in frames_data:
                if 'frame_image' in frame and frame['frame_image']:
                    # Decode base64 image
                    img_data = base64.b64decode(frame['frame_image'])
                    img = Image.open(io.BytesIO(img_data))
                    img_array = np.array(img)
                    
                    # Check if the image has colored rectangles
                    # This is a simple heuristic - we look for rectangular patterns of colors
                    # that match our defect color scheme
                    
                    # Convert to HSV for better color detection
                    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
                    
                    # Define color ranges for our defect colors
                    color_ranges = [
                        # Red (cracks)
                        (np.array([0, 100, 100]), np.array([10, 255, 255])),
                        # Blue (water damage)
                        (np.array([100, 100, 100]), np.array([130, 255, 255])),
                        # Green (mold)
                        (np.array([40, 100, 100]), np.array([80, 255, 255])),
                    ]
                    
                    for lower, upper in color_ranges:
                        mask = cv2.inRange(hsv, lower, upper)
                        if np.sum(mask) > 1000:  # Threshold for detection
                            frames_with_annotations += 1
                            break
            
            if frames_with_annotations > 0:
                self.tests_passed += 1
                print(f"✅ Passed - Found visual annotations in {frames_with_annotations} frames")
                return True
            else:
                print("❌ Failed - No visual annotations found in any frames")
                return False
                
        except Exception as e:
            print(f"❌ Error testing visual annotations: {str(e)}")
            return False

    def test_get_inspections(self):
        """Test getting all inspections"""
        success, response = self.run_test(
            "Get All Inspections",
            "GET",
            "inspections",
            200
        )
        if success:
            print(f"Retrieved {len(response)} inspections")
        return success

    def test_get_inspection_by_id(self, inspection_id):
        """Test getting a specific inspection by ID"""
        success, response = self.run_test(
            "Get Inspection by ID",
            "GET",
            f"inspection/{inspection_id}",
            200
        )
        if success:
            print(f"Retrieved inspection: {response.get('id')}")
            
            # Test bounding box data in this specific inspection
            if 'defects_found' in response:
                self.test_bounding_box_data(response['defects_found'])
                self.test_frame_annotations(response['defects_found'])
                
                # Store a frame number for correction testing
                if len(response['defects_found']) > 0:
                    self.test_frame_number = response['defects_found'][0]['frame_number']
                    
                    # Find a box ID to test with
                    for frame in response['defects_found']:
                        for defect in frame['defects']:
                            if 'boxes' in defect and defect['boxes']:
                                for box in defect['boxes']:
                                    if isinstance(box, dict) and 'id' in box:
                                        self.test_box_id = box['id']
                                        self.test_defect_type = defect['type']
                                        break
                                if hasattr(self, 'test_box_id'):
                                    break
                        if hasattr(self, 'test_box_id'):
                            break
        return success

    def cleanup(self):
        """Clean up any test resources"""
        if self.test_video_path and os.path.exists(self.test_video_path):
            try:
                os.unlink(self.test_video_path)
                print(f"Removed test video: {self.test_video_path}")
            except Exception as e:
                print(f"Error removing test video: {str(e)}")
                
    def test_save_corrections(self):
        """Test saving user corrections for a frame"""
        if not hasattr(self, 'last_inspection_id') or not self.last_inspection_id:
            print("❌ No inspection ID available for correction test")
            return False
            
        if not hasattr(self, 'test_frame_number') or not hasattr(self, 'test_box_id'):
            print("❌ No frame number or box ID available for correction test")
            return False
            
        print(f"\n🔍 Testing User Corrections API...")
        self.tests_run += 1
        
        try:
            # Create correction data
            correction_data = {
                "inspection_id": self.last_inspection_id,
                "frame_number": self.test_frame_number,
                "corrections": [
                    {
                        "box_id": self.test_box_id,
                        "defect_type": self.test_defect_type,
                        "is_hidden": True,
                        "user_feedback": "Test correction from API test"
                    }
                ]
            }
            
            # Send correction
            success, response = self.run_test(
                "Save User Corrections",
                "POST",
                f"inspection/{self.last_inspection_id}/corrections",
                200,
                data=json.dumps(correction_data),
                files=None
            )
            
            if success:
                print("✅ Successfully saved user corrections")
                self.tests_passed += 1
                
                # Now test regenerating the frame with corrections
                self.test_regenerate_frame()
                
            return success
            
        except Exception as e:
            print(f"❌ Error testing corrections API: {str(e)}")
            return False
            
    def test_regenerate_frame(self):
        """Test regenerating a frame with applied corrections"""
        if not hasattr(self, 'last_inspection_id') or not self.last_inspection_id:
            print("❌ No inspection ID available for regeneration test")
            return False
            
        if not hasattr(self, 'test_frame_number'):
            print("❌ No frame number available for regeneration test")
            return False
            
        print(f"\n🔍 Testing Frame Regeneration API...")
        self.tests_run += 1
        
        try:
            # Request regenerated frame
            success, response = self.run_test(
                "Regenerate Frame with Corrections",
                "GET",
                f"inspection/{self.last_inspection_id}/frame/{self.test_frame_number}/regenerate",
                200
            )
            
            if success:
                print("✅ Successfully regenerated frame with corrections")
                
                # Check if the response contains the expected data
                if 'defects' in response and 'frame_image' in response:
                    print("✅ Regenerated frame contains defects and image data")
                    self.tests_passed += 1
                else:
                    print("❌ Regenerated frame missing expected data")
                    
            return success
            
        except Exception as e:
            print(f"❌ Error testing frame regeneration: {str(e)}")
            return False
            
    def test_available_models(self):
        """Test the available models endpoint"""
        print(f"\n🔍 Testing Available Models API...")
        self.tests_run += 1
        
        try:
            success, response = self.run_test(
                "Get Available Models",
                "GET",
                "models/available",
                200
            )
            
            if success:
                print("✅ Successfully retrieved available models")
                if 'models' in response:
                    print(f"Available models: {', '.join(response['models'])}")
                    self.tests_passed += 1
                else:
                    print("❌ Response missing 'models' field")
                    
            return success
            
        except Exception as e:
            print(f"❌ Error testing available models API: {str(e)}")
            return False
            
    def test_analyze_video_with_models(self):
        """Test video analysis with multiple model selection"""
        if not self.test_video_path or not os.path.exists(self.test_video_path):
            print("❌ No test video available")
            return False
            
        print(f"\n🔍 Testing Video Analysis with Multiple Models API...")
        self.tests_run += 1
        
        try:
            with open(self.test_video_path, 'rb') as video_file:
                # Create model selection data
                model_selection = {
                    "selected_models": ["basic_cv", "yolov8n"],
                    "ensemble_method": "weighted_average",
                    "confidence_threshold": 0.4,
                    "use_ensemble": True
                }
                
                # Create multipart form data
                files = {
                    'file': ('test_video.mp4', video_file, 'video/mp4'),
                    'model_selection': (None, json.dumps(model_selection), 'application/json')
                }
                
                print("Uploading video for analysis with multiple models (this may take some time)...")
                success, response = self.run_test(
                    "Video Analysis with Multiple Models",
                    "POST",
                    "analyze-video-with-models",
                    200,
                    files=files
                )
                
                if success:
                    print("✅ Video analysis with multiple models completed successfully")
                    print(f"Defects found: {response.get('summary', {}).get('total_defects_found', 0)}")
                    print(f"Selected models: {', '.join(response.get('selected_models', []))}")
                    
                    # Store the inspection ID for later use
                    self.last_inspection_id = response.get('id')
                    
                    # Check if model results are included
                    if 'model_results' in response:
                        print("✅ Response includes model-specific results")
                        self.tests_passed += 1
                    else:
                        print("❌ Response missing model-specific results")
                        
                return success
                
        except Exception as e:
            print(f"❌ Error testing video analysis with multiple models: {str(e)}")
            return False
            
    def test_export_pdf(self):
        """Test PDF export functionality"""
        if not hasattr(self, 'last_inspection_id') or not self.last_inspection_id:
            print("❌ No inspection ID available for PDF export test")
            return False
            
        print(f"\n🔍 Testing PDF Export API...")
        self.tests_run += 1
        
        try:
            # Create PDF export request
            pdf_request = {
                "include_images": True,
                "include_model_comparison": True,
                "include_recommendations": True
            }
            
            # Request PDF export
            url = f"{self.api_url}/inspection/{self.last_inspection_id}/export-pdf"
            headers = {'Content-Type': 'application/json'}
            
            print(f"Requesting PDF export for inspection {self.last_inspection_id}...")
            response = requests.post(url, data=json.dumps(pdf_request), headers=headers)
            
            if response.status_code == 200:
                print("✅ Successfully received PDF export")
                
                # Save PDF to temporary file for validation
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                    temp_file.write(response.content)
                    pdf_path = temp_file.name
                
                # Validate PDF content
                try:
                    with open(pdf_path, 'rb') as pdf_file:
                        pdf_reader = PyPDF2.PdfReader(pdf_file)
                        num_pages = len(pdf_reader.pages)
                        
                        print(f"PDF has {num_pages} pages")
                        
                        # Check if PDF has content
                        if num_pages > 0:
                            print("✅ PDF contains valid content")
                            self.tests_passed += 1
                        else:
                            print("❌ PDF appears to be empty")
                            
                    # Clean up temp file
                    os.unlink(pdf_path)
                    
                except Exception as e:
                    print(f"❌ Error validating PDF: {str(e)}")
                    return False
                    
                return True
            else:
                print(f"❌ Failed - Expected 200, got {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error testing PDF export: {str(e)}")
            return False
            
    def test_model_comparison(self):
        """Test model comparison endpoint"""
        if not hasattr(self, 'last_inspection_id') or not self.last_inspection_id:
            print("❌ No inspection ID available for model comparison test")
            return False
            
        print(f"\n🔍 Testing Model Comparison API...")
        self.tests_run += 1
        
        try:
            success, response = self.run_test(
                "Get Model Comparison",
                "GET",
                f"inspection/{self.last_inspection_id}/model-comparison",
                200
            )
            
            if success:
                print("✅ Successfully retrieved model comparison data")
                
                # Check if the response contains the expected data
                if 'performance_metrics' in response:
                    print("✅ Response includes performance metrics")
                    
                    # Print some metrics for verification
                    metrics = response.get('performance_metrics', {})
                    for model, data in metrics.items():
                        print(f"Model: {model}")
                        print(f"  - Detections: {data.get('total_detections', 0)}")
                        print(f"  - Avg Confidence: {data.get('average_confidence', 0):.2f}")
                        
                    self.tests_passed += 1
                else:
                    print("❌ Response missing performance metrics")
                    
            return success
            
        except Exception as e:
            print(f"❌ Error testing model comparison API: {str(e)}")
            return False

def main():
    # Setup
    tester = HomeInspectorAPITester()
    
    try:
        # Test API root endpoint
        if not tester.test_root_endpoint():
            print("❌ Root API test failed, stopping tests")
            return 1
            
        # Create test video
        if not tester.create_test_video():
            print("❌ Failed to create test video, stopping tests")
            return 1
            
        # Test video analysis with bounding box feature
        if not tester.test_analyze_video():
            print("❌ Video analysis test failed")
            # Continue with other tests
            
        # Test getting all inspections
        if not tester.test_get_inspections():
            print("❌ Get inspections test failed")
            # Continue with other tests
            
        # If we have a successful video analysis, test getting that inspection
        if tester.last_inspection_id:
            if not tester.test_get_inspection_by_id(tester.last_inspection_id):
                print("❌ Get inspection by ID test failed")
                
            # Test the corrections API
            if hasattr(tester, 'test_frame_number') and hasattr(tester, 'test_box_id'):
                if not tester.test_save_corrections():
                    print("❌ Save corrections test failed")
        
        # Print results
        print(f"\n📊 Tests passed: {tester.tests_passed}/{tester.tests_run}")
        return 0 if tester.tests_passed == tester.tests_run else 1
        
    finally:
        # Clean up resources
        tester.cleanup()

if __name__ == "__main__":
    sys.exit(main())
      