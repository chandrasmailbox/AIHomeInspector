import React, { useState, useRef } from "react";
import "./App.css";
import axios from "axios";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const VideoUpload = ({ onAnalysisComplete }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModels, setSelectedModels] = useState(['basic_cv']);
  const [showModelSelection, setShowModelSelection] = useState(false);
  const [ensembleMethod, setEnsembleMethod] = useState('weighted_average');
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.5);
  const fileInputRef = useRef(null);

  // Fetch available models on component mount
  useEffect(() => {
    fetchAvailableModels();
  }, []);

  const fetchAvailableModels = async () => {
    try {
      const response = await axios.get(`${API}/models/available`);
      setAvailableModels(response.data.available_models || []);
    } catch (error) {
      console.error('Failed to fetch available models:', error);
      // Fallback to basic models if API fails
      setAvailableModels([
        { id: 'basic_cv', name: 'Basic Computer Vision', description: 'Traditional CV methods', enabled: true },
        { id: 'yolov8n', name: 'YOLOv8 Nano', description: 'Fast object detection', enabled: true }
      ]);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('video/')) {
        setSelectedFile(file);
      } else {
        alert('Please select a video file');
      }
    }
  };

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const analyzeVideo = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post(`${API}/analyze-video`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      onAnalysisComplete(response.data);
    } catch (error) {
      console.error('Analysis failed:', error);
      alert('Analysis failed. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-6">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold text-gray-800 mb-2">HomeInspector AI</h1>
        <p className="text-gray-600">Upload a video to detect house defects using AI</p>
      </div>

      <div
        className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
          dragActive
            ? 'border-blue-400 bg-blue-50'
            : 'border-gray-300 hover:border-gray-400'
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*"
          onChange={handleFileSelect}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        />
        
        <div className="space-y-4">
          <div className="text-6xl text-gray-400">📹</div>
          <div>
            <p className="text-lg font-medium text-gray-700">
              {selectedFile ? selectedFile.name : 'Drop your video here or click to select'}
            </p>
            <p className="text-sm text-gray-500 mt-2">
              Supports MP4, AVI, MOV, MKV files
            </p>
          </div>
        </div>
      </div>

      {selectedFile && (
        <div className="mt-6 text-center">
          <button
            onClick={analyzeVideo}
            disabled={isAnalyzing}
            className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-bold py-3 px-8 rounded-lg transition-colors"
          >
            {isAnalyzing ? (
              <span className="flex items-center justify-center">
                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Analyzing Video...
              </span>
            ) : (
              'Analyze Video'
            )}
          </button>
        </div>
      )}
    </div>
  );
};

const ResultsDisplay = ({ results, onBack }) => {
  const [selectedFrame, setSelectedFrame] = useState(null);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });
  const [hiddenBoxes, setHiddenBoxes] = useState(new Set());
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  const [isMaximized, setIsMaximized] = useState(false);

  // Reset zoom and pan when frame changes
  const resetImageView = () => {
    setZoomLevel(1);
    setPanOffset({ x: 0, y: 0 });
    setIsMaximized(false);
  };

  // Handle frame selection
  const handleFrameSelect = (frame) => {
    setSelectedFrame(frame);
    resetImageView();
    setHiddenBoxes(new Set()); // Reset hidden boxes for new frame
  };

  // Zoom functions
  const zoomIn = () => {
    setZoomLevel(prev => Math.min(prev * 1.5, 5));
  };

  const zoomOut = () => {
    setZoomLevel(prev => Math.max(prev / 1.5, 0.5));
  };

  const resetZoom = () => {
    setZoomLevel(1);
    setPanOffset({ x: 0, y: 0 });
  };

  const maximizeImage = () => {
    setIsMaximized(!isMaximized);
    if (!isMaximized) {
      setZoomLevel(2);
      setPanOffset({ x: 0, y: 0 });
    } else {
      resetZoom();
    }
  };

  // Pan functions
  const handleMouseDown = (e) => {
    if (zoomLevel > 1) {
      setIsPanning(true);
      setPanStart({ x: e.clientX - panOffset.x, y: e.clientY - panOffset.y });
    }
  };

  const handleMouseMove = (e) => {
    if (isPanning && zoomLevel > 1) {
      setPanOffset({
        x: e.clientX - panStart.x,
        y: e.clientY - panStart.y
      });
    }
  };

  const handleMouseUp = () => {
    setIsPanning(false);
  };

  // Box management functions
  const toggleBoxVisibility = async (boxId, defectType) => {
    const newHiddenBoxes = new Set(hiddenBoxes);
    
    if (hiddenBoxes.has(boxId)) {
      newHiddenBoxes.delete(boxId);
    } else {
      newHiddenBoxes.add(boxId);
    }
    
    setHiddenBoxes(newHiddenBoxes);

    // Save to backend
    try {
      const corrections = [{
        box_id: boxId,
        defect_type: defectType,
        is_hidden: !hiddenBoxes.has(boxId),
        user_feedback: "User manually adjusted defect visibility"
      }];

      await axios.post(`${API}/inspection/${results.id}/corrections`, {
        inspection_id: results.id,
        frame_number: selectedFrame.frame_number,
        corrections: corrections
      });
      
      console.log(`Box ${boxId} visibility toggled. Now ${newHiddenBoxes.has(boxId) ? 'hidden' : 'visible'}`);
    } catch (error) {
      console.error('Error saving correction:', error);
      // Revert the local change if saving failed
      setHiddenBoxes(hiddenBoxes);
    }
  };

  const showAllBoxes = () => {
    setHiddenBoxes(new Set());
    console.log('All boxes are now visible');
  };

  const hideAllBoxes = () => {
    if (selectedFrame) {
      const allBoxIds = new Set();
      selectedFrame.defects.forEach(defect => {
        defect.boxes?.forEach(box => {
          const boxId = box.id || `${defect.type}_${selectedFrame.defects.indexOf(defect)}_${defect.boxes.indexOf(box)}`;
          allBoxIds.add(boxId);
        });
      });
      setHiddenBoxes(allBoxIds);
      console.log(`All ${allBoxIds.size} boxes are now hidden`);
    }
  };

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'high': return 'text-red-600 bg-red-100';
      case 'medium': return 'text-yellow-600 bg-yellow-100';
      case 'low': return 'text-green-600 bg-green-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getDefectIcon = (type) => {
    switch (type) {
      case 'cracks': return '🔧';
      case 'water_damage': return '💧';
      case 'mold on wall': return '🦠';
      case 'paint peeling off wall': return '🎨';
      default: return '⚠️';
    }
  };

  const getDefectColor = (type) => {
    const colors = {
      'cracks': 'bg-red-500',
      'water_damage': 'bg-blue-500',
      'mold': 'bg-green-500',
      'paint': 'bg-yellow-500',
      'rust': 'bg-orange-500',
      'tiles': 'bg-purple-500',
      'flooring': 'bg-pink-500'
    };
    return colors[type.split(' ')[0]] || 'bg-gray-500';
  };

  const getDefectStrokeColor = (type) => {
    const colors = {
      'cracks': '#ef4444',
      'water_damage': '#3b82f6',
      'mold': '#10b981',
      'paint': '#eab308',
      'rust': '#f97316',
      'tiles': '#8b5cf6',
      'flooring': '#ec4899'
    };
    return colors[type.split(' ')[0]] || '#6b7280';
  };

  const getTextBackgroundColor = (type) => {
    // Use darker colors for better contrast
    const colors = {
      'cracks': '#dc2626',
      'water_damage': '#1d4ed8',
      'mold': '#047857',
      'paint': '#ca8a04',
      'rust': '#ea580c',
      'tiles': '#7c3aed',
      'flooring': '#db2777'
    };
    return colors[type.split(' ')[0]] || '#374151';
  };

  // Color legend for defect types
  const DefectLegend = () => (
    <div className="bg-white rounded-lg shadow-lg p-4 mb-6">
      <h3 className="text-lg font-semibold mb-3">Defect Detection Legend</h3>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 bg-red-500 border"></div>
          <span className="text-sm">Cracks</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 bg-blue-500 border"></div>
          <span className="text-sm">Water Damage</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 bg-green-500 border"></div>
          <span className="text-sm">Mold</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 bg-yellow-500 border"></div>
          <span className="text-sm">Paint Issues</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 bg-orange-500 border"></div>
          <span className="text-sm">Rust</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 bg-purple-500 border"></div>
          <span className="text-sm">Broken Tiles</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 bg-pink-500 border"></div>
          <span className="text-sm">Damaged Flooring</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 bg-gray-500 border"></div>
          <span className="text-sm">Other</span>
        </div>
      </div>
      <p className="text-xs text-gray-500 mt-2">
        * Colored boxes on images show exact locations of detected defects
      </p>
    </div>
  );

  return (
    <div className="max-w-6xl mx-auto p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-3xl font-bold text-gray-800">Inspection Results</h2>
        <button
          onClick={onBack}
          className="bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded-lg"
        >
          ← Back to Upload
        </button>
      </div>

      {/* Color Legend */}
      <DefectLegend />

      {/* Summary Card */}
      <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
        <h3 className="text-xl font-semibold mb-4">Inspection Summary</h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">{results.summary.total_defects_found}</div>
            <div className="text-sm text-gray-600">Total Defects</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-600">{results.summary.frames_analyzed}</div>
            <div className="text-sm text-gray-600">Frames Analyzed</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-orange-600">{results.summary.high_confidence_detections}</div>
            <div className="text-sm text-gray-600">High Confidence</div>
          </div>
          <div className="text-center">
            <span className={`inline-block px-3 py-1 rounded-full text-sm font-medium ${getSeverityColor(results.summary.severity)}`}>
              {results.summary.severity.toUpperCase()}
            </span>
            <div className="text-sm text-gray-600 mt-1">Severity</div>
          </div>
        </div>
      </div>

      {/* Defect Types */}
      {results.summary.defect_types.length > 0 && (
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <h3 className="text-xl font-semibold mb-4">Detected Issues</h3>
          <div className="flex flex-wrap gap-2">
            {results.summary.defect_types.map((type, index) => (
              <span
                key={index}
                className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-red-100 text-red-800"
              >
                <div className={`w-3 h-3 rounded-full mr-2 ${getDefectColor(type)}`}></div>
                {getDefectIcon(type)} {type.replace('_', ' ')}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Frame Analysis */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h3 className="text-xl font-semibold mb-4">Frame-by-Frame Analysis with Defect Locations</h3>
        <p className="text-sm text-gray-600 mb-4">
          Click on any frame to see detailed defect locations. Colored boxes show exact defect positions.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {results.defects_found
            .filter(frame => frame.defects.length > 0)
            .slice(0, 12)
            .map((frame, index) => (
            <div
              key={index}
              className="border rounded-lg p-4 cursor-pointer hover:shadow-md transition-shadow"
              onClick={() => handleFrameSelect(frame)}
            >
              <img
                src={`data:image/jpeg;base64,${frame.frame_image}`}
                alt={`Frame ${frame.frame_number} with defect annotations`}
                className="w-full h-32 object-cover rounded mb-2"
              />
              <div className="text-sm text-gray-600">Frame {frame.frame_number}</div>
              <div className="text-sm font-medium">
                Confidence: {(frame.confidence_score * 100).toFixed(1)}%
              </div>
              <div className="text-xs text-gray-500 mt-1">
                {frame.defects.length} defect(s) with boxes
              </div>
              <div className="flex flex-wrap gap-1 mt-2">
                {frame.defects.slice(0, 3).map((defect, idx) => (
                  <div key={idx} className={`w-2 h-2 rounded-full ${getDefectColor(defect.type)}`}></div>
                ))}
                {frame.defects.length > 3 && (
                  <span className="text-xs text-gray-500">+{frame.defects.length - 3}</span>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Enhanced Selected Frame Modal with Zoom and Interactive Boxes */}
      {selectedFrame && (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center p-4 z-50">
          <div className={`bg-white rounded-lg ${isMaximized ? 'max-w-7xl max-h-screen' : 'max-w-5xl max-h-[90vh]'} overflow-hidden flex flex-col`}>
            {/* Modal Header */}
            <div className="flex justify-between items-center p-4 border-b bg-gray-50">
              <div className="flex items-center space-x-4">
                <h3 className="text-xl font-semibold">Frame {selectedFrame.frame_number} - Interactive Analysis</h3>
                <div className="flex items-center space-x-2 text-sm text-gray-600">
                  <span>Zoom: {Math.round(zoomLevel * 100)}%</span>
                  {zoomLevel > 1 && <span className="text-blue-600">• Click and drag to pan</span>}
                </div>
              </div>
              <button
                onClick={() => setSelectedFrame(null)}
                className="text-gray-500 hover:text-gray-700 text-2xl p-1"
              >
                ×
              </button>
            </div>

            {/* Controls Bar */}
            <div className="p-3 border-b bg-gray-50 flex flex-wrap items-center justify-between gap-2">
              {/* Zoom Controls */}
              <div className="flex items-center space-x-2">
                <span className="text-sm font-medium text-gray-700">Zoom:</span>
                <button
                  onClick={zoomOut}
                  disabled={zoomLevel <= 0.5}
                  className="px-2 py-1 bg-gray-200 hover:bg-gray-300 disabled:bg-gray-100 disabled:text-gray-400 rounded text-sm"
                >
                  −
                </button>
                <button
                  onClick={resetZoom}
                  className="px-3 py-1 bg-blue-500 hover:bg-blue-600 text-white rounded text-sm"
                >
                  Reset
                </button>
                <button
                  onClick={zoomIn}
                  disabled={zoomLevel >= 5}
                  className="px-2 py-1 bg-gray-200 hover:bg-gray-300 disabled:bg-gray-100 disabled:text-gray-400 rounded text-sm"
                >
                  +
                </button>
                <button
                  onClick={maximizeImage}
                  className="px-3 py-1 bg-purple-500 hover:bg-purple-600 text-white rounded text-sm"
                >
                  {isMaximized ? 'Normal' : 'Maximize'}
                </button>
              </div>

              {/* Box Controls */}
              <div className="flex items-center space-x-2">
                <span className="text-sm font-medium text-gray-700">Boxes:</span>
                <button
                  onClick={showAllBoxes}
                  className="px-3 py-1 bg-green-500 hover:bg-green-600 text-white rounded text-sm control-button"
                >
                  Show All
                </button>
                <button
                  onClick={hideAllBoxes}
                  className="px-3 py-1 bg-red-500 hover:bg-red-600 text-white rounded text-sm control-button"
                >
                  Hide All
                </button>
                <span className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded">
                  {selectedFrame.defects.reduce((acc, defect) => acc + (defect.boxes?.length || 0), 0) - hiddenBoxes.size} visible / {selectedFrame.defects.reduce((acc, defect) => acc + (defect.boxes?.length || 0), 0)} total
                </span>
              </div>
            </div>

            {/* Image Container */}
            <div className="flex-1 flex overflow-hidden">
              {/* Main Image Area */}
              <div className="flex-1 relative overflow-hidden bg-gray-100">
                <div className="relative h-full flex items-center justify-center">
                  <div
                    className="relative inline-block cursor-grab active:cursor-grabbing select-none"
                    style={{
                      transform: `scale(${zoomLevel}) translate(${panOffset.x / zoomLevel}px, ${panOffset.y / zoomLevel}px)`,
                      transformOrigin: 'center center'
                    }}
                    onMouseDown={handleMouseDown}
                    onMouseMove={handleMouseMove}
                    onMouseUp={handleMouseUp}
                    onMouseLeave={handleMouseUp}
                  >
                    <img
                      src={`data:image/jpeg;base64,${selectedFrame.frame_image}`}
                      alt={`Frame ${selectedFrame.frame_number} with annotations`}
                      className="max-w-none block"
                      style={{ 
                        maxHeight: isMaximized ? '70vh' : '50vh',
                        userSelect: 'none',
                        pointerEvents: 'none'
                      }}
                      draggable={false}
                    />
                    
                    {/* Interactive Bounding Boxes Overlay */}
                    <svg
                      className="absolute inset-0 w-full h-full pointer-events-none"
                      style={{ 
                        pointerEvents: zoomLevel > 1 ? 'none' : 'auto'
                      }}
                    >
                      {selectedFrame.defects.map((defect, defectIndex) => 
                        defect.boxes?.map((box, boxIndex) => {
                          const boxId = box.id || `${defect.type}_${defectIndex}_${boxIndex}`;
                          const isHidden = hiddenBoxes.has(boxId);
                          const coords = box.coords || box;
                          
                          // Only skip rendering if explicitly hidden
                          if (!coords) return null;
                          
                          const [x, y, w, h] = Array.isArray(coords) ? coords : [coords.x, coords.y, coords.w, coords.h];
                          const strokeColor = getDefectStrokeColor(defect.type);
                          const bgColor = getTextBackgroundColor(defect.type);
                          
                          return (
                            <g key={boxId} className={isHidden ? 'hidden-defect-box' : 'visible-defect-box'}>
                              {/* Bounding Box Rectangle */}
                              <rect
                                x={x}
                                y={y}
                                width={w}
                                height={h}
                                fill="transparent"
                                stroke={strokeColor}
                                strokeWidth="3"
                                className="hover:stroke-4 cursor-pointer transition-all defect-box"
                                style={{ 
                                  pointerEvents: 'auto',
                                  opacity: isHidden ? 0.2 : 1,
                                  strokeDasharray: isHidden ? '10,5' : 'none',
                                  strokeOpacity: isHidden ? 0.5 : 1
                                }}
                                onClick={(e) => {
                                  e.stopPropagation();
                                  toggleBoxVisibility(boxId, defect.type);
                                }}
                              />
                              
                              {/* Delete Button - only show if not hidden */}
                              {!isHidden && (
                                <>
                                  <circle
                                    cx={x + w - 12}
                                    cy={y + 12}
                                    r="10"
                                    fill="rgba(239, 68, 68, 0.9)"
                                    stroke="white"
                                    strokeWidth="2"
                                    className="hover:fill-red-600 cursor-pointer delete-button"
                                    style={{ pointerEvents: 'auto' }}
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      toggleBoxVisibility(boxId, defect.type);
                                    }}
                                  />
                                  <text
                                    x={x + w - 12}
                                    y={y + 12}
                                    textAnchor="middle"
                                    dominantBaseline="central"
                                    fill="white"
                                    fontSize="12"
                                    fontWeight="bold"
                                    className="cursor-pointer"
                                    style={{ pointerEvents: 'auto' }}
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      toggleBoxVisibility(boxId, defect.type);
                                    }}
                                  >
                                    ×
                                  </text>
                                </>
                              )}
                              
                              {/* Label with improved visibility - only show if not hidden */}
                              {!isHidden && (
                                <>
                                  <rect
                                    x={x}
                                    y={y - 30}
                                    width={Math.min(w, 200)}
                                    height="25"
                                    fill={bgColor}
                                    stroke="white"
                                    strokeWidth="1"
                                    rx="3"
                                    opacity="0.95"
                                  />
                                  <text
                                    x={x + 6}
                                    y={y - 10}
                                    fill="white"
                                    fontSize="12"
                                    fontWeight="bold"
                                    className="defect-label"
                                    style={{ 
                                      filter: 'drop-shadow(1px 1px 2px rgba(0,0,0,0.8))',
                                      fontFamily: 'Arial, sans-serif'
                                    }}
                                  >
                                    {defect.type.replace('_', ' ')}: {Math.round(defect.confidence * 100)}%
                                  </text>
                                </>
                              )}
                              
                              {/* Hidden indicator */}
                              {isHidden && (
                                <text
                                  x={x + w/2}
                                  y={y + h/2}
                                  textAnchor="middle"
                                  dominantBaseline="central"
                                  fill="rgba(107, 114, 128, 0.8)"
                                  fontSize="14"
                                  fontWeight="bold"
                                  className="cursor-pointer"
                                  style={{ pointerEvents: 'auto' }}
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    toggleBoxVisibility(boxId, defect.type);
                                  }}
                                >
                                  👁️‍🗨️
                                </text>
                              )}
                            </g>
                          );
                        })
                      )}
                    </svg>
                  </div>
                </div>
              </div>

              {/* Side Panel with Defect List */}
              <div className="w-80 border-l bg-gray-50 p-4 overflow-y-auto">
                <h4 className="font-semibold mb-3 text-gray-800">Detected Defects ({selectedFrame.defects.length})</h4>
                
                {/* Instructions */}
                <div className="mb-4 p-3 bg-blue-50 rounded-lg text-sm">
                  <p className="text-blue-800 font-medium mb-1">💡 How to use:</p>
                  <ul className="text-blue-700 space-y-1 text-xs">
                    <li>• Click on colored boxes to hide/show defects</li>
                    <li>• Use zoom controls to inspect details</li>
                    <li>• Drag to pan when zoomed in</li>
                    <li>• Changes are saved automatically</li>
                  </ul>
                </div>

                <div className="space-y-3">
                  {selectedFrame.defects.map((defect, index) => (
                    <div key={index} className="bg-white p-3 rounded border">
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-medium flex items-center">
                          <div className={`w-4 h-4 rounded mr-2 ${getDefectColor(defect.type)}`}></div>
                          {getDefectIcon(defect.type)} {defect.type}
                        </span>
                        <span className="text-sm text-gray-600">
                          {(defect.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                      
                      {defect.description && (
                        <p className="text-sm text-gray-600 mb-2">{defect.description}</p>
                      )}
                      
                      {defect.boxes && (
                        <div className="text-xs text-gray-500">
                          <p className="mb-1">Bounding boxes: {defect.boxes.length}</p>
                          <div className="flex flex-wrap gap-1">
                            {defect.boxes.map((box, boxIndex) => {
                              const boxId = box.id || `${defect.type}_${index}_${boxIndex}`;
                              const isHidden = hiddenBoxes.has(boxId);
                              return (
                                <button
                                  key={boxIndex}
                                  onClick={() => toggleBoxVisibility(boxId, defect.type)}
                                  className={`px-2 py-1 rounded text-xs font-medium transition-all hover:scale-105 ${
                                    isHidden 
                                      ? 'bg-gray-200 text-gray-500 border border-gray-300' 
                                      : `bg-blue-100 text-blue-800 border border-blue-300 shadow-sm`
                                  }`}
                                  title={isHidden ? 'Click to show this box' : 'Click to hide this box'}
                                >
                                  {isHidden ? '👁️‍🗨️' : '👁️'} Box {boxIndex + 1}
                                </button>
                              );
                            })}
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

function App() {
  const [analysisResults, setAnalysisResults] = useState(null);

  const handleAnalysisComplete = (results) => {
    setAnalysisResults(results);
  };

  const handleBackToUpload = () => {
    setAnalysisResults(null);
  };

  return (
    <div className="min-h-screen bg-gray-100">
      {!analysisResults ? (
        <VideoUpload onAnalysisComplete={handleAnalysisComplete} />
      ) : (
        <ResultsDisplay results={analysisResults} onBack={handleBackToUpload} />
      )}
    </div>
  );
}

export default App;
