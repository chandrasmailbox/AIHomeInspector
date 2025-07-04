# AIHomeInspector – Project Documentation

---

## Table of Contents

1. [Software Requirements Specification (SRS)](#srs)
2. [User Stories and Acceptance Criteria](#user-stories)
3. [Use Case Diagrams & Descriptions](#use-cases)
4. [System Architecture Document](#architecture)
5. [Installation Guide](#installation)
6. [API Documentation](#api)
7. [Admin/User Guide](#user-guide)
8. [Test Strategy & Test Cases](#test-strategy)
9. [CI/CD Pipeline Explanation](#cicd)
10. [Contribution Guidelines](#contributing)
11. [Glossary and Acronyms](#glossary)

---

<a name="srs"></a>
## 1. Software Requirements Specification (SRS)

### 1.1 Functional Requirements

- Users can upload home inspection videos.
- The backend analyzes the video, frame-by-frame, to detect house defects:
  - Mold
  - Cracks
  - Water leaks
  - Paint problems
  - Rust, broken tiles, damaged flooring (extensible)
- The system uses open-source AI (CLIP, OpenCV, custom logic) for detection.
- Visual feedback: Detected defects are shown with colored bounding boxes on video frames.
- Each defect type has a unique bounding box color and a legend.
- Results include defect type, confidence score, location, and summary.
- Users can export a PDF report of findings.
- Admin/user roles supported (future extension).
- Frontend allows zoom, pan, toggle bounding box visibility, and detailed inspection.
- No external/proprietary AI dependencies.

### 1.2 Non-Functional Requirements

- **Performance:** Video processing must handle typical home inspection video lengths (<5 min) within a reasonable time (<2 min).
- **Security:** Only authenticated users can upload videos and view results (future: role-based access).
- **Scalability:** System can be extended with more detection models and hardware acceleration.
- **Reliability:** Service must handle failed uploads, corrupt videos, and partial processing gracefully.
- **Maintainability:** Codebase follows best practices, modularity, and is well-documented.

### 1.3 System Architecture Overview

- **Frontend:** React + Tailwind, modern UX, drag-and-drop, interactive results.
- **Backend:** FastAPI (Python), ML model integration, video/frame processing.
- **Database:** MongoDB for storing inspection results, user data.
- **AI Models:** Open-source (CLIP for general classification, OpenCV for cracks).
- **Deployment:** Docker/Kubernetes-ready, environment variables for config.

### 1.4 User Roles and Permissions

- **User:** Can upload videos, view and export their results.
- **Admin:** (planned) Can view all inspections, manage users, system settings.

### 1.5 External System Dependencies

- **Open-source AI models (CLIP, OpenCV)**
- **MongoDB**
- **Docker (for deployment)**
- **Create React App (frontend tooling)**
- **Tailwind CSS**

---

<a name="user-stories"></a>
## 2. User Stories and Acceptance Criteria

### Video Upload & Analysis (Core Feature)
- **As a homeowner, I want to upload a video of my house so that the AI can inspect it for defects.**
  - *Acceptance:*
    - Video can be selected via drag-and-drop or file picker.
    - Upload progress is shown.
    - System confirms successful upload or displays errors.

- **As a user, I want to see detected defects visually highlighted on my video so that I can easily locate issues.**
  - *Acceptance:*
    - Bounding boxes are shown on frames with defects.
    - A color legend maps each defect type.
    - Defect details (type, confidence) are visible.

- **As a user, I want to export a PDF report of the inspection so that I can share findings.**
  - *Acceptance:*
    - Export button is available after analysis.
    - PDF contains summary, defect images, and details.

### Interactive Visualization
- **As a user, I want to zoom and pan on video frames so that I can inspect defects closely.**
  - *Acceptance:*
    - Zoom and pan controls are present.
    - UI updates interactively.
  
- **As a user, I want to hide/show specific defect types so that I can focus on relevant issues.**
  - *Acceptance:*
    - Each defect type can be toggled.
    - Only selected boxes are shown.

### Model Transparency & Comparison (Advanced)
- **As a power user, I want to see performance metrics for the AI models so that I can trust the results.**
  - *Acceptance:*
    - Model names and metrics (detections, confidence, types) are shown.

### Error Handling
- **As a user, I want clear error messages if my video upload fails so that I know what to fix.**
  - *Acceptance:*
    - Errors are user-friendly and actionable.

---

**Stories grouped by module:**
- *Upload/Processing:* 1, 4, 5, 7
- *Visualization:* 2, 3
- *Reporting:* 6
- *AI Model Info:* 8
- *Error Handling:* 9

---

<a name="use-cases"></a>
## 3. Use Case Diagrams & Descriptions

### Use Case: Video Inspection

- **Actors:** User (Homeowner)
- **Description:** User uploads a video; system analyzes it and returns annotated results.
- **Precondition:** User is authenticated; valid video file.
- **Postcondition:** Inspection results are available for viewing and export.

### Use Case: View Results

- **Actors:** User
- **Description:** User views results, interacts with visualization (zoom, toggle boxes).
- **Precondition:** Video inspection completed.
- **Postcondition:** User understands defect locations and types.

### Use Case: Export Report

- **Actors:** User
- **Description:** User exports a PDF summary of findings.
- **Precondition:** Inspection completed.
- **Postcondition:** PDF file is downloaded.

#### (Diagram: Text Version)
```
[User] ---> (Upload Video) ---> [System: Process & Analyze] ---> (View Results w/Boxes) ---> (Export Report)
```

---

<a name="architecture"></a>
## 4. System Architecture Document

### 4.1 High-Level Architecture

- **Frontend:** React app (video upload, results display, PDF export)
- **Backend:** FastAPI app (video processing, ML inference, storage)
- **Database:** MongoDB (inspections, users)
- **AI Models:** CLIP (classification), OpenCV (cracks), custom logic (water damage, etc.)

```
User
 |
 V
React Frontend <----> FastAPI Backend <----> MongoDB
                       |         |
              [CLIP, OpenCV, Custom AI]
```

### 4.2 Technology Stack

- Frontend: React, Tailwind CSS, Axios
- Backend: Python, FastAPI, OpenCV, transformers (CLIP), PyTorch, MongoDB
- Deployment: Docker, Kubernetes (optional)

### 4.3 Communication

- REST API between frontend and backend
- JSON for request/response
- Images/frames as base64 or file blobs
- Environment variables for config

### 4.4 Database Schema (Simplified)

**Inspections**
- _id
- user_id
- video_filename
- defects: [ { frame, type, bbox, confidence } ]
- created_at, updated_at

**Users**
- _id
- username, email, password_hash
- role

---

<a name="installation"></a>
## 5. Installation Guide

### 5.1 Prerequisites

- Python 3.8+
- Node.js 18+
- MongoDB instance
- Docker (optional)

### 5.2 Environment Setup

- Clone the repo:
  ```sh
  git clone https://github.com/chandrasmailbox/AIHomeInspector.git
  cd AIHomeInspector
  ```

#### Backend

- Create a Python virtual env & install dependencies:
  ```sh
  python -m venv venv
  source venv/bin/activate
  pip install -r backend/requirements.txt
  ```
- Set environment variables:
  ```
  MONGODB_URI=mongodb://localhost:27017/inspector
  ```

- Run FastAPI backend:
  ```sh
  uvicorn backend.main:app --reload
  ```

#### Frontend

- Install dependencies:
  ```sh
  cd frontend
  npm install
  npm start
  ```
- Access at `http://localhost:3000`

#### Production

- Build frontend:
  ```sh
  npm run build
  ```
- Use Docker Compose or Kubernetes manifests (see `/deploy`) for production deployment.

---

<a name="api"></a>
## 6. API Documentation

### Authentication

- (Planned) JWT-based authentication for user login.
- For MVP: Open endpoints, but recommend securing in production.

### Endpoints

#### Upload Video

- **POST** `/api/upload`
- *Request:* `multipart/form-data` (video file)
- *Response:* `{ inspection_id, status }`
- *Status Codes:* 200, 400, 500

#### Get Analysis Results

- **GET** `/api/inspection/{id}`
- *Response:* `{ defects: [ { frame, type, bbox, confidence } ], summary }`

#### Export PDF Report

- **POST** `/api/inspection/{id}/export-pdf`
- *Body:* `{ inspection_id, include_images, ... }`
- *Response:* PDF file (content-type: application/pdf)

#### Example curl

```sh
curl -F "file=@myhouse.mp4" http://localhost:8000/api/upload
```

---

<a name="user-guide"></a>
## 7. Admin/User Guide

### Using the Application

1. Open the web app.
2. Drag-and-drop or select a home inspection video.
3. Wait for processing (progress bar shown).
4. Review results: defects are highlighted with colored boxes; legend explains colors.
5. Click boxes to toggle visibility, use zoom/pan for details.
6. Download PDF report with one click.

### Troubleshooting

- **Video won’t upload:** Check file type/size; try another browser.
- **No defects found:** Try clearer video; ensure good lighting.
- **UI not loading:** Check backend is running and API URL is correct.

---

<a name="test-strategy"></a>
## 8. Test Strategy & Test Cases

### Test Types

- **Unit Tests:** Python (pytest) for backend logic, defect detection, API endpoints.
- **Integration Tests:** End-to-end with sample videos, validating full pipeline.
- **Frontend Tests:** React Testing Library/Jest for UI components, state, and API calls.

### Sample Test Cases

- *Upload valid video, expect 200 and valid inspection_id.*
- *Upload non-video file, expect 400 error.*
- *Video with clear defects, expect non-empty defects list.*
- *Toggle bounding box, expect UI update.*

### Tools

- Pytest, requests, React Testing Library, Cypress (optional for E2E)

---

<a name="cicd"></a>
## 9. CI/CD Pipeline Explanation

### Stages

1. **Build:** Lint and build frontend/backend code.
2. **Test:** Run all tests (unit, integration).
3. **Dockerize:** Build Docker images for frontend/backend.
4. **Deploy:** Push to registry, deploy with Docker Compose/Kubernetes.

### Tools

- GitHub Actions (primary)
- Example: `.github/workflows/ci.yml`
- Environment-specific jobs (staging, prod deploy)

---

<a name="contributing"></a>
## 10. Contribution Guidelines

- Fork the repository.
- Branch: Use `feature/your-feature` or `fix/your-bug`.
- PR: Make a pull request to `main` with a clear title and description.
- Code style: Black (Python), Prettier (JS), 2-space indent.
- Lint before PR (`npm run lint`, `black .`).
- All PRs require at least 1 review.
- Please add tests for new features/bugfixes.

---

<a name="glossary"></a>
## 11. Glossary and Acronyms

- **AI:** Artificial Intelligence
- **CLIP:** Contrastive Language–Image Pre-training (OpenAI model)
- **OpenCV:** Open Source Computer Vision Library
- **FastAPI:** Modern Python web framework for APIs
- **MVP:** Minimum Viable Product
- **Bounding Box:** Rectangle drawn on an image to highlight a region of interest
- **Defect:** Any flaw in the house (mold, crack, etc.)
- **Inspection:** The process and result of analyzing a video for defects
- **Legend:** UI element explaining the color coding of defect types

---

**Other Useful Artifacts**
- All code is structured with clear separation of concerns (frontend/backend).
- The application is easily extensible (add new defect types, models, or frontend features).
- All configuration is via environment variables (no secrets in code).
- Future improvements: authentication, multi-user support, admin dashboards, more defect types.

---
