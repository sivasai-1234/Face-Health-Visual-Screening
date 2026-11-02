# Face Health – Visual Screening Assistant

An AI-powered computer vision project that analyzes facial images and identifies visible indicators related to eye, skin, and lip appearance using image processing and facial landmark detection.

## About the Project

**Face Health – Visual Screening Assistant** is a computer vision-based application designed to analyze selected facial regions and highlight visual indicators that may be useful for preliminary screening.

The system uses OpenCV and MediaPipe Face Mesh to process facial images and examine visible features.

> **Disclaimer:** This project is for educational and experimental purposes only. It does not provide medical diagnoses or replace professional medical advice.

## Key Features

- Facial landmark detection using MediaPipe Face Mesh.
- Image processing using OpenCV.
- Visual indicator analysis, including:
  - Scleral yellowing
  - Eye redness
  - Pallor
  - Dark circles
  - Lip dryness
- Backend API built with FastAPI.
- Frontend interface for interacting with the application.
- Analysis results accessible through API endpoints.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Backend development |
| OpenCV | Image processing |
| MediaPipe | Facial landmark detection |
| FastAPI | Backend API |
| HTML | Frontend structure |
| CSS | Frontend styling |
| JavaScript | Frontend functionality |
| Vite | Frontend development server |

## Project Structure

```text
Face-Health-Visual-Screening/
│
├── backend/
│   ├── face_health.py
│   ├── fastapi_server.py
│   ├── requirements.txt
│   └── results/
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   ├── app.js
│   ├── components/
│   └── package.json
│
├── test_images/
├── Face_Health_CVIP_Lab_Report.pdf
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/sivasai-1234/Face-Health-Visual-Screening.git
```

### 2. Navigate to the Project

```bash
cd Face-Health-Visual-Screening
```

### 3. Set Up the Python Environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

### 4. Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 5. Start the Backend Server

```bash
uvicorn fastapi_server:app --reload --host 127.0.0.1 --port 8000
```

Backend API:

`http://127.0.0.1:8000`

API documentation:

`http://127.0.0.1:8000/docs`

### 6. Start the Frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL displayed by Vite, usually:

`http://localhost:5173`

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Root endpoint |
| GET | `/health` | Health check |
| POST | `/analyze` | Submit an image for analysis |
| GET | `/results/{request_id}/{filename}` | Retrieve an analysis result |
| DELETE | `/results/{request_id}` | Delete stored results |

## Project Objective

The main objective is to explore how computer vision and facial landmark detection can be applied to visual screening. The project combines image processing, facial feature analysis, and API-based application development.

## Future Improvements

- Improve the accuracy of visual indicator detection.
- Enhance the frontend user experience.
- Add more facial feature analysis.
- Improve result visualization.
- Explore machine learning-based classification.
- Deploy the application for online access.

## Contributors

**Developed by:** Siva Sai Moddolla

Contributions, suggestions, and improvements are welcome.

## License

This project is intended for educational and learning purposes.
