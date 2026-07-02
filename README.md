# Face Recognition Attendance System

An AI Engineer portfolio project for real-time face recognition attendance. The system detects faces, generates deep CNN embeddings with a pretrained FaceNet model, matches identities using cosine similarity, and logs attendance in SQLite through a FastAPI service.

## Features

- Real-time face detection and alignment with MTCNN
- Face recognition with pretrained FaceNet embeddings
- Cosine similarity matching with configurable threshold
- Duplicate attendance prevention within a configurable time window
- SQLite database for users, embeddings, and attendance logs
- REST API for registration, recognition, and attendance retrieval
- Webcam script for local demos
- Dockerfile for reproducible deployment
- Unit tests for matching, persistence, and attendance logic

## Architecture

```text
Webcam / Image Upload
        |
        v
MTCNN face detector -> aligned face crop
        |
        v
FaceNet embedding model -> normalized vector
        |
        v
Cosine similarity matcher -> known user / unknown face
        |
        v
Attendance service -> SQLite log with duplicate protection
```

## Project Structure

```text
face-attendance-ai/
|-- app/
|   |-- api/                 # FastAPI routes and dependencies
|   |-- core/                # Settings
|   |-- db/                  # SQLite schema and connection helper
|   |-- services/            # Face encoder, matcher, attendance logic
|   |-- utils/               # Image decoding helpers
|   `-- main.py              # FastAPI application factory
|-- scripts/
|   |-- register_from_folder.py
|   `-- run_webcam.py
|-- tests/
|-- data/
|-- Dockerfile
|-- requirements.txt
`-- README.md
```

## Setup

Use Python 3.11.

```bash
cd face-attendance-ai
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Start the API:

```bash
uvicorn app.main:app --reload
```

Open the API docs at:

```text
http://127.0.0.1:8000/docs
```

## API Usage

Register a face:

```bash
curl -X POST "http://127.0.0.1:8000/register_face" ^
  -F "external_id=EMP001" ^
  -F "name=Asha Kumar" ^
  -F "image=@samples/asha.jpg"
```

Recognize faces and mark attendance:

```bash
curl -X POST "http://127.0.0.1:8000/recognize?mark_attendance=true" ^
  -F "image=@samples/classroom.jpg"
```

Fetch attendance logs:

```bash
curl "http://127.0.0.1:8000/attendance?limit=100"
```

List registered users:

```bash
curl "http://127.0.0.1:8000/users"
```

## Register Users From a Folder

Create folders using this format:

```text
dataset/
|-- EMP001__Asha_Kumar/
|   |-- 1.jpg
|   `-- 2.jpg
`-- EMP002__Ravi_Shah/
    |-- 1.jpg
    `-- 2.jpg
```

Then run:

```bash
python scripts/register_from_folder.py dataset
```

## Webcam Demo

After registering at least one person:

```bash
python scripts/run_webcam.py --camera 0
```

Press `q` to quit.

## Configuration

Environment variables use the `FACE_ATTENDANCE_` prefix.

```text
FACE_ATTENDANCE_DATABASE_PATH=data/attendance.db
FACE_ATTENDANCE_RECOGNITION_THRESHOLD=0.62
FACE_ATTENDANCE_DUPLICATE_WINDOW_MINUTES=480
FACE_ATTENDANCE_DEVICE=cpu
```

For GPU inference, install the matching CUDA-enabled PyTorch build and set:

```text
FACE_ATTENDANCE_DEVICE=cuda
```

## Docker

```bash
docker build -t face-attendance-ai .
docker run --rm -p 8000:8000 -v "%cd%/data:/app/data" face-attendance-ai
```

## Tests

```bash
pytest
```

The tests use a fake encoder so they validate the system logic without downloading model weights.


