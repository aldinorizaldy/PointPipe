# PointPipe: 3D Point Cloud Segmentation API

A production-ready MLOps implementation of **PointNet** for semantic segmentation of geospatial data, specifically optimized for the **ISPRS Vaihingen 3D Dataset**. 


## 🚀 Key Features
* **FastAPI Wrapper**: Low-latency REST API for model inference.
* **Automated Preprocessing**: Integrated spatial normalization and unit-sphere scaling for raw XYZ data.
* **Containerized**: Fully Dockerized for environment parity and easy cloud deployment.
* **Startup-Ready**: Includes health checks, error handling, and pinned dependencies.

## 🛠 Tech Stack
* **Deep Learning**: PyTorch
* **API**: FastAPI, Uvicorn
* **Infrastructure**: Docker
* **Data**: NumPy, ISPRS Vaihingen (3D)

---

## 🏃 Quick Start

### 1. Clone the Repo
```bash
git clone [https://github.com/aldinorizaldy/PointPipe.git](https://github.com/aldinorizaldy/PointPipe.git)
cd PointPipe
```

### 2. Place trained PointNet weights in the `weights/` directory: `weights/pointnet.pth` 

### 3. Build and Run with Docker 
# Build the container image
`docker build -t pointpipe-api .`

# Start the service
`docker run -p 8000:8000 pointpipe-api`

### 4. Interactive API Testing
Once running, navigate to `http://localhost:8000/docs` to use the interactive Swagger UI. Upload an `.xyz` or `.txt` point cloud and receive segmentation labels in JSON format.

### Project Structure
```
PointPipe/
├── Dockerfile          # Container recipe (Cuda-enabled runtime)
├── requirements.txt    # Pinned production dependencies
├── weights/            # Model checkpoints (.pth)
├── models/             # PointNet architecture definitions
└── src/
    ├── main.py         # FastAPI entry point & state management
    └── preprocess.py   # Spatial normalization & sampling logic
```