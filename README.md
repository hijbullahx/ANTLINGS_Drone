# 🚁 ANTLINGS Drone Computer Vision Pipeline

**Autonomous Drone-Based Detection, Tracking & Counting System**

A high-performance computer vision pipeline for autonomous aerial systems built on YOLOv26m for real-time multi-class object detection, persistent multi-object tracking via ByteTrack, and dynamic zone-based counting for autonomous drone surveillance.

**Performance Metrics (VisDrone Dataset):**
* **mAP@50:** 0.459
* **Precision:** 0.570
* **Recall:** 0.457

## ✨ Features
* **High-Speed Real-Time Analytics:** Processes drone footage at 20-25 FPS using **YOLOv26m** and **ByteTrack** for persistent multi-object ID assignment.
* **Dynamic Zone Counting:** Utilizes the Ultralytics 8.4 Solutions API to create a responsive, full-frame polygon margin, accurately tracking objects entering and exiting the drone's field of view.
* **High-Precision Micro-Detection (SAHI):** Integrates Slicing Aided Hyper Inference (SAHI) as an advanced capability to preserve pixel density and maximize recall for tiny, high-altitude objects that standard downsampling would miss.

## 📁 Repository Structure

```text
ANTLINGS_Drone_CV/
│
├── ANTLINGS_Drone.ipynb           # Main Jupyter notebook with full pipeline
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── .gitignore                     # Git ignore rules
│
├── dataset/                       # Input data
│   ├── archive.zip                # Dataset archive
│   ├── Drone Street Traffic, New York City.mp4
│   └── Recording 2026-05-14 213602.mp4
│
├── runs/                          # Training outputs & model weights
│   ├── yolo26m_visdrone_final/    # First training run
│   │   ├── args.yaml              # Training configuration
│   │   └── weights/               # Model weights
│   │
│   └── yolo26m_visdrone_final-2/  # Final training run (best performance)
│       ├── args.yaml              # Training configuration
│       ├── weights/
│       │   ├── best.pt            # ⭐ Best model weights
│       │   └── last.pt            # Last epoch weights
│       ├── results.csv            # Training metrics
│       ├── results.png            # Metrics visualization
│       ├── confusion_matrix.png
│       └── [training visualizations & batch samples]
│
├── outputs/                       # Inference results
│   ├── result_test_drone.mp4      # Detection results
│   ├── result_sahi_drone.mp4      # SAHI inference results
│   ├── result_tracking_drone.mp4  # ByteTrack results
│   └── result_counting_drone.mp4  # Zone counting results
│
└── .git/                          # Git repository

```

### Folder Details

| Folder | Purpose |
| --- | --- |
| `runs/` | Contains all training runs, model weights (.pt files), training curves, and validation metrics from YOLOv26m training on VisDrone dataset |
| `outputs/` | Stores inference results from running the pipeline on drone video inputs, including detection, tracking, and counting visualizations |
| `dataset/` | Input drone video samples and dataset archives for model inference and validation |

## 🚀 Installation & Usage

### Prerequisites

Ensure you have Python 3.9+ installed. A GPU (NVIDIA CUDA) is highly recommended for real-time video processing.

### Setup

1. **Clone the repository:**
```bash
git clone [https://github.com/hijbullahx/ANTLINGS_Drone](https://github.com/hijbullahx/ANTLINGS_Drone)
cd ANTLINGS_Drone_CV

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


3. **Download model weights & demo videos:**
* 🔗 **[Download Weights & Demo Videos Here](https://drive.google.com/drive/folders/1IyLGmpb3vxCB2ry_W58nrFfKULAvuiG3?usp=sharing)**
* Extract `best.pt` to `runs/yolo26m_visdrone_final-2/weights/`
* Place demo videos in the `outputs/` folder.


### Quick Start Example (Zone Counting + Tracking)

```python
import cv2
from ultralytics.solutions import object_counter

# Initialize the 8.4 Object Counter (Handles model, tracking, and counting natively)
counter = object_counter.ObjectCounter(
    show=False,
    region=[(2, 2), (1918, 2), (1918, 1078), (2, 1078)], # Full-frame adaptive margin
    model='runs/yolo26m_visdrone_final-2/weights/best.pt',
    tracker="bytetrack.yaml",
    conf=0.25
)

# Process video
cap = cv2.VideoCapture('dataset/Drone_Sample.mp4')

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
        
    # Detects, tracks, and counts in a single line
    results = counter(frame)
    
    cv2.imshow('Drone Detection & Counting', results.plot_im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

```

## 📊 Training Details

The model was fine-tuned on the VisDrone 2019 dataset for 50 epochs using an NVIDIA Tesla T4 GPU. The architecture leverages YOLOv26m (Medium) to balance parameter count with real-time inference speed. See `runs/yolo26m_visdrone_final-2/` for detailed precision-recall curves, confusion matrices, and validation batches.

## 🔍 Model Classes (VisDrone Standard)

```text
0: pedestrian
1: people
2: bicycle
3: car
4: van
5: truck
6: tricycle
7: awning-tricycle
8: bus
9: motor

```

## 📦 Dependencies

* `ultralytics>=8.4.0`
* `opencv-python>=4.8.0`
* `sahi>=0.11.0`
* `lapx>=0.5.5`
* `tqdm>=4.66.0`

## 🎯 Features Pipeline

```text
Drone Video Input
    ↓
Frame Extraction
    ↓
YOLOv26m Detection (~25 FPS)
    ↓
ByteTrack Multi-Object Tracking
    ↓
Dynamic Zone-Based Counting (In/Out)
    ↓
Visualization & Output (.mp4)

```

## 🧠 Engineering & Problem Solving Analysis
* **Dataset & Preprocessing:** The VisDrone 2019 dataset presents extreme scale variations. Instead of static image manipulation, this pipeline leverages YOLO's native PyTorch dataloader for dynamic, in-memory letterboxing, Mosaic augmentations, and HSV color-space shifts to ensure robustness against aerial lighting without storage bloat.
* **Challenges Faced:** High-resolution 4K drone footage suffers from "data loss via downsampling," causing micro-objects (like background pedestrians) to vanish when resized to 640x640 for standard inference.
* **System Strengths:** The pipeline offers a dual-capability engine. It can run high-speed real-time tracking (20-25 FPS) using ByteTrack, or it can be switched to SAHI mode to preserve pixel density and vastly increase recall for microscopic objects.
* **System Limitations:** While SAHI solves the downsampling problem, the overlapping inference grids significantly reduce FPS, making it better suited for post-flight deep analytics rather than real-time edge deployment.


## 📥 Download Weights & Demo Videos

⚠️ **Model weights and high-res demo videos are too large for GitHub.**

**🔗 [Download Weights & Demo Videos Here](https://drive.google.com/drive/folders/1IyLGmpb3vxCB2ry_W58nrFfKULAvuiG3?usp=sharing)**

## 🔧 Troubleshooting

| Issue | Solution |
| --- | --- |
| `ModuleNotFoundError: ultralytics` | Run `pip install -r requirements.txt` |
| CUDA out of memory | Reduce frame resolution or use CPU inference |
| No GPU detected | Ensure CUDA drivers installed or add `device=0` to model parameters |
| Zero Detections in Streamlit | Ensure the input image is explicitly converted to BGR format using OpenCV before passing to YOLO |

## 👨‍💻 Author

**Md. Taher Bin Omar Hijbullah** 
