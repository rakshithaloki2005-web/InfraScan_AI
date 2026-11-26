# InfraScan – AI-Powered Crack & Damage Assessment for Civil Infrastructure

An AI-powered system for detecting, segmenting, and assessing cracks in civil infrastructure images using deep learning.

## Overview

InfraScan is a computer vision-based application designed to support the preliminary inspection of civil infrastructure. It uses deep learning models to identify visible cracks and generate segmentation masks that help visualize damaged regions.

The project combines two datasets and two deep learning approaches:

- **MobileNetV2** for binary crack classification.
- **U-Net** for pixel-level crack segmentation.

The application uses a FastAPI backend, a web-based dashboard, and SQLite for storing inspection records.

> **Note:** InfraScan is intended for preliminary visual assessment. It is not a replacement for professional structural inspection or engineering evaluation.

## Key Features

### 1. Crack Classification
- Uses a MobileNetV2-based classification model.
- Classifies input images as cracked or uncracked.
- Provides prediction results and confidence information.

### 2. Crack Segmentation
- Uses a U-Net architecture for pixel-level crack detection.
- Generates segmentation masks to highlight detected crack regions.
- Provides visual overlays to help users understand model predictions.

### 3. Damage Assessment
- Estimates the proportion of detected crack pixels in an image.
- Uses rule-based thresholds to categorize the detected damage level.
- Displays visual results to support preliminary assessment.

### 4. Inspection Management
- Provides a web interface for submitting inspection images.
- Stores inspection information and prediction results.
- Supports reviewing previous inspection records.

### 5. Reports and Analytics
- Generates PDF inspection reports.
- Provides visual summaries of inspection results.
- Supports structured inspection data for further analysis.

## System Architecture

```text
                 INPUT IMAGE
                      |
                      v
              IMAGE PREPROCESSING
                      |
             +--------+--------+
             |                 |
             v                 v
      CLASSIFICATION       SEGMENTATION
       MobileNetV2             U-Net
             |                 |
             v                 v
      Crack / No Crack    Crack Segmentation
             |                 |
             +--------+--------+
                      |
                      v
              DAMAGE ASSESSMENT
                      |
                      v
              FASTAPI BACKEND
                      |
             +--------+--------+
             |                 |
             v                 v
        SQLITE DATABASE    WEB DASHBOARD
             |                 |
             v                 v
       INSPECTION HISTORY  VISUAL RESULTS
                               |
                               v
                         PDF REPORTS
```

## Datasets

InfraScan uses two datasets for different computer vision tasks.

### SDNET2018 – Crack Classification

SDNET2018 is used to train and evaluate the binary crack classification model.

- **Task:** Image classification.
- **Classes:** Cracked and Uncracked.
- **Image categories:**
  - CD – Cracked Deck
  - UD – Uncracked Deck
  - CP – Cracked Pavement
  - UP – Uncracked Pavement
  - CW – Cracked Wall
  - UW – Uncracked Wall
- **Images analyzed during EDA:** 56,092.

### CRACK500 – Crack Segmentation

CRACK500 is used to train and evaluate the crack segmentation model.

- **Task:** Pixel-level image segmentation.
- **Input:** Infrastructure images.
- **Target:** Corresponding crack masks.
- **Purpose:** Identifying and highlighting crack regions within images.

**Dataset availability:** The original datasets are not included in this repository because of their size and distribution requirements. Download them from their respective official or authorized sources and organize them according to the dataset-loading paths configured in the project.

## Model Performance

The following results are the reported evaluation values and should be interpreted in the context of the project's test datasets.

### Classification – MobileNetV2

| Metric | Result |
|---|---:|
| Accuracy | 85.31% |
| Sensitivity / Recall | 74.06% |
| ROC-AUC | 0.8881 |

### Segmentation – U-Net

| Metric | Result |
|---|---:|
| Pixel Accuracy | 94.25% |
| Recall | 65.81% |
| Parameters | Approximately 483K |

**Important:** Pixel accuracy alone does not establish segmentation quality, particularly when crack pixels occupy a small portion of an image. Precision, recall, Dice score, and Intersection over Union (IoU) should also be considered when comparing segmentation models.

## Technology Stack

| Component | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | TensorFlow, Keras |
| Classification | MobileNetV2 |
| Segmentation | U-Net |
| Image Processing | OpenCV, Pillow |
| Data Analysis | NumPy, Pandas |
| Backend | FastAPI, Uvicorn, Pydantic |
| Database | SQLite |
| Report Generation | ReportLab |
| Frontend | HTML, CSS, JavaScript |
| Visualization | Chart.js |

## Project Structure

The following is a high-level representation. Keep the paths aligned with the actual folders in the repository.

```text
InfraScan_AI/
│
├── backend/
│   ├── main.py
│   ├── database.py
│   ├── routes/
│   ├── schemas/
│   └── services/
│
├── src/
│   ├── classification/
│   │   ├── dataset.py
│   │   ├── model.py
│   │   └── train.py
│   │
│   ├── segmentation/
│   │   ├── dataset.py
│   │   ├── model.py
│   │   └── train.py
│   │
│   └── dataset verification and evaluation scripts
│
├── models/
│   ├── classification/
│   │   └── best_classifier.keras
│   │
│   └── segmentation/
│       └── best_unet.keras
│
├── outputs/
│   ├── eda/
│   └── model_evaluation/
│
├── templates/
│   ├── index.html
│   └── login.html
│
├── static/
│
├── docs/
│   └── PROJECT_DOCUMENTATION.md
│
├── requirements.txt
├── README.md
├── .gitignore
├── sdnet_eda.ipynb
└── CRACK500_eda.ipynb
```

## Installation and Setup

### Prerequisites

- Python version compatible with the dependencies in `requirements.txt`.
- Git.
- VS Code or another Python development environment.

### 1. Clone the repository

```bash
git clone https://github.com/rakshithaloki2005-web/InfraScan_AI.git
cd InfraScan_AI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

**Windows CMD:**

```cmd
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Configure the datasets

Download SDNET2018 and CRACK500 from their authorized sources.

Place the datasets in the directory structure expected by the dataset-loading scripts. Check the dataset paths in:

- `src/classification/dataset.py`
- `src/segmentation/dataset.py`

### 6. Run the application

From the project root directory:

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Open the application in your browser:

```text
http://127.0.0.1:8000
```

FastAPI documentation:

```text
http://127.0.0.1:8000/docs
```

## Model Training

The classification and segmentation training scripts are maintained separately.

Run these commands from the project root when the datasets are configured:

**Classification:**

```bash
python -m src.classification.train
```

**Segmentation:**

```bash
python -m src.segmentation.train
```

Check the training scripts for configured dataset paths, hyperparameters, and output locations before starting a new training run.

## Evaluation and Outputs

The project maintains evaluation and analysis outputs, including:

- Dataset exploration and summary reports.
- Dataset verification results.
- Classification confusion matrix and training curves.
- Classification metrics and reports.
- Segmentation prediction visualizations.
- Segmentation evaluation metrics.
- Training logs and comparison images.

These outputs help document dataset quality, model training, and evaluation results.

## Testing

The repository includes scripts for application and API testing.

Examples:

```bash
python -m src.show_metrics
python -m src.test_full_app
python -m src.test_api
```

Run the scripts from the project root. Their results depend on the configured environment, installed dependencies, model files, and application state.

## Limitations

- Model predictions depend on image quality, lighting, resolution, and dataset characteristics.
- The system identifies visible crack patterns; it does not determine actual structural integrity.
- Pixel coverage is an image-based estimate, not a physical measurement of crack width or depth.
- Accurate physical measurements require image calibration and known reference dimensions.
- The system is intended for preliminary screening and should not be used as the sole basis for structural safety decisions.

## Future Enhancements

Potential improvements include:

- **Crack Progression Tracking:** Compare inspection results for the same infrastructure over time.
- **Crack Morphology Analysis:** Analyze crack length, orientation, branching, and shape.
- **Image Quality Assessment:** Detect blurry, poorly lit, or unsuitable inspection images.
- **Explainable AI:** Improve interpretability through model-focused visual explanations.
- **Maintenance Prioritization:** Provide preliminary maintenance recommendations based on inspection records.
- **Digital Inspection Records:** Maintain structured digital histories for infrastructure assets.

These are proposed enhancements and should not be considered implemented unless they are added and tested in the application.

## Disclaimer

InfraScan is an academic and development project for AI-assisted visual crack detection and preliminary damage assessment.

It does not replace professional civil engineering inspection, structural analysis, non-destructive testing, or expert judgment. All predictions and damage categories are indicative and should be verified by qualified professionals before maintenance or safety decisions are made.

## Author

**Rakshitha L**

Computer Science and Engineering

GitHub: [rakshithaloki2005-web](https://github.com/rakshithaloki2005-web)

---

If you find this project useful, consider giving the repository a star.