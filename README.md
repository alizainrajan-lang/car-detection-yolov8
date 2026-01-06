# Car Detection using YOLOv8

## Introduction
This project implements a car detection system using the YOLOv8 (You Only Look Once version 8) algorithm. It is designed to detect vehicles in static images with high accuracy and efficiency. This solution addresses the need for robust vehicle detection in intelligent transportation systems, parking management, and road safety analysis.

## Methodology
The system pipeline consists of:
1.  **Data Preparation**: Using the COCO format (or converting custom datasets like KITTI/Stanford Cars). Images are resized to 640x640.
2.  **Model Selection**: YOLOv8n (Network) for a balance of speed and accuracy.
3.  **Training**: Transfer learning from pretrained weights.
4.  **Inference**: Filtering results to detect specific vehicle classes (Car, Bus, Truck).

## System Architecture
Input Image -> YOLOv8 Feature Extraction -> Bounding Box Regression & Classification -> Filtered Output (Cars)

## Tools
-   **Python 3.x**
-   **Ultralytics YOLOv8**
-   **OpenCV**
-   **Matplotlib**

## Installation
1.  Navigate to the project directory.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Training the Model
To train the model on the dataset (default: `coco128`):
```bash
python train_model.py
```
This will download the `coco128` dataset automatically on the first run and train the model for 5 epochs. The best weights will be saved in `car_detection_runs/yolov8n_car/weights/best.pt`.

### 2. Evaluating the Model
To evaluate the trained model's performance (mAP, Precision, Recall):
```bash
python eval_model.py
```

### 3. Detecting Cars
To run detection on an image:
```bash
python detect_cars.py --image path/to/your/image.jpg
```
If no image is provided, it will attempt to use a sample image from the downloaded dataset. The output image with bounding boxes will be saved in the `car_detection_runs/predict` folder.

## Custom Datasets
To use a custom dataset (like KITTI):
1.  Prepare your dataset in YOLO format (images and .txt labels).
2.  Create a strict `custom.yaml` file pointing to your train/val paths and defining class names.
3.  Update `train_model.py` to point to `custom.yaml` instead of `coco128.yaml`.

## Results
The model outputs bounding boxes around detected vehicles with confidence scores. Initial testing on `coco128` demonstrates the capability to identify cars, buses, and trucks.
