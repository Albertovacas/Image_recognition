# 🧠 Face Recognition and Matching System

This project implements a system to detect faces in images, extract them, and then search for matches in other photos using the Structural Similarity Index (SSIM). It's designed for use cases like automatically recognizing people in albums or photo collections.

## 📁 Project Structure

```
.
├── data/
│   ├── people/          # Images with reference people
│   └── photos/          # Images where face matches are searched
├── model/               # Pretrained detection models
├── src/  # Main logic (FaceDetector class)
├── notebooks/    # Usage example (notebook)
```

## 🚀 Features

- Face detection using OpenCV DNN-based models.
- Automatic storage and organization of detected faces.
- Face comparison using SSIM.
- Search for images containing all reference faces.

## 🛠️ Requirements

- Python ≥ 3.8
- OpenCV
- NumPy
- scikit-image

### Installation

```bash
pip install -r requirements.txt
```

Recommended `requirements.txt` content:

```
opencv-python
numpy
scikit-image
```

## 🧪 Example Usage

```python
import cv2
from image_recognition_utils import FaceDetector

# Load model
model = cv2.dnn.readNetFromCaffe('model/deploy.prototxt', 'model/deploy.caffemodel')

# Initialize detector
detector = FaceDetector(model)

# Extract faces from data/people/
detector.get_faces()

# Find photos that contain all the reference faces
detector.find_images_with_all_faces(threshold=0.75)
```

Also check out the notebook `image_recongnition.ipynb` for a visual example.

## 📓 Notes

- The system automatically creates result directories (`results/faces`, `results/matched/...`).
- You can adjust the `threshold` to control match sensitivity.
