THE LINK TO MY CODE -
**https://colab.research.google.com/drive/1O12WsXxS_TTu4yjF0NfRDv98nGVf1ngR?usp=sharing**

# DETECTRON2-TRAFFIC
Used detectron2 for vehicle detection
# Traffic Vehicle Detection with Detectron2

A computer vision project for detecting and tracking vehicles in traffic videos using Facebook's Detectron2 framework with Faster R-CNN architecture.

## Overview

This project implements a vehicle detection system that can process traffic video footage and identify vehicles in real-time. The model is trained on a custom traffic dataset and can be applied to various traffic monitoring and analysis tasks.

## Features

- Vehicle detection in traffic videos using Faster R-CNN
- YOLO to COCO format conversion for training data
- Real-time video processing and annotation
- Model evaluation with COCO metrics
- Frame-by-frame output generation
- Support for custom traffic datasets

## Requirements

- Python 3.7+
- PyTorch
- Detectron2
- OpenCV
- NumPy
- Matplotlib
- PyYAML
- tqdm

## Installation

1. Install Detectron2:
```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

2. Install additional dependencies:
```bash
pip install torch torchvision torchaudio
pip install opencv-python-headless
pip install pyyaml tqdm numpy matplotlib
```

## Dataset Structure

The project expects the following directory structure:

```
traffic_data/
├── traffic_wala_dataset/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   └── data.yaml
```

## Usage

### Training

1. Prepare your dataset in YOLO format with corresponding `data.yaml` file
2. Update the paths in the configuration section
3. Run the training script to convert YOLO annotations to COCO format and train the model

### Inference

1. Load the trained model weights
2. Process video files for vehicle detection
3. Generate annotated output videos and individual frames

### Model Configuration

The model uses Faster R-CNN with ResNet-50 FPN backbone:
- Base learning rate: 0.00025
- Batch size: 2 images per batch
- ROI batch size: 128 per image
- Score threshold: 0.5 for inference

## File Structure

- `train_coco.json` - Training annotations in COCO format
- `valid_coco.json` - Validation annotations in COCO format
- `model_final.pth` - Trained model weights
- `output/` - Directory containing training outputs and results
- `frames/` - Individual annotated frames from video processing

## Model Performance

The model is evaluated using standard COCO metrics including:
- Average Precision (AP) at IoU thresholds 0.5:0.95
- Average Precision at IoU 0.5 (AP50)
- Average Precision at IoU 0.75 (AP75)
- Average Recall (AR) metrics

## Video Processing

The system processes input videos and generates:
- Annotated output video with bounding boxes around detected vehicles
- Individual frames with detection results
- Support for various video formats (MP4, AVI)

## Technical Details

### Data Preprocessing
- Converts YOLO format annotations to COCO JSON format
- Handles bounding box coordinate transformations
- Maintains class mapping consistency

### Model Architecture
- Faster R-CNN with ResNet-50 FPN backbone
- Region Proposal Network (RPN) for object proposals
- ROI Head for classification and bounding box regression

### Training Process
- Uses transfer learning from COCO pre-trained weights
- Custom trainer with COCO evaluation metrics
- Configurable training iterations and learning parameters

## Output

The system generates:
- Trained model file (`model_final.pth`)
- Annotated video files
- Individual annotated frames
- Evaluation metrics and performance statistics

## Customization

To adapt the system for different vehicle types or traffic scenarios:
1. Update the class names in `data.yaml`
2. Modify `NUM_CLASSES` in the configuration
3. Adjust detection threshold based on requirements
4. Fine-tune hyperparameters for specific use cases

## Performance Considerations

- GPU acceleration recommended for training and inference
- Video processing time depends on input resolution and length
- Memory usage scales with batch size and image resolution
- Model size approximately 160MB for the trained weights

---
  All rights reserved ©
