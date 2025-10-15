import argparse
import sys
from ultralytics import YOLO

def train_model(config_file):
    """
    Train the model using the specified configuration file
    """
    # Load the model
    model = YOLO(config_file)
    
    # Start training
    model.train(data='dataset/data.yaml', epochs=100, imgsz=640)
    
    # Validate the model
    metrics = model.val()
    print(f"mAP50: {metrics['mAP50']}, mAP50-95: {metrics['mAP50-95']}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train EPL-DETR model')
    parser.add_argument('--config', type=str, required=True, help='Path to the model config file')
    
    args = parser.parse_args()
    
    train_model(args.config)