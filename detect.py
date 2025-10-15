import argparse
import torch
from ultralytics import YOLO

def detect_image(model_path, image_path, conf_threshold=0.5):
    """
    Detect objects in an image using a trained model
    """
    # Load the model
    model = YOLO(model_path)
    
    # Perform inference
    results = model(image_path, conf=conf_threshold)
    
    # Display results
    for result in results:
        result.show()
        
        # Save results
        result.save(filename=f"result_{image_path.split('/')[-1]}")
        
    print(f"Detection completed, results saved as result_{image_path.split('/')[-1]}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect objects in an image using EPL-DETR model')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model file')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold for detection')
    
    args = parser.parse_args()
    
    detect_image(args.model, args.image, args.conf)