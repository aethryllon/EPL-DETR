import argparse
from ultralytics import YOLO

def validate_model(model_path, data_config):
    """
    Validate the model using the specified model and data configuration
    """
    # Load the model
    model = YOLO(model_path)
    
    # Perform validation
    metrics = model.val(data=data_config)
    
    # Output validation results
    print(f"mAP50: {metrics['mAP50']}")
    print(f"mAP50-95: {metrics['mAP50-95']}")
    print(f"mAP75: {metrics['mAP75']}")
    
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate EPL-DETR model')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model file')
    parser.add_argument('--data', type=str, required=True, help='Path to the data config file')
    
    args = parser.parse_args()
    
    validate_model(args.model, args.data)