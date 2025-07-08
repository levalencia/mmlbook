
import json
import os
import sys
import logging
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init():
    """Initialize the model for inference"""
    global model
    
    try:
        logger.info("Starting model initialization...")
        
        import torch
        import torch.nn as nn
        
        # Define model architecture
        class SimpleNN(nn.Module):
            def __init__(self, input_size=784, hidden_size=128, num_classes=10):
                super(SimpleNN, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, num_classes)
              
            def forward(self, x):
                x = x.view(x.size(0), -1)
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        
        # Find model file
        model_path = None
        search_paths = [
            "/var/azureml-app/azureml-models/SimpleNN-MNIST/*/simple_nn_state_dict.pth",
            "models/simple_nn_state_dict.pth",
            "simple_nn_state_dict.pth"
        ]
        
        import glob
        for pattern in search_paths:
            matches = glob.glob(pattern)
            if matches:
                model_path = matches[0]
                break
        
        if not model_path:
            raise FileNotFoundError("Model file not found")
        
        # Load model
        model = SimpleNN()
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        
        logger.info("Model initialization complete")
        
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        raise

def run(raw_data):
    """Process inference requests"""
    try:
        import torch
        import numpy as np
        
        data = json.loads(raw_data)
        
        # Handle different input formats
        if 'data' in data:
            input_data = data['data']
            if isinstance(input_data, list) and len(input_data) == 784:
                input_tensor = torch.tensor(input_data, dtype=torch.float32).view(1, 1, 28, 28)
            else:
                input_tensor = torch.tensor(input_data, dtype=torch.float32)
                if len(input_tensor.shape) == 2:
                    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
        else:
            # Default test input
            input_tensor = torch.randn(1, 1, 28, 28)
        
        # Perform inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
        
        response = {
            'predicted_digit': int(predicted_class[0]),
            'confidence': float(torch.max(probabilities[0])),
            'probabilities': probabilities[0].tolist(),
            'status': 'success'
        }
        
        return json.dumps(response)
        
    except Exception as e:
        error_response = {
            'error': str(e),
            'status': 'error'
        }
        return json.dumps(error_response)
