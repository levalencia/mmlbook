# Complete PyTorch Model Development Pipeline

This directory contains a comprehensive model development pipeline that demonstrates the complete lifecycle of a PyTorch model from training to deployment.

## 🚀 Quick Start

### Option 1: Interactive Runner (Recommended)
```bash
python run_pipeline.py
```

### Option 2: Direct Execution
```bash
# Install dependencies
pip install -r requirements.txt

# Run the complete pipeline
python fullscript.py
```

## 📋 What This Pipeline Does

The pipeline performs the following steps automatically:

1. **Environment Setup** - Imports PyTorch and creates necessary directories
2. **Data Loading** - Downloads and prepares MNIST dataset
3. **Model Training** - Trains SimpleNN with both SGD and Adam optimizers
4. **Model Evaluation** - Tests both models and selects the best performer
5. **Model Saving** - Saves models in PyTorch's recommended formats
6. **Model Optimization**:
   - **Quantization**: Reduces model size using INT8 quantization
   - **Pruning**: Removes less important parameters
7. **Model Conversion**:
   - **TorchScript**: Converts for production deployment
   - **ONNX**: Creates ONNX format for cross-platform compatibility
8. **Model Versioning** - Creates timestamped versions with metadata
9. **Azure ML Integration** (Optional):
   - Connects to Azure ML workspace
   - Registers model in Azure ML
   - Creates deployment endpoint

## 📁 Output Files

After running the pipeline, you'll find:

```
Chapter20/
├── models/                          # Model files
│   ├── simple_nn_state_dict.pth    # PyTorch state dictionary
│   ├── simple_nn_complete.pth      # Complete PyTorch model
│   ├── simple_nn_traced.pt         # TorchScript model
│   └── simple_nn.onnx             # ONNX model
├── model_versions/                  # Versioned models
│   └── SimpleNN_MNIST_v{timestamp}_{hash}/
│       ├── model.pth
│       └── metadata.json
├── data/                           # MNIST dataset
├── model_pipeline.log             # Complete execution log
├── environment.yml                # Azure ML environment
├── score.py                       # Azure ML scoring script
└── config.json                    # Azure ML config (if connected)
```

## 🔧 Configuration

### Azure ML Setup (Optional)

To enable Azure ML deployment, update the credentials in `fullscript.py`:

```python
# In setup_azure_ml() function
SUBSCRIPTION_ID = "your-subscription-id"
RESOURCE_GROUP = "your-resource-group"
WORKSPACE_NAME = "your-workspace-name"
```

### Model Architecture

The SimpleNN architecture can be modified in the `create_simple_nn_class()` function:

```python
def __init__(self, input_size=784, hidden_size=128, num_classes=10):
    # Modify hidden_size or add more layers here
```

## 📊 Monitoring and Debugging

### Log Files
- **Terminal Output**: Real-time progress with emojis and clear sections
- **model_pipeline.log**: Complete detailed log file with timestamps and function context

### Log Levels
The pipeline uses comprehensive logging with different levels:
- ✓ Success messages
- ⚠️ Warnings (non-critical issues)
- ❌ Error messages
- ℹ️ Information messages

### Common Issues and Solutions

1. **PyTorch Installation Issues**:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

2. **ONNX Conversion Errors**:
   ```bash
   pip install onnx onnxruntime
   ```

3. **Azure ML Connection Issues**:
   ```bash
   pip install azureml-core azure-ai-ml azure-identity
   az login  # Ensure you're logged into Azure CLI
   ```

4. **Quantization Platform Issues**:
   - Quantization may not work on all platforms (especially Apple Silicon)
   - The pipeline will continue if quantization fails

## 🧪 Testing Individual Components

You can test individual parts of the pipeline by importing functions:

```python
from fullscript import setup_environment, load_mnist_data, train_model

# Set up environment
torch, nn, optim, datasets, transforms, DataLoader, quantization, prune, jit, onnx = setup_environment()

# Load data
train_loader, test_loader = load_mnist_data(datasets, transforms, DataLoader)

# Train a single model
model = train_model('Adam', train_loader, nn, optim, torch)
```

## 📈 Performance Expectations

On a typical machine, the pipeline should complete in:
- **CPU**: 5-10 minutes
- **GPU**: 2-5 minutes

The MNIST model typically achieves:
- **SGD**: ~90% accuracy
- **Adam**: ~96% accuracy

## 🔄 Model Versioning

Each model run creates a unique version with:
- Timestamp
- Model hash (for integrity)
- Training metadata (accuracy, optimizer, hyperparameters)
- Full model state dictionary

## 🌐 Azure ML Deployment

If Azure ML is configured, the pipeline will:
1. Register the model in Azure ML Model Registry
2. Create a managed online endpoint
3. Deploy the model with automatic scaling
4. Provide a REST API for inference

### Testing the Deployed Model

```python
import requests
import json

# Test data (28x28 flattened to 784 values)
test_data = {"data": [0.0] * 784}  # All zeros (background)

response = requests.post(
    "YOUR_ENDPOINT_URL/score",
    headers={"Authorization": "Bearer YOUR_KEY"},
    json=test_data
)

print(response.json())
```

## 🛠️ Customization

The pipeline is designed to be easily customizable:

1. **Change the model**: Modify `create_simple_nn_class()`
2. **Add new optimizers**: Update `train_model()`
3. **Different datasets**: Replace the MNIST loading in `load_mnist_data()`
4. **Additional optimizations**: Add new functions to the pipeline
5. **Different cloud providers**: Replace Azure ML functions with AWS SageMaker, etc.

## 📝 License

This code is part of the PyTorch learning materials and is intended for educational purposes.

## 🤝 Contributing

Feel free to modify and extend this pipeline for your own projects. The modular design makes it easy to add new features or replace components. 