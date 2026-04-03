# Torchly - Simple PyTorch Wrapper Library
[![Python](https://img.shields.io/badge/Python-3.8+-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
A comprehensive yet simple wrapper over PyTorch for creating, training, and managing neural networks with minimal code.

## Features

- **Simple API**: Train models with one-line commands
- **Flexible Architecture**: Support for various layer types and activations
- **Easy Weight Access**: Extract and modify layer weights effortlessly
- **Save/Load**: Save and load complete models or just weights
- **Training Utilities**: Early stopping, learning rate scheduling, callbacks
- **Device Management**: Automatic GPU/CPU handling
- **Normalization**: Built-in data preprocessing
- **History Tracking**: Automatic logging of training metrics
- **Regularization**: L1/L2 regularization, dropout, batch normalization

## Installation

```bash
pip install torch numpy
```

Place `Torchly.py` in your project directory or Python path.

## Quick Start

```python
from Torchly import Model
import numpy as np

# Generate sample data
X_train = np.random.randn(1000, 10)
y_train = np.random.randn(1000, 1)

# Create model
model = Model([10, 64, 32, 1], activation="relu")

# Train
model.train([X_train], y_train, epochs=100)

# Extract weights from layer 1
weights = model.extract_layer(1)

# Save model
model.save("my_model.pt")

# Load model
loaded_model = Model.load("my_model.pt")
```

## Detailed Usage Examples

### 1. Creating Models

#### Basic Sequential Network
```python
# Simple 3-layer network
model = Model([2, 16, 16, 2], activation="relu")

# With dropout
model = Model([2, 16, 16, 2], activation="relu", dropout=0.3)

# With batch normalization
model = Model([2, 16, 16, 2], activation="relu", batch_norm=True)

# Different activation per layer
model = Model([2, 16, 16, 2], activation=["relu", "relu", "sigmoid"])

# With regularization
model = Model([2, 16, 16, 2], activation="relu", l1_reg=0.01, l2_reg=0.01)

# Specify optimizer and learning rate
model = Model([2, 16, 16, 2], activation="relu", optimizer="adam", lr=0.001)

# Set random seed for reproducibility
model = Model([2, 16, 16, 2], activation="relu", seed=42)
```

#### View Model Architecture
```python
model.summary()
# Output:
# ======================================================================
# Layer                Type                 Output Shape        
# ======================================================================
# 0                    Linear               16                  
#                      ReLU                 -                   
# 1                    Linear               16                  
#                      ReLU                 -                   
# 2                    Linear               2                   
# ======================================================================
# Total parameters: 562
# ======================================================================

# Count parameters
total = model.count_parameters()
trainable = model.count_parameters(trainable_only=True)
```

### 2. Training Models

#### Basic Training
```python
# Simple training
model.train([X_train], y_train, epochs=100)

# With batch size
model.train([X_train], y_train, epochs=100, batch_size=32)

# With validation data
model.train(
    [X_train], y_train,
    validation_data=([X_val], y_val),
    epochs=100
)

# Specify loss function
model.train([X_train], y_train, epochs=100, loss="mse")  # or "cross_entropy", "bce"
```

#### Advanced Training Options
```python
# With early stopping
model.train(
    [X_train], y_train,
    validation_data=([X_val], y_val),
    epochs=100,
    early_stopping=True,
    patience=10
)

# With learning rate scheduling
model.train(
    [X_train], y_train,
    epochs=100,
    lr_schedule="cosine"  # or "step", "exponential", "plateau"
)

# With gradient clipping
model.train(
    [X_train], y_train,
    epochs=100,
    grad_clip=1.0
)

# Control verbosity
model.train([X_train], y_train, epochs=100, verbose=0)  # Silent
model.train([X_train], y_train, epochs=100, verbose=1)  # Progress
model.train([X_train], y_train, epochs=100, verbose=2)  # Detailed
```

#### Custom Loss Function
```python
import torch.nn as nn

def custom_loss(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean() + 0.1 * torch.abs(y_pred - y_true).mean()

model.train([X_train], y_train, loss=custom_loss, epochs=100)
```

### 3. Making Predictions

```python
# Basic prediction
predictions = model.predict([X_test])

# Batch prediction (for large datasets)
predictions = model.predict([X_test], batch_size=64)

# For classification - get probabilities
probabilities = model.predict_proba([X_test])

# For classification - get class labels
classes = model.predict_classes([X_test])
```

### 4. Model Evaluation

```python
# Basic evaluation
loss, metrics = model.evaluate([X_test], y_test)

# With specific metrics
loss, metrics = model.evaluate(
    [X_test], y_test,
    metrics=["accuracy", "mse", "mae"]
)
print(f"Test Loss: {loss}")
print(f"Accuracy: {metrics['accuracy']}")
```

### 5. Layer Weight Access and Manipulation

#### Extract Weights
```python
# Extract specific layer
weights_dict = model.extract_layer(1)
print(weights_dict['weights'])  # Weight matrix
print(weights_dict['bias'])     # Bias vector

# Extract without bias
weights_dict = model.extract_layer(1, include_bias=False)

# Extract all layers
all_weights = model.extract_all_layers()
for layer_idx, layer_weights in all_weights.items():
    print(f"Layer {layer_idx}: {layer_weights['weights'].shape}")
```

#### Modify Weights
```python
# Set weights for a layer
new_weights = np.random.randn(16, 16)
new_bias = np.random.randn(16)
model.set_layer_weights(1, new_weights, new_bias)
```

#### Freeze/Unfreeze Layers
```python
# Freeze specific layer (stop training)
model.freeze_layer(0)

# Unfreeze
model.unfreeze_layer(0)

# Freeze all layers
model.freeze_all()

# Unfreeze all
model.unfreeze_all()

# Freeze multiple layers
model.freeze_layers([0, 1, 2])
```

#### Layer Information
```python
# Get layer details
info = model.layer_info(1)
print(info)
# Output: {'type': 'linear', 'input_size': 16, 'output_size': 16, 'has_bias': True}
```

### 6. Save and Load Models

#### Save Complete Model
```python
# Save everything
model.save("model.pt")

# Save with optimizer state and history
model.save("model.pt", include_optimizer=True, include_history=True)

# Save only weights
model.save_weights("weights.pt")

# Save architecture only (JSON)
model.save_architecture("architecture.json")
```

#### Load Models
```python
# Load complete model
model = Model.load("model.pt")

# Load and resume training
model = Model.load("model.pt", resume_training=True)

# Load only weights
model.load_weights("weights.pt")

# Create model from architecture JSON
model = Model.from_architecture("architecture.json")
```

### 7. Training History and Visualization

```python
# Access training history
history = model.history
print(history['loss'])      # Training loss per epoch
print(history['val_loss'])  # Validation loss per epoch

# Get specific metric
losses = model.get_history("loss")

# Plot training curves (requires matplotlib)
model.plot_history()

# Export history to CSV
model.export_history("training_history.csv")
```

### 8. Optimizer and Learning Rate Management

```python
# Change optimizer
model.set_optimizer("sgd", lr=0.01, momentum=0.9)
model.set_optimizer("adam", lr=0.001)
model.set_optimizer("rmsprop", lr=0.001)

# Get current learning rate
current_lr = model.get_lr()

# Set learning rate
model.set_lr(0.0001)
```

### 9. Data Normalization

```python
# Normalize data
X_normalized = model.normalize(X_train, method="standard")  # Z-score
X_normalized = model.normalize(X_train, method="minmax")    # Min-max

# Fit normalizer on training data
model.fit_normalizer(X_train, method="standard")

# Transform test data using fitted normalizer
X_test_normalized = model.transform(X_test)
```

### 10. Device Management (GPU/CPU)

```python
# Move to GPU
model.to_gpu()
model.to_gpu(device=1)  # Specific GPU

# Move to CPU
model.to_cpu()

# Automatic device selection
model.auto_device()  # Uses GPU if available

# Check current device
device = model.get_device()
print(f"Model is on: {device}")
```

### 11. Advanced Features

#### Get Intermediate Activations
```python
# Get activations from all layers
activations = model.get_activations([X_sample])
print(activations['layer_0'])  # Activations from layer 0

# Get activations from specific layer
activations = model.get_activations([X_sample], layer=2)
```

#### Model Cloning
```python
# Clone model with same weights
model_copy = model.clone()

# Clone with reinitialized weights
model_new = model.clone(reinitialize=True, seed=123)
```

#### Model Comparison
```python
# Compare two models
diff = Model.compare(model1, model2)
print(diff)

# Check if models are identical
is_same = model1.equals(model2)
```

#### Create DataLoader
```python
# Create PyTorch DataLoader
dataloader = model.create_dataloader(
    [X_train], y_train,
    batch_size=32,
    shuffle=True
)

# Use in custom training loop
for batch_X, batch_y in dataloader:
    # Your custom training code
    pass
```

### 12. Quick Training Mode

```python
# One-liner training with auto-configuration
model = Model.quick_fit(
    [X_train], y_train,
    task="classification",  # or "regression"
    epochs=100
)

# With custom hidden layers
model = Model.quick_fit(
    [X_train], y_train,
    task="regression",
    hidden_layers=[128, 64, 32],
    epochs=100
)
```

### 13. Random Seed Control

```python
# Set seed for reproducibility
model.set_seed(42)

# Get current seed
seed = model.get_seed()
```

## Complete Example: Binary Classification

```python
from Torchly import Model, one_hot_encode
import numpy as np

# Generate synthetic data
np.random.seed(42)
X_train = np.random.randn(1000, 20)
y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
y_train_encoded = one_hot_encode(y_train, num_classes=2)

X_test = np.random.randn(200, 20)
y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int)
y_test_encoded = one_hot_encode(y_test, num_classes=2)

# Create model
model = Model(
    [20, 64, 32, 2],
    activation="relu",
    dropout=0.3,
    batch_norm=True,
    optimizer="adam",
    lr=0.001
)

# Print architecture
model.summary()

# Train with early stopping
model.train(
    [X_train], y_train_encoded,
    validation_data=([X_test], y_test_encoded),
    epochs=200,
    batch_size=32,
    loss="cross_entropy",
    early_stopping=True,
    patience=15,
    verbose=1
)

# Evaluate
loss, metrics = model.evaluate([X_test], y_test_encoded, metrics=["accuracy"])
print(f"\nTest Accuracy: {metrics['accuracy']:.4f}")

# Make predictions
predictions = model.predict_classes([X_test])
print(f"Sample predictions: {predictions[:10]}")
print(f"Actual labels: {y_test[:10]}")

# Extract and analyze weights
layer1_weights = model.extract_layer(0)
print(f"\nFirst layer weights shape: {layer1_weights['weights'].shape}")

# Save model
model.save("classification_model.pt")

# Load and verify
loaded_model = Model.load("classification_model.pt")
loaded_predictions = loaded_model.predict_classes([X_test])
print(f"\nPredictions match: {np.array_equal(predictions, loaded_predictions)}")
```

## Complete Example: Regression

```python
from Torchly import Model
import numpy as np

# Generate synthetic regression data
np.random.seed(42)
X_train = np.random.randn(1000, 10)
y_train = np.sum(X_train[:, :3], axis=1, keepdims=True) + np.random.randn(1000, 1) * 0.1

X_test = np.random.randn(200, 10)
y_test = np.sum(X_test[:, :3], axis=1, keepdims=True) + np.random.randn(200, 1) * 0.1

# Normalize data
X_train_norm = np.copy(X_train)
model_temp = Model([10, 1])
model_temp.fit_normalizer(X_train_norm, method="standard")
X_train_norm = model_temp.transform(X_train_norm)
X_test_norm = model_temp.transform(X_test)

# Create and train model
model = Model(
    [10, 128, 64, 32, 1],
    activation=["relu", "relu", "relu", "none"],
    dropout=0.2,
    l2_reg=0.001
)

model.train(
    [X_train_norm], y_train,
    validation_data=([X_test_norm], y_test),
    epochs=100,
    batch_size=32,
    loss="mse",
    lr_schedule="plateau",
    verbose=1
)

# Evaluate
test_loss, metrics = model.evaluate([X_test_norm], y_test, metrics=["mse", "mae"])
print(f"\nTest MSE: {metrics['mse']:.4f}")
print(f"Test MAE: {metrics['mae']:.4f}")

# Make predictions
predictions = model.predict([X_test_norm])

# Compare predictions
print("\nSample Predictions vs Actual:")
for i in range(5):
    print(f"Predicted: {predictions[i][0]:.3f}, Actual: {y_test[i][0]:.3f}")

# Plot training history
model.plot_history()

# Save
model.save("regression_model.pt")
```

## Activation Functions

Supported activation functions:
- `relu` - Rectified Linear Unit
- `tanh` - Hyperbolic Tangent
- `sigmoid` - Sigmoid
- `leaky_relu` - Leaky ReLU
- `elu` - Exponential Linear Unit
- `selu` - Scaled Exponential Linear Unit
- `softmax` - Softmax (use for multi-class classification output)
- `none` - No activation (Identity)

## Loss Functions

Supported loss functions:
- `mse` - Mean Squared Error (regression)
- `mae` - Mean Absolute Error (regression)
- `cross_entropy` - Cross Entropy (multi-class classification)
- `bce` - Binary Cross Entropy (binary classification)
- `bce_logits` - Binary Cross Entropy with Logits
- `huber` - Huber Loss (robust regression)
- Custom functions (provide callable)

## Optimizers

Supported optimizers:
- `adam` - Adam (adaptive learning rate)
- `sgd` - Stochastic Gradient Descent
- `rmsprop` - RMSProp
- `adamw` - AdamW (Adam with weight decay)

## Learning Rate Schedules

Supported LR schedules:
- `step` - Step decay
- `exponential` - Exponential decay
- `cosine` - Cosine annealing
- `plateau` - Reduce on plateau

## API Reference Summary

### Model Creation
- `Model(architecture, ...)` - Create new model
- `Model.load(filepath)` - Load model from file
- `Model.from_architecture(filepath)` - Load architecture from JSON
- `Model.quick_fit(X, y, task)` - Quick training

### Training
- `train(X, y, epochs, ...)` - Train model
- `evaluate(X, y, metrics)` - Evaluate model
- `predict(X)` - Make predictions
- `predict_proba(X)` - Get probabilities
- `predict_classes(X)` - Get class labels

### Layer Operations
- `extract_layer(layer)` - Get layer weights
- `extract_all_layers()` - Get all weights
- `set_layer_weights(layer, weights, bias)` - Set weights
- `freeze_layer(layer)` - Freeze layer
- `unfreeze_layer(layer)` - Unfreeze layer
- `freeze_all()` - Freeze all layers
- `unfreeze_all()` - Unfreeze all layers
- `layer_info(layer)` - Get layer information

### Model Information
- `summary()` - Print model architecture
- `count_parameters()` - Count parameters
- `get_device()` - Get current device
- `get_lr()` - Get learning rate
- `get_history(metric)` - Get training history

### Save/Load
- `save(filepath)` - Save complete model
- `save_weights(filepath)` - Save only weights
- `save_architecture(filepath)` - Save architecture JSON
- `load_weights(filepath)` - Load weights

### Device Management
- `to_gpu()` - Move to GPU
- `to_cpu()` - Move to CPU
- `auto_device()` - Auto select device

### Optimizer Management
- `set_optimizer(type, lr)` - Set optimizer
- `set_lr(lr)` - Set learning rate

### Data Utilities
- `normalize(X, method)` - Normalize data
- `fit_normalizer(X)` - Fit normalizer
- `transform(X)` - Transform with normalizer
- `create_dataloader(X, y, ...)` - Create DataLoader

### Advanced
- `get_activations(X, layer)` - Get intermediate activations
- `clone()` - Clone model
- `set_seed(seed)` - Set random seed
- `plot_history()` - Plot training curves
- `export_history(filepath)` - Export history to CSV

### Static Methods
- `Model.compare(model1, model2)` - Compare models
- `one_hot_encode(y, num_classes)` - One-hot encode labels

## Requirements

- Python 3.7+
- PyTorch 1.7+
- NumPy 1.19+
- (Optional) Matplotlib for plotting

## License

MIT License

## Contributing

Feel free to extend this library with additional features!

## Tips and Best Practices

1. **Always normalize your data** for better training stability
2. **Use early stopping** to prevent overfitting
3. **Start with a simple architecture** and gradually increase complexity
4. **Set random seed** for reproducible experiments
5. **Monitor validation loss** to detect overfitting
6. **Use dropout and regularization** for better generalization
7. **Experiment with learning rates** - use learning rate scheduling
8. **Save checkpoints** during long training runs

## Troubleshooting

### Out of Memory (GPU)
```python
# Reduce batch size
model.train([X], y, batch_size=16)  # instead of 32

# Move to CPU
model.to_cpu()
```

### Model Not Learning
```python
# Try different learning rate
model.set_lr(0.01)

# Try different optimizer
model.set_optimizer("sgd", lr=0.01, momentum=0.9)

# Check for dead ReLU neurons - try different activation
model = Model([...], activation="leaky_relu")
```

### Overfitting
```python
# Add dropout
model = Model([...], dropout=0.5)

# Add regularization
model = Model([...], l2_reg=0.01)

# Use early stopping
model.train([X], y, early_stopping=True, patience=10)
```

## Future Enhancements

Planned features:
- Convolutional layers (Conv2D, MaxPool, etc.)
- Recurrent layers (LSTM, GRU)
- Hyperparameter tuning (grid search, random search)
- Model ensemble support
- TensorBoard integration
- ONNX export
- More advanced callbacks
- Cross-validation utilities

---

Happy Training! 🚀
