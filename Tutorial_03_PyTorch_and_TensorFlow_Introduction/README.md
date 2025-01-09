
# PyTorch and TensorFlow Beginner's Guide

This guide introduces how to install PyTorch and TensorFlow, along with basic tensor operation examples.

---

## Table of Contents
1. [PyTorch Installation Guide](#1-pytorch-installation-guide)  
2. [Introduction to PyTorch Tensors](#2-introduction-to-pytorch-tensors)  
3. [Basic PyTorch Tensor Operations](#3-basic-pytorch-tensor-operations)  
4. [TensorFlow Installation Guide](#4-tensorflow-installation-guide)  
5. [Introduction to TensorFlow Tensors](#5-introduction-to-tensorflow-tensors)  
6. [Basic TensorFlow Tensor Operations](#6-basic-tensorflow-tensor-operations)  

---

## 1. PyTorch Installation Guide

PyTorch is a popular deep learning framework that supports both CPU and GPU.

### 1.1 Install PyTorch (CPU Version)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 1.2 Install PyTorch (GPU Version)

Ensure that **CUDA** and **cuDNN** are installed, then run:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

> **Note**:  
> `cu118` corresponds to CUDA 11.8. Replace it with your appropriate CUDA version, such as `cu117` for CUDA 11.7.  
> Use the `nvidia-smi` command to confirm your CUDA version.

### Official Installation Guide Link

Refer to the PyTorch official installation page for the latest instructions:  
[PyTorch Official Installation Guide](https://pytorch.org/get-started/locally/)

---

## 2. Introduction to PyTorch Tensors

The core data structure in PyTorch is the **Tensor**, which represents multi-dimensional arrays similar to NumPy arrays but with GPU acceleration.

### Example: Create a Simple Tensor

```python
import torch

# Create a 2x2 tensor
tensor = torch.tensor([[1, 2], [3, 4]])
print("PyTorch Tensor:
", tensor)
```

---

## 3. Basic PyTorch Tensor Operations

### 3.1 Tensor Addition

```python
import torch

tensor1 = torch.tensor([[1, 2], [3, 4]])
tensor2 = torch.tensor([[5, 6], [7, 8]])

# Tensor addition
result = tensor1 + tensor2
print("Tensor Addition Result:
", result)
```

### 3.2 Element-wise Multiplication

```python
import torch

tensor1 = torch.tensor([[1, 2], [3, 4]])
tensor2 = torch.tensor([[2, 3], [4, 5]])

# Element-wise multiplication
result = tensor1 * tensor2
print("Element-wise Multiplication Result:
", result)
```

### 3.3 Matrix Multiplication

```python
import torch

matrix1 = torch.tensor([[1, 2], [3, 4]])
matrix2 = torch.tensor([[5, 6], [7, 8]])

# Matrix multiplication
result = torch.matmul(matrix1, matrix2)
print("Matrix Multiplication Result:
", result)
```

---

## 4. TensorFlow Installation Guide

TensorFlow is another popular deep learning framework that supports both CPU and GPU.

### 4.1 Install TensorFlow (CPU Version)

```bash
pip install tensorflow --upgrade
```

### 4.2 Install TensorFlow (GPU Version)

Ensure **CUDA** and **cuDNN** are installed, then run:

```bash
pip install tensorflow --upgrade
```

> **Note**: TensorFlow 2.0+ automatically detects GPU support when the correct NVIDIA drivers and CUDA toolkit are installed.

### Official Installation Guide Link

Refer to the TensorFlow official installation page for the latest instructions:  
[TensorFlow Official Installation Guide](https://www.tensorflow.org/install)

---

## 5. Introduction to TensorFlow Tensors

The core data structure in TensorFlow is the **Tensor**, which represents multi-dimensional arrays (similar to matrices).

### Example: Create a Constant Tensor

```python
import tensorflow as tf

# Create a constant tensor
tensor = tf.constant([[1, 2], [3, 4]])
print("TensorFlow Tensor:
", tensor)
```

---

## 6. Basic TensorFlow Tensor Operations

### 6.1 Tensor Addition

```python
import tensorflow as tf

tensor1 = tf.constant([[1, 2], [3, 4]])
tensor2 = tf.constant([[5, 6], [7, 8]])

# Tensor addition
result = tf.add(tensor1, tensor2)
print("Tensor Addition Result:
", result)
```

### 6.2 Element-wise Multiplication

```python
import tensorflow as tf

tensor1 = tf.constant([[1, 2], [3, 4]])
tensor2 = tf.constant([[2, 3], [4, 5]])

# Element-wise multiplication
result = tf.multiply(tensor1, tensor2)
print("Element-wise Multiplication Result:
", result)
```

### 6.3 Matrix Multiplication

```python
import tensorflow as tf

matrix1 = tf.constant([[1, 2], [3, 4]])
matrix2 = tf.constant([[5, 6], [7, 8]])

# Matrix multiplication
result = tf.matmul(matrix1, matrix2)
print("Matrix Multiplication Result:
", result)
```

---

## Next Steps

Refer to the following tutorials to practice regression and classification tasks with PyTorch and TensorFlow:  
- **`tutorial_03_torch_regression_and_classification.ipynb`**  
- **`tutorial_03_tf_regression_and_classification.ipynb`**  


## Acknowledgments
- François Chollet
- Tensorflow.org
- pytorch.org