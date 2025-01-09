# Environment Setup
---
Welcome to this tutorial! This session will guide you through setting up a powerful environment for deep learning using Python, Anaconda, and Jupyter Notebook.

## 1. Why Python for Deep Learning?
Python has emerged as the most widely used programming language for deep learning due to:
- **Extensive Libraries:** Frameworks like TensorFlow, PyTorch, and Keras simplify deep learning tasks.
- **Readable Syntax:** Python's syntax is intuitive, making it accessible for all skill levels.
- **Versatility:** Python is used for data preprocessing, modeling, visualization, and deployment.

Popular libraries include:
| Library       | Description                          | Installation Command       |
|---------------|--------------------------------------|-----------------------------|
| **NumPy**     | Numerical computations               | `pip install numpy`        |
| **Pandas**    | Data manipulation & analysis         | `pip install pandas`       |
| **Matplotlib**| Visualization of graphs and charts   | `pip install matplotlib`   |
| **Scikit-Learn** | Machine learning algorithms       | `pip install scikit-learn` |

By leveraging these libraries, Python becomes a robust tool for deep learning workflows.

## 2. What is Anaconda?
**Anaconda** is a free, open-source distribution of Python for scientific computing and machine learning. Key features include:
- **Package Manager:** Simplifies installation of libraries with `conda`.
- **Pre-installed Libraries:** Comes with 250+ packages for data science.
- **Integrated Tools:** Tools like Jupyter Notebook, Spyder, and VS Code are included.

**Installation Steps:**
1. Go to the [Anaconda Download Page](https://www.anaconda.com/products/individual).
2. Download the installer for your OS (Windows, macOS, Linux).
3. Follow on-screen instructions.

**Verify Installation:**
```bash
conda --version
python --version
```

## 3. Jupyter Notebook Overview
**Jupyter Notebook** is an interactive, web-based development tool for data science and deep learning.

### Strengths
- **Interactive Coding:** Execute code in blocks (cells) for modular development.
- **Rich Visualizations:** Supports inline graphs using libraries like Matplotlib and Plotly.
- **Documentation:** Markdown cells allow combining code with explanations.

### Install Jupyter Notebook:
If Jupyter Notebook is not installed, use the following commands:
```bash
conda install jupyter
```
Or with pip:
```bash
pip install notebook
```

### Launch Jupyter Notebook:
```bash
jupyter notebook
```
This opens the notebook interface in your web browser.

### Example Code Cell:
```python
# Test Code in Jupyter Notebook
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.title('Sine Wave')
plt.show()
```

### Verify Environment Setup:
```python
# Verify Python Environment
import sys
print('Python Version:', sys.version)

# Check NumPy Installation
import numpy as np
print('Numpy Version:', np.__version__)

# Plot a Simple Chart
import matplotlib.pyplot as plt
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
plt.plot(x, y)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Test Plot')
plt.show()
```

## 4. Install Essential Libraries
You can use `pip` or `conda` to install libraries:

### Pip Example:
```bash
pip install numpy pandas matplotlib scikit-learn
```

### Conda Example:
```bash
conda install numpy pandas matplotlib scikit-learn
```

These libraries enable efficient data processing, visualization, and machine learning workflows.

### Test Installed Libraries:
```python
# Install Libraries (if not already installed)
# Uncomment and run these lines in Jupyter Notebook
# !pip install numpy pandas matplotlib scikit-learn

import pandas as pd

# Test Pandas Library
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'Profession': ['Engineer', 'Doctor', 'Teacher']}

df = pd.DataFrame(data)
print("Sample DataFrame:")
print(df)
```

Sample Output:
```
      Name  Age Profession
0    Alice   25   Engineer
1      Bob   30     Doctor
2  Charlie   35    Teacher
```

## 5. Conclusion
You have successfully set up your Python environment using Anaconda and tested Jupyter Notebook. You also installed essential libraries and verified their functionality.
