# Text and Sequences: TensorFlow and PyTorch Tutorials

This tutorial provides a comprehensive introduction to processing and modeling text data using TensorFlow and PyTorch. Learn how to represent text numerically, train word embeddings, and build sentiment classification models.

## Overview
- **Text Representation**: Strategies to convert text into numeric data.
- **Word Embeddings**: Dense, trainable vectors representing words in high-dimensional space.
- **Embedding Layers**:
  - TensorFlow: Keras `Embedding` layer.
  - PyTorch: `nn.Embedding` layer.
- **Natural Language Processing (NLP)**:
  - Sentiment classification.
  - Skip-Gram and Negative Sampling models.

## Key Topics
### 1. Text Representation
- **One-Hot Encoding**:
  - Sparse representation of words.
- **Integer Encoding**:
  - Assign unique numbers to words.
- **Word Embeddings**:
  - Efficient and dense representation learned during training.

### 2. Building and Training Models
- **TensorFlow**:
  - "Continuous Bag of Words" (CBOW) style sentiment classification model.
  - Key layers: `TextVectorization`, `Embedding`, `GlobalAveragePooling1D`, and `Dense`.
  - Save and visualize trained embeddings.
- **PyTorch**:
  - Using `nn.Embedding` for word embeddings.
  - Techniques to train and visualize word vectors.

### 3. Advanced NLP Concepts
- **Bag of Words (BoW)**:
  - Binary vectors to indicate word presence in documents.
- **Word2Vec**:
  - Continuous Bag-of-Words and Skip-Gram models for learning word representations.
- **Skip-Gram with Negative Sampling**:
  - Predict context words for a target word.

## Learning Outcomes
- Understand how to preprocess and represent text data for machine learning.
- Build and train models using TensorFlow and PyTorch embedding layers.
- Explore advanced NLP techniques, including Word2Vec and Skip-Gram models.

## Acknowledgments
- François Chollet
- Tensorflow.org
- pytorch.org