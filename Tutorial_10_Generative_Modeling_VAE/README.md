# Generative Modeling: Autoencoders and Variational Autoencoders (VAEs)

Autoencoders and Variational Autoencoders (VAEs) are neural network architectures widely used in generative modeling tasks, including image generation, data compression, and feature learning. These models are designed to learn and generate data by capturing meaningful representations of the input.

## Autoencoder

An autoencoder is a type of neural network with two main components: an encoder and a decoder. Its primary goal is to learn a compressed representation (latent space) of the input data while maintaining the ability to reconstruct the input.

- **Encoder**: Maps the input data to a lower-dimensional representation (latent space).  
- **Decoder**: Reconstructs the input from the encoded representation.  
- **Training Objective**: The training process minimizes reconstruction error, often using loss functions like mean squared error (MSE). This enables the network to identify essential features and patterns, effectively compressing the data into its latent space.

Once trained, the decoder can generate data by sampling from the latent space, making autoencoders useful for representation learning and data reconstruction.

## Variational Autoencoder (VAE)

A VAE extends the autoencoder by introducing probabilistic concepts, making it more suitable for generative modeling tasks. It learns not only a latent representation but also the uncertainty associated with it.

- **Encoder**: Maps input data to a probability distribution in the latent space, typically outputting the mean and variance of a multivariate Gaussian distribution.  
- **Decoder**: Generates data points from a sample drawn from the latent space.  

### Training Objectives:
1. **Reconstruction Loss**: Similar to autoencoders, it measures how well the input is reconstructed.  
2. **Regularization Loss (KL Divergence)**: Ensures the distribution in the latent space matches a standard Gaussian distribution.

The combined losses encourage the VAE to learn a structured and smooth latent space, making it easier to generate novel data points by sampling from the learned distribution.

### Key Difference:
- Autoencoders produce a deterministic latent representation for reconstruction.  
- VAEs learn a probabilistic latent space, enabling them to generate new data samples and capture data uncertainty effectively.

## Modern Autoencoders

Autoencoders have evolved significantly from their traditional applications in dimensionality reduction and feature learning. Modern autoencoders generalize the encoding and decoding processes using stochastic mappings, such as \( p_{\text{encoder}}(h | x) \) and \( p_{\text{decoder}}(x | h) \). This connection to latent variable models has elevated autoencoders as a cornerstone in generative modeling.

### Characteristics:
- Autoencoders are trained to approximate \( g(f(x)) \approx x \), focusing only on aspects of the input that resemble the training data.  
- This constraint helps the model prioritize useful properties of the data, making it highly effective for feature extraction.

Training is typically performed using techniques like minibatch gradient descent and backpropagation.

## Examples and Applications

In this tutorial, we will explore autoencoders and VAEs through the following steps:
1. **Autoencoders**:
   - Basic autoencoder concepts.
   - Image denoising.
   - Anomaly detection.
2. **Variational Autoencoder (VAE)**:
   - Introduction to VAEs.
   - Practical implementation of VAEs for generative modeling.

## Acknowledgments
- François Chollet
- Tensorflow.org
- pytorch.org

