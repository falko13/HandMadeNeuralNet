# Handmade Neural Network Framework

## Project Status: In Progress

### Overview

This repository is dedicated to the ongoing development of a fully handmade neural network framework constructed from the ground up using pure mathematics and basic Python programming principles. The primary objective of this endeavor is to delve deeply into the mechanisms of machine learning (ML) by building the fundamental components of neural networks from scratch. This approach facilitates a comprehensive understanding of the underlying mathematics and algorithms that drive modern ML models.

### Purpose

The project serves as an platform for experimenting, exploring and mastering the intricacies of machine learning algorithms and their implementation without reliance on high-level ML libraries. By avoiding these libraries in favor of direct mathematical implementations, we aim to achieve a granular understanding of ML operations and open the door to creative experimentation that could lead to the discovery of new, more advanced models.

### Goals

- **To Build Deep Foundation**: Establish a robust foundation in neural network theory and application by manually coding the framework's components.
- **Flexibility for Experimentation**: Create a flexible environment conducive to experimentation with various neural network architectures, including Recurrent Neural Networks (RNNs) and Linear Neural Networks (LNNs).
- **Future Expansion**: Lay the groundwork for future expansions into more complex ML models and frameworks, enabling the design and testing of innovative neural network models and learning algorithms.

### Current Implementation

The repository currently includes implementations of the following components:
- **CustomNormalization**: A class for data normalization, using only numpy for numerical operations.
- **Activation Functions**: Implementation of the sigmoid function as the neural network's activation function.
- **Dense Layer**: A foundational dense (fully connected) layer implementation, capable of performing linear transformations followed by activation.
- **Sequential Model Construction**: A rudimentary framework to construct simple neural networks using sequential layer stacking.
- **Prediction Function**: A function to predict outputs for given inputs using the constructed neural network model.

### Future Directions

As this project evolves, we anticipate incorporating additional features, including but not limited to:
- Implementation of more activation functions (ReLU, tanh, softmax).
- Introduction of mechanisms for backpropagation and optimization (SGD, Adam).
- Expansion into more complex architectures for RNNs and LNNs.
- Development of a comprehensive library of loss functions and metrics for model evaluation.
