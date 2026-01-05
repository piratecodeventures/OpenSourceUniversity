# Appendix C: The Glossary

## A
*   **Activation Function**: A non-linear function (ReLU, Sigmoid) applied to neuron outputs to enable learning complex patterns.
*   **Attention Mechanism**: A technique allowing models to focus on specific parts of the input sequence when generating output. $Softmax(QK^T / \sqrt{d})V$.
*   **Autoencoder**: A neural network trained to copy its input to its output via a bottleneck (compression).

## B
*   **Backpropagation**: The algorithm for calculating gradients of the loss function w.r.t weights by applying the Chain Rule backwards.
*   **Batch Normalization**: Identifying and normalizing layer inputs to mean 0 and var 1 to stabilize training.
*   **Bias (Model)**: The simplifying assumptions made by a model (e.g., assuming linear).
*   **Bias (Ethical)**: Systematic prejudice in data or predictions.

## C
*   **CNN (Convolutional Neural Network)**: A network using filters to capture spatial hierarchies (edges -> shapes -> objects).
*   **Cross-Entropy**: The standard loss function for classification tasks, measuring simple distance between probability distributions.

## D
*   **Data Augmentation**: Artificially increasing training data by modifying existing data (rotations, flips).
*   **Dropout**: A regularization technique where random neurons are ignored during training to prevent overfitting.

## E
*   **Embedding**: A dense vector representation of a discrete variable (word, user ID) where similar items are close in space.
*   **Epoch**: One complete pass through the entire training dataset.

## F
*   **Fine-Tuning**: Taking a pre-trained model and training it further on a specific task/dataset.
*   **Forward Pass**: The computation of output from input.

## G
*   **GAN (Generative Adversarial Network)**: A framework where a Generator and Discriminator compete.
*   **Gradient Descent**: An optimization algorithm that iteratively moves weights in the opposite direction of the gradient.

## H
*   **Hyperparameter**: A configuration external to the model (Learning Rate, Batch Size) set before training.

## L
*   **Latent Space**: A compressed, abstract representation of data found in the bottleneck of AE/GANs.
*   **Loss Function**: A metric quantifying how "wrong" the model's prediction is (e.g., MSE).

## O
*   **Overfitting**: When a model learns noise in the training data and fails to generalize to new data.

## P
*   **Parameter**: Internal variables (Weights, Biases) learned by the model.
*   **Perceptron**: The simplest unit of a neural network. Linear classifier: $f(x) = \text{step}(w \cdot x + b)$.

## R
*   **Regularization**: Techniques (L1, L2, Dropout) used to penalize complex models and reduce overfitting.
*   **RNN (Recurrent Neural Network)**: A network with loops, allowing information to persist. Good for sequences.

## S
*   **Softmax**: A function that turns a vector of $K$ real values into a probability distribution summing to 1.
*   **Stochastic Gradient Descent (SGD)**: Gradient descent using a single (or mini-batch) sample to estimate the gradient.

## T
*   **Tensor**: A multidimensional array. The fundamental data structure in PyTorch/TensorFlow.
*   **Token**: A chunk of text (word, sub-word, character) processed by an NLP model.
*   **Transformer**: An architecture based solely on self-attention, parallelizable and dominant in NLP.

## U
*   **Underfitting**: When a model is too simple to capture the underlying structure of the data.

## V
*   **Validation Set**: A subset of data used to tune hyperparameters, distinct from Train and Test sets.
*   **Vanishing Gradient**: A problem where gradients become too small during backprop, stopping early layers from learning (Fixed by ReLU, ResNets).
