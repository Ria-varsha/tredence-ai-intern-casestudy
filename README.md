# tredence-ai-intern-casestudy
Self-Pruning Neural Network for CIFAR-10
Overview

This project implements a self-pruning neural network that learns to eliminate unnecessary connections during training. Unlike traditional pruning methods applied after training, this approach integrates a learnable gating mechanism directly into the model, enabling dynamic sparsification.

The implementation is built using PyTorch and evaluated on the CIFAR-10 dataset to demonstrate the trade-off between model accuracy and sparsity.

Core Idea

Each weight in the network is paired with a learnable gate parameter. During the forward pass, weights are scaled by the sigmoid of their corresponding gate scores, producing an effective weight.

As training progresses, the model learns to push unimportant gates toward zero, effectively pruning those connections without requiring a separate pruning step.

Loss Formulation

The training objective combines classification performance with sparsity enforcement:

Total Loss = CrossEntropyLoss + lambda * SparsityLoss

The sparsity loss is defined as the sum of all gate values across the network. This L1-based penalty encourages many gates to approach zero, resulting in a sparse model while preserving important connections.

Model Design

The network is a fully connected feed-forward architecture operating on flattened CIFAR-10 images.

Each hidden block consists of:

Prunable linear layer
Batch normalization
GELU activation
Dropout

All standard linear layers are replaced with custom prunable layers, ensuring that every weight in the network can be selectively pruned.

Training and Evaluation

The model is trained using the Adam optimizer with a cosine annealing learning rate schedule. Standard data augmentation is applied during training.

Performance is evaluated using:

Test accuracy
Sparsity level, defined as the proportion of weights with gate values below a small threshold

Multiple values of the sparsity coefficient (lambda) are used to study the trade-off between model performance and compression.

Results Summary

The model successfully learns to prune itself during training. Increasing the sparsity coefficient leads to higher compression, while excessively large values can impact accuracy. The results demonstrate a clear and expected trade-off between efficiency and predictive performance.

Implementation Highlights
Custom prunable linear layer built from scratch with full gradient flow
Joint optimization of weights and gate parameters
Integrated sparsity regularization within the training loop
Modular and reproducible training pipeline
Visualization of gate distributions to validate pruning behavior
Running the Project

Install dependencies:

pip install torch torchvision matplotlib numpy

Run the training script:

python solution.py --epochs 30 --lambdas 1e-5 1e-4 1e-3

The script will train the model, evaluate performance, and generate output artifacts including metrics and visualizations.

Conclusion

This project demonstrates how sparsity can be learned directly during training through a principled combination of gating mechanisms and L1 regularization. The approach provides an effective way to reduce model complexity while maintaining competitive performance.
