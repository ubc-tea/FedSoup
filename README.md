# FedSoup
The official Pytorch implementation of paper "FedSoup: Improving Generalization and Personalization in Federated Learning via Selective Model Interpolation" accepted by MICCAI 2023

Authors: Minghui Chen, Meirui Jiang, Qi Dou, Zehua Wang, Xiaoxiao Li.

## Abstract
Cross-silo federated learning (FL) enables the development of machine learning models on datasets distributed across data centers such as hospitals and clinical research laboratories. However, recent research has found that current FL algorithms face a trade-off between local and global performance when confronted with distribution shifts. Specifically, personalized FL methods have a tendency to overfit to local data, leading to a sharp valley in the local model and inhibiting its ability to generalize to out-of-distribution data. In this paper, we propose a novel federated model soup method (\textit{i.e.}, selective interpolation of model parameters) to optimize the trade-off between local and global performance. Specifically, during the federated training phase, each client maintains its own global model pool by monitoring the performance of the interpolated model between the local and global models. This allows us to alleviate overfitting and seek flat minima, which can significantly improve the model's generalization performance.

## Dataset Preparation

## Local and Global Performance Evaluation
Prepare logs and saved model directories
```
mkdir results
cd system
mkdir models
```

```
sh scripts/run_tiny_camelyon17.sh
```

## Out-of-Domain Performance Evaluation
Specify the hold-out client index (e.g., 0)
```
sh scripts/run_tiny_camelyon17_hoid0.sh
```
