# **LightGC²N: A Lightweight Graph Capsule Convolutional Network with Subspace Alignment (Tensorflow)** 

<p align="left">
  <img src='https://img.shields.io/badge/python-3.8.18-blue'>
  <img src='https://img.shields.io/badge/nvidia_tensorflow-1.15.4+nv20.12-blue'>
  <img src='https://img.shields.io/badge/numPy-1.17.3-brightgreen'>
  <img src='https://img.shields.io/badge/pandas-1.1.5-brightgreen'>
  <img src='https://img.shields.io/badge/scipy-1.7.3-brightgreen'>
</p> 

## **Overall description** 
- This repository contains the codebase for the LightGC²N project, which is associated with our research paper titled "Lightweight yet Fine-grained: A Graph Capsule Convolutional Network with Subspace Alignment for Shared-account Sequential Recommendation" accepted AAAI 2025. The datasets are also released at [ [https://bitbucket.org/jinyuz1996/lightgc2n_data/src/main/](https://bitbucket.org/jinyuz1996/lightgc2n_data/src/main/) ].
## **Code description** 
### **Implementation details**
We implemented LightGC²N with TensorFlow and accelerated the model training using an Intel® Xeon® Silver 4210 CPU @ 2.20GHz CPU and NVIDIA RTX 3090 (24G) GPU. The operating system is Ubuntu 22.04, the system memory is 126G, and the coding platform is Pycharm.

### **Vesion of packages**
The following versions of the programming language and libraries are used in this project:

1. python = 3.8.18
2. nvidia-tensorflow = 1.15.4+nv20.12
3. tensorboard = 1.15.4
4. scipy = 1.7.3
5. pandas = 1.1.5
6. numpy = 1.17.3
### **Source code of LightGC²N**
The source code for the LightGC²N is organized as follows:

1. Main components definition: Located in `LightGC2N/LightGC2N/Light_model.py`.
2. Parameter settings: Found in `LightGC2N/LightGC2N/Light_config.py`.
3. Training process: Detailed in `LightGC2N/LightGC2N/Light_train.py`.
4. Execution entry point: Provided by `LightGC2N/LightGC2N/Light_main.py`.


-- The `check_points` directory is designated for storing trained models.
Please note that the code is structured for clarity and ease of review. We encourage reviewers to examine the code thoroughly and provide constructive feedback.





