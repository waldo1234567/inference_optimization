### GPU Inference Optimization

* This project implements a systematic approach to optimizing transformer inference on consumer GPU hardware. Starting with a baseline PyTorch implementation of DistilBERT for sentiment classification, I explored the complete optimization pipeline: ONNX export, TensorRT graph optimization, and INT8 quantization with calibration.

* The goal was to achieve 2-3x inference speedup while maintaining acceptable accuracy, demonstrating production-grade optimization techniques within the constraints of entry-level hardware (GTX 1650 4GB Memory).

### Prerequisites

* python 3.8 +
* Pytorch (With CUDA)
* Onnx & Onnxruntime-gpu
* Tensorrt (Match with CUDA version)
* Polygraph
* Sklearn
* PyCuda

### Performance Result

## Pytorch
![alt text](https://github.com/waldo1234567/inference_optimization/blob/main/screenshots/pytorch.png)
