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

Batch-size 12, Seq_len = 64 [12x64] warmup 200 with 1000 iterations

!

## Pytorch
![alt text](https://github.com/waldo1234567/inference_optimization/blob/main/screenshots/pytorch.png)

## Onnx
![alt text](https://github.com/waldo1234567/inference_optimization/blob/main/screenshots/onnx.png)

## TensorRt FP32
![alt text](https://github.com/waldo1234567/inference_optimization/blob/main/screenshots/trt_fp32.png)

## TensorRt INT8
![alt text](https://github.com/waldo1234567/inference_optimization/blob/main/screenshots/trt_int8.png)

## Accuracy Report (FP32 V INT8)
![alt text](https://github.com/waldo1234567/inference_optimization/blob/main/screenshots/accuracy.png)

I didnt use FP16, because of my gpu (gtx 1650 does not support TensorCores). I have built and tested FP16, but the performance are the same with FP32

### INT 8 Quantization

* Initially the accuracy loss was up to 18-20% from FP32 (I did not calibrate the engine)
* After that I tried again with 1000 samples calibration, and the accuracy loss went down to ~10%
* Finally i tried with bigger samples; 12000. This resulting the accuracy loss went down even further, up to ~5%




