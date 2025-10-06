### GPU Inference Optimization

* This project implements a systematic approach to optimizing transformer inference on consumer GPU hardware. Starting with a baseline PyTorch implementation of DistilBERT for sentiment classification, I explored the complete optimization pipeline: ONNX export, TensorRT graph optimization, and INT8 quantization with calibration.

* The goal was to achieve 2-3x inference speedup while maintaining acceptable accuracy, demonstrating production-grade optimization techniques within the constraints of entry-level hardware (GTX 1650 4GB Memory).

### Prerequisites

* python 3.8 +
* Pytorch (With CUDA)
* Onnx & Onnxruntime-gpu
* Tensorrt (Match with CUDA version)
* Polygraphy
* Sklearn
* PyCuda

### Performance Result

All was tested with Batch-size 12, Seq_len = 64 (input_ids:[12x64] and attention_mask:[12x64]),  warmup 200 with 1000 iterations

![alt text](https://github.com/waldo1234567/inference_optimization/blob/main/screenshots/table.png)

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
* Out of desperation, i attempt to build the engine manually (using trt api & python) but the runtime was identical to fp32 and no accuracy loss, i suspect the engine wasnt actually quantized, due to its identical runtime and file size
* Finally i tried with bigger samples; 12000. This resulting the accuracy loss went down even further, up to ~5%
* I also tried using different calibration method, MinMaxCalibrator (on same dataset, 12000) but only achieved ~8% accuracy loss
  

### HardWare Limitations (GTX 1650)

* Memory Limits, gtx 1650 only has 4GB of VRAM. This prevents testing on larger model (e.g GPT2)
* No Tensor Cores, Thus dont support Mixed Precision

### Guide

## Pytorch(Baseline) Benchmark

```bash
    python distil_load.py

```

## Onnx Export and benchmark

```bash
    python export_distil_onnx.py # wait till it finsihed exporting

    python disitl_runtime.py
```

## Export Onnx to Trt (polygraphy)

FP32

```bash
    polygraphy convert distilbert_model.onnx -o distilbert_fp32.trt --trt-min-shapes input_ids:[1,64] attention_mask:[1,64] --trt-opt-shapes input_ids:[12,64] attention_mask:[12,64] --trt-max-shapes input_ids:[20,64] attention_mask:[20,64] --convert-to trt --verbose

```

INT 8

```bash
    polygraphy convert distilbert_model.onnx --int8 --data-loader-script load_data.py --calibration-cache disitlbert_calib.cache -o distilbert_int8.trt --trt-min-shapes input_ids:[1,64] attention_mask:[1,64] --trt-opt-shapes input_ids:[12,64] attention_mask:[12,64] --trt-max-shapes input_ids:[20,64] attention_mask:[20,64] --convert-to trt --verbose

```

## Benchmark Tensorrt Engine

FP32

```bash
    trtexec --loadEngine=distilbert_fp32.trt --shapes=input_ids:12x64,attention_mask:12x64 --iterations=1000 --warmUp=200 --useCudaGraph --useSpinWait --verbose

```

INT 8

```bash
    trtexec --loadEngine=distilbert_int8.trt --shapes=input_ids:12x64,attention_mask:12x64 --iterations=1000 --warmUp=200 --useCudaGraph --useSpinWait --verbose    

```

## Accuracy Score

```bash
    python compare.py --onnx-ref distilbert_model.onnx --engines distilbert_fp32.trt distilbert_int8.trt  --dataset glue/sst2 --max-samples 1000 --batch-size 12 --max-length 64

```

### What's Next

* if i have the budget, i will try to use cloud GPU (or just buy a better laptop)
* Use Mixed Precision for enhancing accuracy and speed up
* Benchmark on production-scale level



