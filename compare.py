
import argparse
import json
import os
import time
from typing import Dict, List, Tuple
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSequenceClassification, AutoModel, AutoTokenizer
import onnxruntime as ort
import torch

def load_trt_engine(engine_path: str):
    if trt is None:
        raise RuntimeError("trt package not available")
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
    with open(engine_path , 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")
        context = engine.create_execution_context()
        return engine, context


def trt_run(engine, context, inputs_np: Dict[str, np.ndarray]):
    bindings = []
    stream = cuda.Stream()
    device_mem = []
    host_mem = []
    output_names = []
    
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        if is_input:
            if name not in inputs_np:
                raise ValueError(f"Engine expects input named '{name}' but inputs dict does not contain it.")
            arr = np.ascontiguousarray(inputs_np[name])
            
            if -1 in tuple(engine.get_tensor_shape(name)):
                context.set_input_shape(name, tuple(arr.shape))
                
            nbytes = arr.nbytes
            dptr = cuda.mem_alloc(nbytes)
            device_mem.append(dptr)
            host_mem.append(arr)
            bindings.append(int(dptr))
        else:
            shape = context.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            out_nbytes = int(np.prod(shape) * np.dtype(dtype).itemsize)
            out_host = np.empty(shape, dtype=dtype)
            dptr = cuda.mem_alloc(out_nbytes)
            device_mem.append(dptr)
            host_mem.append(out_host)
            bindings.append(int(dptr))
            output_names.append(name)
            
    input_idx = 0
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            h_arr = host_mem[input_idx]
            cuda.memcpy_htod_async(device_mem[input_idx], h_arr, stream)
            input_idx += 1
    
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        d_ptr = device_mem[i]
        context.set_tensor_address(name, int(d_ptr))
                                   
    success = context.execute_async_v3(stream_handle = stream.handle)
    
    if not success:
        raise RuntimeError("Inference execution failed.")
    
    outputs ={}
    input_count = sum(1 for i in range(engine.num_io_tensors) if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT)
    for out_i, binding_i in enumerate(range(input_count, engine.num_io_tensors)):
        host_arr = host_mem[binding_i]
        cuda.memcpy_dtoh_async(host_arr, device_mem[binding_i], stream)
        outputs[engine.get_tensor_name(binding_i)] = host_arr.copy()
    stream.synchronize()
    return outputs


def run_pytorch_reference(model_name_or_path: str, tokenizer, inputs_list: List[Dict], task: str, device='cuda'):
    model = None
    if task == 'sequence-classification':
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path).to(device).eval()
    elif task in ('feature-extraction', 'token-classification', 'masked-lm'):
        model = AutoModel.from_pretrained(model_name_or_path).to(device).eval()
    else:
        raise ValueError("Unsupported task for PyTorch reference.")
    outputs = []
    with torch.no_grad():
        for batch in inputs_list:
            batch_t = {k: torch.tensor(v).to(device) for k, v in batch.items()}
            model_out = model(**batch_t)
            if task == 'sequence-classification':
                logits = model_out.logits.detach().cpu().numpy()
                outputs.append(logits)
            elif task == 'feature-extraction':
                if hasattr(model_out, 'pooler_output'):
                    emb = model_out.pooler_output.detach().cpu().numpy()
                else:
                    last = model_out.last_hidden_state.detach().cpu().numpy()
                    emb = last.mean(axis=1)
                outputs.append(emb)
            elif task == 'token-classification' or task == 'masked-lm':
                logits = model_out.last_hidden_state.detach().cpu().numpy()
                outputs.append(logits)
    return np.concatenate(outputs, axis=0)

def run_onnx_reference(onnx_path: str, inputs_list: List[Dict]):
    if ort is None:
        raise RuntimeError("onnxruntime not available.")
    sess = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    outputs = []
    for batch in inputs_list:
        ort_inputs = {}
        for k, v in batch.items():
            ort_inputs[k] = v
        out = sess.run(None, ort_inputs)
        if len(out) == 1:
            outputs.append(out[0])
        else:
            outputs.append(out[0])
    return np.concatenate(outputs, axis=0)


def classification_metrics(ref_logits: np.ndarray , test_logits: np.ndarray, topk=(1,3)):
    results={}
    ref_arg = np.argmax(ref_logits, axis=-1)
    test_arg = np.argmax(test_logits, axis=-1)
    results['accuracy'] = float((ref_arg == test_arg).mean())
    
    for k in topk:
        def topk_acc(a_logits, b_logits, k):
            a_topk = np.argsort(a_logits, axis=-1)[:, -k:]
            b_arg = np.argmax(b_logits, axis=-1)
            return float(np.mean([1 if b_arg[i] in a_topk[i] else 0 for i in range(len(b_arg))]))
        results[f'top{k}_ref_contains_test'] = topk_acc(ref_logits, test_logits, k)
        results[f'top{k}_test_contains_ref'] = topk_acc(test_logits, ref_logits, k)
        
    results.update(logit_difference(ref_logits, test_logits))
    return results

def logit_difference(a: np.ndarray , b : np.ndarray):
    print(f"a shape : {a.shape} \n")
    print(f"b shape: {b.shape}")
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"
    
    diff = a - b
    return{
        'max_abs_diff': float(np.max(np.abs(diff))),
        'mean_abs_diff': float(np.mean(np.abs(diff))),
        'rmse' : float(np.sqrt(np.mean(diff ** 2)))
    }
    
def embedding_metrics(ref_emb: np.ndarray, test_emb: np.ndarray):
    assert ref_emb.shape == test_emb.shape
    cos = np.diag(cosine_similarity(ref_emb, test_emb))
    return {
        'cosine_mean': float(np.mean(cos)),
        'cosine_median': float(np.median(cos)),
        'cosine_min': float(np.min(cos)),
        **logit_difference(ref_emb, test_emb)
    }


def batchify(inputs_np: Dict[str, np.ndarray], batch_size: int):
    N = list(inputs_np.values())[0].shape[0]
    batches = []
    for i in range(0, N , batch_size):
        batch = {k : v[i: i + batch_size] for k, v in inputs_np.items()}
        batches.append(batch)
        
    return batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', default='distilbert-base-uncased')
    parser.add_argument('--ref-hf', default=None, help='HF model name for PyTorch reference (e.g. distilbert-base-uncased)')
    parser.add_argument('--onnx-ref', default=None, help='ONNX reference path (optional)')
    parser.add_argument('--engines', nargs='+', required=True, help='Paths to TRT engines to test (fp32 fp16 int8).')
    parser.add_argument('--task', choices=['sequence-classification','token-classification','masked-lm','feature-extraction'], default='sequence-classification')
    parser.add_argument('--input-file', default=None, help='Plain text file (one sentence per line). If classification, separate label<tab>text per line.')
    parser.add_argument('--dataset', default=None, help='HuggingFace dataset identifier like glue/sst2 (optional)')
    parser.add_argument('--max-samples', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--max-length', type=int, default=60)
    parser.add_argument('--out', default='trt_accuracy_report.json')
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    texts = []
    labels = None
    ds = None
    ds_full= None
    
    
    if args.dataset:
        from datasets import load_dataset
        
        ds_id =args.dataset
        print(f"Loading HF dataset '{ds_id}' (will prefer 'validation' split)...")
        try:
            if '/' in ds_id:
                ds_main, ds_config = ds_id.split('/', 1)
                ds_full = load_dataset(ds_main, ds_config)
            else:
                ds_full = load_dataset(ds_id)
        except Exception:
            try:
                ds = load_dataset(ds_id, split="validation")
            except Exception as e:
                raise RuntimeError(f"Failed to load dataset '{ds_id}': {e}")
        
        if ds is None:
            if isinstance(ds_full, dict) or hasattr(ds_full, "keys"):
                for s in ("validation", "test", "train"):
                    if s in ds_full:
                        ds = ds_full[s]
                        break
                else:
                    first_split = list(ds_full.keys())[0]
                    ds = ds_full[first_split]
            else:
                ds = ds_full
                
        texts=[]
        labels=[]
        for item in ds:
            if len(texts) >= args.max_samples:
                break
            if "sentence" in item and isinstance(item["sentence"], str):
                texts.append(item["sentence"])
            elif "text" in item and isinstance(item["text"], str):
                texts.append(item["text"])
            elif "sentence1" in item and isinstance(item["sentence1"], str):
                texts.append(item["sentence1"])
            else:    
                picked =None
                for k, v in item.items():
                    if isinstance(v, str):
                        picked = v
                        break
                if picked is not None:
                    texts.append(picked)
                else:
                    continue
            
            if "label" in item:
                try:
                    labels.append(int(item["label"]))
                except Exception:
                    labels.append(-1)
            else:
                labels.append(-1)
    
        if len(texts) == 0:
            raise ValueError(f"No text fields found in dataset '{ds_id}'. Provide --input-file instead.")
        print(f"Loaded {len(texts)} samples from dataset '{ds_id}' (clipped to --max-samples={args.max_samples}).")
        
    elif args.input_file:
        lines = open(args.input_file, 'r', encoding='utf-8').read().strip().splitlines()
        if args.task == 'sequence-classification':
            # expect label \t sentence
            parsed = [l.split('\t',1) for l in lines if l.strip()]
            labels = [int(p[0]) for p in parsed]
            texts = [p[1] for p in parsed]
        else:
            texts = [l for l in lines if l.strip()]
    
    else:
        texts =[
            "This is a test sentence.",
            "The quick brown fox jumps over the lazy dog.",
            "I love machine learning and model optimization.",
            "TensorRT gives huge speedups when engineered properly.",
            "Quantization may change outputs slightly but often preserves labels."
            "Another Test for the input",
            "I swear this is the last one",
            "Wait wait there's more test",
            "Okay this is the last last last one",
            "Just Kidding im adding another test"
        ] 
    
    texts= texts[:args.max_samples]
    if labels is not None:
        labels = labels[:len(texts)]
        labels = np.array(labels, dtype=np.int64)
        
    enc = tokenizer(texts, padding='max_length', truncation=True, max_length=args.max_length, return_tensors='np')
    inputs_np={
        'input_ids': enc['input_ids'].astype(np.int64),
        'attention_mask' : enc['attention_mask'].astype(np.int64)
    }
    print(f"Prepared inputs: input_ids shape = {inputs_np['input_ids'].shape}, attention_mask shape = {inputs_np['attention_mask'].shape}")
    if labels is not None:
        print(f"Loaded labels: {labels.shape[0]} labels available (values sample: {labels[:10].tolist()})")

    batches=batchify(inputs_np, args.batch_size)
    
    print("Running reference model...")
    if args.onnx_ref:
        ref_out = run_onnx_reference(args.onnx_ref, batches)
    elif args.ref_hf:
        ref_out = run_pytorch_reference(args.ref_hf, tokenizer, batches, task=args.task, device='cuda' if torch.cuda.is_available() else 'cpu')
    else:
        raise ValueError("Provide either --ref-hf (HF model) or --onnx-ref (ONNX model) for reference.")
    
    
    summary = {'task' : args.task, 'num_samples': len(texts), 'engines':{}}
    
    for engine_path in args.engines:
        print(f"\nTesting engine: {engine_path}")
        engine, context = load_trt_engine(engine_path)
        
        tensor_info = [] 
        num_tensors = getattr(engine, "num_io_tensors", None) or getattr(engine, "num_tensors", None) or engine.num_bindings
        for i in range(num_tensors):
            try:
                tname = engine.get_tensor_name(i)
            except Exception:
                tname = engine.get_binding_name(i)
            try:
                tdtype = engine.get_tensor_dtype(tname)
            except Exception:
                tdtype = engine.get_binding_dtype(i)
            try:
                tmode = engine.get_tensor_mode(tname)
            except Exception:
                tmode = trt.TensorIOMode.INPUT if engine.binding_is_input(i) else trt.TensorIOMode.OUTPUT

            tensor_info.append({'name': tname, 'dtype': tdtype, 'mode': tmode, 'index': i})

        print("Engine tensor summary (name, dtype, mode, index):")
        for t in tensor_info:
            print(f"  {t['index']:2d}  name='{t['name']}' dtype={t['dtype']} mode={t['mode']}")

        name_map = {}

        input_tensors = [t for t in tensor_info if t['mode'] == trt.TensorIOMode.INPUT]
        def expected_dtype_for_key(key):
            if key in name_map:
                tname = name_map[key]
                return engine.get_tensor_dtype(tname)
            for t in input_tensors:
                if t['name'] == key:
                    return t['dtype']
            if len(input_tensors) > 0:
                return input_tensors[0]['dtype']
            return None

        engine_outputs = []
        for b in batches:   
            trt_inputs = {}

            for key, arr in b.items():
                expected_dtype = expected_dtype_for_key(key)
                if expected_dtype == trt.DataType.INT32:
                    arr_cast = arr.astype(np.int32)
                elif expected_dtype == trt.DataType.INT64:
                    arr_cast = arr.astype(np.int64)
                elif expected_dtype == trt.DataType.FLOAT:
                    arr_cast = arr.astype(np.float32)
                elif expected_dtype == trt.DataType.HALF:
                    arr_cast = arr.astype(np.float16)
                else:
                    arr_cast = arr
                trt_inputs[key] = np.ascontiguousarray(arr_cast)

            for key, arr in trt_inputs.items():
                if key in name_map:
                    tname = name_map[key]
                else:
                    match = next((t['name'] for t in input_tensors if t['name'] == key), None)
                    tname = match or (input_tensors[0]['name'] if input_tensors else None)

                if tname is None:
                    continue

                shape_tuple = tuple(arr.shape)
                if hasattr(context, "set_tensor_shape"):
                    try:
                        context.set_tensor_shape(tname, shape_tuple)
                    except Exception as e:
                        idx = next((t['index'] for t in tensor_info if t['name']==tname), None)
                        if idx is not None and hasattr(context, "set_binding_shape"):
                            context.set_binding_shape(idx, shape_tuple)
                else:
                    idx = next((t['index'] for t in tensor_info if t['name']==tname), None)
                    if idx is not None and hasattr(context, "set_binding_shape"):
                        context.set_binding_shape(idx, shape_tuple)
            
                out = trt_run(engine, context, trt_inputs)
            if 'logits' in out:
                out_arr = out['logits']
            else:
                ref_ndim = ref_out.ndim
                out_arr = None
                if ref_ndim == 2:
                    num_labels = ref_out.shape[-1]
                    for name, arr in out.items():
                        if 'logit' in name.lower() and arr.ndim >= 2 and arr.shape[-1] == num_labels:
                            out_arr = arr.reshape(arr.shape[0], -1)
                            break
               
                    if out_arr is None:
                        for name, arr in out.items():
                            if arr.ndim >= 2 and arr.shape[-1] == num_labels:
                                out_arr = arr.reshape(arr.shape[0], -1)
                                break
                if out_arr is None:
                    for name, arr in out.items():
                        if arr.shape == ref_out.shape:
                            out_arr = arr
                            break

                if out_arr is None:
                    print("Warning: couldn't find logits-like output. Engine outputs were:")
                    for name, arr in out.items():
                        print(f"  name='{name}' shape={arr.shape}")
                    out_arr = next(iter(out.values()))

            print(f"Selected engine output shape: {out_arr.shape}")
            
            engine_outputs.append(out_arr)
        engine_out_concat = np.concatenate(engine_outputs, axis=0)
        if args.task == 'sequence-classification':
            metrics = classification_metrics(ref_out, engine_out_concat, topk=(1,3))
        elif args.task == 'feature-extraction':
            metrics = embedding_metrics(ref_out, engine_out_concat)
        elif args.task in ('token-classification', 'masked-lm'):
            metrics = logit_difference(ref_out, engine_out_concat)
        else:
            metrics = logit_difference(ref_out, engine_out_concat)
        summary['engines'][os.path.basename(engine_path)] = metrics
        print("Metrics:", json.dumps(metrics, indent=2))
    
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved report to {args.out}")
    
if __name__ == '__main__':
    main()