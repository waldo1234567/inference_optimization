import time
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
session = ort.InferenceSession("distilbert_model.onnx", providers=['CUDAExecutionProvider'])

test_texts = [
    "This is a test sentence.",
    "This is a longer test sentence with more words to process.",
    "An even longer sentence that contains significantly more tokens to really test the performance characteristics.",
] * 4

def benchmark_onnx(session, texts, num_runs=100, iterations=10):
    inputs = tokenizer(texts, return_tensors="np", padding=True, truncation=True)
    print(f"Actual sequence length: {inputs['input_ids'].shape[1]}")
    onnx_inputs = {
        'input_ids': inputs['input_ids'].astype(np.int64),
        'attention_mask': inputs['attention_mask'].astype(np.int64)
    }
    
    for _ in range(iterations):
        _ = session.run(None, onnx_inputs)
        
    latencies = []
    for _ in range(num_runs):
        start = time.time()
        _ = session.run(None, onnx_inputs)
        latencies.append((time.time() - start) * 1000)  

    latencies = np.array(latencies)
    
    print(f"ONNX Runtime - Batch size: {len(texts)}")
    print(f"  Mean latency: {latencies.mean():.2f} ms")
    print(f"  P50: {np.percentile(latencies, 50):.2f} ms")
    print(f"  P95: {np.percentile(latencies, 95):.2f} ms")
    print(f"  Throughput: {1000 * len(texts) / latencies.mean():.2f} samples/sec")
    print(f"  Speedup vs PyTorch: {17.68 / latencies.mean():.2f}x")
    

benchmark_onnx(session, test_texts)  

