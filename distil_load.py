from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

model.to(device)
model.eval()

test_texts = [
    "This is a test sentence.",
    "This is a longer test sentence with more words to process.",
    "An even longer sentence that contains significantly more tokens to really test the performance characteristics."
]


def benchmark_model(model, texts, num_runs=100, iterations=10):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)  
    
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(**inputs)
        
    torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _ = model(**inputs)
            torch.cuda.synchronize()
            times.append(time.time() - start)
    
    times = np.array(times) * 1000  # convert to milliseconds
    
    print(f"PyTorch Baseline - Batch size: {len(texts)}")
    print(f"  Mean latency: {times.mean():.2f} ms")
    print(f"  Std latency: {times.std():.2f} ms")
    print(f"  P50: {np.percentile(times, 50):.2f} ms")
    print(f"  P95: {np.percentile(times, 95):.2f} ms")
    print(f"  P99: {np.percentile(times, 99):.2f} ms")
    print(f"  Throughput: {1000 * len(texts) / times.mean():.2f} samples/sec")

    
    return times.mean()


baseline_latency = benchmark_model(model, test_texts * 4) # Batch size of 12

