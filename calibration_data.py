from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
import random

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

dataset = load_dataset("glue", "sst2", split="train")

target_n = 12000
label_to_indice={}

for i, ex in enumerate(dataset):
    label_to_indice.setdefault(int(ex["label"]), []).append(i)

chosen_idxs = []
for lbl, idxs in label_to_indice.items():
    k = target_n // len(label_to_indice)
    chosen_idxs += random.sample(idxs, min(k, len(idxs)))
    
subset = dataset.select(chosen_idxs)
batch_size=12

calibration_data = []

for i in range(0, len(subset), batch_size):
    batch = subset.select(range(i, min(i + batch_size, len(subset))))
    if len(batch) < batch_size:
        break
    input_ids_list = []
    attention_mask_list = []

    for sample in batch:
        inputs = tokenizer(
            sample['sentence'],
            padding='max_length',
            max_length=64,
            truncation=True,
            return_tensors="np"
        )
        input_ids_list.append(inputs['input_ids'])
        attention_mask_list.append(inputs['attention_mask'])

    batch_input_ids = np.vstack(input_ids_list).astype(np.int64)
    batch_attention_mask = np.vstack(attention_mask_list).astype(np.int64)

    calibration_data.append({
        'input_ids': batch_input_ids,
        'attention_mask': batch_attention_mask
    })

np.save('calibration_data.npy', calibration_data)
print(f"Saved {len(calibration_data)} calibration batches")