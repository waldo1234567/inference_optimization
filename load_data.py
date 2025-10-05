import numpy as np

def load_data():
    calib_data = np.load('calibration_data.npy', allow_pickle=True)
    
    for sample in calib_data:
        batch_input_ids = sample["input_ids"].astype(np.int64)
        batch_attention_mask = sample["attention_mask"].astype(np.int64)
        yield {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask
        }