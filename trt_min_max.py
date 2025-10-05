import numpy as np
import tensorrt as trt
from polygraphy.backend.trt import (
    Calibrator, NetworkFromOnnxPath, CreateConfig, EngineFromNetwork, SaveEngine, Profile
)
from polygraphy.backend.trt import util as trt_util
import os


ONNX_PATH = "distilbert_model.onnx"
OUT_ENGINE = "distilbert_int8_minmax.trt"
CALIB_CACHE = "distilbert_minmax.cache"
CALIB_NPY = "calibration_data.npy"

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

def data_loader():
    data = np.load(CALIB_NPY, allow_pickle=True)
    for batch in data:
        yield {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
        
calib = Calibrator(
    data_loader(),
    cache=CALIB_CACHE,
    BaseClass=trt.IInt8MinMaxCalibrator,
    algo=trt.CalibrationAlgoType.MINMAX_CALIBRATION
)

profile = Profile()
profile.add("input_ids", (1, 64), (12, 64), (20, 64))
profile.add("attention_mask", (1, 64), (12, 64), (20, 64))

config = CreateConfig()
config.int8 = True
config.calibrator = calib
config.profiles = [profile]

builder = trt_util.get_trt_logger()

network = NetworkFromOnnxPath(ONNX_PATH)
engine_from_network=EngineFromNetwork(network, config)
print("Starting engine build (this runs calibration)...")
try:
    engine = engine_from_network()
except Exception as e:
    engine = None
    print("Engine build raised an exception:", repr(e))

if engine is None:
    print("Engine build failed (engine is None).")
    if os.path.exists(CALIB_CACHE):
        print(f"Calibration cache exists: {CALIB_CACHE} ({os.path.getsize(CALIB_CACHE)} bytes)")
    else:
        print("Calibration cache was not created.")
    print("Check the above TensorRT / Polygraphy verbose logs for parsing/build errors.")
    raise SystemExit("Failed to build engine — see logs above.")

print("Engine built successfully. Saving engine...")
try:
    SaveEngine(engine, OUT_ENGINE)()  
    print("Saved engine:", OUT_ENGINE)
    print("Saved calibration cache:", CALIB_CACHE)
except Exception as e:
    print("Failed to save engine:", repr(e))
    raise