import torch
from faster_whisper import WhisperModel
import time

print("Checking CUDA...")
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Running on CPU.")

print("\nTesting Model Load Speeds...")
for size in ["tiny", "base", "small"]:
    t0 = time.time()
    try:
        model = WhisperModel(size, device="auto", compute_type="default")
        print(f"Loaded {size} in {time.time() - t0:.2f}s")
    except Exception as e:
         print(f"Err loading {size}: {e}")
