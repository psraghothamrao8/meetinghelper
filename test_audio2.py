import pyaudiowpatch as pyaudio
import numpy as np

p = pyaudio.PyAudio()

print("Testing direct 16000Hz MME Open:")
try:
    stream = p.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=4000
    )
    print("SUCCESS: Windows resampled the 48000Hz hardware to 16000Hz native mono!")
    stream.close()
except Exception as e:
    print("FAILED:", e)

p.terminate()
