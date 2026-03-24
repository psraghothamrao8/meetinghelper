import pyaudiowpatch as pyaudio
import numpy as np
import time

p = pyaudio.PyAudio()

mic_device = p.get_default_input_device_info()
mic_rate = int(mic_device["defaultSampleRate"])
mic_channels = mic_device["maxInputChannels"]
print("Opening:", mic_device["name"], " at ", mic_rate, "Hz", mic_channels, "Channels")

buffer = []
def callback(in_data, frame_count, time_info, status):
    audio_data = np.frombuffer(in_data, dtype=np.float32)
    if mic_channels > 1:
        audio_data = np.mean(audio_data.reshape(-1, mic_channels), axis=1)
    buffer.append(audio_data)
    return (in_data, pyaudio.paContinue)

stream = p.open(
    format=pyaudio.paFloat32,
    channels=mic_channels,
    rate=mic_rate,
    input=True,
    input_device_index=mic_device["index"],
    stream_callback=callback
)

print("Listening for 3 seconds...")
time.sleep(3)
stream.stop_stream()
stream.close()
p.terminate()

if len(buffer) > 0:
    combined = np.concatenate(buffer)
    rms = np.sqrt(np.mean(combined**2))
    print(f"Captured {len(combined)} frames. RMS Volume: {rms:.5f}")
else:
    print("NO AUDIO CAPTURED (CALLBACK NEVER FIRED)")
