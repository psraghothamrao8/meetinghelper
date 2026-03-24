import pyaudiowpatch as pyaudio
import numpy as np
import queue
import threading
import time

class AudioEngine:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.mic_stream = None
        self.loopback_stream = None
        self.chunk_size = 1024
        self.rate = 16000 # Whisper needs 16kHz
        
        # Audio buffers for mixing
        self.mic_buffer = np.array([], dtype=np.float32)
        self.loopback_buffer = np.array([], dtype=np.float32)
        self.buffer_lock = threading.Lock()
        
    def get_default_wasapi_loopback(self):
        try:
            wasapi_info = self.p.get_host_api_info_by_type(pyaudio.paWASAPI)
            default_speakers = self.p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
            
            if not default_speakers["isLoopbackDevice"]:
                for loopback in self.p.get_loopback_device_info_generator():
                    if default_speakers["name"] in loopback["name"]:
                        return loopback
            else:
                return default_speakers
        except Exception:
            return None
        return None
        
    def get_default_mic(self):
        try:
            return self.p.get_default_input_device_info()
        except Exception:
            return None
        
    def start_recording(self):
        self.is_recording = True
        
        loopback_device = self.get_default_wasapi_loopback()
        
        # Start mic via implicit MME Windows layer for perfect native OS resampling back to 16000Hz mono.
        # This completely solves the "static robotic noise" aliasing caused by Python array subsetting.
        try:
            self.mic_stream = self.p.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._mic_callback
            )
        except Exception as e:
            print(f"Failed to open microphone: {e}")
            
        # Start loopback
        try:
            if loopback_device:
                self.loop_rate = int(loopback_device["defaultSampleRate"])
                self.loop_channels = loopback_device["maxInputChannels"]
                self.loopback_stream = self.p.open(
                    format=pyaudio.paFloat32,
                    channels=self.loop_channels,
                    rate=self.loop_rate,
                    input=True,
                    input_device_index=loopback_device["index"],
                    stream_callback=self._loopback_callback
                )
        except Exception as e:
            print(f"Failed to open loopback: {e}")

        # Start a thread to mix audio and queue it periodically
        threading.Thread(target=self._mix_audio_thread, daemon=True).start()

    def stop_recording(self):
        self.is_recording = False
        if self.mic_stream:
            self.mic_stream.stop_stream()
            self.mic_stream.close()
            self.mic_stream = None
        if self.loopback_stream:
            self.loopback_stream.stop_stream()
            self.loopback_stream.close()
            self.loopback_stream = None
            
    def _mic_callback(self, in_data, frame_count, time_info, status):
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        with self.buffer_lock:
            self.mic_buffer = np.append(self.mic_buffer, audio_data)
        return (in_data, pyaudio.paContinue)

    def _loopback_callback(self, in_data, frame_count, time_info, status):
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        # Convert to mono
        if self.loop_channels > 1:
            audio_data = np.mean(audio_data.reshape(-1, self.loop_channels), axis=1)
        
        # Resample to 16000
        if self.loop_rate != self.rate:
            indices = np.round(np.arange(0, len(audio_data), self.loop_rate / self.rate)).astype(int)
            indices = indices[indices < len(audio_data)]
            audio_data = audio_data[indices]
            
        with self.buffer_lock:
            self.loopback_buffer = np.append(self.loopback_buffer, audio_data)
        return (in_data, pyaudio.paContinue)
        
    def _mix_audio_thread(self):
        # 4-second continuous chunks (no overlap).
        # Ensures enough SOV context without enforcing a 5+ second rigid UI delay.
        chunk_sec = 4
        chunk_samples = self.rate * chunk_sec
        
        while self.is_recording:
            # Check frequently to reduce latency
            time.sleep(0.5) 
            with self.buffer_lock:
                len_mic = len(self.mic_buffer)
                len_loop = len(self.loopback_buffer)
                
                max_len = max(len_mic, len_loop)
                if max_len >= chunk_samples:
                    mic_chunk = np.zeros(chunk_samples, dtype=np.float32)
                    loop_chunk = np.zeros(chunk_samples, dtype=np.float32)
                    
                    if len_mic >= chunk_samples:
                        mic_chunk = self.mic_buffer[:chunk_samples]
                        self.mic_buffer = self.mic_buffer[chunk_samples:]
                        
                    if len_loop >= chunk_samples:
                        loop_chunk = self.loopback_buffer[:chunk_samples]
                        self.loopback_buffer = self.loopback_buffer[chunk_samples:]
                    
                    # Mix audio
                    mixed_chunk = mic_chunk + loop_chunk
                    
                    # Compute RMS on the block to see if we should transcribe
                    rms = np.sqrt(np.mean(mixed_chunk**2))
                    
                    # 0.005 threshold (lowered from 0.02) carefully ignores ~0.003 ambient silence 
                    # while successfully capturing naturally quiet hardware microphones.
                    if rms > 0.005:
                        self.audio_queue.put(mixed_chunk)
                    
                elif max_len < chunk_samples and not self.p: # safeguard breaking loop
                    break
