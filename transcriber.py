from faster_whisper import WhisperModel
import queue
import threading

class TranscriberEngine:
    def __init__(self, model_size="small", device="cpu", compute_type="int8"):
        # Explicit CPU compute_type="int8" using "small" is required for robust foreign language translation (e.g., Korean)
        try:
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        except Exception:
             # Fallback if int8 fails
             self.model = WhisperModel(model_size, device="cpu", compute_type="default")
        self.is_running = False
        self.transcription_callback = None
        self.history = []
        
    def start(self, audio_queue, callback):
        self.is_running = True
        self.transcription_callback = callback
        threading.Thread(target=self._process_loop, args=(audio_queue,), daemon=True).start()
        
    def stop(self):
        self.is_running = False
        
    def _process_loop(self, audio_queue):
        while self.is_running:
            try:
                # To prevent 10-second delays, we aggressively CLEAR the queue if CPU is falling behind.
                # If there are >1 pending chunks, drop the old ones to catch up to live real-time audio.
                while audio_queue.qsize() > 1:
                    _ = audio_queue.get_nowait()
                
                # Get audio chunk (timeout to allow breaking the loop)
                audio_chunk = audio_queue.get(timeout=1.0)
                
                # task="translate" translates any speech to English
                # condition_on_previous_text=True is restored to give Whisper sentence context across chunks.
                # Hallucinations are prevented now because audio_engine strictly deletes silent frames.
                segments, info = self.model.transcribe(
                    audio_chunk, 
                    task="translate", 
                    language=None,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500),
                    condition_on_previous_text=True
                )
                
                valid_texts = []
                for segment in segments:
                    # Whisper hallucinates on static/typing and spits out captions.
                    # no_speech_prob > 0.4 means Whisper thinks it's likely just noise. Drop it!
                    if segment.no_speech_prob > 0.4:
                        continue
                    valid_texts.append(segment.text)
                
                text = " ".join(valid_texts).strip()
                if text and self.transcription_callback:
                    self.history.append(text)
                    self.transcription_callback(text)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Transcription error: {e}")

