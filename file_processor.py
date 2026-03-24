try:
    import moviepy.editor as mp
except ImportError:
    # MoviePy v2.x compatibility
    from moviepy.video.io.VideoFileClip import VideoFileClip
    class mp:
        VideoFileClip = VideoFileClip
import pandas as pd
import os
import threading

class FileProcessor:
    def __init__(self, transcriber, llm_client):
        self.transcriber = transcriber
        self.llm_client = llm_client
        
    def process_media_file(self, file_path, output_dir="."):
        # Run in a separate thread so UI doesn't freeze
        threading.Thread(target=self._process_media_task, args=(file_path, output_dir), daemon=True).start()
        
    def _process_media_task(self, file_path, output_dir):
        print(f"Processing {file_path}...")
        try:
            # 1. Extract audio if it's a video file, or directly transcribe if it's audio
            ext = file_path.lower().split('.')[-1]
            if ext in ['mp4', 'avi', 'mkv', 'mov']:
                video = mp.VideoFileClip(file_path)
                audio_path = "temp_audio.wav"
                video.audio.write_audiofile(audio_path, logger=None, fps=16000)
                file_to_transcribe = audio_path
                video.close()
            else:
                file_to_transcribe = file_path
                
            # 2. Transcribe using faster-whisper natively
            print("Transcribing...")
            segments, info = self.transcriber.model.transcribe(
                file_to_transcribe, 
                task="translate",
                language=None,
                vad_filter=True
            )
            
            history = []
            for segment in segments:
                history.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text
                })
                print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
                
            # 3. Export to CSV
            base_name = os.path.basename(file_path).split('.')[0]
            csv_path = os.path.join(output_dir, f"{base_name}_transcript.csv")
            df = pd.DataFrame(history)
            df.to_csv(csv_path, index=False)
            print(f"Saved transcript to {csv_path}")
            
            # 4. Generate MOM
            print("Generating MOM...")
            full_text = " ".join([h["text"] for h in history])
            mom_text = self.llm_client.generate_mom(full_text)
            
            mom_path = os.path.join(output_dir, f"{base_name}_MOM.txt")
            with open(mom_path, "w", encoding="utf-8") as f:
                f.write(mom_text)
            print(f"Saved MOM to {mom_path}")
            
            # cleanup
            if 'temp_audio.wav' in file_to_transcribe and os.path.exists(file_to_transcribe):
                try:
                    os.remove(file_to_transcribe)
                except:
                    pass
                
            print(f"Finished processing {file_path}")
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            
    def export_to_csv(self, transcription_data, output_path="history.csv"):
        # For real-time meeting history
        df = pd.DataFrame(transcription_data, columns=["Transcription Text"])
        df.to_csv(output_path, index=False)
        return output_path
