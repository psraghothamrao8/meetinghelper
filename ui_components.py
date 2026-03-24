import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import os
from datetime import datetime
import threading

class CaptionsOverlay(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        
        self.overrideredirect(True)
        self.wm_attributes("-topmost", True)
        self.wm_attributes("-transparentcolor", "black")
        self.config(bg="black")
        
        # Position at the bottom of the screen
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        self.geometry(f"{int(screen_width*0.8)}x250+{int(screen_width*0.1)}+{screen_height - 300}")
        
        self.label = tk.Label(self, text="Waiting for audio...", font=("Arial", 28, "bold"), 
                              fg="yellow", bg="black", wraplength=int(screen_width*0.75), justify="center")
        self.label.pack(expand=True, fill="both")
        
        self.text_buffer = []

    def update_text(self, text):
        self.text_buffer.append(text)
        if len(self.text_buffer) > 3:  # Keep last 3 sentences
            self.text_buffer = self.text_buffer[-3:]
        display_text = "\n".join(self.text_buffer)
        self.label.config(text=display_text)

class MeetingHelperApp:
    def __init__(self, audio_engine, transcriber, llm_client, file_processor):
        self.audio_engine = audio_engine
        self.transcriber = transcriber
        self.llm_client = llm_client
        self.file_processor = file_processor
        
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")
        
        self.root = ctk.CTk()
        self.root.title("Meeting Helper Control Panel")
        self.root.geometry("400x350")
        
        self.overlay = None
        self.is_meeting_active = False
        self.build_ui()
        
    def build_ui(self):
        self.title_lbl = ctk.CTkLabel(self.root, text="Meeting Helper", font=("Arial", 24, "bold"))
        self.title_lbl.pack(pady=20)
        
        self.start_btn = ctk.CTkButton(self.root, text="Start Meeting", command=self.start_meeting)
        self.start_btn.pack(pady=10)
        
        self.stop_btn = ctk.CTkButton(self.root, text="Stop & Generate MOM", command=self.stop_meeting, state="disabled")
        self.stop_btn.pack(pady=10)
        
        self.file_btn = ctk.CTkButton(self.root, text="Process Media File", command=self.process_file)
        self.file_btn.pack(pady=10)
        
        self.status_lbl = ctk.CTkLabel(self.root, text="Ready", text_color="green")
        self.status_lbl.pack(pady=20)
        
    def update_status(self, text, color="white"):
        self.status_lbl.configure(text=text, text_color=color)
        
    def on_transcription(self, text):
        if self.overlay:
            self.overlay.update_text(text)
            
    def start_meeting(self):
        if not self.is_meeting_active:
            self.is_meeting_active = True
            
            # Reset history
            self.transcriber.history = []
            
            # Start Overlay
            self.overlay = CaptionsOverlay(self.root)
            
            # Start Transcriber
            self.transcriber.start(self.audio_engine.audio_queue, self.on_transcription)
            
            # Start Audio Engine
            self.audio_engine.start_recording()
            
            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
            self.file_btn.configure(state="disabled")
            self.update_status("Meeting in progress...", "green")
            
    def stop_meeting(self):
        if self.is_meeting_active:
            self.is_meeting_active = False
            self.update_status("Processing MOM... Please wait", "orange")
            self.root.update()
            
            # Stop Engines
            self.audio_engine.stop_recording()
            self.transcriber.stop()
            
            if self.overlay:
                self.overlay.destroy()
                self.overlay = None
                
            history = self.transcriber.history
            if history:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Export history
                csv_path = self.file_processor.export_to_csv(history, f"meeting_{timestamp}.csv")
                
                # Generate MOM in background thread
                threading.Thread(target=self._generate_mom_task, args=(history, timestamp), daemon=True).start()
            else:
                self.update_status("No audio captured.", "red")
                self.reset_ui()
                
    def _generate_mom_task(self, history, timestamp):
        full_text = " ".join(history)
        mom_text = self.llm_client.generate_mom(full_text)
        mom_path = f"meeting_{timestamp}_MOM.txt"
        with open(mom_path, "w", encoding="utf-8") as f:
            f.write(mom_text)
            
        # Update UI from main thread
        self.root.after(0, lambda: self.update_status(f"Saved to {mom_path}", "green"))
        self.root.after(0, self.reset_ui)
        
    def reset_ui(self):
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.file_btn.configure(state="normal")
            
    def process_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Media File",
            filetypes=(("Media Files", "*.mp4 *.avi *.mov *.mkv *.mp3 *.wav"), ("All Files", "*.*"))
        )
        if file_path:
            self.update_status("Processing file... (Check console)", "orange")
            self.file_processor.process_media_file(file_path)

    def start(self):
        self.root.mainloop()
