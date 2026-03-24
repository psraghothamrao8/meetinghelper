import os
import queue
import threading
from ui_components import MeetingHelperApp
from audio_engine import AudioEngine
from transcriber import TranscriberEngine
from llm_client import LlmClient
from file_processor import FileProcessor

def main():
    audio_engine = AudioEngine()
    transcriber = TranscriberEngine()
    llm_client = LlmClient()
    file_processor = FileProcessor(transcriber, llm_client)

    app = MeetingHelperApp(audio_engine, transcriber, llm_client, file_processor)
    app.start()

if __name__ == "__main__":
    main()
