import unittest
import queue
import time
import numpy as np
from audio_engine import AudioEngine
from transcriber import TranscriberEngine
from llm_client import LlmClient

class TestMeetingHelper(unittest.TestCase):
    def test_audio_engine_init(self):
        engine = AudioEngine()
        self.assertFalse(engine.is_recording)
        self.assertIsNotNone(engine.p)

    def test_llm_client_fallback(self):
        client = LlmClient(endpoint="http://localhost:9999/api/generate") # Bad endpoint
        res = client.generate_mom("Hello test")
        self.assertTrue("Error connect" in res or "Error parsing" in res)

    def test_transcriber_init(self):
        # We test initialization loads correctly
        try:
            transcriber = TranscriberEngine(model_size="tiny") # Use tiny for fast test
            self.assertIsNotNone(transcriber.model)
        except Exception as e:
            self.fail(f"Transcriber loaded with error: {e}")

if __name__ == "__main__":
    unittest.main()
