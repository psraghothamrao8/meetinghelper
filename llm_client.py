import requests
import json

class LlmClient:
    def __init__(self, endpoint="http://localhost:11434/api/generate", model="llama3:latest"):
        self.endpoint = endpoint
        self.model = model
        
    def generate_mom(self, transcript_text):
        if not transcript_text.strip():
            return "No transcript available to generate MOM."
            
        prompt = f"""You are an executive assistant analyzing a meeting transcription.
Please create comprehensive Minutes of Meeting (MOM).
Include:
1. Executive Summary
2. Key Discussion Points
3. Action Items (if any identified)

Transcript:
{transcript_text}
"""
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(self.endpoint, json=payload, timeout=120)
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "Error parsing response.")
            else:
                return f"Error: Ollama API returned HTTP {response.status_code}"
        except Exception as e:
            return f"Error connecting to Ollama ({self.endpoint}): {e}\nEnsure Ollama is running locally and the 'llama3' model is pulled."
