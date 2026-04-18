import whisper
import os
from typing import Tuple

class AudioProcessor:
    def __init__(self, model_size: str = "medium", device: str = "cpu"):
        print(f"[*] Loading Whisper '{model_size}' model on {device}...")
        self.model = whisper.load_model(model_size, device=device)
        print("[+] Whisper ready.")

    def transcribe(self, file_path: str) -> Tuple[str, str]:
        """
        Transcribe audio file and detect language.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
            
        print(f"[*] Transcribing {os.path.basename(file_path)}...")
        result = self.model.transcribe(file_path)
        text = result["text"].strip()
        lang = result["language"]
        
        print(f"[+] Detected language: {lang}")
        return text, lang

if __name__ == "__main__":
    pass
