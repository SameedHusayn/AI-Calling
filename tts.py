import tempfile
import os
import subprocess
import threading
import platform
from config import TTS_RATE, MIN_CHUNK_LENGTH, MAX_BUFFER_SIZE

class StreamingTTS:
    """A streaming TTS class that can speak text incrementally"""
    def __init__(self, rate=TTS_RATE):
        self.rate = rate
        self.system = platform.system()
        print(f"Initializing StreamingTTS on {self.system} platform")
        
        # For tracking the sentence-in-progress
        self.text_buffer = ""
        self.min_chunk_length = MIN_CHUNK_LENGTH
        self.max_buffer_size = MAX_BUFFER_SIZE
        
        # Semaphore to control concurrent TTS calls
        self.tts_semaphore = threading.Semaphore(1)
    
    def speak_chunk(self, text, final=False):
        """Speak a chunk of text, buffering until we have a good stopping point"""
        if not text and not final:
            return False
            
        # Add this text to buffer
        self.text_buffer += text
        
        # If buffer is empty, nothing to do
        if not self.text_buffer:
            return True
            
        # Only speak if we have enough text, hit a sentence boundary, or this is the final chunk
        should_speak = (
            len(self.text_buffer) >= self.max_buffer_size or
            final or
            (len(self.text_buffer) > self.min_chunk_length and 
             any(p in self.text_buffer for p in ['.', '!', '?', ';', ':', '\n']))
        )
        
        if should_speak:
            # Launch in a thread to avoid blocking
            threading.Thread(
                target=self._speak_text_now,
                args=(self.text_buffer,),
                daemon=True
            ).start()
            # Clear the buffer after speaking
            self.text_buffer = ""
            
        return True
    
    def _speak_text_now(self, text):
        """Immediately speak the given text"""
        if not text:
            return
        
        # Acquire semaphore to limit concurrent TTS operations
        with self.tts_semaphore:
            if self.system == "Windows":
                self._speak_windows(text)
    
    def _speak_windows(self, text):
        """Use PowerShell to speak on Windows with better character escaping"""
        try:
            # Create a temporary file to avoid command line escaping issues
            with tempfile.NamedTemporaryFile(suffix='.txt', delete=False, mode='w', encoding='utf-8') as f:
                f.write(text)
                temp_file = f.name
            
            # PowerShell command to read from file and speak
            cmd = f'powershell -Command "Add-Type -AssemblyName System.Speech; ' \
                  f'$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; ' \
                  f'$speak.Rate = {int((self.rate - 1) * 10)}; ' \
                  f'$speak.Speak([System.IO.File]::ReadAllText(\'{temp_file}\'));"'
            
            print(f"🔊 Speaking: \"{text}\"")
            subprocess.call(cmd, shell=True)
            
            # Clean up temp file
            os.unlink(temp_file)
            return True
        except Exception as e:
            print(f"Error with Windows TTS: {e}")
            return False