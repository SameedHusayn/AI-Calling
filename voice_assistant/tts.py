import tempfile
import os
import subprocess
import threading
import platform
import queue
from voice_assistant.config import TTS_RATE

class SeamlessTTS:
    """TTS class that eliminates pauses between chunks"""
    def __init__(self, rate=TTS_RATE):
        self.rate = rate
        self.system = platform.system()
        print(f"Initializing SeamlessTTS on {self.system} platform")
        
        # For collecting the complete response
        self.current_response = ""
        self.response_queue = queue.Queue()
        
        # Flag for completion
        self.response_complete = False
        
        # Start background processor thread
        self.processor_thread = threading.Thread(target=self._process_responses, daemon=True)
        self.processor_thread.start()
    
    def add_text(self, text, final=False):
        """Add text to the current response"""
        if text:
            self.current_response += text
            
        if final:
            # Mark that we're done with this response
            self.response_complete = True
            
            # Only queue non-empty responses
            if self.current_response:
                self.response_queue.put(self.current_response)
                print(f"Queued for speech: \"{self.current_response}\"")
                
            # Reset for next response
            self.current_response = ""
    
    def _process_responses(self):
        """Background thread that processes complete responses"""
        while True:
            try:
                # Wait for a complete response
                response = self.response_queue.get()
                
                # Speak the entire response at once
                if self.system == "Windows":
                    self._speak_windows(response)
                
                # Mark task as done
                self.response_queue.task_done()
                
            except Exception as e:
                print(f"Error in TTS processor thread: {e}")
    
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
            
            print(f"🔊 Speaking full response")
            subprocess.call(cmd, shell=True)
            
            # Clean up temp file
            os.unlink(temp_file)
            return True
        except Exception as e:
            print(f"Error with Windows TTS: {e}")
            return False