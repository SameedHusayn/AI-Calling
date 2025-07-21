"""Text-to-speech functionality with improved chunking"""
import tempfile
import os
import subprocess
import threading
import platform
import re
from config import TTS_RATE

class StreamingTTS:
    """A streaming TTS class that can speak text incrementally"""
    def __init__(self, rate=TTS_RATE):
        self.rate = rate
        self.system = platform.system()
        print(f"Initializing StreamingTTS on {self.system} platform")
        
        # For tracking the sentence-in-progress
        self.text_buffer = ""
        self.min_chunk_length = 4       # Much shorter minimum (was 10)
        self.max_buffer_size = 60       # Smaller max buffer size (was 150)
        self.max_wait_time = 0.5        # Max seconds to wait before speaking
        self.last_chunk_time = 0        # Timestamp of last chunk
        
        # Phrase boundary detection
        self.phrase_boundaries = [',', '.', '!', '?', ':', ';', '\n', ' - ', ' and ', ' but ', ' or ']
        
        # Semaphore to control concurrent TTS calls
        self.tts_semaphore = threading.Semaphore(1)
    
    def speak_chunk(self, text, final=False):
        """Speak a chunk of text, buffering until we have a good stopping point"""
        import time
        current_time = time.time()
        
        if not text and not final:
            return False
            
        # Add this text to buffer
        self.text_buffer += text
        
        # If buffer is empty, nothing to do
        if not self.text_buffer:
            return True
            
        # Calculate time since last chunk was spoken
        time_since_last_chunk = current_time - getattr(self, 'last_spoken_time', 0)
        
        # Check if we've reached a natural phrase boundary
        has_boundary = any(boundary in self.text_buffer for boundary in self.phrase_boundaries)
        
        # Only speak if:
        # 1. We've reached a phrase boundary and have enough text, or
        # 2. Buffer is getting large, or
        # 3. It's been too long since we last spoke, or
        # 4. This is the final chunk
        should_speak = (
            (has_boundary and len(self.text_buffer) > self.min_chunk_length) or
            len(self.text_buffer) >= self.max_buffer_size or
            (len(self.text_buffer) > self.min_chunk_length and time_since_last_chunk > self.max_wait_time) or
            final
        )
        
        if should_speak:
            # Find the best place to break the text
            break_point = self._find_best_break_point(self.text_buffer)
            
            # Split the buffer at the break point
            text_to_speak = self.text_buffer[:break_point].strip()
            self.text_buffer = self.text_buffer[break_point:].strip()
            
            if text_to_speak:
                # Launch in a thread to avoid blocking
                threading.Thread(
                    target=self._speak_text_now,
                    args=(text_to_speak,),
                    daemon=True
                ).start()
                
                # Update last spoken time
                self.last_spoken_time = time.time()
            
        return True
    
    def _find_best_break_point(self, text):
        """Find the best point to break the text for natural-sounding speech"""
        # If text is short, just speak all of it
        if len(text) <= self.min_chunk_length:
            return len(text)
            
        # Try to find phrase boundaries from right to left
        for boundary in self.phrase_boundaries:
            pos = text.rfind(boundary)
            if pos > self.min_chunk_length:
                # Include the boundary in the text to speak
                return pos + len(boundary)
        
        # If no good boundary, try to break at a word boundary
        words = re.findall(r'\S+\s*', text)
        if words:
            # Find a word boundary near the middle of the buffer
            target_length = min(len(text) // 2, self.max_buffer_size // 2)
            cumulative_length = 0
            for i, word in enumerate(words):
                cumulative_length += len(word)
                if cumulative_length >= target_length:
                    return cumulative_length
        
        # If all else fails, break at max_buffer_size or the end
        return min(self.max_buffer_size, len(text))
    
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