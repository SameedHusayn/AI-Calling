"""Optimized speech-to-text functionality"""
import asyncio
import os
import numpy as np
import time
from faster_whisper import WhisperModel
from config import DEFAULT_WHISPER_MODEL, DEFAULT_LANGUAGE

class Transcriber:
    """Class for speech-to-text transcription using Whisper"""
    
    def __init__(self, model_size=DEFAULT_WHISPER_MODEL, language=DEFAULT_LANGUAGE):
        self.model_size = model_size
        self.language = language
        
        # Initialize the Whisper model directly
        print(f"Loading Whisper {model_size} model...")
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print("Model loaded successfully!")
        
        # Buffer for continuous audio processing
        self.audio_buffer = np.array([], dtype=np.float32)
        self.min_audio_length = 16000  # 1 second at 16kHz
        self.sample_rate = 16000
        
    async def process_audio_chunk(self, audio_data):
        """Add audio chunk to buffer and determine if we should transcribe"""
        # Append to buffer
        self.audio_buffer = np.append(self.audio_buffer, audio_data)
        
        # Only transcribe if we have enough audio
        if len(self.audio_buffer) >= self.min_audio_length:
            return await self.transcribe_buffer()
        return None
        
    async def transcribe_buffer(self):
        """Transcribe the current audio buffer directly in memory"""
        if len(self.audio_buffer) < self.min_audio_length:
            return None
            
        try:
            # Process the current buffer
            start_time = time.time()
            print(f"Transcribing {len(self.audio_buffer)/self.sample_rate:.1f}s of audio...")
            
            # Run transcription in a thread pool to avoid blocking
            # FIX: Correctly call transcribe with audio and options
            segments, _ = await asyncio.to_thread(
                self.model.transcribe, 
                audio=self.audio_buffer,  # Named parameter
                language=self.language,
                vad_filter=True
            )
            
            # Process the segments
            text = ""
            for segment in segments:
                text += segment.text + " "
            
            text = text.strip()
            if text:
                print(f"Transcription ({time.time() - start_time:.2f}s): {text}")
            
            # Clear the buffer after transcription
            self.audio_buffer = np.array([], dtype=np.float32)
            
            return text
            
        except Exception as e:
            print(f"Error transcribing buffer: {e}")
            import traceback
            traceback.print_exc()
            # Keep the buffer in case of error
            return None
            
    async def transcribe_file(self, file_path):
        """Legacy method to transcribe from file - for compatibility"""
        try:
            chunk_name = os.path.basename(file_path)
            print(f"Transcribing {chunk_name}...")
            
            # Run transcription in a thread pool
            segments, _ = await asyncio.to_thread(
                self.model.transcribe, 
                audio=file_path,  # Named parameter
                language=self.language,
                vad_filter=True
            )
            
            # Process the segments
            text = ""
            for segment in segments:
                text += segment.text + " "
            
            text = text.strip()
            if text:
                print(f"Transcription: {text}")
            
            # Return the transcription and the path for cleanup
            return text, file_path
            
        except Exception as e:
            print(f"Error transcribing file {file_path}: {e}")
            return "", file_path
            
    async def finalize(self):
        """Transcribe any remaining audio in the buffer"""
        if len(self.audio_buffer) > 0:
            return await self.transcribe_buffer()
        return None