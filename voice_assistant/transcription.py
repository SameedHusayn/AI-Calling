import asyncio
import numpy as np
import time
import threading
from faster_whisper import WhisperModel
from voice_assistant.config import SAMPLE_RATE

# Global transcription model (loaded once)
_whisper_model = None
_model_lock = threading.Lock()

def get_whisper_model(model_size="large-v3", compute_type="int8"):
    """Get or create the Whisper model (singleton)"""
    global _whisper_model
    if _whisper_model is None:
        with _model_lock:
            if _whisper_model is None:
                print(f"Loading Whisper {model_size} model...")
                _whisper_model = WhisperModel(model_size, device="cpu", compute_type=compute_type)
                print(f"Model loaded successfully!")
    return _whisper_model

class FastTranscriber:
    """Ultra-optimized transcriber for lowest latency"""
    
    def __init__(self, model_size="large-v3", language="en"):
        self.model_size = model_size
        self.language = language
        self.sample_rate = SAMPLE_RATE
        
        # Pre-load the model during init
        self.model = get_whisper_model(model_size)
        
        # Audio buffer and queue for parallel processing
        self.audio_buffer = np.array([], dtype=np.float32)
        self.min_buffer_size = self.sample_rate * 1.5  # 1.5 seconds at 16kHz
        
        # For segmenting audio by VAD
        self.vad_threshold = 0.01
        self.vad_window = int(self.sample_rate * 0.03)  # 30ms window
        
        # Transcription results
        self.results = []
        self.result_queue = asyncio.Queue()
        
        # Processing state
        self.is_processing = False
        self._processor_task = None
        
    def start_processing(self):
        """Start background processing"""
        self.is_processing = True
        self._processor_task = asyncio.create_task(self._process_queue())
        
    async def stop_processing(self):
        """Stop background processing"""
        self.is_processing = False
        if self._processor_task:
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        
        # Process any remaining audio
        if len(self.audio_buffer) > 0:
            await self._transcribe_buffer()
            
    async def add_audio(self, audio_data):
        """Add audio to buffer and process if needed"""
        # Add to buffer
        self.audio_buffer = np.append(self.audio_buffer, audio_data)
        
        # Check if we have enough for processing
        if len(self.audio_buffer) >= self.min_buffer_size:
            # Process the buffer in the background
            await self._transcribe_buffer()
    
    async def _transcribe_buffer(self):
        """Transcribe current buffer without clearing it"""
        if len(self.audio_buffer) < self.min_buffer_size:
            return
            
        # Extract audio for processing (keep last 0.5s for context)
        buffer_to_process = self.audio_buffer.copy()
        overlap = int(self.sample_rate * 0.2)  # 0.5 seconds overlap
        self.audio_buffer = self.audio_buffer[-overlap:] if len(self.audio_buffer) > overlap else np.array([], dtype=np.float32)
        
        # Start transcription in a thread pool
        start_time = time.time()
        
        # Use a separate thread for model inference
        def _transcribe():
            try:
                segments, _ = self.model.transcribe(
                    audio=buffer_to_process,
                    language=self.language,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=1000)
                )
                
                # Collect all text
                text = ""
                for segment in segments:
                    text += segment.text + " "
                
                text = text.strip()
                return text
            except Exception as e:
                print(f"Transcription error: {e}")
                return ""
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(None, _transcribe)
        
        # Add result if we got text
        if text:
            duration = time.time() - start_time
            print(f"Transcription ({duration:.2f}s): {text}")
            self.results.append({
                "timestamp": time.time(),
                "text": text,
                "duration": duration
            })
            
            # Add to queue for real-time processing
            await self.result_queue.put(text)
    
    async def _process_queue(self):
        """Background task to process transcription results"""
        while self.is_processing:
            # Process any available audio
            if len(self.audio_buffer) >= self.min_buffer_size:
                await self._transcribe_buffer()
                
            # Short sleep to avoid CPU spinning
            await asyncio.sleep(0.01)
    
    def get_results(self):
        """Get all transcription results"""
        return self.results
        
    def get_transcribed_text(self):
        """Get the full transcribed text"""
        return " ".join([r["text"] for r in self.results])