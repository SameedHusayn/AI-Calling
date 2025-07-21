"""Audio recording and processing"""
import numpy as np
import sounddevice as sd
import asyncio
import time
import os
from voice_assistant.config import SAMPLE_RATE, CHANNELS, SILENCE_THRESHOLD, SILENCE_DURATION
from voice_assistant.transcription import FastTranscriber

class AudioRecorder:
    """Class to handle audio recording and silence detection"""
    
    def __init__(self, silence_threshold=SILENCE_THRESHOLD, silence_duration=SILENCE_DURATION):
        self.sample_rate = SAMPLE_RATE
        self.channels = CHANNELS
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        
        self.is_recording = False
        self.frames = []
        self.last_sound_time = 0
        self.auto_stop = False
        self.stream = None
        
        # Tracking if speech has been detected
        self.speech_detected = False
        
        # For faster processing
        self.processing_task = None
        self.transcriber = None
    
    def audio_callback(self, indata, frames, time_info, status):
        """Store audio chunks and detect silence"""
        if status:
            print(f"Audio status: {status}")
            
        # Add to frames buffer for processing
        self.frames.append(indata.copy())
        
        # Calculate RMS amplitude
        rms = np.sqrt(np.mean(indata**2))
        
        # Check if there's sound in this chunk
        if self.auto_stop and self.is_recording:
            # If amplitude is above threshold, we've detected speech
            if rms > self.silence_threshold:
                if not self.speech_detected:
                    self.speech_detected = True
                    print("🎙️ Speech detected, listening...")
                
                # Update last sound time only if speech has been detected
                self.last_sound_time = time.time()
            else:
                # Only check for silence after speech has been detected
                if self.speech_detected:
                    silence_time = time.time() - self.last_sound_time
                    if silence_time >= self.silence_duration:
                        print(f"Detected {silence_time:.2f}s of silence. Auto-stopping recording.")
                        self.is_recording = False
    
    async def start_recording(self, auto_stop=True):
        """Start recording audio"""
        # Initialize the optimized transcriber
        self.transcriber = FastTranscriber(model_size="tiny.en")
        self.transcriber.start_processing()
        
        self.is_recording = True
        self.frames = []  # Clear any old frames
        self.auto_stop = auto_stop
        self.last_sound_time = time.time()  # Initialize with current time
        self.speech_detected = False  # Reset speech detection flag
        
        # Start audio recording
        print("Starting audio recording...")
        self.stream = sd.InputStream(
            callback=self.audio_callback,
            channels=self.channels,
            samplerate=self.sample_rate,
            dtype='float32',
            blocksize=int(self.sample_rate * 0.2)  # 200ms blocks for lower latency
        )
        self.stream.start()
        
        print(f"Recording started! Speak into your microphone...")
        print(f"Waiting for speech, then will auto-stop after {self.silence_duration} seconds of silence")
        
        # Start processing audio in the background
        self.processing_task = asyncio.create_task(self._process_audio_stream())
        
        # Check recording status in a loop
        await self._check_recording_status()
        
        # Cleanup
        self.stream.stop()
        self.stream.close()
        
        # Stop the transcriber
        await self.transcriber.stop_processing()
        
        # Cancel processing task
        if self.processing_task:
            self.processing_task.cancel()
    
    async def _check_recording_status(self):
        """Check if recording should be stopped"""
        while self.is_recording:
            await asyncio.sleep(0.1)
    
    async def _process_audio_stream(self):
        """Process audio frames in real-time for transcription"""
        while self.is_recording:
            # Get all new frames
            if self.frames:
                new_frames = self.frames.copy()
                self.frames = []
                
                # Process each frame
                for frame in new_frames:
                    # Get mono audio
                    if len(frame.shape) > 1 and frame.shape[1] > 1:
                        audio_data = frame[:, 0]  # First channel
                    else:
                        audio_data = frame.flatten()
                    
                    # Feed to transcriber
                    await self.transcriber.add_audio(audio_data)
            
            # Short sleep to avoid CPU spinning
            await asyncio.sleep(0.01)
    
    def get_transcription_results(self):
        """Get all transcription results"""
        if self.transcriber:
            return self.transcriber.get_results()
        return []
        
    def get_transcribed_text(self):
        """Get the full transcribed text"""
        if self.transcriber:
            return self.transcriber.get_transcribed_text()
        return ""